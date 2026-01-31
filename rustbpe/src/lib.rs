use std::cmp::Ordering;

use std::collections::HashMap as StdHashMap;

use dary_heap::OctonaryHeap;
use pyo3::prelude::*;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use rayon::prelude::*;

use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{Read, Write},
    path::PathBuf,
};

// Default GPT-4 style regex pattern for splitting text
const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

const DELIMITERS: [&str; 7] = ["\n", "\t", "=", ":", ".", ",", " "];
// const DELIM_IDS: [&u32; 7] = [&10, &9, &61, &58, &46, &44, &32];
/// Open / close pairs that the tokenizer treats as a single token pair.
const PAIRS: &[(&str, &str)] = &[("(", ")"), ("[", "]"), ("{", "}")];

const DOWN_ID: u32 = 256;
const UP_ID: u32 = 257;

// Special tokens for conversation formatting
const BOS_TOKEN: &str = "<|bos|>";
const USER_START_TOKEN: &str = "<|user_start|>";
const USER_END_TOKEN: &str = "<|user_end|>";
const ASSISTANT_START_TOKEN: &str = "<|assistant_start|>";
const ASSISTANT_END_TOKEN: &str = "<|assistant_end|>";
const PYTHON_START_TOKEN: &str = "<|python_start|>";
const PYTHON_END_TOKEN: &str = "<|python_end|>";
const OUTPUT_START_TOKEN: &str = "<|output_start|>";
const OUTPUT_END_TOKEN: &str = "<|output_end|>";

type Pair = (u32, u32);

/// The actual tokenizer – exposed to Python.
#[pyclass]
pub struct Tokenizer {
    /// Maps pairs of token IDs to their merged token ID
    #[pyo3(get, set)]
    pub merges: StdHashMap<Pair, u32>,

    /// The regex pattern used for text splitting
    #[pyo3(get, set)]
    pub pattern: String,

    /// Size of the vocabulary
    #[pyo3(get, set)]
    pub n_vocab: u32,

    /// Cached token ID to bytes mapping for fast decoding
    /// Built after training or loading from disk
    token_bytes_cache: Option<Vec<Vec<u8>>>,
}

/// A helper struct that contains only the serialisable fields.
/// `compiled_pattern` is omitted because it can be rebuilt from `pattern`.
#[derive(Serialize, Deserialize)]
struct TokenizerData {
    merges: Vec<(Pair, u32)>,
    pattern: String,
    n_vocab: u32,
}

// ------------------------ internal helpers ------------------------

#[derive(Clone, Debug)]
struct Word {
    ids: Vec<u32>,
}

impl Word {
    #[inline]
    fn new(ids: Vec<u32>) -> Self {
        Self { ids }
    }

    #[inline]
    fn pairs<'a>(&'a self) -> impl Iterator<Item = Pair> + 'a {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    /// Merge all non-overlapping occurrences of pair -> new_id.
    /// Returns a small Vec of local pair-count deltas for THIS word only:
    ///   -1 for removed pairs, +1 for newly created pairs.
    ///
    /// NOTE: this version deliberately avoids a HashMap in the hot loop.
    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
        let (a, b) = pair;
        let n = self.ids.len();
        if n < 2 {
            return Vec::new();
        }

        let mut out: Vec<u32> = Vec::with_capacity(n);
        let mut deltas: Vec<(Pair, i32)> = Vec::with_capacity(6);

        let mut i = 0;
        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
                let left = out.last().copied();
                let right = if i + 2 < n {
                    Some(self.ids[i + 2])
                } else {
                    None
                };

                // remove old pairs
                if let Some(x) = left {
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1));
                if let Some(y) = right {
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }

                // write merged token
                out.push(new_id);
                i += 2; // skip 'a' and 'b'
            } else {
                out.push(self.ids[i]);
                i += 1;
            }
        }

        self.ids = out;
        deltas
    }
}

#[derive(Debug, Eq)]
struct MergeJob {
    pair: Pair,
    count: u64,
    /// set of word indices where this pair may occur and needs processing
    pos: AHashSet<usize>,
}

impl PartialEq for MergeJob {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for MergeJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by count; tie-break to ascending pair order (deterministic)
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // ascending order on the pair when counts tie
            other.pair.cmp(&self.pair)
        }
    }
}

#[inline]
fn count_pairs_parallel(
    words: &[Word],
    counts: &[i32],
) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
    words
        .par_iter()
        .enumerate()
        .map(|(i, w)| {
            let mut local_pc: AHashMap<Pair, i32> = AHashMap::new();
            let mut local_wtu: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            if w.ids.len() >= 2 && counts[i] != 0 {
                for (a, b) in w.pairs() {
                    *local_pc.entry((a, b)).or_default() += counts[i];
                    local_wtu.entry((a, b)).or_default().insert(i);
                }
            }
            (local_pc, local_wtu)
        })
        .reduce(
            || (AHashMap::new(), AHashMap::new()),
            |(mut acc_pc, mut acc_wtu), (pc, wtu)| {
                for (k, v) in pc {
                    *acc_pc.entry(k).or_default() += v;
                }
                for (k, s) in wtu {
                    acc_wtu.entry(k).or_default().extend(s);
                }
                (acc_pc, acc_wtu)
            },
        )
}

// ------------------------ END helpers ------------------------

impl Tokenizer {
    /// Return `(token, id)` if `text` starts with a known special token.
    fn leading_special_token(&self, text: &str) -> Option<(&'static str, u32)> {
        const SPECIALS: &[(&str, u32)] = &[
            (BOS_TOKEN, 0),
            (USER_START_TOKEN, 1),
            (USER_END_TOKEN, 2),
            (ASSISTANT_START_TOKEN, 3),
            (ASSISTANT_END_TOKEN, 4),
            (PYTHON_START_TOKEN, 5),
            (PYTHON_END_TOKEN, 6),
            (OUTPUT_START_TOKEN, 7),
            (OUTPUT_END_TOKEN, 8),
        ];

        for &(token, id) in SPECIALS {
            if text.starts_with(token) {
                return Some((token, id));
            }
        }
        None
    }

    /// Recursively split a string into the *leaf* words (no delimiters left).
    fn split_into_words(text: &str) -> Vec<CompactString> {
        let mut out = Vec::new();
        Self::split_recursive(text, 0, &mut out);
        out
    }

    fn split_recursive(text: &str, idx: usize, out: &mut Vec<CompactString>) {
        if idx == DELIMITERS.len() - 1 {
            // last delimiter is a space
            for w in text.split(' ') {
                if !w.is_empty() {
                    out.push(CompactString::from(w));
                }
            }
        } else {
            let delim = DELIMITERS[idx];
            if !text.contains(delim) {
                // nothing to split – go deeper
                Self::split_recursive(text, idx + 1, out);
            } else {
                for part in text.split(delim) {
                    // recurse on every sub‑segment
                    if !part.is_empty() {
                        Self::split_recursive(part, idx + 1, out);
                    }
                }
            }
        }
    }

    /// Build the token_bytes cache for fast decoding.
    /// This should be called after training or loading from disk.
    fn build_decode_cache(&mut self) {
        // Build vocabulary incrementally from low to high token IDs
        let mut token_bytes: Vec<Vec<u8>> = (0..256_u32).map(|i| vec![i as u8]).collect();

        // Add DOWN (↘) and UP (↗) markers
        token_bytes.push(vec![0xE2, 0x86, 0x98]); // DOWN – U+2198
        token_bytes.push(vec![0xE2, 0x86, 0x97]); // UP   – U+2197

        // Add special tokens
        let special_tokens = [
            BOS_TOKEN,
            USER_START_TOKEN,
            USER_END_TOKEN,
            ASSISTANT_START_TOKEN,
            ASSISTANT_END_TOKEN,
            PYTHON_START_TOKEN,
            PYTHON_END_TOKEN,
            OUTPUT_START_TOKEN,
            OUTPUT_END_TOKEN,
        ];

        for token in special_tokens.iter() {
            token_bytes.push(token.as_bytes().to_vec());
        }

        // Sort merges by token id (so we can reconstruct bytes progressively)
        let mut sorted_merges: Vec<_> = self.merges.iter().collect();
        sorted_merges.sort_by_key(|&(_, &token_id)| token_id);

        for (&pair, &merged_id) in sorted_merges {
            let (left, right) = pair;
            let mut merged_bytes = token_bytes[left as usize].clone();
            merged_bytes.extend(&token_bytes[right as usize]);

            if token_bytes.len() <= merged_id as usize {
                token_bytes.resize(merged_id as usize + 1, Vec::new());
            }
            token_bytes[merged_id as usize] = merged_bytes;
        }

        self.token_bytes_cache = Some(token_bytes);
    }

    /// Core incremental BPE training given unique words and their counts.
    /// `words`: one entry per unique chunk (Vec<u32> of token-ids/bytes).
    /// `counts`: same length as `words`, count per chunk.
    fn train_core_incremental(&mut self, mut words: Vec<Word>, counts: Vec<i32>, vocab_size: u32) {
        assert!(vocab_size > 256 + 2 + 9, "vocab_size must be at least 267");
        let num_merges = vocab_size - 256 - 2 - 9; // 256 base bytes + DOWN/UP
        log::info!("Starting BPE training: {} merges to compute", num_merges);
        self.merges.clear();

        // ---- Initial pair_counts and where_to_update (parallel) ----
        log::info!(
            "Computing initial pair counts from {} unique sequences",
            words.len()
        );
        let (mut pair_counts, mut where_to_update) = count_pairs_parallel(&words, &counts);

        // ---- Build heap ----
        log::info!("Building heap with {} unique pairs", pair_counts.len());
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, pos) in where_to_update.drain() {
            let c = *pair_counts.get(&pair).unwrap_or(&0);
            if c > 0 {
                heap.push(MergeJob {
                    pair,
                    count: c as u64,
                    pos,
                });
            }
        }

        // ---- Merge loop ----
        log::info!("Starting merge loop");
        let mut merges_done = 0u32;
        let mut last_log_percent = 0u32;

        while merges_done < num_merges {
            let Some(mut top) = heap.pop() else {
                break;
            };

            // Lazy refresh
            let current = *pair_counts.get(&top.pair).unwrap_or(&0);
            if top.count != current as u64 {
                top.count = current as u64;
                if top.count > 0 {
                    heap.push(top);
                }
                continue;
            }
            if top.count == 0 {
                break;
            }

            // Record merge
            let new_id = 256 + 2 + 9 + merges_done;
            self.merges.insert(top.pair, new_id);

            // Merge this pair in all words where it occurs
            let mut local_pos_updates: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            for &word_idx in &top.pos {
                // Apply merge to this word and collect pair-count deltas
                let changes = words[word_idx].merge_pair(top.pair, new_id);
                // Update global pair counts based on this word's count
                for (pair, delta) in changes {
                    let delta_total = delta * counts[word_idx];
                    if delta_total != 0 {
                        *pair_counts.entry(pair).or_default() += delta_total;
                        if delta > 0 {
                            local_pos_updates.entry(pair).or_default().insert(word_idx);
                        }
                    }
                }
            }

            // Add the updated pair counts back to the heap
            for (pair, pos) in local_pos_updates {
                let cnt = *pair_counts.get(&pair).unwrap_or(&0);
                if cnt > 0 {
                    heap.push(MergeJob {
                        pair,
                        count: cnt as u64,
                        pos,
                    });
                }
            }

            merges_done += 1;

            // Log progress every 1%
            let current_percent = (merges_done * 100) / num_merges;
            if current_percent > last_log_percent {
                log::info!(
                    "Progress: {}% ({}/{} merges) - Last merge: {:?} -> {} (frequency: {})",
                    current_percent,
                    merges_done,
                    num_merges,
                    top.pair,
                    new_id,
                    top.count
                );
                last_log_percent = current_percent;
            }
            self.n_vocab = new_id;
        }

        log::info!("Finished training: {} merges completed", merges_done);

        // Build decode cache for fast decoding
        self.build_decode_cache();
    }

    /// Encode a *single* word using the already‑trained BPE merges.
    /// Optimized to avoid O(n) remove() operations by building a new vector.
    fn encode_word(&self, word: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = word.bytes().map(|b| b as u32).collect();

        loop {
            if ids.len() < 2 {
                break;
            }

            // Find the best (lowest ID) merge pair
            let mut best_pair: Option<(usize, Pair, u32)> = None;
            for i in 0..ids.len() - 1 {
                let pair: Pair = (ids[i], ids[i + 1]);
                if let Some(&new_id) = self.merges.get(&pair) {
                    if best_pair.is_none() || new_id < best_pair.unwrap().2 {
                        best_pair = Some((i, pair, new_id));
                    }
                }
            }

            // If no merge found, we're done
            let Some((merge_idx, _pair, new_id)) = best_pair else {
                break;
            };

            // Build new vector with the merge applied (avoids O(n) shift from remove())
            let mut new_ids = Vec::with_capacity(ids.len() - 1);
            let mut i = 0;
            while i < ids.len() {
                if i == merge_idx {
                    new_ids.push(new_id);
                    i += 2; // skip both tokens of the pair
                } else {
                    new_ids.push(ids[i]);
                    i += 1;
                }
            }
            ids = new_ids;
        }

        ids
    }

    /// Flush the current buffer as a BPE‑encoded word surrounded by
    /// DOWN_ID / UP_ID.
    #[inline]
    fn flush_word(&self, out: &mut Vec<u32>, buf: &String) {
        if !buf.is_empty() {
            out.push(DOWN_ID);
            out.extend(self.encode_word(buf));
            out.push(UP_ID);
        }
    }

    /// Tokenise `text` while honouring paired delimiters.
    ///
    ///  * Opening delimiters are emitted as a token and pushed onto the stack.
    ///  * Closing delimiters pop from the stack – if they don’t match we
    ///    treat them as normal characters.
    ///  * All other delimiters (space, newline, `=`, `:`, etc.) are
    ///    emitted as raw bytes and trigger a flush of the current word.
    ///
    /// The output is a flat `Vec<u32>` that can be fed straight into the
    /// rest of the pipeline (no recursion needed).
    fn tokenize_with_stack(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();

        // 1️⃣ Handle a special token at the very beginning.
        if let Some((token, id)) = self.leading_special_token(text) {
            // 258 + id is the hard‑coded mapping used in the original code.
            out.push(258 + id);
            // Recurse on the remainder of the string.
            let remainder = &text[token.len()..];
            out.extend(self.tokenize_with_stack(remainder));
            return out;
        }

        // 2️⃣ Build maps for paired delimiters.
        let opener_to_closer: StdHashMap<char, char> = PAIRS
            .iter()
            .map(|&(o, c)| (o.chars().next().unwrap(), c.chars().next().unwrap()))
            .collect();
        let closer_to_opener: StdHashMap<char, char> = PAIRS
            .iter()
            .map(|&(o, c)| (c.chars().next().unwrap(), o.chars().next().unwrap()))
            .collect();

        // 3️⃣ Scan the string.
        let mut stack: Vec<char> = Vec::new(); // currently open delimiters
        let mut buf: String = String::new(); // current “word”

        for ch in text.chars() {
            if opener_to_closer.contains_key(&ch) {
                // Opening delimiter
                self.flush_word(&mut out, &buf);
                buf.clear();

                out.push(ch as u32); // emit the opening token
                stack.push(ch);
            } else if closer_to_opener.contains_key(&ch) {
                // Closing delimiter
                self.flush_word(&mut out, &buf);
                buf.clear();

                if let Some(&top) = stack.last() {
                    if opener_to_closer[&top] == ch {
                        out.push(ch as u32); // emit the closing token
                        stack.pop();
                    } else {
                        buf.push(ch); // mismatched – treat as text
                    }
                } else {
                    buf.push(ch); // no opener – treat as text
                }
            } else{
                if DELIMITERS.iter().any(|&d| d.contains(ch)) {
                    // Regular (single‑byte) delimiter
                    self.flush_word(&mut out, &buf);
                    buf.clear();
                    out.push(ch as u32);
                } else {
                    // Normal character
                    buf.push(ch);
                }
            }
        }

        // 4️⃣ Flush the last word, if any.
        self.flush_word(&mut out, &buf);

        // 5️⃣ Return the flattened token list.
        out
    }

    fn recursive_encode(&self, text: &str) -> Vec<u32> {
        // 1️⃣ – get the raw encoded vector
        let mut res = self.tokenize_with_stack(text);
//         let mut res = self.recursive_encode_with_delim(text, 0);

        //         // 2️⃣ – strip *every* leading DOWN_ID
        //         while res.first().map_or(false, |&x| x == DOWN_ID) {
        //             // `remove(0)` is O(n), but we only hit it once per element
        //             res.remove(0);
        //         }
        //
        // 3️⃣ – strip *every* trailing UP_ID
        while res.last().map_or(false, |&x| x == UP_ID) {
            // `pop()` is O(1)
            res.pop();
        }
        //         // 3️⃣ – strip *every* trailing DOWN_ID
        //         while res.last().map_or(false, |&x| x == DOWN_ID) {
        //             // `pop()` is O(1)
        //             res.pop();
        //         }

        res
    }
}

/// Public methods for the Tokenizer class that will be exposed to Python.
#[pymethods]
impl Tokenizer {
    #[new]
    pub fn new() -> Self {
        Self {
            merges: StdHashMap::new(),
            pattern: String::new(),
            n_vocab: 0,
            token_bytes_cache: None,
        }
    }

    #[getter]
    fn compiled_pattern_str(&self) -> PyResult<String> {
        Ok(self.pattern.clone())
    }

    pub fn save(&self, folder: &str) -> PyResult<()> {
        let mut path = PathBuf::from(folder);
        fs::create_dir_all(&path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Could not create directory `{}`: {}",
                path.display(),
                e
            ))
        })?;

        // Convert the map to a Vec before serialising.
        let data = TokenizerData {
            merges: self.merges.iter().map(|(k, v)| (*k, *v)).collect(),
            pattern: self.pattern.clone(),
            n_vocab: self.n_vocab + 1,
        };

        let json = serde_json::to_string_pretty(&data).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to serialise tokenizer: {}", e))
        })?;

        path.push("tokenizer.json");
        let mut file = fs::File::create(&path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Could not create file `{}`: {}",
                path.display(),
                e
            ))
        })?;
        file.write_all(json.as_bytes()).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!(
                "Could not write to file `{}`: {}",
                path.display(),
                e
            ))
        })?;

        Ok(())
    }

    #[staticmethod]
    pub fn load(folder: &str) -> PyResult<Self> {
        let mut path = PathBuf::from(folder);
        path.push("tokenizer.json");

        let mut json = String::new();
        fs::File::open(&path)
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Could not open file `{}`: {}",
                    path.display(),
                    e
                ))
            })?
            .read_to_string(&mut json)
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!(
                    "Could not read file `{}`: {}",
                    path.display(),
                    e
                ))
            })?;

        let data: TokenizerData = serde_json::from_str(&json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to deserialise tokenizer: {}",
                e
            ))
        })?;

        // Convert the Vec back into a HashMap.
        let merges = data.merges.into_iter().collect::<StdHashMap<_, _>>();

        let mut tokenizer = Tokenizer {
            merges,
            pattern: data.pattern,
            n_vocab: data.n_vocab,
            token_bytes_cache: None,
        };

        // Build decode cache for fast decoding
        tokenizer.build_decode_cache();

        Ok(tokenizer)
    }

    /// Train from a streaming iterator (parallel ingestion).
    /// We refill a Rust Vec<String> buffer under the GIL, then release the GIL
    /// to do the heavy splitting and counting **in parallel** with rayon.
    #[pyo3(signature = (iterator, vocab_size, buffer_size=8192, pattern=None))]
    #[pyo3(text_signature = "(self, iterator, vocab_size, buffer_size=8192, pattern=None)")]
    pub fn train_from_iterator(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        vocab_size: u32,
        buffer_size: usize,
        pattern: Option<String>,
    ) -> PyResult<()> {
        // Use provided pattern or default to GPT-4 pattern
        let pattern_str = pattern.unwrap_or_else(|| GPT4_PATTERN.to_string());

        // Update the stored pattern and compile it
        self.pattern = pattern_str.clone();

        // Prepare a true Python iterator object
        let py_iter: pyo3::Py<pyo3::PyAny> = unsafe {
            pyo3::Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
        };

        // Global chunk counts
        let mut counts: AHashMap<CompactString, i32> = AHashMap::new();

        // Temporary buffer we refill under the GIL
        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);

        log::info!(
            "Processing sequences from iterator (buffer_size: {})",
            buffer_size
        );
        let mut total_sequences = 0u64;

        // Helper: refill `buf` with up to `buffer_size` strings from the Python iterator.
        // Returns Ok(true) if the iterator is exhausted, Ok(false) otherwise.
        let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
            pyo3::Python::with_gil(|py| {
                buf.clear();
                let it = py_iter.bind(py);
                loop {
                    if buf.len() >= buffer_size {
                        return Ok(false);
                    }
                    // next(it)
                    let next_obj = unsafe {
                        pyo3::Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(it.as_ptr()))
                    };
                    match next_obj {
                        Some(obj) => {
                            let s: String = obj.extract()?;
                            buf.push(s);
                        }
                        None => {
                            if pyo3::PyErr::occurred(py) {
                                return Err(pyo3::PyErr::fetch(py));
                            } else {
                                return Ok(true); // exhausted
                            }
                        }
                    }
                }
            })
        };

        // Stream ingestion loop: refill under GIL, process without GIL (parallel)
        loop {
            let exhausted = refill(&mut buf)?;
            if buf.is_empty() && exhausted {
                break;
            }

            total_sequences += buf.len() as u64;

            let local: AHashMap<CompactString, i32> = py.allow_threads(|| {
                buf.par_iter()
                    .map(|s| {
                        let mut m: AHashMap<CompactString, i32> = AHashMap::new();
                        for w in Tokenizer::split_into_words(s) {
                            *m.entry(w).or_default() += 1;
                        }
                        m
                    })
                    .reduce(
                        || AHashMap::new(),
                        |mut a, b| {
                            for (k, v) in b {
                                *a.entry(k).or_default() += v;
                            }
                            a
                        },
                    )
            });

            // Merge local into global (single-threaded)
            for (k, v) in local {
                *counts.entry(k).or_default() += v;
            }

            if exhausted {
                break;
            }
        }
        log::info!(
            "Processed {} sequences total, {} unique",
            total_sequences,
            counts.len()
        );

        // Materialize words & counts
        let mut words = Vec::with_capacity(counts.len());
        let mut cvec = Vec::with_capacity(counts.len());
        for (chunk, c) in counts.into_iter() {
            words.push(Word::new(
                chunk.as_bytes().iter().map(|&b| b as u32).collect(),
            ));
            cvec.push(c);
        }

        self.train_core_incremental(words, cvec, vocab_size);
        Ok(())
    }

    /// Return the regex pattern
    pub fn get_pattern(&self) -> String {
        self.pattern.clone()
    }

    /// Return the mergeable ranks (token bytes -> token id / rank)
    pub fn get_mergeable_ranks(&self) -> Vec<(Vec<u8>, u32)> {
        let mut mergeable_ranks = Vec::new();

        // Build vocabulary incrementally from low to high token IDs
        let mut token_bytes: Vec<Vec<u8>> = (0..256_u32).map(|i| vec![i as u8]).collect();

        // reserve DOWN (↘) and UP (↗)
//         token_bytes.push(vec![3]);
//         token_bytes.push(vec![3]);
        token_bytes.push(vec![0xE2, 0x86, 0x98]); // DOWN – U+2198
        token_bytes.push(vec![0xE2, 0x86, 0x97]); // UP   – U+2197

        // Add special tokens
        let special_tokens = vec![
            BOS_TOKEN,
            USER_START_TOKEN,
            USER_END_TOKEN,
            ASSISTANT_START_TOKEN,
            ASSISTANT_END_TOKEN,
            PYTHON_START_TOKEN,
            PYTHON_END_TOKEN,
            OUTPUT_START_TOKEN,
            OUTPUT_END_TOKEN,
        ];

        for (i, token) in special_tokens.iter().enumerate() {
            let token_bytes = token.as_bytes().to_vec();
            mergeable_ranks.push((token_bytes, (256 + 2 + i) as u32));
        }

        for (i, bytes) in token_bytes.iter().enumerate() {
            mergeable_ranks.push((bytes.clone(), i as u32));
        }

        // Sort merges by token id (so we can reconstruct bytes progressively)
        let mut sorted_merges: Vec<_> = self.merges.iter().collect();
        sorted_merges.sort_by_key(|&(_, &token_id)| token_id);

        for (&pair, &merged_id) in sorted_merges {
            let (left, right) = pair;
            let mut merged_bytes = token_bytes[left as usize].clone();
            merged_bytes.extend(&token_bytes[right as usize]);

            if token_bytes.len() <= merged_id as usize {
                token_bytes.resize(merged_id as usize + 1, Vec::new());
            }
            token_bytes[merged_id as usize] = merged_bytes.clone();

            mergeable_ranks.push((merged_bytes, merged_id));
        }

        mergeable_ranks
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.recursive_encode(text)
    }

    pub fn decode(&self, token_ids: Vec<u32>) -> String {
        // Use cached token_bytes for fast lookup
        let token_bytes = if let Some(ref cache) = self.token_bytes_cache {
            cache
        } else {
            // Fallback: build on-the-fly if cache not available
            // This shouldn't happen in normal usage after training/loading
            let mergeable = self.get_mergeable_ranks();
            let mut temp_bytes: Vec<Vec<u8>> = vec![Vec::new(); mergeable.len()];
            for (bytes, id) in mergeable {
                temp_bytes[id as usize] = bytes;
            }
            // Can't return reference to local variable, so we need to handle this differently
            // For now, just panic with a helpful message
            panic!("Decode cache not built. Call build_decode_cache() after training/loading.");
        };

        let mut raw: Vec<u8> = Vec::new();
        for id in token_ids {
            // Skip the recursion markers – they carry no textual content.
            if id == DOWN_ID || id == UP_ID {
                continue;
            }
            // Safety: the id must exist because it was produced by this tokenizer.
            if (id as usize) < token_bytes.len() {
                raw.extend(&token_bytes[id as usize]);
            }
        }

        // Convert the concatenated bytes back to a UTF‑8 string.
        String::from_utf8(raw).unwrap_or_else(|_| "<invalid utf‑8>".to_string())
    }

    /// Get the vocabulary size
    pub fn get_vocab_size(&self) -> u32 {
        self.n_vocab
    }

    /// Get the special tokens
    pub fn get_special_tokens(&self) -> Vec<&'static str> {
        vec![
            BOS_TOKEN,
            USER_START_TOKEN,
            USER_END_TOKEN,
            ASSISTANT_START_TOKEN,
            ASSISTANT_END_TOKEN,
            PYTHON_START_TOKEN,
            PYTHON_END_TOKEN,
            OUTPUT_START_TOKEN,
            OUTPUT_END_TOKEN,
        ]
    }
}

#[pymodule]
fn rdt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init(); // forwards Rust `log` to Python's `logging`
    m.add_class::<Tokenizer>()?;
    Ok(())
}
