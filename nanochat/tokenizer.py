"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""

import os
import json
import copy
from functools import lru_cache

import torch

from nanochat.dataset import parquets_iter_batched, base_dir

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I haven't validated that this is actually a good idea, TODO.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        bos = self.encode_special("<|bos|>")
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
import pickle
import rdt
import tiktoken

class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        tokenizer = rdt.Tokenizer()
        assert vocab_size >= 267, f"vocab_size_no_special must be at least 267, got {vocab_size}"
        tokenizer.train_from_iterator(text_iterator, vocab_size)
        # 2) construct the associated tiktoken encoding for inference
        # pattern = tokenizer.get_pattern()
        # mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        # mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        # tokens_offset = len(mergeable_ranks)
        # special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        # enc = tiktoken.Encoding(
        #     name="rustbpe",
        #     pat_str=pattern,
        #     mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
        #     special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        # )
        return cls(tokenizer, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, '<|bos|>')

    def get_vocab_size(self):
        return self.enc.get_vocab_size()

    def get_special_tokens(self):
        return self.enc.get_special_tokens()

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        try:
            return self.enc.encode_special(text)
        except:
            return self.encode(text)[0]

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = [self.enc.encode(t) for t in text]
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        # pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        # with open(pickle_path, "wb") as f:
        #     pickle.dump(self.enc, f)
        self.enc.save(tokenizer_dir)
        print(f"Saved tokenizer encoding to {tokenizer_dir}")

    def render_conversation(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
    ) -> tuple[list[int], list[int]]:
        """
        Tokenise a single chat conversation.
    
        Parameters
        ----------
        messages : list[dict]
            The raw message objects from the dataset.  Each dict is expected to
            contain at least ``role`` (one of "system", "user", "assistant") and
            ``content`` (a string or a list of parts).  Optional keys
            ``tools``, ``tool_calls`` and ``name`` are ignored.
        max_tokens : int, default 2048
            Truncate the final token list to this length.
    
        Returns
        -------
        ids   : list[int]
            Token IDs of the rendered conversation.
        mask  : list[int]
            Binary mask (1 = supervised token, 0 = ignored).
        """
        # ------------------------------------------------------------------
        # 1️⃣  Initialise the output containers and helper
        # ------------------------------------------------------------------
        ids, mask = [], []
    
        def add_tokens(token_ids: list[int] | int, mask_val: int) -> None:
            """Append token IDs and corresponding mask values."""
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))
    
        # ------------------------------------------------------------------
        # 2️⃣  Handle an optional leading system message
        # ------------------------------------------------------------------
        if messages[0]["role"] == "system":
            # Merge the system message with the first user message
            messages = copy.deepcopy(messages)          # avoid mutating the caller
            sys_msg   = messages.pop(0)
            assert (
                messages[0]["role"] == "user"
            ), "System message must be followed by a user message"
    
            # Prepend system content to the next user message
            messages[0]["content"] = (
                sys_msg["content"] + "\n\n" + messages[0]["content"]
            )
    
        # At this point we have only user / assistant pairs
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"
    
        # ------------------------------------------------------------------
        # 3️⃣  Special tokens
        # ------------------------------------------------------------------
        bos = self.get_bos_token_id()
    
        user_start, user_end   = (
            self.encode_special("<|user_start|>"),
            self.encode_special("<|user_end|>"),
        )
        assistant_start, assistant_end = (
            self.encode_special("<|assistant_start|>"),
            self.encode_special("<|assistant_end|>"),
        )
        python_start, python_end = (
            self.encode_special("<|python_start|>"),
            self.encode_special("<|python_end|>"),
        )
        output_start, output_end = (
            self.encode_special("<|output_start|>"),
            self.encode_special("<|output_end|>"),
        )
    
        # ------------------------------------------------------------------
        # 4️⃣  Tokenise the conversation
        # ------------------------------------------------------------------
        add_tokens(bos, 0)                      # BOS is never supervised
    
        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
    
            # Sanity: user / assistant should alternate
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert (
                role == expected_role
            ), f"Message {i} is from {role} but should be from {expected_role}"
    
            if role == "user":
                # User messages are always plain strings
                assert isinstance(content, str), (
                    "User messages must be simple strings (got "
                    f"{type(content).__name__})"
                )
                add_tokens(user_start, 0)
                add_tokens(self.encode(content), 0)
                add_tokens(user_end, 0)
    
            elif role == "assistant":
                add_tokens(assistant_start, 0)
    
                if isinstance(content, str):
                    # Simple assistant reply
                    add_tokens(self.encode(content), 1)
    
                elif isinstance(content, list):
                    # Assistant reply with tool calls / parts
                    for part in content:
                        text = part["text"]
                        typ  = part["type"]
    
                        if typ == "text":
                            add_tokens(self.encode(text), 1)
    
                        elif typ == "python":
                            add_tokens(python_start, 1)
                            add_tokens(self.encode(text), 1)
                            add_tokens(python_end, 1)
    
                        elif typ == "python_output":
                            # These tokens are generated by the model at inference time
                            add_tokens(output_start, 0)
                            add_tokens(self.encode(text), 0)
                            add_tokens(output_end, 0)
    
                        else:
                            raise ValueError(f"Unknown part type: {typ}")
    
                else:
                    raise ValueError(
                        f"Unsupported assistant content type: {type(content).__name__}"
                    )
    
                add_tokens(assistant_end, 1)
    
        # ------------------------------------------------------------------
        # 5️⃣  Truncate to max_tokens
        # ------------------------------------------------------------------
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
    
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

# def get_tokenizer():
#     from nanochat.common import get_base_dir
#     base_dir = get_base_dir()
#     tokenizer_dir = os.path.join(base_dir, "tokenizer")
#     # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
#     return RustBPETokenizer.from_directory(tokenizer_dir)

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > 10000:
                doc_text = doc_text[:10000]
            nchars += len(doc_text)
            yield doc_text
            if nchars > int(1e8):
                return
def get_tokenizer():
    tokenizer_path = os.path.join(base_dir, "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = rdt.Tokenizer.load(tokenizer_path)
    else:
        text_iter = text_iterator()
        vocab_size = 1024
        # 1) train using rustbpe
        tokenizer = rdt.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iter, vocab_size_no_special, pattern=SPLIT_PATTERN)

    return RustBPETokenizer(tokenizer, "<|bos|>")

def get_token_bytes(device="cpu"):
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
    # return torch.Tensor([2]*1024).to(device).to(torch.long)
