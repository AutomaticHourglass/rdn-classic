"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

# def parquets_iter_batched(split, start=0, step=1):
#     """
#     Iterate through the dataset, in batches of underlying row_groups for efficiency.
#     - split can be "train" or "val". the last parquet file will be val.
#     - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
#     """
#     assert split in ["train", "val"], "split must be 'train' or 'val'"
#     parquet_paths = list_parquet_files()
#     parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
#     for filepath in parquet_paths:
#         pf = pq.ParquetFile(filepath)
#         for rg_idx in range(start, pf.num_row_groups, step):
#             rg = pf.read_row_group(rg_idx)
#             texts = rg.column('text').to_pylist()
#             yield texts

from datasets import load_dataset, load_from_disk

# The dataset comes pre‑split into the keys you request.  At least
# a `train` split is guaranteed; if you need a validation set look
# for the key that contains it (often `test` or `validation`).
# DS = load_dataset("danilopeixoto/pandora-tool-calling")
# DS['val']=DS['validation']
DS = load_from_disk('generator/calc')
DS['val'] = DS['test']

# ------------------------------------------------------------------
# 2️⃣  Utility – list the available splits (useful for debugging)
# ------------------------------------------------------------------
def list_splits() -> list[str]:
    """
    Return a list of split names that are available in the loaded dataset.
    """
    return list(DS.keys())


# ------------------------------------------------------------------
# 3️⃣  Iteration helper – yields batches of the ``messages`` column
# ------------------------------------------------------------------
def parquets_iter_batched(
    split: str = "train",
    *,
    start: int = 0,
    step: int = 1,
    batch_size: int | None = None,
):
    """
    Iterate over a HuggingFace `datasets` split in batches, mirroring the
    behaviour of your original ``parquets_iter_batched`` helper.

    Parameters
    ----------
    split : str, optional
        Which split to iterate over.  Must be a key in ``DS``.
    start : int, optional
        Index of the first row to include (used for DDP sharding).
    step : int, optional
        Stride between rows.  For DDP this is typically the world size.
    batch_size : int | None, optional
        Number of rows to return in each yielded list.  If ``None`` the
        helper will yield one row at a time (i.e. `batch_size = 1`).

    Yields
    ------
    list[str]
        A list of the ``messages`` field from the dataset, with length
        equal to ``batch_size`` (or 1 if ``batch_size is None``).

    Notes
    -----
    * The helper uses the *exact* order that ``datasets`` returns – no
      shuffling is performed.  If you need random access or a shuffled
      iterator, call ``DS[split].shuffle(...)`` before passing the split.
    * If you are using DDP, set ``start=rank`` and ``step=world_size``.
    * For a validation split you can pass whatever key the dataset
      exposes – commonly ``"test"``, but inspect ``list_splits()`` to be
      sure.
    """
    # ------------------------------------------------------------------
    # Validate the requested split
    # ------------------------------------------------------------------
    if split not in DS:
        raise ValueError(
            f"Requested split `{split}` is not available. "
            f"Available splits: {list_splits()}"
        )

    # ------------------------------------------------------------------
    # Grab the split (it may be a `Dataset` or an `IterableDataset`)
    # ------------------------------------------------------------------
    dataset = DS[split]

    # ------------------------------------------------------------------
    # Compute the indices that will be iterated over.
    # ------------------------------------------------------------------
    total = len(dataset)
    if step <= 0:
        raise ValueError("`step` must be a positive integer")
    indices = list(range(start, total, step))

    # ------------------------------------------------------------------
    # Decide on a sensible default batch size
    # ------------------------------------------------------------------
    if batch_size is None:
        batch_size = 1

    # ------------------------------------------------------------------
    # Yield in chunks
    # ------------------------------------------------------------------
    for i in range(0, len(indices), batch_size):
        chunk_indices = indices[i : i + batch_size]
        # Pull the ``messages`` field from each example
        texts = [generate_random_problem() for j in chunk_indices]
        yield texts

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
