from collections import deque

import torch
import pyarrow.parquet as pq

from generator.unified import generate_random_problem
from rdn.common import get_dist_info
from rdn.dataset import list_parquet_files, parquets_iter_batched
from rdn.tokenizer import get_tokenizer

from datasets import load_dataset, load_from_disk

# The dataset comes pre‑split into the keys you request.  At least
# a `train` split is guaranteed; if you need a validation set look
# for the key that contains it (often `test` or `validation`).
# DS = load_dataset("danilopeixoto/pandora-tool-calling")
# DS['val']=DS['validation']
# DS = load_from_disk('generator/calc')
# DS['val'] = DS['test']

# ------------------------------------------------------------------
# 2️⃣  Utility – list the available splits (useful for debugging)
# ------------------------------------------------------------------
def list_splits() -> list[str]:
    """
    Return a list of split names that are available in the loaded dataset.
    """
    return list(DS.keys())


def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None, use_generator=True, use_recursive_markers=True):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    def document_batches(
            split: str = "train",
            batch_size: int = 1024,
            resume_state_dict: dict | None = None,
    ):
        ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

        if use_generator:
            # Generator-based mode (calculator/tictactoe agents)
            while True:  # Infinite epoch loop
                batch_indices = []
                for i in range(batch_size):
                    global_pos = ddp_rank + (i * ddp_world_size)
                    batch_indices.append(global_pos)

                if not batch_indices:
                    break

                batch = [generate_random_problem() for _ in batch_indices]
                e_texts = [b['prompt'] for b in batch]
                d_texts = [b['json'] for b in batch]

                e_texts = [str(e) if isinstance(e, dict) else e for e in e_texts]
                d_texts = [str(d) if isinstance(d, dict) else d for d in d_texts]

                yield e_texts, d_texts
        else:
            # Parquet-based mode (normal pretraining on text)
            for batch_texts in parquets_iter_batched(split, start=ddp_rank, step=ddp_world_size):
                # For text pretraining, we just have one stream of text (no prompt/response split)
                # So we return empty e_texts and put everything in d_texts
                yield [], batch_texts

    batches = document_batches(split)

    # Now emit batches of tokens.
    needed_tokens = B * (T + 1) # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer(use_recursive_markers=use_recursive_markers)
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    t_buffer = deque() # we stream tokens on the right and pop from the left
    # e_buffer = deque()  # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(t_buffer) < needed_tokens:
            e_batch, d_batch = next(batches)

            if use_generator:
                # Generator mode: tokenize prompts and responses separately
                e_lists = tokenizer.encode(e_batch, prepend=bos_token, num_threads=tokenizer_threads)
                d_lists = tokenizer.encode(d_batch, num_threads=tokenizer_threads)
                for d_tokens, e_tokens in zip(d_lists, e_lists):
                    t_buffer.extend(e_tokens)
                    t_buffer.extend(d_tokens)
            else:
                # Parquet mode: just tokenize text documents
                d_lists = tokenizer.encode(d_batch, prepend=bos_token, num_threads=tokenizer_threads)
                for tokens in d_lists:
                    t_buffer.extend(tokens)



        # Move tokens from the deque into the scratch buffer
        tokens = []
        for i in range(needed_tokens):
            tokens += [t_buffer.popleft()]
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        use_cuda_optimizations = device == "cuda"
        all_cpu = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        # Create the inputs/targets as 1D tensors
        # inputs_cpu = scratch[:-1]
        # targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = all_cpu.view(B, T + 1)[:,:-1].to(device=device, non_blocking=use_cuda_optimizations)
        targets = all_cpu.view(B, T + 1)[:,1:].to(device=device, non_blocking=use_cuda_optimizations)
        yield inputs, targets

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
