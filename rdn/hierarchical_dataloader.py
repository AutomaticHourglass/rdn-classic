"""
Hierarchical Dataloader

Modified dataloaders that handle both tokens and hierarchical coordinates.
Based on the original dataloader but extended to track coordinates.

Coordinates are now calculated on GPU for maximum performance.
"""

import torch
from rdn.dataloader import _document_batches
from rdn.gpu_coordinates import create_coordinate_calculator


def tokenizing_distributed_data_loader_with_coords(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    n_coord_dim=8, bracket_pairs=None
):
    """
    Stream pretraining text with hierarchical coordinates calculated on GPU.

    Args:
        tokenizer: Base BPE tokenizer (RustBPETokenizer)
        n_coord_dim: Number of coordinate dimensions (default: 8)
        bracket_pairs: List of (open, close) token ID tuples for hierarchy detection

    Returns: (inputs, targets, coords_inputs, coords_targets, state_dict)
    where coords_inputs/targets are tensors of shape (B, T, n_coord_dim)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    needed_tokens = B * T + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()
    token_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    # Create GPU coordinate calculator
    vocab_size = tokenizer.get_vocab_size()
    coord_calculator = create_coordinate_calculator(
        n_dim=n_coord_dim,
        bracket_pairs=bracket_pairs,
        bos_token=bos_token,
        device=device,
        vocab_size=vocab_size
    )

    while True:
        # Accumulate enough tokens (coordinates calculated later on GPU)
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx, epoch) = next(batches)

            # Encode to tokens only (fast BPE)
            token_lists = tokenizer.encode(
                doc_batch,
                prepend=bos_token,
                num_threads=tokenizer_threads
            )

            # Accumulate tokens
            for tokens in token_lists:
                token_buffer.extend(tokens)

        # Take B*T+1 tokens
        tokens = token_buffer[:needed_tokens]
        token_buffer = token_buffer[B*T:]

        # Package into tensors
        use_cuda = device == "cuda"

        # Token tensors - send to GPU
        scratch_tokens = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)
        scratch_tokens = scratch_tokens.to(device=device, non_blocking=use_cuda)

        # Calculate coordinates on GPU
        coords_flat = coord_calculator(scratch_tokens)  # [T+1, n_coord_dim]

        # Split into inputs and targets
        inputs = scratch_tokens[:-1].view(B, T)
        targets = scratch_tokens[1:].view(B, T)
        coords_inputs = coords_flat[:-1].view(B, T, n_coord_dim)
        coords_targets = coords_flat[1:].view(B, T, n_coord_dim)

        yield inputs, targets, coords_inputs, coords_targets, {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}


def tokenizing_distributed_data_loader_with_coords_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000, n_coord_dim=8, bracket_pairs=None
):
    """
    BOS-aligned dataloader with Best-Fit Cropping and GPU coordinate calculation.

    Every row starts with BOS, and coordinates are calculated on GPU for maximum speed.

    Args:
        tokenizer: Base BPE tokenizer (RustBPETokenizer)
        n_coord_dim: Number of coordinate dimensions (default: 8)
        bracket_pairs: List of (open, close) token ID tuples for hierarchy detection
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()

    # Create GPU coordinate calculator
    vocab_size = tokenizer.get_vocab_size()
    coord_calculator = create_coordinate_calculator(
        n_dim=n_coord_dim,
        bracket_pairs=bracket_pairs,
        bos_token=bos_token,
        device=device,
        vocab_size=vocab_size
    )

    # Buffer of token lists (coordinates calculated later on GPU)
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def fetch_docs():
        """Fetch and tokenize a batch of documents."""
        nonlocal doc_buffer, pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)

        # Encode to tokens only (fast BPE)
        token_lists = tokenizer.encode(
            doc_batch,
            prepend=bos_token,
            num_threads=tokenizer_threads
        )

        # Add to buffer
        for tokens in token_lists:
            doc_buffer.append(tokens)

    def build_row():
        """Build a row using best-fit algorithm."""
        row_tokens = []
        remaining = row_capacity

        while remaining > 0:
            # Ensure we have docs in buffer
            if not doc_buffer:
                fetch_docs()
                if not doc_buffer:
                    # Still empty - shouldn't happen but handle gracefully
                    break

            # Find largest doc that fits
            best_idx = None
            best_size = 0
            for i, tokens in enumerate(doc_buffer):
                size = len(tokens)
                if size <= remaining and size > best_size:
                    best_idx = i
                    best_size = size

            if best_idx is not None:
                # Use this doc
                tokens = doc_buffer.pop(best_idx)
                row_tokens.extend(tokens)
                remaining -= len(tokens)
            else:
                # No doc fits - crop one to fill exactly
                tokens = doc_buffer.pop(0)
                row_tokens.extend(tokens[:remaining])
                remaining = 0

        # Pad to row_capacity if needed (shouldn't happen with proper best-fit)
        if len(row_tokens) < row_capacity:
            pad_len = row_capacity - len(row_tokens)
            row_tokens.extend([0] * pad_len)

        return row_tokens

    # Main loop
    use_cuda = device == "cuda"

    while True:
        # Ensure buffer has enough docs
        while len(doc_buffer) < buffer_size:
            fetch_docs()

        # Build B rows (each row is guaranteed to be row_capacity length)
        batch_tokens = []

        for i in range(B):
            row_tokens = build_row()
            assert len(row_tokens) == row_capacity, f"Row {i}: tokens length {len(row_tokens)} != {row_capacity}"
            batch_tokens.append(row_tokens)

        # Convert to tensor and send to GPU
        tokens_tensor = torch.tensor(batch_tokens, dtype=torch.long, pin_memory=use_cuda)
        tokens_tensor = tokens_tensor.to(device=device, non_blocking=use_cuda)  # [B, T+1]

        # Calculate coordinates on GPU in one batch operation
        coords_tensor = coord_calculator(tokens_tensor)  # [B, T+1, n_coord_dim]

        # Split into inputs and targets
        inputs = tokens_tensor[:, :-1]  # [B, T]
        targets = tokens_tensor[:, 1:]  # [B, T]
        coords_inputs = coords_tensor[:, :-1]  # [B, T, n_coord_dim]
        coords_targets = coords_tensor[:, 1:]  # [B, T, n_coord_dim]

        yield inputs, targets, coords_inputs, coords_targets, {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
