# ------------------------------------------------------------------
#  NumPy/Numba implementation – same public API as compute_coords()
# ------------------------------------------------------------------
import numba
import numpy as np
import torch

# constants (copy from your original file)
NUM_DIMENSIONS = 8
DOWN = 256  # token ID that means “move pointer forward”
UP = 257  # token ID that means “move pointer backward”


@numba.njit(parallel=False, fastmath=True)
def _coords_numba(
    tokens: np.ndarray, num_dims: int, down_token: int, up_token: int
) -> np.ndarray:
    """
    tokens : (B, T)  dtype=int64
    returns: (B, T, num_dims)  dtype=int64
    """
    B, T = tokens.shape

    # result buffer (will be cast to torch later if desired)
    out = np.zeros((B, T, num_dims), dtype=np.int64)

    # per‑batch state
    coords = np.zeros((B, num_dims), dtype=np.int64)
    dim_ptrs = np.full(B, num_dims - 1, dtype=np.int64)

    for t in range(T):  # sequential over time
        # --- update state for every batch element -----------------
        for b in numba.prange(B):  # parallel over the batch
            tok = tokens[b, t]

            if tok == down_token:  # DOWN
                new_dim = (dim_ptrs[b] + 1) % num_dims
                dim_ptrs[b] = new_dim
                coords[b, new_dim] += 1

            elif tok == up_token:  # UP
                old = dim_ptrs[b]
                new_dim = num_dims - 1 if dim_ptrs[b] == 0 else dim_ptrs[b] - 1
                coords[b, old] = 0  # clear the old coordinate
                dim_ptrs[b] = new_dim
                coords[b, new_dim] += 1

            else:
                coords[b, dim_ptrs[b]] += 1

        # --- write the state after this time step ----------------
        for b in range(B):
            out[b, t, :] = coords[b]

    return out


def compute_coords(
    tokens,
    num_dims: int = NUM_DIMENSIONS,
    down_token: int = DOWN,
    up_token: int = UP,
) -> torch.Tensor | np.ndarray:
    """
    Parameters
    ----------
    tokens : list[Sequence[int]]  or torch.Tensor (B, T)
        If a list of sequences is given the function pads them to equal length.

    Returns
    -------
    np.ndarray or torch.Tensor
        Shape (B, T, num_dims).  If the input was a torch tensor,
        the output will be a torch tensor on the same device.
    """
    # ----- 1. convert to padded NumPy array --------------------------------
    if isinstance(tokens, torch.Tensor):
        # already a tensor – just move to CPU and convert
        tokens_np = tokens.cpu().numpy()
    else:  # assume list / iterable of sequences
        B = len(tokens)
        T_max = max(len(seq) for seq in tokens)
        tokens_np = np.zeros((B, T_max), dtype=np.int64)

        for i, seq in enumerate(tokens):
            tokens_np[i, : len(seq)] = np.asarray(seq, dtype=np.int64)

    # ----- 2. run the Numba kernel ----------------------------------------
    out_np = _coords_numba(tokens_np, num_dims, down_token, up_token)

    # ----- 3. return in the same type as the input ------------------------
    if isinstance(tokens, torch.Tensor):
        # keep device / dtype
        out_torch = torch.from_numpy(out_np).to(tokens.device)
        return out_torch
    else:
        return out_np
