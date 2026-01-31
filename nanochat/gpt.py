"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nanochat.adamw import DistAdamW
from nanochat.common import get_dist_info
from nanochat.muon import DistMuon, Muon
from nanochat.utils import compute_coords


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    n_dimensions: int = 8           # Number of recursive tokenization levels
    page_size: int = 8192
    n_streams: int = 4
    # MoE Specifics
    moe_enabled: bool = False
    n_routed_experts: int = 8      # Total number of selectable experts
    n_shared_experts: int = 2       # Number of experts that always run
    num_experts_per_tok: int = 6    # Top-K experts selected per token
    moe_intermediate_size: int = 128 # Dimension of a single expert (usually smaller than dense MLP)


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last time into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(x.dtype)  # ensure input/output dtypes match
    return out

# ------------------------------------------------------------------
#  Sinkhorn–Knopp projection onto the Birkhoff polytope
# ------------------------------------------------------------------

def sinkhorn_knopp(log_alpha: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """
    Projects a square matrix (in log‑space) onto the set of doubly stochastic matrices.
    Args:
        log_alpha: shape (M, M) – raw logits that will be exponentiated.
        n_iters: number of Sinkhorn iterations (DeepSeek uses 20).
    Returns:
        A doubly‑stochastic matrix of shape (M, M).
    """
    # Exponentiate once – all subsequent ops are in the normal domain.
    A = torch.exp(log_alpha)

    # Row‑/column‑normalisation
    for _ in range(n_iters):
        A = A / (A.sum(dim=1, keepdim=True) + 1e-6)
        A = A / (A.sum(dim=0, keepdim=True) + 1e-6)

    return A

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, _ = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = (
            apply_rotary_emb(q, cos, sin),
            apply_rotary_emb(k, cos, sin),
        )  # QK rotary embedding
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)  # number of queries in this forward pass
        Tk = k.size(
            2
        )  # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = (
            self.n_head != self.n_kv_head
        )  # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=enable_gqa
            )
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, enable_gqa=enable_gqa
            )
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros(
                (Tq, Tk), dtype=torch.bool, device=q.device
            )  # True = keep, False = mask
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa
            )

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

# ------------------------------------------------------------------
# DeepSeek Mixture of Experts
# ------------------------------------------------------------------
class Expert(nn.Module):
    """A single expert: A small MLP with Gated Linear Unit (usually SwiGLU or similar)"""
    def __init__(self, n_embd, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(n_embd, intermediate_size, bias=False) # Gate
        self.w2 = nn.Linear(n_embd, intermediate_size, bias=False) # Value
        self.w3 = nn.Linear(intermediate_size, n_embd, bias=False) # Output

    def forward(self, x):
        # SwiGLU variant
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DeepSeekMoE(nn.Module):
    """
    DeepSeek-style MoE with Shared Experts + Routed Experts.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # 1. Shared Experts (Always Active)
        # DeepSeek often implements this as one larger MLP equivalent to N_shared small ones
        shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = Expert(config.n_embd, shared_intermediate)

        # 2. Routed Experts (Selected via Top-K)
        self.routed_experts = nn.ModuleList([
            Expert(config.n_embd, config.moe_intermediate_size)
            for _ in range(self.n_routed_experts)
        ])

        # 3. Router
        self.router = nn.Linear(config.n_embd, self.n_routed_experts, bias=False)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.size()
        x_flat = x.view(-1, C) # (N, C) where N = B*T

        # --- Shared Path ---
        # "Common knowledge" path that processes every token
        out_shared = self.shared_experts(x_flat)

        # --- Routed Path ---
        router_logits = self.router(x_flat) # (N, n_routed_experts)
        routing_probs = F.softmax(router_logits, dim=-1)

        # Select Top-K
        # weights: (N, k), indices: (N, k)
        topk_weights, topk_indices = torch.topk(routing_probs, self.num_experts_per_tok, dim=-1)

        # Re-normalize weights to sum to 1.0 (optional, but common in MoE)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Efficient Dispatch (Python Loop Implementation)
        # For production, use specialized kernels (e.g., Triton, MegaBlocks)
        out_routed = torch.zeros_like(x_flat)

        # We iterate over experts to avoid massive sparse tensor creation
        # This is reasonably fast for num_experts < 128 on GPU
        for i, expert in enumerate(self.routed_experts):
            # Find tokens that selected this expert
            # batch_idx, k_idx = where(token chose expert i)
            batch_mask = (topk_indices == i) # (N, k) boolean

            if batch_mask.any():
                # We need to flatten the mask to find which tokens (dim 0) selected this expert
                # However, a token might select expert i in different 'k' slots.
                # Simplification: A token effectively sends a scaled copy of itself to the expert.

                # Get indices of tokens that selected expert `i`
                # (We collapse the K dimension because we just need to know "did token n select expert i?")
                token_indices = batch_mask.any(dim=-1).nonzero(as_tuple=True)[0]

                # Compute expert output for these tokens
                selected_x = x_flat[token_indices]
                expert_out = expert(selected_x)

                # Determine the weight for this expert for each selected token
                # We need to gather the specific weight corresponding to where 'i' appeared in topk_indices
                # simpler: gather weights from topk_weights using the mask

                # Get the position in Top-K (0..K-1) where expert i was selected
                # Note: deepseek usually ensures an expert is selected at most once per token
                k_pos = (topk_indices[token_indices] == i).nonzero(as_tuple=True)[1]
                weights = topk_weights[token_indices, k_pos].unsqueeze(-1) # (Subset_Size, 1)

                # Accumulate weighted output
                out_routed.index_add_(0, token_indices, expert_out * weights)

        final_out = out_shared + out_routed
        return final_out.view(B, T, C)

# ------------------------------------------------------------------
# MHC Block with DeepSeekMoE
# ------------------------------------------------------------------
class MHCBlock(nn.Module):
    """
    A transformer block that implements DeepSeek’s mHC architecture.
    The residual stream is a tensor of shape (B, T, M, C).

    Includes DeepSeekMoE (Shared + Routed Experts) for the FFN layer.
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Components
        # Assuming CausalSelfAttention is available in your context
        self.attn = CausalSelfAttention(config, layer_idx)

        # Use DeepSeekMoE instead of standard MLP
        if getattr(config, 'moe_enabled', False):
            self.mlp = DeepSeekMoE(config)
        else:
            self.mlp = MLP(config) # Fallback to dense if configured

        # mHC‑specific parameters
        self.n_streams = config.n_streams
        self.mix_logits    = nn.Parameter(torch.randn(self.n_streams, self.n_streams))
        self.input_weights = nn.Parameter(torch.randn(self.n_streams, 1))
        self.output_weights= nn.Parameter(torch.randn(1, self.n_streams))

        # Compile the per‑chunk step
        self.compiled_step = torch.compile(self._step, mode="default")

    # ------------------------------------------------------------------
    # One‑chunk step
    # ------------------------------------------------------------------
    def _step(self, X_chunk: torch.Tensor,
              cos_chunk: torch.Tensor,
              sin_chunk: torch.Tensor,
              kv_cache):
        B, T, M, C = X_chunk.shape
        assert M == self.n_streams

        # 1. Compression → single residual
        w_in   = torch.sigmoid(self.input_weights)          # (M,1)
        x_layer= (X_chunk * w_in.view(1,1,M,1)).sum(dim=2)  # (B,T,C)

        # 2. Standard attention + MoE
        # Note: Using self.norm1/2 explicitly instead of global norm()
        x_norm   = norm(x_layer)
        attn_out = self.attn(x_norm, (cos_chunk, sin_chunk), kv_cache)

        x_norm2  = norm(x_layer + attn_out)

        # MoE Forward Pass
        mlp_out  = self.mlp(x_norm2)

        layer_update = attn_out + mlp_out  # (B,T,C)

        # 3. Expansion → streams
        w_out   = torch.sigmoid(self.output_weights)         # (1,M)
        update_streams = layer_update.unsqueeze(2) * w_out.view(1,1,M,1)

        # 4. Mixing (A @ X_prev)
        # mHC core: Doubly Stochastic Matrix mixing via Sinkhorn-Knopp
        A      = sinkhorn_knopp(self.mix_logits)             # (M,M)
        X_flat = X_chunk.view(-1, M, C)                      # (B*T,M,C)

        # Broadcasting A over the batch*seq dimension
        # Optimize: bmm (N, M, M) x (N, M, C) -> (N, M, C)
        # Since A is shared, we can use linear projection logic or simple expansion
        A_batch= A.unsqueeze(0).expand(X_flat.size(0),-1,-1)
        X_mixed= torch.bmm(A_batch, X_flat).view(B,T,M,C)

        # 5. Residual addition
        return X_mixed + update_streams

    # ------------------------------------------------------------------
    # Paged forward
    # ------------------------------------------------------------------
    def _paged_forward(self, X: torch.Tensor,
                       cos_chunk: torch.Tensor,
                       sin_chunk: torch.Tensor,
                       kv_cache,
                       page_size):
        out_chunks = []
        for i in range(0, X.size(1), page_size):
            start, end = i, min(i + page_size, X.size(1))
            X_chunk   = X[:, start:end, :, :]
            cos_c     = cos_chunk[:, start:end, ...]
            sin_c     = sin_chunk[:, start:end, ...]
            out_chunks.append(self.compiled_step(X_chunk, cos_c, sin_c, kv_cache))
        return torch.cat(out_chunks, dim=1)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(self, X: torch.Tensor,
                cos_sin: tuple[torch.Tensor, torch.Tensor],
                kv_cache,
                page_size=None,
                use_checkpointing=True):
        cos_chunk, sin_chunk = cos_sin

        if page_size is None or X.size(1) <= page_size:
            # Process the whole sequence in one go
            if use_checkpointing:
                return torch.utils.checkpoint.checkpoint(
                    self._step, X, cos_chunk, sin_chunk, kv_cache, use_reentrant=False
                )
            else:
                return self.compiled_step(X, cos_chunk, sin_chunk, kv_cache)

        # Paged processing
        if use_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self._paged_forward,
                X, cos_chunk, sin_chunk, kv_cache, page_size, use_reentrant=False
            )
        else:
            return self._paged_forward(X,
                                       cos_chunk, sin_chunk,
                                       kv_cache, page_size)



class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        # <--- MST change: use MiniSequenceMLP instead of plain MLP
        self.mlp = MLP(config)

        # Compile the single-step calculation.
        # This makes the re-computation during backward very fast.
        self.compiled_step = torch.compile(self._step, mode="default")

    def _step(self, x_chunk, cos_chunk, sin_chunk, kv_cache):
        """
        The heavy computation for a single page.
        """
        # Attention
        attn_out = self.attn(norm(x_chunk), (cos_chunk, sin_chunk), kv_cache)
        x_chunk = x_chunk + attn_out

        # MLP
        mlp_out = self.mlp(norm(x_chunk))
        x_chunk = x_chunk + mlp_out
        return x_chunk

    def _paged_forward(self, x, cos, sin, kv_cache, page_size):
        """
        The logic that iterates over pages.
        We separate this so we can wrap it (or parts of it) if needed,
        though usually we checkpoint the BLOCK, not the loop.
        """
        B, T, C = x.size()
        out_chunks = []

        # Loop runs in Python (Eager) to force memory cleanup
        for i in range(0, T, page_size):
            start, end = i, min(i + page_size, T)

            x_chunk = x[:, start:end, :]
            cos_chunk = cos[:, start:end, ...]
            sin_chunk = sin[:, start:end, ...]

            # Run the compiled step
            out_chunk = self.compiled_step(x_chunk, cos_chunk, sin_chunk, kv_cache)
            out_chunks.append(out_chunk)

        return torch.cat(out_chunks, dim=1)

    @torch.compiler.disable
    def forward(self, x, cos_sin, kv_cache, page_size=None, use_checkpointing=True):
        # 1. Standard Path (Fastest for Inference/Small Models)
        if page_size is None or x.size(1) <= page_size:
            if use_checkpointing:
                # Checkpoint the whole block as one unit
                return checkpoint(
                    self._step, x, cos_sin[0], cos_sin[1], kv_cache, use_reentrant=False
                )
            else:
                return self.compiled_step(x, cos_sin[0], cos_sin[1], kv_cache)

        # 2. Paged Path (Low Memory)
        # If we want to save memory during BACKWARD, we must checkpoint
        # EACH CHUNK individually, or the whole loop.

        # OPTION A: Checkpoint the ENTIRE paged loop (simplest)
        # This saves inputs (x) and re-runs the loop during backward.
        if use_checkpointing:
            return checkpoint(
                self._paged_forward,
                x,
                cos_sin[0],
                cos_sin[1],
                kv_cache,
                page_size,
                use_reentrant=False,
            )

        # OPTION B: No checkpointing, just paging (Low Inference Memory, High Training Memory)
        return self._paged_forward(x, cos_sin[0], cos_sin[1], kv_cache, page_size)


import torch
from torch import nn

class CoordGatedMLPWithRoPE(nn.Module):
    """
    CoordGatedMLP that also embeds relative positions in a RoPE‑style
    sinusoidal fashion, with *per‑dimension* base frequencies.

    Parameters
    ----------
    n_dim : int
        Number of coordinate dimensions (e.g. 3 for 3‑D points).
    hidden_dim : int
        Size of the output embedding.  Must be divisible by `n_dim * 2`
        because we interleave sin/cos for each dimension.
    hidden_hidden : int, default 256
        Hidden size of the gating MLP (same as original).
    seq_len : int, default 2048
        Normalisation divisor – keeps coordinates in the range [-0.5, 0.5].
    base_frequencies : list[float] or torch.Tensor, optional
        One base frequency per coordinate dimension.  If omitted the
        defaults are `10000**(i / n_dim)` for i in [0, …, n_dim-1].
    """

    def __init__(
        self,
        n_dim: int,
        hidden_dim: int,
        seq_len: int = 2048,
        base_frequencies=None,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # ---- sanity checks -------------------------------------------------
        if hidden_dim % (n_dim * 2) != 0:
            raise ValueError(
                "hidden_dim must be divisible by n_dim * 2 "
                "(so each dimension gets an even number of sin/cos slots)."
            )

        # # ---- base frequencies per axis ------------------------------------
        # if base_frequencies is None:
        #     base_freqs = torch.tensor(
        #         [10000.0 ** ((i+1) / n_dim) for i in range(n_dim)],
        #         dtype=torch.float32,
        #     )
        # else:
        #     base_freqs = torch.tensor(base_frequencies, dtype=torch.float32)
        #     if base_freqs.shape[0] != n_dim:
        #         raise ValueError("base_frequencies must have length n_dim")
        #
        # # Register as a buffer so it is moved with the module but never trained
        # self.register_buffer("base_frequencies", base_freqs)

        # ---- original gating + projection ---------------------------------
        self.gate = nn.Sequential(
            nn.Linear(n_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),  # gating mask in (0,1)
        )
        self.proj = nn.Linear(n_dim, hidden_dim)
        self.init(base_frequencies)

    def init(self, base_frequencies=None):
        # ---- base frequencies per axis ------------------------------------
        common_base = (2 * 3 * 5 * 7) ** 2 * (2 * 3)
        
        if base_frequencies is None:
            base_freqs = torch.tensor(
                [common_base ** (i / self.n_dim) for i in range(self.n_dim, 0, -1)],
                dtype=torch.float32,
            )
        else:
            base_freqs = torch.tensor(base_frequencies, dtype=torch.float32)
            if base_freqs.shape[0] != self.n_dim:
                raise ValueError("base_frequencies must have length n_dim")
        self.register_buffer("base_frequencies", base_freqs)


    # ----------------------------------------------------------------------
    def _sin_cos_per_axis(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute a sinusoidal embedding for *each* coordinate dimension
        with its own base frequency.

        Parameters
        ----------
        coords : Tensor, shape (B, n_dim)
            Normalised coordinates in [-0.5, 0.5].

        Returns
        -------
        Tensor of shape (B, hidden_dim)
            Concatenated sin/cos vectors for all axes.
        """
        B = coords.shape[0]
        # number of sin/cos pairs per axis
        pairs_per_axis = self.hidden_dim // (self.n_dim * 2)
        # frequencies for one axis: shape (pairs_per_axis,)
        freq = (1.0
            / (
                self.base_frequencies.unsqueeze(-1)
                ** (torch.arange(1, pairs_per_axis+1, dtype=torch.float32, device=coords.device)
                    * 2
                    / (pairs_per_axis))
            )
        )  # shape: (n_dim, pairs_per_axis)

        # scale coords by frequency: shape (B, n_dim, pairs_per_axis)
        scaled = coords.unsqueeze(-1) * freq  # broadcasting over n_dim

        sin_emb = torch.sin(scaled)
        cos_emb = torch.cos(scaled)

        # interleave sin & cos along the last dimension:
        emb = torch.cat([sin_emb, cos_emb], dim=-1)  # (B, n_dim, pairs_per_axis*2)
        emb = emb.view(*emb.shape[:-2], -1)
        return emb

    # ----------------------------------------------------------------------
    def forward(
        self,
        coords: torch.Tensor,
        coords_rel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        coords : Tensor, shape (B, n_dim)
            Absolute coordinates.
        coords_rel : Tensor or None, shape (B, n_dim)
            If provided, the network will embed *relative* positions
            as `coords - coords_rel`.  This is useful for attention‑style
            query/key pairs.

        Returns
        -------
        Tensor, shape (B, hidden_dim)
            Final gated embedding.
        """
        # normalise to [-0.5, 0.5]
        coords_norm = (coords - self.seq_len / 2) / self.seq_len
        if coords_rel is not None:
            # relative coordinates – optional
            rel = (coords_rel - self.seq_len / 2) / self.seq_len
            coords_norm = coords_norm - rel  # B × n_dim

        # absolute projection (MLP)
        abs_emb = self.proj(coords_norm.float())

        # gating mask
        gate_mask = self.gate(coords_norm.float())

        # sinusoidal (RoPE‑style) embedding
        sin_cos_emb = self._sin_cos_per_axis(coords_norm.float())

        # combine absolute & sinusoidal parts
        combined = abs_emb + sin_cos_emb

        return combined * gate_mask  # element‑wise gating



class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": CoordGatedMLPWithRoPE(
                    n_dim=self.config.n_dimensions,
                    hidden_dim=self.config.n_embd,
                    seq_len=self.config.sequence_len,
                ),
                "h": nn.ModuleList(
                    [MHCBlock(config, layer_idx) if self.config.n_streams > 1 else Block(config, layer_idx) for layer_idx in range(config.n_layer)]
                ),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = (
            config.sequence_len * 10 * (self.config.n_dimensions+1)
        )  # 100X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer(
            "cos", cos, persistent=False
        )  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.attn.c_proj.weight)

            # torch.nn.init.zeros_(block.mlp.shared_experts.w1.weight)
            # torch.nn.init.zeros_(block.mlp.shared_experts.w2.weight)
            # torch.nn.init.zeros_(block.mlp.shared_experts.w3.weight)

            # for expert in block.mlp.routed_experts:
            #     torch.nn.init.zeros_(expert.w1.weight)
            #     torch.nn.init.zeros_(expert.w2.weight)
            #     torch.nn.init.zeros_(expert.w3.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

        common_base = float((2 * 3 * 5 * 7) ** 2 * (2 * 3))
        # common_base = 264593.0
        self.transformer.wpe.base_frequencies = torch.tensor(
            [common_base ** (i / self.config.n_dimensions) for i in range(self.config.n_dimensions, 0, -1)],
            dtype=torch.float32, device=self.lm_head.weight.device
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    # def estimate_flops(self):
    #     """Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311"""
    #     nparams = sum(p.numel() for p in self.parameters())
    #     nparams_embedding = self.transformer.wte.weight.numel()
    #     l, h, q, t = (
    #         self.config.n_layer,
    #         self.config.n_head,
    #         self.config.n_embd // self.config.n_head,
    #         self.config.page_size
    #         if self.config.page_size
    #         else self.config.sequence_len,
    #     )
    #     num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
    #     return num_flops_per_token

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model.
        Adjusts for MoE sparsity by counting only active parameters.
        Ref: https://arxiv.org/abs/2204.02311
        """
        # 1. Calculate total parameters (includes all experts, even dormant ones)
        nparams = sum(p.numel() for p in self.parameters())

        # 2. Subtract Embedding (standard practice, as they are lookups not heavy compute)
        nparams_embedding = self.transformer.wte.weight.numel()

        # 3. Calculate Dormant Parameters (MoE) to subtract
        # For DeepSeekMoE, we only count Shared Experts + Top-K Routed Experts as active.
        n_dormant_params = 0
        if getattr(self.config, 'moe_enabled', False):
            # Config values
            n_routed = getattr(self.config, 'n_routed_experts', 0)
            n_active_routed = getattr(self.config, 'num_experts_per_tok', 0)
            d_model = self.config.n_embd
            d_ff = self.config.moe_intermediate_size

            # Number of routed experts NOT selected per token
            # Note: Shared experts are always active, so we don't subtract them.
            n_dormant_experts_per_layer = max(0, n_routed - n_active_routed)

            # Parameter count per expert.
            # DeepSeekMoE and modern LLMs typically use SwiGLU (Gate, Up, Down).
            # SwiGLU params = 3 * d_model * d_ff (Gate + Up + Down projections)
            # If using standard GeLU, this would be 2 * ...
            params_per_expert = 3 * d_model * d_ff

            # Total dormant parameters across all layers
            n_dormant_params = self.config.n_layer * n_dormant_experts_per_layer * params_per_expert

        # 4. Effective Active Parameters
        # The 6N formula applies to the parameters actually participating in the fwd/bwd pass.
        n_active_params = nparams - n_dormant_params

        # 5. Apply Kaplan Formula: 6 * N_active + Attention Overhead
        # Attention overhead is linear with context length 't' per token (12 * L * d_model * t)
        l, h, q, t = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.page_size if self.config.page_size else self.config.sequence_len,
        )

        # We use n_active_params for the dense approximation
        num_flops_per_token = 6 * (n_active_params - nparams_embedding) + 12 * l * h * q * t

        return num_flops_per_token


    def setup_optimizers(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        accumulation_steps=1,
    ):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = (
            list(self.transformer.wte.parameters())
            + list(self.transformer.wpe.gate.parameters())
            + list(self.transformer.wpe.proj.parameters())
        )
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(
            embedding_params
        ) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(
                f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
            )
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(
            lr=matrix_lr, momentum=0.95, accumulation_steps=accumulation_steps
        )
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), (
            f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        )
        assert idx.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        )
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (
            self.cos[:, T0 : T0 + T],
            self.sin[:, T0 : T0 + T],
        )  # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        with torch.no_grad():
            coords = compute_coords(idx.to(torch.long))
            coords[coords >= self.config.sequence_len] = self.config.sequence_len - 1

        x = self.transformer.wte(idx) + self.transformer.wpe(coords)
        x = norm(x)
        # ------------------------------------------------------------------
        # 2. Expand into M streams before the block loop
        # ------------------------------------------------------------------
        if self.config.n_streams > 1:
            X = x.unsqueeze(2).expand(-1, -1, self.config.n_streams, -1)  # (B,T,M,C)
        else:
            X = x

        for block in self.transformer.h:
            X = block(X, cos_sin, kv_cache)

        # ------------------------------------------------------------------
        # 3. Collapse streams (average) – the LM head expects a single residual
        # ------------------------------------------------------------------
        if self.config.n_streams > 1:
            X = X.mean(dim=2)  # (B,T,C)
        x = norm(X)

        # ------------------------------------------------------------------
        #  Mini‑Sequence LM head – compute logits in chunks
        # ------------------------------------------------------------------
        # Forward the lm_head (compute logits)
        softcap = 15  # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(
            x
        )  # (B, T, vocab_size) <- very big tensor, large amount of memory
        logits = logits.float()  # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap)  # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=258,
                reduction=loss_reduction,
            )
            if loss_reduction == 'none':
                loss = loss.view(targets.shape)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

    def generate_with_ttt(model, prompt_text, steps=5, lr=1e-2, max_new_tokens=128):
        print(f"\nPrompt: '{prompt_text}'")
        
        # Prepare Input
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        
        # 1. Save state
        original_state = copy.deepcopy(model.state_dict())
        
        # 2. TTT Adaptation Loop
        # We fine-tune on the PROMPT ITSELF to minimize prediction error on the prompt's own sequence
        # This helps the model "grok" the specific syntax/style of the prompt before generating.
        model.train() # Enable grads
        # Freeze classification head, only update internal representations (Encoder)
        # This is a common TTT strategy: don't change how you read the output, change how you think.
        for param in model.lm_head.parameters():
            param.requires_grad = False
            
        ttt_optimizer = torch.optim.SGD(model.transformer_decoder.parameters(), lr=lr)
        
        print(f"Adapting to prompt for {steps} steps...")
        for _ in range(steps):
            # Predict next tokens *within* the prompt
            # In: "The cat sat on" -> Target: "cat sat on the"
            prompt_inputs = input_ids[:, :-1]
            prompt_targets = input_ids[:, 1:]
            
            logits = model(prompt_inputs) # Causal mask applied internally
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), prompt_targets.reshape(-1))
            
            ttt_optimizer.zero_grad()
            loss.backward()
            ttt_optimizer.step()
            
        # 3. Generation (Standard Autoregressive)
        model.eval()
        generated = input_ids
        
        print("Generating...", end=" ")
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Forward pass
                logits = model(generated)
                
                # Get last token logits
                next_token_logits = logits[:, -1, :]
                
                # Greedy decode
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                
                # Append
                generated = torch.cat((generated, next_token), dim=1)
                print(".", end="")
                
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\nResult: {decoded}")
        
        # 4. Restore
        model.load_state_dict(original_state)
