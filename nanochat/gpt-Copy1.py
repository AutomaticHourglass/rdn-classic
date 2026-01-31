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
    n_dimensions: int = 32           # Number of recursive tokenization levels
    page_size: int = 512
    n_streams: int = 4

    # MoE Specifics
    moe_enabled: bool = True
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


# class MLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
#         self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

#     def forward(self, x):
#         x = self.c_fc(x)
#         x = F.relu(x).square()
#         x = self.c_proj(x)
#         return x

class SquaredReLU(nn.Module):
    def forward(self, x):
        return F.relu(x).square()

class DeepSeekMoE(nn.Module):
    """
    Implements the DeepSeekMoE architecture (Shared + Routed Experts).
    Replaces the standard dense MLP.
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        # Config defaults if not present
        self.num_experts = getattr(config, 'num_routed_experts', 8)    # Total routed experts (N)
        self.top_k = getattr(config, 'num_experts_per_tok', 2)         # Active experts (K)
        self.n_shared = getattr(config, 'n_shared_experts', 1)         # Always active experts
        
        # Expert intermediate dimension. 
        # DeepSeek often scales this down so (K + Shared) * dim ≈ Dense dim
        # For a drop-in, you can use (4 * n_embd) / (top_k + n_shared) or a fixed dim.
        # Here we assume a config param or default to a smaller shard.
        self.expert_dim = getattr(config, 'moe_intermediate_size', config.n_embd // 4) 

        # 1. Shared Experts (Always Active)
        # Treated as one large MLP for efficiency (conceptually N shared experts)
        shared_dim = self.expert_dim * self.n_shared
        self.shared_experts = nn.Sequential(
            nn.Linear(self.n_embd, shared_dim, bias=False),
            SquaredReLU(),
            nn.Linear(shared_dim, self.n_embd, bias=False)
        )

        # 2. Routed Experts (Top-K Selected)
        # Using ModuleList for flexibility. 
        # (For extreme speed with 100+ experts, use GroupedGEMM, but this is fast for N=8..16)
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.n_embd, self.expert_dim, bias=False),
                SquaredReLU(),
                nn.Linear(self.expert_dim, self.n_embd, bias=False)
            ) for _ in range(self.num_experts)
        ])

        # 3. Router
        self.router = nn.Linear(self.n_embd, self.num_experts, bias=False)

    def forward(self, x):
        # x: (B, T, C)
        shape = x.shape
        x_flat = x.view(-1, self.n_embd) # (N_tokens, C)

        # --- Shared Path ---
        # "Common knowledge" flows here
        out_shared = self.shared_experts(x_flat)

        # --- Routed Path ---
        router_logits = self.router(x_flat) # (N_tokens, num_experts)
        
        # Top-K selection
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).type_as(x)

        # We need to map tokens to experts.
        # Efficient "Masked Dispatch" for small num_experts (N <= 64).
        # This is easier for torch.compile to optimize than sparse scatter/gather.
        out_routed = torch.zeros_like(out_shared).to(x.dtype)

        # Iterate over all experts (compile unrolls this for small N)
        for i, expert in enumerate(self.routed_experts):
            # Create a mask for tokens that selected this expert
            # selected_experts is (N_tokens, K)
            # mask: (N_tokens, K) -> any -> (N_tokens)
            is_active = (selected_experts == i).any(dim=-1)
            
            # Optimization: Skip expert if no token selected it (graph break in eager, but handled in compile)
            # For strict compile compliance without graph breaks, remove the `if` check
            # or use torch.compiler.disable around the check if needed. 
            # Here we rely on masking which is compile-safe.
            
            # Mask out inputs
            # To preserve gradients/shapes for compile, we multiply by mask (0 or 1)
            # instead of advanced indexing (x[mask]), unless we use specialized kernels.
            # However, standard PyTorch MoE usually does:
            #   subset = x_flat[mask]
            #   out = expert(subset)
            #   accumulate back.
            # Below is the "Matmul-Mask" style (wastes FLOPs on zeros but avoids indexing graph breaks)
            # BUT: indexing is preferred for actual speed. Let's use Indexing.
            
            batch_mask = torch.any(selected_experts == i, dim=1)
            
            if batch_mask.any():
                # Extract tokens assigned to expert 'i'
                # Note: A token might select expert 'i' as its 1st OR 2nd choice.
                # We need the weight associated with this choice.
                
                # Where is 'i' in the top-k selections?
                # (N_tokens, K) boolean mask
                loc_mask = (selected_experts == i)
                
                # Get the weight for this expert assignment
                # (N_tokens) - will be zero if not selected
                prob_i = (routing_weights * loc_mask).sum(dim=1)
                
                # Select inputs (Compress)
                # Note: Standard boolean indexing causes graph breaks or dynamic shapes. 
                # For now, we assume dynamic shapes support is enabled or we accept the overhead.
                indices = torch.nonzero(batch_mask).squeeze(-1)
                x_sub = x_flat[indices]
                
                # Expert Forward
                expert_out = expert(x_sub).to(x.dtype)
                
                # Scale by routing weight
                w_sub = prob_i[indices].unsqueeze(-1)
                expert_out = expert_out * w_sub.to(indices.dtype)
                
                # Scatter back (Expand)
                # out_routed[indices] += expert_out
                out_routed.index_add_(0, indices, expert_out)

        # Combine
        return (out_shared + out_routed).view(shape)

# ------------------------------------------------------------------
# MHC Block
# ------------------------------------------------------------------
class MHCBlock(nn.Module):
    """
    A transformer block that implements DeepSeek’s mHC architecture.
    The residual stream is a tensor of shape (B, T, M, C).
    """
    def __init__(self, config: "GPTConfig", layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = DeepSeekMoE(config)

        # mHC‑specific parameters
        self.n_streams = config.n_streams
        self.mix_logits     = nn.Parameter(torch.randn(self.n_streams, self.n_streams))
        self.input_weights  = nn.Parameter(torch.randn(self.n_streams, 1))
        self.output_weights = nn.Parameter(torch.randn(1, self.n_streams))

        # Compile the per‑chunk step
        self.compiled_step = torch.compile(self._step, mode="default")

    # ------------------------------------------------------------------
    # One‑chunk step
    # ------------------------------------------------------------------
    def _step(
        self,
        X_chunk: torch.Tensor,      # (B, t, M, C)
        cos_chunk: torch.Tensor,
        sin_chunk: torch.Tensor,
        kv_cache,
        A: torch.Tensor,            # (M, M) precomputed mixing matrix
    ):
        B, t, M, C = X_chunk.shape
        assert M == self.n_streams

        # 1) Compression → single residual
        w_in = torch.sigmoid(self.input_weights)  # (M,1)
        x_layer = (X_chunk * w_in.view(1, 1, M, 1)).sum(dim=2)  # (B,t,C)

        # 2) Standard attention + MLP
        x_norm = norm(x_layer)
        attn_out = self.attn(x_norm, (cos_chunk, sin_chunk), kv_cache)  # (B,t,C)

        x_norm2 = norm(x_layer + attn_out)
        mlp_out = self.mlp(x_norm2)  # (B,t,C)

        layer_update = attn_out + mlp_out  # (B,t,C)

        # 3) Expansion → streams
        w_out = torch.sigmoid(self.output_weights)  # (1,M)
        update_streams = layer_update.unsqueeze(2) * w_out.view(1, 1, M, 1)  # (B,t,M,C)

        # 4) Mixing (A @ X_prev) without materializing A_batch
        # X_mixed: (B,t,M,C)
        X_mixed = torch.einsum("mn,btmc->btnc", A, X_chunk)

        # 5) Residual addition
        return X_mixed + update_streams

    # ------------------------------------------------------------------
    # Paged forward
    # ------------------------------------------------------------------
    def _paged_forward(
        self,
        X: torch.Tensor,            # (B,T,M,C)
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache,
        page_size: int,
        A: torch.Tensor,            # (M,M)
    ):
        B, T, M, C = X.shape
        out = torch.empty_like(X)

        for i in range(0, T, page_size):
            start, end = i, min(i + page_size, T)

            X_chunk = X[:, start:end, :, :]
            cos_c = cos[:, start:end, ...]
            sin_c = sin[:, start:end, ...]

            out[:, start:end, :, :] = self.compiled_step(X_chunk, cos_c, sin_c, kv_cache, A)

        return out

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(
        self,
        X: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        kv_cache,
        page_size: int | None = None,
        use_checkpointing: bool = True,
    ):
        cos, sin = cos_sin

        # Compute Sinkhorn ONCE per forward (amortize across chunks)
        A = sinkhorn_knopp(self.mix_logits)  # (M,M)

        if page_size is None or X.size(1) <= page_size:
            if use_checkpointing:
                return checkpoint(self._step, X, cos, sin, kv_cache, A, use_reentrant=False)
            return self.compiled_step(X, cos, sin, kv_cache, A)

        if use_checkpointing:
            return checkpoint(self._paged_forward, X, cos, sin, kv_cache, page_size, A, use_reentrant=False)
        return self._paged_forward(X, cos, sin, kv_cache, page_size, A)


# ------------------------------------------------------------------
# Standard Block (paged)
# ------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = DeepSeekMoE(config)

        # Compile the single-step calculation.
        self.compiled_step = torch.compile(self._step, mode="default")

    def _step(self, x_chunk, cos_chunk, sin_chunk, kv_cache):
        attn_out = self.attn(norm(x_chunk), (cos_chunk, sin_chunk), kv_cache)
        x_chunk = x_chunk + attn_out

        mlp_out = self.mlp(norm(x_chunk))
        x_chunk = x_chunk + mlp_out
        return x_chunk

    def _paged_forward(self, x, cos, sin, kv_cache, page_size):
        B, T, C = x.size()
        out = torch.empty_like(x)

        for i in range(0, T, page_size):
            start, end = i, min(i + page_size, T)

            x_chunk = x[:, start:end, :]
            cos_chunk = cos[:, start:end, ...]
            sin_chunk = sin[:, start:end, ...]

            out[:, start:end, :] = self.compiled_step(x_chunk, cos_chunk, sin_chunk, kv_cache)

        return out

    # NOTE: removed @torch.compiler.disable to avoid forcing graph breaks around forward [page:0]
    def forward(self, x, cos_sin, kv_cache, page_size=None, use_checkpointing=True):
        cos, sin = cos_sin

        if page_size is None or x.size(1) <= page_size:
            if use_checkpointing:
                return checkpoint(self._step, x, cos, sin, kv_cache, use_reentrant=False)
            return self.compiled_step(x, cos, sin, kv_cache)

        if use_checkpointing:
            return checkpoint(self._paged_forward, x, cos, sin, kv_cache, page_size, use_reentrant=False)
        return self._paged_forward(x, cos, sin, kv_cache, page_size)


class CoordGatedMLP(nn.Module):
    def __init__(self, n_dim, hidden_dim, hidden_hidden=256, seq_len=2048):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_dim, hidden_hidden),
            nn.ReLU(),
            nn.Linear(hidden_hidden, hidden_dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(n_dim, hidden_dim)
        self.seq_len = seq_len

    def forward(self, coords):
        coords = (coords - self.seq_len/2) / self.seq_len
        return self.proj(coords.float()) * self.gate(coords.float())


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": CoordGatedMLP(
                    n_dim=config.n_dimensions,
                    hidden_dim=config.n_embd,
                    seq_len=self.config.sequence_len,
                ),
                "h": nn.ModuleList([MHCBlock(config, layer_idx) for layer_idx in range(config.n_layer)]) if self.config.n_streams > 1 else
            nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]
                ),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = (
            config.sequence_len * 10
        )  # 10X over-compute should be enough, TODO make nicer?
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
    #     l, h, q, t, s = (
    #         self.config.n_layer,
    #         self.config.n_head,
    #         self.config.n_embd // self.config.n_head,
    #         self.config.page_size
    #         if self.config.page_size
    #         else self.config.sequence_len,
    #         self.config.n_streams
    #     )
    #     num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t * s
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
        l, h, q, t, s = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
            self.config.n_streams
        )

        # We use n_active_params for the dense approximation
        num_flops_per_token = 6 * (n_active_params - nparams_embedding) + 12 * l * h * q * t * s

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
            x = x.unsqueeze(2).expand(-1, -1, self.config.n_streams, -1).contiguous()  # (B,T,M,C)

        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)

        # ------------------------------------------------------------------
        # 3. Collapse streams (average) – the LM head expects a single residual
        # ------------------------------------------------------------------
        if self.config.n_streams > 1:
            x = x.mean(dim=2)  # (B,T,C)
        x = norm(x)

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
                ignore_index=-1,
                reduction=loss_reduction,
            )
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
