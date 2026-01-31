# RDN7-TTT Optimization Plan
## Fusing rdn-cc Performance into rdn7-ttt

**Date:** 2026-01-31
**Goal:** Port performance optimizations from rdn-cc into rdn7-ttt while preserving MoE and all experimental features
**Expected Speedup:** 3-5x (from ~750ms to ~150ms per step)

---

## Executive Summary

This plan merges the blazing-fast implementation from `/Users/unsalgokdag/git/rdns/rdn-cc` into the feature-complete `/Users/unsalgokdag/git/rdns/rdn7-ttt` codebase. The rdn-cc repo is 5x faster but simpler, while rdn7-ttt has all the correct architectural pieces (MoE, generators, experimental features).

**Key Bottlenecks in rdn7-ttt:**
1. ❌ No GPU coordinate calculation (CPU bottleneck, 50-150ms/batch)
2. ❌ Missing Polar Express optimizer (suboptimal convergence)
3. ❌ No prefetch dataloader (GPU starvation)
4. ❌ Missing value embeddings (ResFormer-style)
5. ❌ Missing sliding window attention
6. ❌ Suboptimal Flash Attention usage
7. ❌ --gpt flag crashes with inplace operation error

---

## Priority Order

1. **GPU Coordinate Calculation** (biggest speedup: 50-150ms/batch)
2. **Optimizer Improvements** (Polar Express + variance reduction)
3. **Fix --gpt Flag** (critical for baseline comparisons)
4. **Prefetch Dataloader** (20-40% throughput improvement)
5. **Model Features** (value embeddings, sliding window attention)
6. **Flash Attention 3** (parked until H100 available)

---

## Repository Restructuring

### Phase 0: Rename and Reorganize (1-2 hours)

**A. Rename Core Directories**
- [ ] Rename `nanochat/` → `rdn/`
- [ ] Rename `rustbpe/` → `rdt/`
- [ ] Update all imports across the codebase
- [ ] Update `pyproject.toml` and `Cargo.toml` files

**B. Update Configuration Files**

Files to update:
- [ ] `pyproject.toml`: Change package name from `recursive_dimensional_network` to `rdn`
- [ ] `rdt/Cargo.toml`: Update crate name and paths
- [ ] All `scripts/*.py`: Update imports `from nanochat` → `from rdn`
- [ ] `tasks/*.py`: Update imports
- [ ] `generator/*.py`: Update imports
- [ ] `tests/*.py`: Update imports

**C. Analyze and Delete Unused Files**

Compare file structure between repos:
```bash
# rdn-cc structure (24 modules in rdn/):
rdn/
├── gpt.py
├── gpu_coordinates.py          # NEW - need to port
├── hierarchical_dataloader.py  # NEW - need to port
├── prefetch_dataloader.py      # NEW - need to port
├── flash_attention.py          # NEW - need to port
├── optim.py                    # REPLACE muon.py + adamw.py
├── dataloader.py
├── tokenizer.py
├── engine.py
├── checkpoint_manager.py
├── common.py
├── configurator.py
├── dataset.py
├── loss_eval.py
└── report.py

# rdn7-ttt structure (21 modules in nanochat/):
nanochat/
├── gpt.py
├── muon.py                     # REPLACE with optim.py
├── adamw.py                    # MERGE into optim.py
├── dataloader.py               # ENHANCE
├── utils.py                    # CHECK if needed
├── checkpoint_manager.py
└── ... (rest are similar)
```

Files to delete/merge:
- [ ] `nanochat/muon.py` → merge into `rdn/optim.py`
- [ ] `nanochat/adamw.py` → merge into `rdn/optim.py`
- [ ] `nanochat/utils.py` → check if `compute_coords()` still needed
- [ ] Any other unused utility files

---

## Phase 1: GPU Coordinate Calculation (CRITICAL - Day 1, 3-4 hours)

### 1.1 Port GPU Coordinate Calculator

**Source:** `/Users/unsalgokdag/git/rdns/rdn-cc/rdn/gpu_coordinates.py`
**Destination:** `/Users/unsalgokdag/git/rdns/rdn7-ttt/rdn/gpu_coordinates.py`

**What to port:**
```python
class GPUCoordinateCalculator:
    """GPU-accelerated hierarchical coordinate calculation using torch.cumsum"""
    def __init__(self, n_dim=8, bracket_pairs=None, bos_token=None, device='cuda', vocab_size=50000)
    def calculate_coordinates_batch(self, token_ids: torch.Tensor) -> torch.Tensor
    def calculate_coordinates(self, token_ids: torch.Tensor) -> torch.Tensor
    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor

def create_coordinate_calculator(n_dim=8, bracket_pairs=None, bos_token=None, device='cuda', vocab_size=50000)
```

**Key Features:**
- Uses `torch.cumsum()` for parallel bracket depth tracking
- Vectorized operations, no Python loops
- Supports batched input: `[B, T]` → `[B, T, n_dim]`
- Handles BOS token specially: gets `[0, 0, 0, ...]`
- Default brackets: `()`, `[]`, `{}`, `<>`

**Task Checklist:**
- [ ] Copy `gpu_coordinates.py` to `rdn/`
- [ ] Verify torch.compile compatibility
- [ ] Add type hints
- [ ] Test on single sequence
- [ ] Test on batch
- [ ] Benchmark throughput (target: >15M tokens/sec)

### 1.2 Port Hierarchical Dataloader

**Source:** `/Users/unsalgokdag/git/rdns/rdn-cc/rdn/hierarchical_dataloader.py`
**Destination:** `/Users/unsalgokdag/git/rdns/rdn7-ttt/rdn/hierarchical_dataloader.py`

**What to port:**
```python
def tokenizing_distributed_data_loader_with_coords(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    n_coord_dim=8, bracket_pairs=None
)

def tokenizing_distributed_data_loader_with_coords_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000, n_coord_dim=8, bracket_pairs=None
)
```

**Key Features:**
- Yields `(inputs, targets, coords_inputs, coords_targets, state_dict)`
- GPU coordinate calculation after BPE tokenization
- BOS-aligned best-fit packing (minimizes cropping)
- Pre-allocated pinned memory buffers

**Task Checklist:**
- [ ] Copy `hierarchical_dataloader.py` to `rdn/`
- [ ] Integrate with existing `_document_batches()` helper
- [ ] Preserve generator support for calculator/tictactoe
- [ ] Test coordinate alignment with tokens
- [ ] Verify DDP compatibility
- [ ] Benchmark: batch generation should be <100ms

### 1.3 Update Main Dataloader

**File:** `/Users/unsalgokdag/git/rdns/rdn7-ttt/rdn/dataloader.py`

**Changes:**
- [ ] Add `get_dataloader()` factory function (from rdn-cc)
- [ ] Add parameters: `use_coordinates=True`, `n_coord_dim=8`, `bracket_pairs=None`
- [ ] Integrate hierarchical dataloader when `use_coordinates=True`
- [ ] Keep generator support (calculator/tictactoe) working
- [ ] Add BOS-aligned best-fit option: `use_bos_aligned=True`

**Signature:**
```python
def get_dataloader(
    tokenizer, B, T, split,
    device="cuda",
    resume_state_dict=None,
    use_bos_aligned=True,
    use_coordinates=True,
    n_coord_dim=8,
    bracket_pairs=None,
    prefetch=True,  # Added in Phase 4
    use_generator=False,  # PRESERVE this for calculator/tictactoe
    use_recursive_markers=True,  # PRESERVE this
    **kwargs
):
    """
    Get RDN dataloader with optional GPU-accelerated coordinates.

    Returns:
        - If use_coordinates=True: (inputs, targets, coords_inputs, coords_targets, state_dict)
        - If use_coordinates=False: (inputs, targets, state_dict)
    """
```

### 1.4 Update Training Scripts

**Files to update:**
- [ ] `scripts/base_train.py`
- [ ] `scripts/mid_train.py`
- [ ] `scripts/chat_sft.py`
- [ ] `scripts/chat_rl.py`

**Changes in base_train.py:**
```python
# Line ~250-260: Replace manual dataloader calls with get_dataloader()
from rdn.dataloader import get_dataloader

train_loader = get_dataloader(
    tokenizer,
    args.device_batch_size,
    args.max_seq_len,
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
    use_coordinates=config.use_coordinate_embeddings,  # Respects --gpt flag
    n_coord_dim=config.n_dimensions,
    tokenizer_threads=16,
    tokenizer_batch_size=256,
    use_generator=args.use_generator,  # PRESERVE
    use_recursive_markers=args.use_recursive_markers,  # PRESERVE
)

# Unpack with coordinates
if config.use_coordinate_embeddings:
    x, y, coords_x, coords_y, dataloader_state_dict = next(train_loader)
else:
    x, y, dataloader_state_dict = next(train_loader)
    coords_x = None

# Forward pass
loss = model(x, y, coords=coords_x)
```

### 1.5 Update Model to Accept Coordinates

**File:** `/Users/unsalgokdag/git/rdns/rdn7-ttt/rdn/gpt.py`

**Current signature (line ~912):**
```python
def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
```

**New signature:**
```python
def forward(self, idx, targets=None, coords=None, kv_cache=None, loss_reduction='mean'):
    """
    Args:
        idx: Token IDs [B, T]
        targets: Target token IDs [B, T] (optional)
        coords: Hierarchical coordinates [B, T, n_dimensions] (optional)
        kv_cache: KV cache for inference (optional)
        loss_reduction: Loss reduction mode
    """
```

**Changes:**
- [ ] Add `coords` parameter to forward()
- [ ] Pass coords to coordinate embedding logic (already exists in rdn7-ttt!)
- [ ] Ensure --gpt flag skips coordinate embeddings (line ~915)

### 1.6 Testing GPU Coordinates

**Create:** `tests/test_gpu_coordinates.py`

```python
def test_gpu_coordinate_calculator():
    """Test GPU coordinate calculation on single sequence"""
    pass

def test_gpu_coordinate_batch():
    """Test GPU coordinate calculation on batch"""
    pass

def test_coordinate_alignment():
    """Verify coordinates align with tokens"""
    pass

def test_bos_token_coordinates():
    """Verify BOS token gets [0,0,0,...]"""
    pass

def test_bracket_depth_tracking():
    """Test bracket nesting levels"""
    pass

def test_throughput():
    """Benchmark coordinate calculation throughput (target: >15M tok/sec)"""
    pass
```

**Create:** `tests/test_hierarchical_dataloader.py`

```python
def test_dataloader_yields_coordinates():
    """Verify dataloader returns coordinate tensors"""
    pass

def test_bos_aligned_packing():
    """Verify every row starts with BOS"""
    pass

def test_coordinate_shape():
    """Verify coords shape is [B, T, n_coord_dim]"""
    pass

def test_generator_compatibility():
    """Verify calculator/tictactoe generator still works"""
    pass
```

---

## Phase 2: Optimizer Improvements (CRITICAL - Day 2, 4-6 hours)

### 2.1 Port Unified MuonAdamW Optimizer

**Source:** `/Users/unsalgokdag/git/rdns/rdn-cc/rdn/optim.py`
**Destination:** `/Users/unsalgokdag/git/rdns/rdn7-ttt/rdn/optim.py`

**What to port:**
```python
# Lines 76-84: Polar Express coefficients
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    """Fused AdamW with decoupled weight decay"""
    pass

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    """Fused Muon with Polar Express + variance reduction + cautious weight decay"""
    pass

class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others"""
    pass

class DistMuonAdamW(torch.optim.Optimizer):
    """Distributed version with ZeRO-2 style sharding"""
    pass
```

**Key Improvements over rdn7-ttt:**
1. **Polar Express orthogonalization** (vs Newton-Schulz)
   - Better convergence properties
   - Optimized coefficients for 5 iterations
   - Distinguishes tall vs wide matrices

2. **Variance reduction**
   - Factored second moment buffer (per-row or per-column)
   - Adaptive step size scaling
   - Missing entirely in rdn7-ttt!

3. **Cautious weight decay**
   - Only applies WD when gradient and param have same sign
   - `mask = (g * stacked_params) >= 0`
   - Prevents optimizer "thrashing"

4. **Fused kernels**
   - Single torch.compile graph for each optimizer
   - Uses `lerp_()` for cleaner fusion
   - 0-D CPU tensors for hyperparameters (avoid recompilation)

5. **Unified interface**
   - Single optimizer handles both Muon and AdamW
   - No need to loop over multiple optimizers
   - Cleaner training loop

**Task Checklist:**
- [ ] Copy `optim.py` to `rdn/`
- [ ] Replace `rdn/muon.py` and `rdn/adamw.py`
- [ ] Add MoE parameter support to setup_optimizer()
- [ ] Verify torch.compile works
- [ ] Test gradient accumulation compatibility
- [ ] Benchmark: ensure no performance regression

### 2.2 Update Model setup_optimizer()

**File:** `/Users/unsalgokdag/git/rdns/rdn7-ttt/rdn/gpt.py`

**Current approach (lines 851-868):**
```python
# Returns TWO separate optimizers
adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
return [adamw_optimizer, muon_optimizer]
```

**New approach (from rdn-cc lines 386-437):**
```python
def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                   weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
    model_dim = self.config.n_embd
    ddp, rank, local_rank, world_size = get_dist_info()

    # Separate parameters into groups
    matrix_params = list(self.transformer.h.parameters())
    embedding_params = list(self.transformer.wte.parameters())
    lm_head_params = list(self.lm_head.parameters())

    # NEW: Add MoE parameters if enabled
    moe_params = []
    if self.config.moe_enabled:
        for block in self.transformer.h:
            if hasattr(block, 'moe') and block.moe is not None:
                moe_params.extend(list(block.moe.parameters()))

    # NEW: Add scalar parameters if they exist (will add in Phase 5)
    # resid_params = [self.resid_lambdas] if hasattr(self, 'resid_lambdas') else []
    # x0_params = [self.x0_lambdas] if hasattr(self, 'x0_lambdas') else []

    # Scale LR by sqrt(dmodel)
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    # Build param_groups with explicit 'kind' field
    param_groups = [
        # AdamW groups
        dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale,
             betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale,
             betas=adam_betas, eps=1e-10, weight_decay=0.0),
    ]

    # Add MoE parameters to AdamW (treat like embeddings)
    if moe_params:
        param_groups.append(
            dict(kind='adamw', params=moe_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0)
        )

    # Muon groups (matrix params, grouped by shape for stacking)
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(
            kind='muon', params=group_params, lr=matrix_lr,
            momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
        ))

    Factory = DistMuonAdamW if ddp else MuonAdamW
    optimizer = Factory(param_groups)

    # Store initial_lr for scheduling
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    return optimizer  # Single optimizer, not a list!
```

**Task Checklist:**
- [ ] Replace setup_optimizers() with setup_optimizer()
- [ ] Add MoE parameter group handling
- [ ] Add coordinate embedding parameter group
- [ ] Update all training scripts to use single optimizer
- [ ] Verify parameter counts match

### 2.3 Update Training Loop

**Files to update:**
- [ ] `scripts/base_train.py` (lines 540-555)
- [ ] `scripts/mid_train.py`
- [ ] `scripts/chat_sft.py`
- [ ] `scripts/chat_rl.py`

**Current pattern:**
```python
for opt in optimizers:  # Loop over [adamw, muon]
    for group in opt.param_groups:
        group["lr"] = group["initial_lr"] * lrm
muon_momentum = get_muon_momentum(step)
for group in muon_optimizer.param_groups:
    group["momentum"] = muon_momentum
for opt in optimizers:
    opt.step()
```

**New pattern:**
```python
# Update hyperparameters
lrm = get_lr_multiplier(step)
muon_momentum = get_muon_momentum(step)
muon_weight_decay = get_weight_decay(step)  # NEW: decay to zero

for group in optimizer.param_groups:
    group["lr"] = group["initial_lr"] * lrm
    if group['kind'] == 'muon':
        group["momentum"] = muon_momentum
        group["weight_decay"] = muon_weight_decay  # Dynamic WD!

optimizer.step()  # Single call
model.zero_grad(set_to_none=True)
```

**Task Checklist:**
- [ ] Update all training loops to use single optimizer
- [ ] Add weight decay scheduler: `wd * (1 - step / num_iterations)`
- [ ] Verify momentum scheduling still works
- [ ] Test gradient accumulation

### 2.4 Add Logging for Optimizer Stats

**From rdn-cc base_train.py (lines 436-469):**

Add to training loop:
```python
# Log detailed optimizer stats
log_data = {
    "step": step,
    "total_training_flops": flops_so_far,
    "total_training_time": total_training_time,
    "train/loss": debiased_smooth_loss,
    "train/lrm": lrm,
    "train/dt": dt,
    "train/tok_per_sec": tok_per_sec,
    "train/mfu": mfu,
    "train/epoch": epoch,
    "train/muon_momentum": muon_momentum,  # NEW
    "train/muon_weight_decay": muon_weight_decay,  # NEW
}
```

### 2.5 Testing Optimizer

**Create:** `tests/test_optimizer.py`

```python
def test_muonadamw_single_gpu():
    """Test MuonAdamW on single GPU"""
    pass

def test_muonadamw_distributed():
    """Test DistMuonAdamW with DDP"""
    pass

def test_polar_express_orthogonalization():
    """Verify orthogonalization produces orthogonal matrices"""
    pass

def test_variance_reduction():
    """Test factored second moment buffer"""
    pass

def test_cautious_weight_decay():
    """Verify WD only applied when gradient and param agree"""
    pass

def test_fused_kernels():
    """Verify torch.compile fusion works"""
    pass

def test_parameter_grouping():
    """Test params grouped correctly by shape"""
    pass

def test_moe_parameter_handling():
    """Verify MoE params get correct optimizer treatment"""
    pass
```

---

## Phase 3: Fix --gpt Flag (CRITICAL - Day 3, 2-3 hours)

### 3.1 Diagnose the Crash

**Error:**
```
RuntimeError: one of the variables needed for gradient computation has been modified by an
inplace operation: [MPSLongType [64, 256]] is at version 352; expected version 351 instead.
```

**Root Cause Analysis:**

From exploration, the crash happens because:
1. MHCBlock uses `.expand()` to create stream views (line 387-388)
2. Operations on expanded tensors can cause inplace modifications
3. With `torch.utils.checkpoint` and `use_reentrant=False`, backward pass fails
4. The issue is in MHCBlock's `_step` method (lines 323-355) with `torch.bmm()`

**Solution Strategy:**
- Make tensor operations fully contiguous before checkpointing
- Add explicit handling when `use_coordinate_embeddings=False`
- Ensure Block and MHCBlock have compatible signatures

### 3.2 Fix MHCBlock Tensor Operations

**File:** `/Users/unsalgokdag/git/rdns/rdn7-ttt/rdn/gpt.py` (lines 320-400)

**Changes:**

```python
class MHCBlock(nn.Module):
    def forward(self, X, cos_sin, kv_cache, page_size=None, use_checkpointing=True):
        B, N, S, D = X.size()

        # ORIGINAL (line 387-388):
        # X_chunked = X.expand(B, N, self.n_streams, S, D)

        # FIX: Make contiguous copy instead of expand
        X_chunked = X.unsqueeze(2).expand(B, N, self.n_streams, S, D).contiguous()

        # Apply streams
        outputs = []
        for stream_idx in range(self.n_streams):
            X_stream = X_chunked[:, :, stream_idx, :, :]  # [B, N, S, D]
            if use_checkpointing:
                # Ensure input is contiguous
                X_stream = X_stream.contiguous()
                output = checkpoint(
                    self._step,
                    X_stream,
                    stream_idx,
                    cos_sin,
                    kv_cache,
                    page_size,
                    use_reentrant=False
                )
            else:
                output = self._step(X_stream, stream_idx, cos_sin, kv_cache, page_size)
            outputs.append(output)

        # Stack and mean (ORIGINAL line 421)
        # X = torch.stack(outputs, dim=2).mean(dim=2)

        # FIX: Ensure output is contiguous
        X = torch.stack(outputs, dim=2)
        X = X.mean(dim=2).contiguous()

        return X
```

### 3.3 Add --gpt Flag Support to Dataloader

**File:** `/Users/unsalgokdag/git/rdns/rdn7-ttt/rdn/dataloader.py`

Ensure dataloader respects `use_coordinates=False`:

```python
def get_dataloader(..., use_coordinates=True, ...):
    if use_coordinates:
        # Use hierarchical dataloader with GPU coordinates
        loader = tokenizing_distributed_data_loader_with_coords_bos_bestfit(...)
    else:
        # Use simple dataloader without coordinates
        loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(...)

    if prefetch:
        loader = PrefetchDataLoader(loader, prefetch_count=3)

    return loader
```

### 3.4 Update base_train.py to Pass use_coordinates

**File:** `/Users/unsalgokdag/git/rdns/rdn7-ttt/scripts/base_train.py`

```python
# Line ~240: Determine if using coordinates (respects --gpt flag)
use_coordinates = config.use_coordinate_embeddings  # True by default, False with --gpt

# Line ~250: Pass to dataloader
train_loader = get_dataloader(
    tokenizer,
    args.device_batch_size,
    args.max_seq_len,
    split="train",
    device=device,
    use_coordinates=use_coordinates,  # Respects --gpt flag
    n_coord_dim=config.n_dimensions,
    ...
)

# Line ~270: Conditional unpacking
if use_coordinates:
    x, y, coords_x, coords_y, dataloader_state_dict = next(train_loader)
else:
    x, y, dataloader_state_dict = next(train_loader)
    coords_x = None

# Line ~410: Forward pass
loss = model(x, y, coords=coords_x)  # Works with or without coords
```

### 3.5 Testing --gpt Flag

**Create:** `tests/test_gpt_flag.py`

```python
def test_gpt_mode_forward():
    """Test forward pass with --gpt flag (no coordinates)"""
    config = GPTConfig(use_coordinate_embeddings=False)
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (2, 128))
    logits = model(x, coords=None)
    assert logits.shape == (2, 128, config.vocab_size)

def test_gpt_mode_backward():
    """Test backward pass with --gpt flag (critical!)"""
    config = GPTConfig(use_coordinate_embeddings=False)
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (2, 128))
    y = torch.randint(0, config.vocab_size, (2, 128))
    loss = model(x, y, coords=None)
    loss.backward()  # Should not crash!

def test_gpt_mode_mhc():
    """Test MHCBlock with --gpt flag"""
    config = GPTConfig(use_coordinate_embeddings=False, n_streams=4)
    model = GPT(config)
    # ... test backward pass

def test_gpt_mode_dataloader():
    """Verify dataloader respects use_coordinates=False"""
    loader = get_dataloader(..., use_coordinates=False)
    batch = next(loader)
    assert len(batch) == 3  # (inputs, targets, state_dict) - no coords

def test_rdn_vs_gpt_training():
    """Compare training step between RDN and GPT modes"""
    # Train 100 steps in each mode, verify both converge
    pass
```

**Create:** `test_runs/test_gpt_flag.sh`

```bash
#!/bin/bash
# Test --gpt flag end-to-end

# Small model, 100 iterations
python -m scripts.base_train \
  --depth=4 \
  --max-seq-len=512 \
  --device-batch-size=2 \
  --total-batch-size=1024 \
  --num-iterations=100 \
  --eval-every=50 \
  --save-every=-1 \
  --gpt \
  --run=test_gpt_flag

echo "✓ --gpt flag test passed"
```

---

## Phase 4: Prefetch Dataloader (HIGH IMPACT - Day 3, 1-2 hours)

### 4.1 Port PrefetchDataLoader

**Source:** `/Users/unsalgokdag/git/rdns/rdn-cc/rdn/prefetch_dataloader.py`
**Destination:** `/Users/unsalgokdag/git/rdns/rdn7-ttt/rdn/prefetch_dataloader.py`

**What to port:**
```python
import threading
import queue

class PrefetchDataLoader:
    """
    Wraps any dataloader to prefetch batches in background thread.
    Prevents GPU starvation from CPU-bound tokenization.
    """
    def __init__(self, dataloader, prefetch_count=3):
        self.dataloader = dataloader
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        """Background thread that fills the queue"""
        try:
            for batch in self.dataloader:
                self.queue.put(batch)
        except Exception as e:
            self.queue.put(e)

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.queue.get(timeout=30)
        if isinstance(batch, Exception):
            raise batch
        return batch
```

**Key Features:**
- Queue-based prefetching (default 3 batches ahead)
- Background thread (non-blocking)
- Exception handling (re-raises in main thread)
- Timeout protection (30 seconds)

**Task Checklist:**
- [ ] Copy `prefetch_dataloader.py` to `rdn/`
- [ ] Test with hierarchical dataloader
- [ ] Test with generator support
- [ ] Verify exception handling works
- [ ] Benchmark: should eliminate GPU starvation

### 4.2 Integrate Prefetch into get_dataloader()

**File:** `/Users/unsalgokdag/git/rdns/rdn7-ttt/rdn/dataloader.py`

```python
from rdn.prefetch_dataloader import PrefetchDataLoader

def get_dataloader(..., prefetch=True, prefetch_count=3, ...):
    # Create base dataloader
    if use_coordinates:
        if use_bos_aligned:
            loader = tokenizing_distributed_data_loader_with_coords_bos_bestfit(...)
        else:
            loader = tokenizing_distributed_data_loader_with_coords(...)
    else:
        if use_bos_aligned:
            loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(...)
        else:
            loader = tokenizing_distributed_data_loader_with_state(...)

    # Wrap with prefetching
    if prefetch:
        loader = PrefetchDataLoader(loader, prefetch_count=prefetch_count)

    return loader
```

### 4.3 Testing Prefetch

**Create:** `tests/test_prefetch_dataloader.py`

```python
def test_prefetch_basic():
    """Test prefetch with simple dataloader"""
    pass

def test_prefetch_with_coordinates():
    """Test prefetch with hierarchical dataloader"""
    pass

def test_prefetch_exception_handling():
    """Verify exceptions are re-raised in main thread"""
    pass

def test_prefetch_timeout():
    """Test timeout protection"""
    pass

def test_prefetch_throughput():
    """Benchmark: prefetch should improve throughput 20-40%"""
    pass
```

---

## Phase 5: Model Features (MEDIUM PRIORITY - Day 4, 4-6 hours)

### 5.1 Add Value Embeddings (ResFormer-style)

**Source:** `/Users/unsalgokdag/git/rdns/rdn-cc/rdn/gpt.py` (lines 77-78, 92-93, 191-194, 240-247)

**Changes to `/Users/unsalgokdag/git/rdns/rdn7-ttt/rdn/gpt.py`:**

```python
# Line ~50: Add helper function
def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always)."""
    return layer_idx % 2 == (n_layer - 1) % 2

# Line ~87: Add ve_gate to CausalSelfAttention.__init__
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # ... existing code
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) \
                       if has_ve(layer_idx, config.n_layer) else None

# Line ~102: Update forward signature to accept ve
    def forward(self, x, ve, cos_sin, kv_cache):
        # ... project q, k, v

        # Value residual (ResFormer): mix in value embedding with gating
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # ... rest of attention

# Line ~138: Update Block.forward signature
class Block(nn.Module):
    def forward(self, x, ve, cos_sin, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x

# Line ~190: Add value_embeds to GPT.__init__
class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        # ... existing code

        # Value embeddings (ResFormer-style)
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        })

# Line ~235: Initialize value embeddings
    @torch.no_grad()
    def init_weights(self):
        # ... existing init code

        # Value embeddings (init like c_v: uniform)
        s = 3**0.5 * self.config.n_embd**-0.5
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero (gates start at 0.5, scaled by 2 -> 1.0 neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Cast embeddings to bf16
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

# Line ~470: Update forward loop to pass ve
    def forward(self, idx, targets=None, coords=None, kv_cache=None, loss_reduction='mean'):
        # ... existing code

        for i, block in enumerate(self.transformer.h):
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, kv_cache)

# Line ~850: Update setup_optimizer to include value_embeds
    def setup_optimizer(...):
        # ... existing code

        value_embeds_params = list(self.value_embeds.parameters())

        param_groups = [
            # ... existing groups
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
```

**Task Checklist:**
- [ ] Add `has_ve()` helper function
- [ ] Add `ve_gate` to CausalSelfAttention
- [ ] Add `value_embeds` ModuleDict to GPT
- [ ] Update attention forward to use value embeddings
- [ ] Update initialization
- [ ] Update optimizer parameter grouping
- [ ] Test with and without value embeddings

### 5.2 Add Per-Layer Scalars

**Source:** `/Users/unsalgokdag/git/rdns/rdn-cc/rdn/gpt.py` (lines 185-190, 237-238, 470)

```python
# Line ~185: Add scalars to GPT.__init__
class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        # ... existing code

        # Per-layer learnable scalars
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

# Line ~237: Initialize scalars
    @torch.no_grad()
    def init_weights(self):
        # ... existing init code

        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for x0 skip

# Line ~467: Use scalars in forward loop
    def forward(self, idx, targets=None, coords=None, kv_cache=None, loss_reduction='mean'):
        x = norm(x)
        x0 = x  # Save initial normalized embedding

        for i, block in enumerate(self.transformer.h):
            # Apply per-layer scalars
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, kv_cache)

# Line ~850: Update optimizer
    def setup_optimizer(..., scalar_lr=0.5):
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]

        param_groups = [
            # ... existing groups
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01,
                 betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr,
                 betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # Higher beta1!
        ]
```

### 5.3 Add Sliding Window Attention

**Source:** `/Users/unsalgokdag/git/rdns/rdn-cc/rdn/gpt.py` (lines 36-39, 286-313)

**Note:** Requires Flash Attention 3 or SDPA with window support. Since we're parking FA3 until H100 is available, we can add the infrastructure but keep it disabled for now.

```python
@dataclass
class GPTConfig:
    # ... existing fields
    window_pattern: str = "L"  # Default: all full context
    # "L" = long (full), "S" = short (half window)
    # Examples: "L" = all full, "SL" = alternating, "SSSL" = pattern

# Line ~286: Compute per-layer window sizes
class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        # ... existing code
        self.window_sizes = self._compute_window_sizes(config)

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.
        Returns list of (left, right) tuples: (-1, 0) for full, (N, 0) for sliding.
        Pattern is tiled across layers, final layer always gets full context.
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid pattern: {pattern}"

        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),   # Full context (or -1 for unlimited)
            "S": (short_window, 0),  # Half context
        }

        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])

        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)

        return window_sizes
```

**NOTE:** Actual window size usage requires Flash Attention 3, which we'll add later.

### 5.4 Testing Model Features

**Create:** `tests/test_model_features.py`

```python
def test_value_embeddings():
    """Test value embeddings are used in alternating layers"""
    pass

def test_value_embedding_gating():
    """Test ve_gate produces values in (0, 2) range"""
    pass

def test_per_layer_scalars():
    """Test resid_lambdas and x0_lambdas"""
    pass

def test_x0_skip_connection():
    """Verify x0 skip connection preserves initial embedding"""
    pass

def test_window_size_computation():
    """Test window size patterns: L, SSL, SSSL"""
    pass

def test_model_with_all_features():
    """Test model with VE + scalars + window patterns together"""
    pass
```

---

## Phase 6: Logging & Profiling (Day 5, 2-3 hours)

### 6.1 Add Verbose Logging from rdn-cc

**Source:** `/Users/unsalgokdag/git/rdns/rdn-cc/scripts/base_train.py` (lines 436-469)

**Add to `/Users/unsalgokdag/git/rdns/rdn7-ttt/scripts/base_train.py`:**

```python
# Line ~540: Enhanced training loop logging
pct_done = 100 * step / num_iterations
tok_per_sec = int(args.total_batch_size / dt)
flops_per_sec = num_flops_per_token * args.total_batch_size / dt
mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)

if step > 1:
    total_training_time += dt

# Calculate ETA
steps_done = step - 1
if steps_done > 0:
    avg_time_per_step = total_training_time / steps_done
    remaining_steps = num_iterations - step
    eta_seconds = remaining_steps * avg_time_per_step
    eta_str = f" | eta: {eta_seconds/60:.1f}m"
else:
    eta_str = ""

epoch = dataloader_state_dict.get("epoch", 1)

# Print with all metrics
print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | "
       f"loss: {debiased_smooth_loss:.6f} | "
       f"lrm: {lrm:.2f} | "
       f"dt: {dt * 1000:.2f}ms | "
       f"tok/sec: {tok_per_sec:,} | "
       f"mfu: {mfu:.2f} | "
       f"epoch: {epoch} | "
       f"total time: {total_training_time/60:.2f}m{eta_str}")

# Log to wandb
log_data = {
    "step": step,
    "total_training_flops": flops_so_far,
    "total_training_time": total_training_time,
    "train/loss": debiased_smooth_loss,
    "train/lrm": lrm,
    "train/dt": dt,
    "train/tok_per_sec": tok_per_sec,
    "train/mfu": mfu,
    "train/epoch": epoch,
    "train/muon_momentum": muon_momentum,
    "train/muon_weight_decay": muon_weight_decay,
}
wandb_run.log(log_data)
```

### 6.2 Add Memory Usage Tracking

```python
# Line ~80: After compute_init
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# Line ~474: After training completes
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")
```

### 6.3 Add Detailed Parameter Counts

**Source:** `/Users/unsalgokdag/git/rdns/rdn-cc/rdn/gpt.py` (lines 349-384)

```python
# In rdn/gpt.py
def num_scaling_params(self):
    """
    Return detailed parameter counts for scaling law analysis.
    Returns dict with counts for each parameter group.
    """
    wte = sum(p.numel() for p in self.transformer.wte.parameters())
    value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
    lm_head = sum(p.numel() for p in self.lm_head.parameters())
    transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
    scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()

    # Add coordinate embeddings if enabled
    coord_embeds = 0
    if self.config.use_coordinate_embeddings:
        # ... count coordinate params

    # Add MoE params if enabled
    moe_params = 0
    if self.config.moe_enabled:
        for block in self.transformer.h:
            if hasattr(block, 'moe') and block.moe is not None:
                moe_params += sum(p.numel() for p in block.moe.parameters())

    total = wte + value_embeds + lm_head + transformer_matrices + scalars + coord_embeds + moe_params
    assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"

    return {
        'wte': wte,
        'value_embeds': value_embeds,
        'lm_head': lm_head,
        'transformer_matrices': transformer_matrices,
        'scalars': scalars,
        'coord_embeds': coord_embeds,
        'moe_params': moe_params,
        'total': total,
    }
```

**Use in base_train.py:**
```python
# Line ~200: After model initialization
param_counts = orig_model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
```

### 6.4 Add MFU Analysis

**Compare MFU calculation between repos:**

**rdn-cc approach (lines 318-347):**
```python
def estimate_flops(self):
    nparams = sum(p.numel() for p in self.parameters())
    # Exclude non-matmul params
    value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
    coord_embeds_numel = sum(ce.weight.numel() for ce in self.coord_embeddings)
    nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                      coord_embeds_numel + self.resid_lambdas.numel() + self.x0_lambdas.numel())

    h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len

    # Sum attention FLOPs per layer, accounting for sliding window
    attn_flops = 0
    for window_size in self.window_sizes:
        window = window_size[0]
        effective_seq = t if window < 0 else min(window, t)
        attn_flops += 12 * h * q * effective_seq

    num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
    return num_flops_per_token
```

**Ensure rdn7-ttt has similar calculation, including MoE params.**

---

## Phase 7: Testing & Validation (Day 6, Full Day)

### 7.1 Create Comprehensive Test Suite

**Directory structure:**
```
tests/
├── test_gpu_coordinates.py        # GPU coordinate calculation
├── test_hierarchical_dataloader.py # Hierarchical dataloader with coords
├── test_prefetch_dataloader.py    # Prefetch wrapper
├── test_optimizer.py              # MuonAdamW optimizer
├── test_model_features.py         # Value embeddings, scalars, windows
├── test_gpt_flag.py               # --gpt flag (critical!)
├── test_moe.py                    # MoE functionality preserved
├── test_generator.py              # Calculator/tictactoe generators
├── test_e2e_training.py           # End-to-end training smoke tests
└── test_performance.py            # Performance benchmarks
```

### 7.2 Create test.sh Runner

**Create:** `tests/test.sh`

```bash
#!/bin/bash
set -e  # Exit on first error

echo "================================"
echo "RDN7-TTT Test Suite"
echo "================================"

# Unit tests
echo ""
echo "[1/10] Testing GPU Coordinates..."
python -m pytest tests/test_gpu_coordinates.py -v -s

echo ""
echo "[2/10] Testing Hierarchical Dataloader..."
python -m pytest tests/test_hierarchical_dataloader.py -v -s

echo ""
echo "[3/10] Testing Prefetch Dataloader..."
python -m pytest tests/test_prefetch_dataloader.py -v -s

echo ""
echo "[4/10] Testing Optimizer..."
python -m pytest tests/test_optimizer.py -v -s

echo ""
echo "[5/10] Testing Model Features..."
python -m pytest tests/test_model_features.py -v -s

echo ""
echo "[6/10] Testing --gpt Flag..."
python -m pytest tests/test_gpt_flag.py -v -s

echo ""
echo "[7/10] Testing MoE..."
python -m pytest tests/test_moe.py -v -s

echo ""
echo "[8/10] Testing Generators..."
python -m pytest tests/test_generator.py -v -s

echo ""
echo "[9/10] End-to-End Training Tests..."
python -m pytest tests/test_e2e_training.py -v -s

echo ""
echo "[10/10] Performance Benchmarks..."
python -m pytest tests/test_performance.py -v -s

echo ""
echo "================================"
echo "✓ All tests passed!"
echo "================================"
```

### 7.3 End-to-End Training Tests

**Create:** `tests/test_e2e_training.py`

```python
def test_rdn_mode_training():
    """Test full training pipeline with RDN (coordinates)"""
    # Train d4 model for 100 steps
    # Verify loss decreases
    # Verify no crashes
    pass

def test_gpt_mode_training():
    """Test full training pipeline with GPT (no coordinates)"""
    # Train d4 model for 100 steps with --gpt
    # Verify loss decreases
    # Verify no crashes (critical!)
    pass

def test_moe_training():
    """Test training with MoE enabled"""
    # Train d4 model with MoE for 100 steps
    # Verify experts are used
    pass

def test_generator_training():
    """Test training with calculator generator"""
    # Train d4 model on calculator data
    # Verify generator works
    pass

def test_distributed_training():
    """Test DDP training on 2 GPUs"""
    # Use torchrun with 2 processes
    # Verify gradients sync correctly
    pass

def test_resume_training():
    """Test checkpoint resume"""
    # Train 50 steps, save checkpoint
    # Resume for 50 more steps
    # Verify continuity
    pass
```

### 7.4 Performance Benchmarks

**Create:** `tests/test_performance.py`

```python
def test_coordinate_calculation_speed():
    """Benchmark GPU coordinate calculation (target: >15M tok/sec)"""
    pass

def test_dataloader_throughput():
    """Benchmark dataloader throughput"""
    pass

def test_training_step_time():
    """Benchmark single training step (target: <150ms on 8xH100)"""
    pass

def test_prefetch_speedup():
    """Verify prefetch improves throughput by 20-40%"""
    pass

def test_optimizer_step_time():
    """Benchmark optimizer step time"""
    pass

def test_end_to_end_tokens_per_sec():
    """Measure sustained tokens/sec over 100 steps"""
    pass
```

### 7.5 Create Smoke Test Scripts

**Create:** `test_runs/smoke_test_rdn.sh`

```bash
#!/bin/bash
# Quick smoke test for RDN mode

python -m scripts.base_train \
  --depth=4 \
  --max-seq-len=512 \
  --device-batch-size=4 \
  --total-batch-size=2048 \
  --num-iterations=50 \
  --eval-every=25 \
  --save-every=-1 \
  --run=smoke_test_rdn

echo "✓ RDN smoke test passed"
```

**Create:** `test_runs/smoke_test_gpt.sh`

```bash
#!/bin/bash
# Quick smoke test for GPT mode (--gpt flag)

python -m scripts.base_train \
  --depth=4 \
  --max-seq-len=512 \
  --device-batch-size=4 \
  --total-batch-size=2048 \
  --num-iterations=50 \
  --eval-every=25 \
  --save-every=-1 \
  --gpt \
  --run=smoke_test_gpt

echo "✓ GPT smoke test passed"
```

**Create:** `test_runs/smoke_test_moe.sh`

```bash
#!/bin/bash
# Quick smoke test for MoE

python -m scripts.base_train \
  --depth=4 \
  --max-seq-len=512 \
  --device-batch-size=4 \
  --total-batch-size=2048 \
  --num-iterations=50 \
  --eval-every=25 \
  --save-every=-1 \
  --moe-enabled \
  --run=smoke_test_moe

echo "✓ MoE smoke test passed"
```

---

## Phase 8: Documentation & Cleanup (Day 7, 2-3 hours)

### 8.1 Update README.md

Add sections:
- [ ] GPU coordinate calculation
- [ ] Polar Express optimizer
- [ ] Prefetch dataloader
- [ ] Value embeddings
- [ ] --gpt flag for baseline comparisons
- [ ] Performance benchmarks (before/after)

### 8.2 Update CLAUDE.md

Update project instructions with:
- [ ] New rdn/ directory structure
- [ ] rdt/ Rust tokenizer
- [ ] Updated commands with new features
- [ ] Testing instructions

### 8.3 Create Performance Documentation

**Create:** `PERFORMANCE.md`

Document:
- [ ] Speedup achieved (target: 3-5x)
- [ ] Bottleneck analysis (before/after)
- [ ] Profiling results
- [ ] Optimization techniques used
- [ ] Future optimization opportunities

### 8.4 Clean Up Unused Files

After confirming everything works:
- [ ] Remove old `nanochat/muon.py` (replaced by `rdn/optim.py`)
- [ ] Remove old `nanochat/adamw.py` (merged into `rdn/optim.py`)
- [ ] Remove `nanochat/utils.py` if `compute_coords()` no longer needed
- [ ] Remove any other obsolete files

---

## Expected Results

### Performance Improvements

| Metric | Before (rdn7-ttt) | After (with rdn-cc optimizations) | Improvement |
|--------|-------------------|-----------------------------------|-------------|
| Step time | ~750ms | ~150ms | **5x faster** |
| Tokens/sec | ~87k | ~400k+ | **4.6x faster** |
| GPU utilization | 60-70% (stalls) | 90-95% (sustained) | **Consistent** |
| Coordinate calc | CPU (1.5M tok/sec) | GPU (15M+ tok/sec) | **10x faster** |
| Batch prefetch | Manual (1 ahead) | Queue (3 ahead) | **No starvation** |

### Feature Completeness

**Preserved from rdn7-ttt:**
- ✓ DeepSeek MoE implementation
- ✓ Generator support (calculator/tictactoe)
- ✓ Experimental features
- ✓ mHC/Streams architecture

**Added from rdn-cc:**
- ✓ GPU coordinate calculation
- ✓ Polar Express optimizer
- ✓ Variance reduction in Muon
- ✓ Cautious weight decay
- ✓ Prefetch dataloader
- ✓ Value embeddings (ResFormer)
- ✓ Per-layer scalars
- ✓ Sliding window attention (infrastructure)
- ✓ BOS-aligned best-fit packing
- ✓ Verbose logging
- ✓ --gpt flag support (fixed)

---

## Risk Mitigation

### Potential Issues

1. **Compatibility Issues**
   - Risk: New optimizer incompatible with MoE
   - Mitigation: Test MoE thoroughly, add special handling in setup_optimizer()

2. **Gradient Accumulation**
   - Risk: Polar Express may not work well with grad accumulation
   - Mitigation: Keep grad accumulation support from rdn7-ttt Muon

3. **--gpt Flag Still Crashes**
   - Risk: Fix doesn't fully resolve inplace operation issue
   - Mitigation: Add extensive testing, try alternative fixes

4. **Performance Regression**
   - Risk: Some optimizations don't translate to rdn7-ttt architecture
   - Mitigation: Benchmark at each phase, roll back if needed

5. **Generator Compatibility**
   - Risk: Hierarchical dataloader breaks calculator/tictactoe
   - Mitigation: Test generators explicitly, maintain separate code paths

### Rollback Plan

Each phase is independent:
- If Phase N fails, can rollback to Phase N-1
- Git branches for each phase
- Keep original files until fully tested

---

## Success Criteria

**Phase 1 (GPU Coordinates):**
- [ ] Coordinate calculation on GPU works
- [ ] Dataloader yields coordinates
- [ ] Training runs without crashes
- [ ] Step time improves by 50-150ms

**Phase 2 (Optimizer):**
- [ ] MuonAdamW replaces separate optimizers
- [ ] Polar Express orthogonalization works
- [ ] Variance reduction improves convergence
- [ ] MoE parameters handled correctly

**Phase 3 (--gpt Flag):**
- [ ] --gpt flag doesn't crash
- [ ] Standard GPT mode trains successfully
- [ ] Can compare RDN vs GPT performance

**Phase 4 (Prefetch):**
- [ ] Prefetch eliminates GPU starvation
- [ ] Throughput improves 20-40%
- [ ] No crashes or hangs

**Phase 5 (Model Features):**
- [ ] Value embeddings work
- [ ] Per-layer scalars work
- [ ] Sliding window infrastructure in place

**Phase 6 (Logging):**
- [ ] All metrics logged (ETA, MFU, throughput, etc.)
- [ ] Memory tracking works
- [ ] Detailed parameter counts

**Phase 7 (Testing):**
- [ ] All unit tests pass
- [ ] End-to-end tests pass
- [ ] Performance benchmarks meet targets
- [ ] test.sh runs without errors

**Phase 8 (Documentation):**
- [ ] README updated
- [ ] CLAUDE.md updated
- [ ] PERFORMANCE.md created
- [ ] Unused files removed

---

## Timeline Estimate

| Phase | Duration | Priority |
|-------|----------|----------|
| Phase 0: Restructuring | 1-2 hours | CRITICAL |
| Phase 1: GPU Coordinates | 3-4 hours | CRITICAL |
| Phase 2: Optimizer | 4-6 hours | CRITICAL |
| Phase 3: Fix --gpt Flag | 2-3 hours | CRITICAL |
| Phase 4: Prefetch | 1-2 hours | HIGH |
| Phase 5: Model Features | 4-6 hours | MEDIUM |
| Phase 6: Logging | 2-3 hours | MEDIUM |
| Phase 7: Testing | 8 hours | CRITICAL |
| Phase 8: Documentation | 2-3 hours | LOW |
| **Total** | **27-37 hours** | **~4-5 days** |

---

## Next Steps

1. **Review this plan** - Verify priorities and approach
2. **Create git branch** - `git checkout -b optimize-rdn7-ttt`
3. **Start Phase 0** - Rename directories (nanochat → rdn, rustbpe → rdt)
4. **Proceed sequentially** - Don't skip phases
5. **Test frequently** - Run smoke tests after each major change
6. **Benchmark early** - Measure performance improvements at each phase

---

## Questions to Resolve Before Starting

1. **Bigram embeddings (engram-lite):** Neither repo has them currently. Should we port from rdn-cc dev logs? (Adds 126M params but improves loss)

2. **Flash Attention 3:** Park until H100 available, or implement SDPA fallback with window support?

3. **rdt/ vs rustbpe/:** Confirm we want to rename and update all Cargo.toml references?

4. **Coordinate calculation speed:** User mentioned rdn-cc tokenizer "struggles to keep up" - should we verify GPU coordinates are actually faster?

5. **Testing on MPS:** The --gpt flag crash happened on MPS (Mac). Should we prioritize CUDA testing or ensure MPS compatibility?

---

## Appendix: File Mapping

### Files to Port from rdn-cc

| Source | Destination | Priority |
|--------|-------------|----------|
| `rdn-cc/rdn/gpu_coordinates.py` | `rdn7-ttt/rdn/gpu_coordinates.py` | P0 |
| `rdn-cc/rdn/hierarchical_dataloader.py` | `rdn7-ttt/rdn/hierarchical_dataloader.py` | P0 |
| `rdn-cc/rdn/prefetch_dataloader.py` | `rdn7-ttt/rdn/prefetch_dataloader.py` | P1 |
| `rdn-cc/rdn/optim.py` | `rdn7-ttt/rdn/optim.py` | P0 |
| `rdn-cc/rdn/flash_attention.py` | `rdn7-ttt/rdn/flash_attention.py` | P2 (parked) |

### Files to Update in rdn7-ttt

| File | Changes | Priority |
|------|---------|----------|
| `rdn/gpt.py` | Add VE, scalars, window sizes, fix --gpt | P0 |
| `rdn/dataloader.py` | Add get_dataloader(), integrate coords | P0 |
| `scripts/base_train.py` | Update dataloader usage, logging, optimizer | P0 |
| `scripts/mid_train.py` | Update dataloader usage, optimizer | P1 |
| `scripts/chat_sft.py` | Update dataloader usage, optimizer | P1 |
| `scripts/chat_rl.py` | Update dataloader usage, optimizer | P1 |

### Files to Delete

| File | Reason |
|------|--------|
| `nanochat/muon.py` | Replaced by `rdn/optim.py` |
| `nanochat/adamw.py` | Merged into `rdn/optim.py` |
| `nanochat/utils.py` | `compute_coords()` replaced by GPU version |

---

## Appendix: Key Code Snippets

### GPU Coordinate Calculation Core

```python
# From rdn-cc/rdn/gpu_coordinates.py
def calculate_coordinates_batch(self, token_ids: torch.Tensor) -> torch.Tensor:
    B, T = token_ids.shape

    # Detect brackets: [B, T] with values {-1, 0, 1}
    clamped_ids = torch.clamp(token_ids, 0, len(self.bracket_map) - 1)
    bracket_deltas = self.bracket_map[clamped_ids]

    # Parallel scan to get depth at each position
    depths = torch.cumsum(bracket_deltas, dim=1)
    depths = torch.clamp(depths, 0, self.n_dim - 1)

    # Initialize coordinates
    coordinates = torch.zeros(B, T, self.n_dim, dtype=torch.long, device=self.device)

    # For each depth level, count positions
    for dim in range(self.n_dim):
        at_depth = (depths == dim)
        position_at_depth = torch.cumsum(at_depth.long(), dim=1)
        coordinates[:, :, dim] = torch.where(at_depth, position_at_depth - 1, 0)

    # BOS token gets [0,0,0,...]
    if self.bos_token is not None:
        bos_mask = (token_ids == self.bos_token)
        if bos_mask.any():
            coordinates[bos_mask] = 0

    return coordinates
```

### Polar Express Orthogonalization

```python
# From rdn-cc/rdn/optim.py
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(...):
    # Nesterov momentum
    momentum_buffer.lerp_(stacked_grads, 1 - momentum_t)
    g = stacked_grads.lerp_(momentum_buffer, momentum_t)

    # Polar Express
    X = g / (X.norm(...) * 1.02 + 1e-6)
    if tall:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    # Variance reduction
    v_mean = X.float().square().mean(dim=red_dim, keepdim=True)
    second_momentum_buffer.lerp_(v_mean, 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    # ... scale X by step_size

    # Cautious weight decay
    mask = (X * stacked_params) >= 0
    stacked_params.sub_(lr * X + lr * wd * stacked_params * mask)
```

---

**End of Plan**

This plan provides a comprehensive roadmap to merge the fast rdn-cc implementation into the feature-complete rdn7-ttt codebase. Follow phases sequentially, test thoroughly, and benchmark at each step. The expected 3-5x speedup should make training significantly more efficient while preserving all experimental features.
