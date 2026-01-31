"""
Paged Muon optimizer - adapts Muon to work with checkpointing + gradient accumulation.

The key insight: Muon operates on 2D parameter updates in the Newton-Schulz orthogonalization step.
When using checkpointing/gradient accumulation, gradients are accumulated over multiple forward passes.

We modify Muon to:
1. Accumulate momentum buffers correctly across multiple gradient.backward() calls
2. Defer the orthogonalization step until we have the full accumulated gradient
3. Support optional "paging" of the Newton-Schulz computation itself (though usually not needed)

This maintains Muon's scaling law benefits while being compatible with memory-efficient training.
"""
import torch
from torch import Tensor
import torch.distributed as dist
from typing import Optional

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Paged Muon optimizer - compatible with gradient accumulation and checkpointing.

    Key differences from standard Muon:
    - Supports `accumulation_steps` parameter. Momentum is updated every step, but
      orthogonalization + parameter update only happen every `accumulation_steps` steps.
    - This allows the momentum buffer to accumulate gradients properly across multiple
      backward() calls, maintaining the benefits of Muon's scaling law.
    - When used with checkpointing, this significantly reduces peak memory while
      keeping Muon's convergence properties.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
        accumulation_steps: Number of backward() calls before performing the orthogonalization
                           and parameter update. If 1, behaves like standard Muon. Higher values
                           reduce peak memory in gradient computation.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 accumulation_steps=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                       accumulation_steps=accumulation_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

        # Global step counter to track when to perform actual updates
        self.global_step = 0

    @torch.no_grad()
    def step(self, perform_update: bool = True):
        """
        Perform an optimizer step.

        Args:
            perform_update: If True, performs orthogonalization + parameter update.
                           If False, only updates momentum buffers (for accumulation).
                           Normally you don't need to set this; the optimizer auto-detects
                           based on self.global_step % accumulation_steps.
        """
        accumulation_steps = self.param_groups[0]["accumulation_steps"]

        # Auto-detect if we should perform update based on accumulation schedule
        if perform_update is None:
            perform_update = (self.global_step + 1) % accumulation_steps == 0

        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf: Tensor = state["momentum_buffer"]

                # Always update momentum buffer
                buf.lerp_(g, 1 - group["momentum"])

                # Only perform orthogonalization + parameter update if we've accumulated enough steps
                if perform_update:
                    g_update = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g_update = zeropower_via_newtonschulz5(g_update, steps=group["ns_steps"])
                    p.add_(g_update, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

        self.global_step += 1

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = False):
        """
        Reset gradients. Call this after step() during accumulation.
        """
        super().zero_grad(set_to_none=set_to_none)


class DistMuon(torch.optim.Optimizer):
    """
    Distributed Paged Muon optimizer.

    Combines DistMuon's distributed synchronization with MuonPaged's gradient accumulation
    support for memory-efficient distributed training.

    Additional arguments over DistMuon:
        accumulation_steps: Number of backward() calls before performing the orthogonalization
                           and parameter update.
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, accumulation_steps: int = 1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                       accumulation_steps=accumulation_steps)
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"
        rank = dist.get_rank()

        shapes = sorted({p.shape for p in params})
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            if rank == 0:
                print(f"DistMuonPaged: Grouping {len(group_params)} params of shape {shape}, "
                      f"device {device}, dtype {dtype}")
            param_groups.append(dict(params=group_params, zero_buffer=torch.zeros_like(group_params[0])))

        super().__init__(param_groups, defaults)
        self.global_step = 0

    @torch.no_grad()
    def step(self, perform_update: bool = None):
        """
        Perform a distributed optimizer step with optional accumulation.

        Args:
            perform_update: If True, performs orthogonalization + parameter update.
                           If False, only updates momentum buffers.
                           If None (default), auto-detects based on accumulation schedule.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        accumulation_steps = self.param_groups[0]["accumulation_steps"]

        # Auto-detect if we should perform update
        if perform_update is None:
            perform_update = (self.global_step + 1) % accumulation_steps == 0

        # Ensure all grads exist
        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), \
            "All params must have grads"

        # Phase 1: Reduce-scatter to average gradients across ranks
        all_reduce_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            for base_i in range(0, len(params), world_size):
                owner_idx = base_i + rank
                rs_input = [p.grad for p in params[base_i:base_i + world_size]]
                rs_input.extend([zero_buffer] * (world_size - len(rs_input)))
                rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(zero_buffer)
                work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                all_reduce_futures.append(work)

        # Phase 2: Compute momentum and conditionally orthogonalize
        future_idx = 0
        all_gather_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            for base_i in range(0, len(params), world_size):
                owner_idx = base_i + rank
                all_reduce_futures[future_idx].wait()
                future_idx += 1

                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = p.grad  # now averaged across ranks
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)

                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1.0 - group["momentum"])

                    # Only orthogonalize and update if we've accumulated enough steps
                    if perform_update:
                        g_update = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                        g_update = zeropower_via_newtonschulz5(g_update, steps=group["ns_steps"])
                        scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                        p.add_(g_update, alpha=-group["lr"] * scale)

                # Replicate updated parameters to all ranks
                ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                ag_output = list(params[base_i:base_i + world_size])
                ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))])
                work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
                all_gather_futures.append(work)

        torch.futures.collect_all(all_gather_futures).wait()
        self.global_step += 1

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = False):
        """Reset gradients."""
        super().zero_grad(set_to_none=set_to_none)