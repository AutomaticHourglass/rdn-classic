"""
Train model. From root directory of the project, run as:

python -m scripts.base_train.py

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os

from thefuzz import fuzz

from generator.generator_calc import safe_eval_math

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse

# -----------------------------------------------------------------------------
# AGENT SELECTION - Change this to train different agents
# -----------------------------------------------------------------------------
AGENT = "rdn"  # Options: "rdn", "calculator", "tictactoe"
# "rdn" = normal pretraining on text data (standard nanochat)
# "calculator"/"tictactoe" = task-specific agents with generators
# -----------------------------------------------------------------------------
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, \
    autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument("--run", type=str, default="", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device_type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect_ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head_dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--n_heads", type=int, default=-1, help="Number of heads, -1 to auto define")
parser.add_argument("--n_kv_heads", type=int, default=-1, help="Number of kv heads, -1 to disable GQA")
parser.add_argument("--max_seq_len", type=int, default=8192, help="max context length")
parser.add_argument("--page_size", type=int, default=512, help="Paged attention page size")
# mHC
parser.add_argument("--n_streams", type=int, default=1, help="mHC num streams, 1 to disable")
# Racursion
parser.add_argument("--recursion", type=int, default=8, help="Recursion depth cap for the tokenizer.")

# Training horizon (only one used, in order of precedence)
parser.add_argument("--num_iterations", type=int, default=-1,
                    help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target_flops", type=float, default=-1.0,
                    help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target_param_data_ratio", type=int, default=20,
                    help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device_batch_size", type=int, default=8, help="per-device batch size")
parser.add_argument("--total_batch_size", type=int, default=65536, help="total batch size in tokens")
parser.add_argument("--embedding_lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding_lr", type=float, default=0.004,
                    help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight_decay", type=float, default=0.0,
                    help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--matrix_lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--adam_beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup_ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown_ratio", type=float, default=0.4, help="ratio of iterations for LR warmdown")
parser.add_argument("--final_lr_frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume_from_step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval_every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval_tokens", type=int, default=524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core_metric_every", type=int, default=-1,
                    help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core_metric_max_per_task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample_every", type=int, default=250, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save_every", type=int, default=250, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model_tag", type=str, default=None, help="override model tag for checkpoint directory name")
# Agent selection
parser.add_argument("--agent", type=str, default=AGENT, choices=["rdn", "calculator", "tictactoe"],
                    help="which agent to train (rdn=pretraining, calculator, tictactoe)")
# GPT mode (standard architecture)
parser.add_argument("--gpt", action="store_true", default=False,
                    help="use standard GPT architecture (RoPE only, no coordinate embeddings)")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type,
                                  dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer(not args.gpt)
if args.gpt:
    print0("Using RustBPE tokenizer (GPT mode: no UP/DOWN markers, no coordinate embeddings)")
else:
    print0("Using RustBPE tokenizer (RDN mode: with UP/DOWN markers and coordinate embeddings)")

token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = args.depth
model_dim = args.n_heads * args.aspect_ratio


def find_num_heads(model_dim, target_head_dim):
    # Find num_heads that divides model_dim evenly, with head_dim closest to target.
    ideal = max(1, round(model_dim / target_head_dim))
    for offset in range(model_dim):
        for candidate in [ideal + offset, ideal - offset]:
            if candidate > 0 and model_dim % candidate == 0:
                return candidate
    return 1


if args.n_heads == -1:
    num_heads = find_num_heads(model_dim, args.head_dim)
    num_kv_heads = num_heads  # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
else:
    num_heads = args.n_heads
assert num_heads % args.n_kv_heads == 0
if args.n_kv_heads > 0:
    num_kv_heads = args.n_kv_heads  # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")
print0(f"num_streams: {args.n_streams}")
print0(f"gpt_mode: {args.gpt} (coordinate embeddings: {not args.gpt})")

# print0(f"n_experts: {args.n_experts}")
# print0(f"n_shared_experts: {args.n_shared_experts}")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
run_name = ["GPT" if args.gpt else "RDN"]
if args.n_streams > 1:
    run_name += ['mHC']
# if args.n_experts > 1:
#     run_name += ['MoE']
# if len(run_name) == 0:
#     run_name = "RDN"
# else:
run_name = " + ".join(run_name)

if args.model_tag:
    run_name = args.model_tag
else:
    param_part = f" D{args.depth}_H{num_heads}_HKV{num_kv_heads}_R{args.head_dim}_V{vocab_size}"
    run_name = run_name + param_part

if args.agent == 'calculator':
    project = 'agent-calc'
elif args.agent == 'tictactoe':
    project = 'agent-tictactoe'
else:
    project = 'RDN-classic'

wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project=project, name=run_name, config=user_config)

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len  # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Batch size scaling for learning rates (hyperparameters were tuned at reference batch size 2^19)
batch_lr_scale = 1.0
reference_batch_size = 2 ** 19
batch_ratio = args.total_batch_size / reference_batch_size
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard
    # Muon: sqrt scaling is an assumption - not fully studied, but it's a second-order-ish optimizer
    batch_lr_scale = batch_ratio ** 0.5
    print0(
        f"Scaling LRs by {batch_lr_scale:.4f} for batch size {args.total_batch_size:,} (reference: {reference_batch_size:,})")

# -----------------------------------------------------------------------------
# Initialize the Model

# Create a new model with random weights
model_config_kwargs = dict(
    sequence_len=args.max_seq_len,
    vocab_size=vocab_size,
    n_head=num_heads,
    n_dimensions=args.recursion,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    n_streams=args.n_streams,
    page_size=args.page_size,
    use_coordinate_embeddings=not args.gpt  # GPT mode disables coordinate embeddings
)
with torch.device("meta"):
    # All tensors are created as meta tensors (they have shape/dtype but no data)
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)  # All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights()  # All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"  # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device,
                                                            load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data  # free up this memory after the copy

orig_model = model  # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
# model = torch.compile(model, dynamic=True) # the inputs to model will never change shape so dynamic=False is safe
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(args.target_flops / (num_flops_per_token * args.total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio (use scaling params per Kaplan et al.)
    target_tokens = args.target_param_data_ratio * num_params
    num_iterations = target_tokens // args.total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = args.total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {args.total_batch_size * num_iterations / num_params:.2f}")  # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
adam_betas = (args.adam_beta1, args.adam_beta2)
optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=args.weight_decay,
)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data  # free up the memory

# -----------------------------------------------------------------------------
# Configure agent (train only selected agent)
use_generator = True  # Default: use generators
if args.agent == "rdn":
    use_generator = False  # Use normal text dataloader
    print0(f"Training mode: RDN (pretraining on text data)")
elif args.agent == "calculator":
    from generator import configure_mix
    configure_mix(calculator=1.0, tictactoe=0.0)
    print0(f"Training agent: CALCULATOR (math expressions)")
elif args.agent == "tictactoe":
    from generator import configure_mix
    configure_mix(calculator=0.0, tictactoe=1.0)
    print0(f"Training agent: TIC-TAC-TOE (game state management)")
else:
    raise ValueError(f"Unknown agent: {args.agent}")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state(args.device_batch_size, args.max_seq_len, split="train",
                                                             device=device,
                                                             resume_state_dict=dataloader_resume_state_dict,
                                                             use_generator=use_generator, use_recursive_markers= not args.gpt)
build_val_loader = lambda: tokenizing_distributed_data_loader(args.device_batch_size, args.max_seq_len, split="val",
                                                              device=device, use_generator=use_generator, use_recursive_markers= not args.gpt)
x, y = next(train_loader)  # kick off load of the very first batch of data


# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac


# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    val_bpb = None  # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0  # EMA of training loss
    total_training_time = 0  # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = step == num_iterations  # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # ONLY for RDN mode (not for generator-based agents)
    results = {}
    if args.agent == "rdn" and args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    # ONLY for generator-based agents (not for RDN pretraining)
    if use_generator and master_process and args.sample_every > 0 and (last_step or step % args.sample_every == 0):
        model.eval()
        engine = Engine(orig_model, tokenizer)
        total_eval = 0
        total_corr = 0
        total_score = 0
        tool_correctness = -1e-2

        # Get agent-specific evaluator
        from generator.evaluator import get_evaluator
        evaluator = get_evaluator(args.agent)

        # 1. Collection Phase: Gather all prompts and expected values
        batch_prompts = []      # List[List[int]]
        batch_expecteds = []    # List[str]
        batch_qs = []           # List[str] (for printing)

        # Iterate through the loader batch to extract all problems
        q = 8192 // args.max_seq_len
        all_generate = []
        val_loader = build_val_loader()
        for pp in next(val_loader)[0].tolist()[:q]:
            s = tokenizer.decode(pp)
            all_ps = s.split('<|bos|>')[1:-1]
            for p in all_ps:
                # Use agent-specific parser
                prompt, expected = evaluator.parse_problem(p)
                if prompt is not None and expected is not None:
                    batch_prompts.append(tokenizer.encode(prompt))
                    batch_expecteds.append(expected)
                    batch_qs.append(prompt)

        # 2. Inference Phase: Run everything in parallel
        if batch_prompts:
            # Initialize results with the prompts (just like generate_batch did)
            batch_results = [p[:] for p in batch_prompts]
            batch_completed = [False] * len(batch_prompts)

            # Cache special tokens for termination check
            assistant_end = tokenizer.encode_special("<|assistant_end|>")
            bos = tokenizer.get_bos_token_id()

            with autocast_ctx:
                # Consume the batched generator
                for token_cols, _ in engine.generate_batched(
                    batch_prompts,
                    max_tokens=args.max_seq_len // 2,
                    temperature=0
                ):
                    for i, token in enumerate(token_cols):
                        if not batch_completed[i]:
                            if token == assistant_end or token == bos:
                                batch_completed[i] = True
                            else:
                                batch_results[i].append(token)

            # 3. Evaluation Phase: Process results using agent-specific evaluator
            for i, sample in enumerate(batch_results):
                total_eval += 1
                prompt = batch_qs[i]
                expected = batch_expecteds[i]
                predicted = tokenizer.decode(sample)

                # Use agent-specific evaluation
                res = evaluator.evaluate(prompt, expected, predicted)
                all_generate.append(res)

                total_corr += res['corr']
                total_score += res['score']

            sorted_all = sorted(
                all_generate,
                key=lambda r: (
                    -int(r['corr']),          # 1 → first, 0 → second
                    -float(r['score'])        # higher score first
                )
            )

            for r in sorted_all:
                print0(f"{r['score']:.2f}-{r['corr']}-{r['prompt']}{r['pred']}")
            
            tool_correctness = total_corr / (total_eval + 1e-6)
            tool_score = total_score / (total_eval + 1e-6)
            wandb_run.log({
                "step": step,
                "tool_correctness": tool_correctness,
                "tool_score": tool_score,
            })
            print0(f"Average tool call correctness: {tool_correctness:.4f}")
            print0(f"Average tool call score: {tool_score:.4f}")
            model.train()

        ddp_rank = int(os.environ.get("RANK", 0))
        if ddp_rank == 0:
            output_dirname = f"d{args.depth}"  # e.g. d12
            checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(),
                [opt.state_dict() for opt in optimizers],  # optimizer states
                {
                    "step": step,
                    # "val_bpb": val_bpb,  # loss at last step
                    # "min-val/bpb": min_val_bpb,
                    "model_config": model_config_kwargs,
                    "user_config": user_config,  # inputs to the training script
                    "device_batch_size": args.device_batch_size,
                    "max_seq_len": args.max_seq_len,
                },
            )

            if args.sample_every > 0 and step > 3 * args.sample_every and step % args.sample_every == 0:
                delete_step = step - 3 * args.sample_every
                try:
                    os.remove(f"{checkpoint_dir}/meta_{delete_step:06d}.json")
                    os.remove(f"{checkpoint_dir}/model_{delete_step:06d}.pt")
                    for wr in range(ddp_world_size):
                        os.remove(
                            f"{checkpoint_dir}/optim_{delete_step:06d}_rank{wr}.pt"
                        )

                except FileNotFoundError:
                    pass

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()  # for logging
        loss = loss / grad_accum_steps  # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y = next(train_loader)  # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizers`
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    ema_beta = 0.9  # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()  # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))  # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size  # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100  # in %
    if step > 1:
        total_training_time += dt  # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 1
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds / 60:.1f}m"
    else:
        eta_str = ""
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.3f}%) | loss: {debiased_smooth_loss:1.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:02.2f} | total time: {total_training_time / 60:.2f}m{eta_str}")
    if step % 1 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        wandb_run.log(log_data)

    # state update
    step += 1

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report

get_report().log(section="Base model training", data=[
    user_config,  # CLI args
    {  # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": args.total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": args.warmup_ratio,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    {  # stats about training outcomes
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time / 60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish()  # wandb run finish
compute_cleanup()
