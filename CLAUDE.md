# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nanochat** is a full-stack implementation of an LLM like ChatGPT in a single, minimal, hackable codebase. It trains models from scratch through tokenization, pretraining, finetuning, evaluation, and inference. The project is designed to be compute-efficient, running complete pipelines on a single 8XH100 node.

Project name in pyproject.toml: `recursive_dimensional_network` (version 0.0.4)

## Common Commands

### Environment Setup
```bash
# Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate

# Build Rust tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Training Pipelines

**Quick speedrun (~4 hours, ~$100 on 8XH100):**
```bash
bash speedrun.sh
```

**Full pipeline (~33 hours, ~$800 on 8XH100):**
```bash
bash run1000.sh
```

**With wandb logging:**
```bash
WANDB_RUN=my_run_name bash speedrun.sh
```

**In screen session (recommended for long runs):**
```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

### Individual Training Stages

**Stage 1: Tokenizer training**
```bash
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval
```

**Stage 2: Base training (pretraining)**
```bash
# Single GPU
python -m scripts.base_train -- --depth=20 --device_batch_size=8

# Multi-GPU (8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --device_batch_size=8

# Evaluate
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval
```

**Stage 3: Midtraining**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=8
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid
```

**Stage 4: Supervised Fine-Tuning (SFT)**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device_batch_size=8
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

**Stage 5: Reinforcement Learning (optional)**
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K
```

### Inference

**CLI chat:**
```bash
# Interactive
python -m scripts.chat_cli

# Single prompt
python -m scripts.chat_cli -p "Why is the sky blue?"
```

**Web UI:**
```bash
python -m scripts.chat_web
# Then visit http://localhost:8000 (or use public IP on cloud instances)
```

### Testing

```bash
# Run all tests
python -m pytest tests/ -v -s

# Run specific test file
python -m pytest tests/test_rustbpe.py -v -s
```

### Data Management

```bash
# Download pretraining data shards (each ~250M chars, ~100MB compressed)
python -m nanochat.dataset -n 8    # Download 8 shards (~2B chars)
python -m nanochat.dataset -n 240  # Download 240 shards (~60B chars)
```

### Reports

```bash
# Generate training report
python -m nanochat.report generate

# View report
cat report.md
```

## High-Level Architecture

### Training Pipeline Flow

The codebase implements a 5-stage pipeline that transforms raw text into a conversational AI:

1. **Tokenizer Training** → Trains BPE tokenizer (vocab size 65,536) using custom Rust implementation (rustbpe)
2. **Base Training (Pretraining)** → Next-token prediction on FinWeb-Edu dataset following Chinchilla scaling (20:1 tokens:params ratio)
3. **Midtraining** → Fine-tunes on conversation/instruction data to learn special tokens and formats
4. **Supervised Fine-Tuning (SFT)** → Trains on curated conversation examples
5. **Reinforcement Learning (RL)** → Optional GRPO-style reward optimization (currently GSM8K math)

Each stage produces checkpoints in dedicated directories: `base_checkpoints/`, `mid_checkpoints/`, `chatsft_checkpoints/`, `chatrl_checkpoints/`.

### Core Components

**Model (`nanochat/gpt.py`):**
- GPT-style Transformer with modern optimizations
- Rotary positional embeddings (no learned positions)
- QK normalization for stability
- ReLU² activation (not GELU)
- Group-Query Attention (GQA) for efficient inference
- Untied embedding/unembedding weights
- Configurable depth (layers), width (n_embd), heads

**Tokenizer (`nanochat/tokenizer.py` + `rustbpe/`):**
- Custom Rust BPE implementation for training efficiency
- GPT-4 style split patterns
- Special tokens: `<|bos|>`, `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`, `<|python_start|>`, `<|python_end|>`, `<|output_start|>`, `<|output_end|>`
- Supports both HuggingFace and RustBPE backends

**Data Loading (`nanochat/dataloader.py`):**
- `tokenizing_distributed_data_loader()` provides infinite stream of tokenized batches
- On-the-fly tokenization with multiple threads (default 4)
- DDP-aware data distribution using stride pattern
- Loads from parquet files containing raw text
- Stateful for checkpoint resume

**Inference Engine (`nanochat/engine.py`):**
- `KVCache`: Manages cached key/value tensors for efficient autoregressive generation
- `RowState`: Per-sequence state tracking (tool use, forced tokens)
- Batched generation with cache replication
- Tool use integration (Python calculator with safe evaluation)
- Left-padding support for batched different-length prompts

**Optimizers (Distributed):**
- `DistAdamW` (`adamw.py`): ZeRO-2 style sharded AdamW for embeddings
  - Different learning rates for embedding (0.3) vs unembedding (0.004)
  - Gradient reduce_scatter for memory efficiency
- `Muon` (`muon.py`): Newton-Schulz orthogonalization optimizer for 2D weight matrices
  - Gradient accumulation support
  - LR typically 0.02

**Task System (`tasks/`):**
- Base `Task` class with eval methods
- `TaskMixture`: Combines multiple tasks with deterministic shuffling
- `TaskSequence`: Sequential training across tasks
- Conversation format: list of messages with roles (user/assistant) and content

### Key Design Patterns

**Configurator Pattern (`nanochat/configurator.py`):**
- Uses `exec()` to inject CLI overrides into global config variables
- Avoids argparse boilerplate while maintaining flexibility
- All hyperparameters defined as module-level globals, overridable via `--param=value`

**Distributed Training:**
- DDP (Distributed Data Parallel) via `torchrun`
- Rank-based data sharding with stride pattern
- AllReduce for gradient averaging
- File-based locking for safe multi-rank downloads

**Memory Management:**
- Dynamic KV cache growth
- Gradient accumulation compensates for VRAM limits
- `torch.compile()` for memory/speed optimization
- Configurable `device_batch_size` (default 32 for speedrun, reduce to 16/8/4/2/1 if OOM)

**Checkpoint Management (`nanochat/checkpoint_manager.py`):**
- Auto-detection of latest checkpoint by tag and step
- Handles torch.compile artifacts (`_orig_mod.` prefix stripping)
- JSON metadata for run configuration
- Directory structure: `{stage}_checkpoints/d{depth}/model_{step:06d}.pt`

## Important Technical Details

### Scaling and Resource Management

**Chinchilla Optimal Scaling:**
- Training tokens = 20 × model parameters
- Used to auto-compute training iterations in base_train.py

**Model Sizes:**
- d20 (depth=20): 561M params, ~$100 budget, 4 hours on 8XH100
- d26 (depth=26): slightly outperforms GPT-2, ~$300 budget, 12 hours
- d32 (depth=32): 1.9B params, ~$800 budget, 33 hours (run1000.sh)

**Data Requirements:**
- Each shard: ~250M chars (~100MB compressed)
- Tokenizer compression: ~4.8 chars/token
- Calculate shards needed: (params × 20 × 4.8) / 250M

**Memory Tuning:**
- If OOM, reduce `device_batch_size` (code auto-adjusts gradient accumulation)
- Example: d32 fits at device_batch_size=8 on 80GB H100
- d26 needs device_batch_size=16, d20 uses device_batch_size=32

### CPU/MPS Support

Can run on CPU or Apple Silicon (MPS) with modified hyperparameters. See `dev/runcpu.sh` for example of scaled-down configurations.

### Tool Use (Python Calculator)

Model can execute Python code for calculations:
- Generates code between `<|python_start|>` and `<|python_end|>` tokens
- Safe execution via `nanochat/execution.py` (no imports, no system calls)
- Output injected back into generation between `<|output_start|>` and `<|output_end|>`
- State machine in Engine tracks tool use flow

### Evaluation Metrics

- **CORE**: Composition, Reasoning, Explaining (from DCLM paper)
- **Bits-per-byte (BPB)**: Language modeling evaluation
- **Task-specific**: Multiple choice accuracy (ARC, MMLU), math correctness (GSM8K), code pass@1 (HumanEval)
- Report consolidates all metrics into `report.md`

## Project Structure Notes

- `nanochat/`: Core library modules (model, optimizer, data, inference)
- `scripts/`: Training and evaluation entry points
- `tasks/`: Dataset definitions and evaluation tasks
- `rustbpe/`: Custom Rust BPE tokenizer (Cargo project)
- `dev/`: Development utilities and examples
- `tests/`: Test suite (primarily tokenizer tests)

## Environment Variables

- `NANOCHAT_BASE_DIR`: Base directory for artifacts (default: `~/.cache/nanochat`)
- `WANDB_RUN`: W&B run name for logging (use "dummy" to skip wandb)
- `OMP_NUM_THREADS`: Set to 1 to avoid CPU thread contention

## Customization

See GitHub Discussions for guides:
- Infusing identity/personality: Modify synthetic data generation
- Adding abilities: Mix custom tasks into midtraining/SFT stages
