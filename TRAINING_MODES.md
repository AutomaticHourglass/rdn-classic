# Training Modes

The training script supports three modes controlled by the `--agent` flag:

## 1. RDN Mode (Normal Pretraining)

**Default nanochat behavior** - Pretrain on text data from parquet files.

```python
AGENT = "rdn"
```

```bash
# Train with text data (normal pretraining)
python -m scripts.base_train -- --agent=rdn --depth=20

# Distributed training
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --agent=rdn --depth=20
```

**Features:**
- ✅ CORE metrics enabled
- ✅ Loads from parquet files (~/.cache/nanochat/data/)
- ✅ Standard pretraining loop
- ✅ Full nanochat behavior

## 2. Calculator Agent

Trains only on **math expression problems** (single-turn tool use).

```python
AGENT = "calculator"
```

```bash
# Train tiny calculator agent
python -m scripts.base_train -- --agent=calculator --depth=4
```

**Features:**
- ❌ CORE metrics disabled (not applicable)
- ✅ Agent-specific evaluation (math correctness)
- ✅ Generates problems on-the-fly
- ✅ Single-turn Q&A format

## 3. Tic-Tac-Toe Agent

Trains only on **game sequences** (multi-turn tool use).

```python
AGENT = "tictactoe"
```

```bash
# Train tiny tic-tac-toe agent
python -m scripts.base_train -- --agent=tictactoe --depth=4
```

**Features:**
- ❌ CORE metrics disabled (not applicable)
- ✅ Agent-specific evaluation (game logic correctness)
- ✅ Generates problems on-the-fly
- ✅ Multi-turn conversation format

## Quick Comparison

| Mode | Data Source | CORE Metrics | Evaluation | Use Case |
|------|-------------|--------------|------------|----------|
| `rdn` | Parquet files | ✅ Yes | Language modeling | Standard pretraining |
| `calculator` | Generated | ❌ No | Math correctness | Tool use training |
| `tictactoe` | Generated | ❌ No | Game logic | Multi-turn training |

## Switching Modes

### Option 1: Edit File

Open `scripts/base_train.py`:
```python
AGENT = "rdn"          # Normal pretraining
# AGENT = "calculator"   # Calculator agent
# AGENT = "tictactoe"    # Tic-tac-toe agent
```

### Option 2: CLI Flag

```bash
python -m scripts.base_train -- --agent=rdn
python -m scripts.base_train -- --agent=calculator
python -m scripts.base_train -- --agent=tictactoe
```

## Complete Examples

### Normal Pretraining (RDN)
```bash
# Download data first
python -m nanochat.dataset -n 8

# Train
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --agent=rdn \
  --depth=20 \
  --device_batch_size=8 \
  --core_metric_every=250
```

### Calculator Agent
```bash
python -m scripts.base_train -- \
  --agent=calculator \
  --depth=4 \
  --max_seq_len=512 \
  --device_batch_size=8 \
  --num_iterations=1000 \
  --sample_every=100
```

### Tic-Tac-Toe Agent
```bash
python -m scripts.base_train -- \
  --agent=tictactoe \
  --depth=4 \
  --max_seq_len=1024 \
  --device_batch_size=4 \
  --num_iterations=1000 \
  --sample_every=100
```

## Adding New Agents

See `generator/README.md` for how to add new agent types beyond calculator and tic-tac-toe.

## Technical Details

### Data Loading

- **RDN**: Uses `parquets_iter_batched()` to stream text from disk
- **Agents**: Uses `generate_random_problem()` to create problems on-the-fly

### Evaluation

- **RDN**: CORE metrics (composition, reasoning, explaining)
- **Calculator**: Numerical correctness + fuzzy string matching
- **Tic-Tac-Toe**: Structure validation + game logic checks

### Checkpoints

All modes save to the same checkpoint structure:
```
~/.cache/nanochat/base_checkpoints/d{depth}/
```

Different depths can be used for different modes without conflict.
