# Training Individual Agents

Each agent trains on a single task type with a tiny network.

## Quick Config (Edit File)

Open `scripts/base_train.py` and change this variable:

```python
AGENT = "calculator"  # Options: "calculator", "tictactoe"
```

Then run:
```bash
python -m scripts.base_train -- --depth=4 --max_seq_len=512 --device_batch_size=8
```

## CLI Override (No File Editing)

```bash
# Train calculator agent
python -m scripts.base_train -- --depth=4 --agent=calculator

# Train tic-tac-toe agent
python -m scripts.base_train -- --depth=4 --agent=tictactoe
```

## Complete Training Examples

### Small Calculator Agent
```bash
python -m scripts.base_train -- \
  --agent=calculator \
  --depth=4 \
  --max_seq_len=512 \
  --device_batch_size=8 \
  --num_iterations=1000
```

### Small Tic-Tac-Toe Agent
```bash
python -m scripts.base_train -- \
  --agent=tictactoe \
  --depth=4 \
  --max_seq_len=512 \
  --device_batch_size=8 \
  --num_iterations=1000
```

### Larger Calculator Agent (Multi-GPU)
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --agent=calculator \
  --depth=20 \
  --device_batch_size=8
```

### Larger Tic-Tac-Toe Agent (Multi-GPU)
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
  --agent=tictactoe \
  --depth=20 \
  --device_batch_size=8
```

## Recommended Tiny Network Configs

For quick experimentation with each agent:

```bash
# CPU-friendly tiny network
python -m scripts.base_train -- \
  --agent=calculator \
  --depth=4 \
  --max_seq_len=512 \
  --device_batch_size=1 \
  --total_batch_size=512 \
  --num_iterations=100

# Single GPU tiny network
python -m scripts.base_train -- \
  --agent=tictactoe \
  --depth=6 \
  --max_seq_len=1024 \
  --device_batch_size=4 \
  --num_iterations=500
```

## Agent Descriptions

### Calculator Agent
- **Task**: Evaluate math expressions
- **Format**: Single-turn Q&A
- **Example**: "Calculate 5 + 3?" → "5 + 3 # Answer: \"8.0\""
- **Trains on**: Random math expression trees

### Tic-Tac-Toe Agent
- **Task**: Play tic-tac-toe games
- **Format**: Multi-turn conversation with tool calls
- **Example**: Full game from start to finish
- **Trains on**: All possible game sequences

## Checkpoints

Each agent saves to its own directory:
```
~/.cache/nanochat/base_checkpoints/d4/
~/.cache/nanochat/base_checkpoints/d6/
~/.cache/nanochat/base_checkpoints/d20/
```

## Adding New Agents

1. Create generator in `generator/generator_<name>.py`
2. Register in `generator/unified.py`
3. Add to choices in `scripts/base_train.py`:
   ```python
   parser.add_argument("--agent", type=str, default=AGENT,
                       choices=["calculator", "tictactoe", "mynewtask"])
   ```
4. Add configuration:
   ```python
   elif args.agent == "mynewtask":
       configure_mix(calculator=0.0, tictactoe=0.0, mynewtask=1.0)
   ```

## Quick Reference

| Agent | Command | Training Data |
|-------|---------|---------------|
| Calculator | `--agent=calculator` | Math expressions only |
| Tic-Tac-Toe | `--agent=tictactoe` | Game sequences only |

Change agent type by editing `AGENT = "..."` in the script or using `--agent=...` flag.
