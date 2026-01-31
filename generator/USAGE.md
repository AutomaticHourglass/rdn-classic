# Quick Usage Guide

## Using the Unified Generator in Training

The dataloader now automatically uses the unified generator. To configure the task mix:

### Default Behavior (No Changes Needed)

```bash
# Trains on mixed data: 67% calculator, 33% tic-tac-toe
python -m scripts.base_train -- --depth=20
```

### Configure Task Mix

Add this to the top of your training script:

```python
from generator import configure_mix

# Option 1: Calculator only
configure_mix(calculator=1.0, tictactoe=0.0)

# Option 2: Equal mix
configure_mix(calculator=1.0, tictactoe=1.0)

# Option 3: Mostly tic-tac-toe
configure_mix(calculator=0.2, tictactoe=1.0)
```

## Training Formats

### Calculator (Single-Turn)

```
PROMPT: "Calculate 5 + 3?"
RESPONSE: "5 + 3 # Answer: \"8.0\""
```

Model learns: Single Q&A pattern with tool use

### Tic-Tac-Toe (Multi-Turn)

```
PROMPT: "Tic-tac-toe. I'm X at center."
RESPONSE: "<|python_start|>
board[1][1]='X'
board[0][0]='O'
print(board)
<|python_end|>
<|output_start|>
O| |
 |X|
 | |
<|output_end|>
I play top-right.
<|python_start|>
board[1][1]='X'
board[0][0]='O'
board[0][2]='X'
board[1][1]='O'
print(board)
check_winner(board)
<|python_end|>
<|output_start|>
O| |X
 |O|
 | |
O wins
<|output_end|>"
```

Model learns: Full multi-turn game interaction from start to finish

## Command Line Examples

```bash
# Train small model with default mix
python -m scripts.base_train -- --depth=4 --max_seq_len=512 --device_batch_size=1

# Train with calculator only (add config to script first)
python -m scripts.base_train -- --depth=20 --device_batch_size=8

# Distributed training
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20
```

## Testing Generators

```bash
# Test unified generator
python -m generator.unified

# Test individual generators
python generator/generator_calc.py
python generator/generator_ttt.py
```

## Adding New Generators

See `generator/README.md` for detailed instructions.

Quick steps:
1. Create `generator/generator_<name>.py` with `generate_random_problem()`
2. Add format converter in `generator/unified.py`
3. Register in `generator/unified.py`
4. Use via `configure_mix()`
