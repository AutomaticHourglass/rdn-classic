# Training Task Configuration

## Quick Config (Edit File)

Open `scripts/base_train.py` and change these variables at the top:

```python
# GENERATOR CONFIGURATION - Change these to control training task mix
CALCULATOR_WEIGHT = 1.0   # Calculator problems (math expressions)
TICTACTOE_WEIGHT = 0.5    # Tic-tac-toe games (multi-turn tool use)
```

Then run training normally:
```bash
python -m scripts.base_train -- --depth=20
```

## Examples

### Calculator Only
```python
CALCULATOR_WEIGHT = 1.0
TICTACTOE_WEIGHT = 0.0
```

### Tic-Tac-Toe Only
```python
CALCULATOR_WEIGHT = 0.0
TICTACTOE_WEIGHT = 1.0
```

### Equal Mix
```python
CALCULATOR_WEIGHT = 1.0
TICTACTOE_WEIGHT = 1.0
```

### Heavy Tic-Tac-Toe Focus
```python
CALCULATOR_WEIGHT = 0.2
TICTACTOE_WEIGHT = 1.0
```

## CLI Override (No File Editing)

Override weights from command line:

```bash
# Calculator only
python -m scripts.base_train -- --depth=20 --calc_weight=1.0 --ttt_weight=0.0

# Tic-tac-toe only
python -m scripts.base_train -- --depth=20 --calc_weight=0.0 --ttt_weight=1.0

# Equal mix
python -m scripts.base_train -- --depth=20 --calc_weight=1.0 --ttt_weight=1.0

# Custom ratio
python -m scripts.base_train -- --depth=20 --calc_weight=0.3 --ttt_weight=1.0
```

## Distributed Training

```bash
# Default mix (calc=1.0, ttt=0.5)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# Calculator only
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --calc_weight=1.0 --ttt_weight=0.0

# Equal mix
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --calc_weight=1.0 --ttt_weight=1.0
```

## Understanding Weights

Weights control **relative sampling probability**:

- `calc=1.0, ttt=0.5` → 67% calculator, 33% tic-tac-toe
- `calc=1.0, ttt=1.0` → 50% calculator, 50% tic-tac-toe
- `calc=2.0, ttt=1.0` → 67% calculator, 33% tic-tac-toe
- `calc=1.0, ttt=0.0` → 100% calculator, 0% tic-tac-toe

Formula: `probability = weight / sum(all_weights)`

## Adding New Tasks

1. Create generator in `generator/generator_<name>.py`
2. Register in `generator/unified.py`
3. Add weight variable in `scripts/base_train.py`:
   ```python
   MYNEWTASK_WEIGHT = 0.5
   ```
4. Add CLI argument:
   ```python
   parser.add_argument("--mytask_weight", type=float, default=MYNEWTASK_WEIGHT)
   ```
5. Configure in the mix:
   ```python
   configure_mix(calculator=args.calc_weight,
                 tictactoe=args.ttt_weight,
                 mytask=args.mytask_weight)
   ```

## Checking Configuration

When training starts, you'll see:
```
Generator mix: calculator=1.0, tictactoe=0.5
```

This confirms which weights are active.

## Best Practices

- Start with calculator only to verify training works
- Add tic-tac-toe gradually (0.2 → 0.5 → 1.0)
- Monitor loss curves for each task type
- If one task dominates, reduce its weight
- For new tasks, start with low weight (0.1-0.2) until validated
