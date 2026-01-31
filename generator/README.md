# Unified Generator System

A flexible, extensible system for generating training problems across multiple task types.

## Overview

The unified generator provides:
- **Single entry point** for all problem types
- **Weighted sampling** to control task distribution
- **Easy extension** for adding new tasks
- **Consistent format** for the training loop

## Quick Start

### Using Default Mix

By default, the generator samples from:
- Calculator problems (weight: 1.0)
- Tic-tac-toe games (weight: 0.5)

```python
from generator.unified import generate_random_problem

problem = generate_random_problem()
# Returns: {'prompt': str, 'json': str}
```

### Configuring Task Mix

```python
from generator import configure_mix

# Use only calculator
configure_mix(calculator=1.0, tictactoe=0.0)

# Equal mix of both
configure_mix(calculator=1.0, tictactoe=1.0)

# Heavily favor tic-tac-toe
configure_mix(calculator=0.2, tictactoe=1.0)
```

### In Training Scripts

The generator is automatically used by `nanochat/dataloader.py`. To configure the mix for training:

```python
# At the top of your training script (e.g., scripts/base_train.py)
from generator import configure_mix

# Configure before creating the dataloader
configure_mix(calculator=1.0, tictactoe=0.5)

# Then proceed with normal training...
train_loader = tokenizing_distributed_data_loader(...)
```

## Available Generators

Current generators:

| Name | Description | Weight | Format |
|------|-------------|--------|--------|
| `calculator` | Math expression problems | 1.0 | Single-turn Q&A |
| `tic-tac-toe` | Game state management | 0.5 | Multi-turn conversation |

List available generators:
```python
from generator import list_generators
print(list_generators())  # ['calculator', 'tic-tac-toe']
```

## Adding New Generators

### Step 1: Create Generator Function

Create `generator/generator_<name>.py`:

```python
import random
import numpy as np
from numba import njit

def generate_random_problem():
    """
    Generate a problem.

    Returns:
        Dict with any structure - will be converted by format function
    """
    # Your generation logic here
    return {
        'prompt': 'Your question',
        'answer': 'Expected output',
        # ... any other fields
    }
```

**Key principles:**
1. Use **Numba** (`@njit`) for fast generation logic
2. Generate first, validate, then format (like `generator_calc.py`)
3. Only format successful problems to avoid wasted work

### Step 2: Create Format Converter

In `generator/unified.py`, add a converter function:

```python
def format_mynewtask_problem(problem: dict) -> Dict[str, str]:
    """
    Convert your problem format to standard training format.

    Returns:
        {'prompt': str, 'json': str}
    """
    return {
        'prompt': problem['prompt'],
        'json': problem['answer']
    }
```

### Step 3: Register Generator

In `generator/unified.py`, register your generator:

```python
from generator.generator_mynewtask import generate_random_problem as gen_mytask

_registry.register('mynewtask',
                  lambda: format_mynewtask_problem(gen_mytask()),
                  weight=0.5)
```

### Step 4: Use It

```python
from generator import configure_mix

configure_mix(calculator=1.0, tictactoe=0.5, mynewtask=1.0)
```

## Format Requirements

All generators must ultimately produce:

```python
{
    'prompt': str,  # The question/context for the model
    'json': str     # The expected response (can include tool calls)
}
```

The training loop will:
1. Tokenize the prompt with `<|bos|>` prefix
2. Tokenize the json response
3. Concatenate them into a training sequence

## Multi-Turn Conversations

For multi-turn tasks (like tic-tac-toe), the converter flattens the conversation:

```python
def format_multiturn_problem(problem: dict) -> Dict[str, str]:
    conversation = problem['conversation']

    # Everything except last assistant response = prompt
    prompt_parts = [msg['content'] for msg in conversation[:-1]]

    # Last assistant response = what model should generate
    response = conversation[-1]['content']

    return {
        'prompt': '\n'.join(prompt_parts),
        'json': response
    }
```

## Examples

### Example 1: Calculator Problem

```
PROMPT: "Calculate 5 + 3?"
JSON: "5 + 3 # Answer: \"8.0\""
```

### Example 2: Tic-Tac-Toe Problem

```
PROMPT:
Tic-tac-toe. I'm X at center.
<|python_start|>
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

JSON:
<|python_start|>
board[1][1]='X'
board[0][0]='O'
board[0][2]='X'
board[2][1]='O'
print(board)
check_winner(board)
<|python_end|>
<|output_start|>
O| |X
 |X|
 |O|
<|output_end|>
```

## Testing

Test the unified generator:

```bash
python -m generator.unified
```

Test individual generators:

```bash
python generator/generator_calc.py
python generator/generator_ttt.py
```

## Future Additions

Planned generators:
- String manipulation (reverse, sort, etc.)
- Date/time calculator
- JSON formatter
- List operations
- Base conversion

To request a new generator, open an issue describing:
1. What task it should perform
2. How to evaluate correctness
3. Example input/output
