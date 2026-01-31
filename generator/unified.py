"""
Unified generator entry point for all problem types.

Provides a single `generate_random_problem()` function that can sample from
multiple generator types (calculator, tic-tac-toe, etc.) and returns a
consistent format for the training loop.

Usage in dataloader:
    from generator.unified import generate_random_problem
"""

import random
from typing import Dict, List, Callable

# Import all generator types
from generator.generator_calc import generate_random_problem as gen_calc
from generator.generator_ttt import generate_random_problem as gen_ttt


# -----------------------------------------------------------------------------
# Generator Registry
# -----------------------------------------------------------------------------

class GeneratorRegistry:
    """
    Registry for managing multiple problem generators.

    Each generator should return problems in a consistent format that can be
    converted to {'prompt': str, 'json': str} for training.
    """

    def __init__(self):
        self.generators: Dict[str, Callable] = {}
        self.weights: Dict[str, float] = {}

    def register(self, name: str, generator_fn: Callable, weight: float = 1.0):
        """
        Register a generator function.

        Args:
            name: Unique identifier for this generator
            generator_fn: Function that returns a problem dict
            weight: Sampling weight (higher = more likely to be sampled)
        """
        self.generators[name] = generator_fn
        self.weights[name] = weight

    def sample_generator(self) -> Callable:
        """Sample a generator based on weights."""
        names = list(self.generators.keys())
        weights = [self.weights[name] for name in names]
        chosen_name = random.choices(names, weights=weights, k=1)[0]
        return self.generators[chosen_name]


# Global registry instance
_registry = GeneratorRegistry()


# -----------------------------------------------------------------------------
# Format Converters
# -----------------------------------------------------------------------------

def format_calc_problem(problem: dict) -> Dict[str, str]:
    """
    Convert calculator problem to standard format.

    Input: {'prompt': str, 'json': str, 'answer': float}
    Output: {'prompt': str, 'json': str}
    """
    return {
        'prompt': problem['prompt'],
        'json': f"{problem['json']} # Answer: \"{problem['answer']}\""
    }


def format_ttt_problem(problem: dict) -> Dict[str, str]:
    """
    Convert tic-tac-toe multi-turn conversation to standard format.

    Input: {'conversation': List[Dict], 'final_board': List, 'outcome': str}
    Output: {'prompt': str, 'json': str}

    Trains on the FULL game sequence from start to end.
    """
    conversation = problem['conversation']

    if len(conversation) < 2:
        # Edge case: need at least user + assistant
        return format_calc_problem(gen_calc())

    # Last message should be assistant (game should end on assistant's move)
    if conversation[-1]['role'] != 'assistant':
        return format_calc_problem(gen_calc())

    # Prompt = first user message (game start)
    prompt = conversation[0]['content']

    # JSON = all subsequent messages (assistant responses + user moves)
    # This trains the model on the full multi-turn interaction
    response_parts = []
    for msg in conversation[1:]:
        response_parts.append(msg['content'])

    response = '\n'.join(response_parts)

    return {
        'prompt': prompt,
        'json': response
    }


# -----------------------------------------------------------------------------
# Register Generators
# -----------------------------------------------------------------------------

# Calculator: weight=1.0 (baseline)
_registry.register('calculator',
                  lambda: format_calc_problem(gen_calc()),
                  weight=1.0)

# Tic-tac-toe: weight=0.5 (train on it but less frequently)
_registry.register('tic-tac-toe',
                  lambda: format_ttt_problem(gen_ttt()),
                  weight=0.5)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def generate_random_problem() -> Dict[str, str]:
    """
    Generate a random problem from registered generators.

    Returns:
        Dict with keys 'prompt' and 'json'
    """
    generator = _registry.sample_generator()
    return generator()


def configure_generators(config: Dict[str, float]):
    """
    Reconfigure generator weights.

    Args:
        config: Dict mapping generator names to weights
                e.g., {'calculator': 1.0, 'tic-tac-toe': 0.5}

    Example:
        # Use only calculator
        configure_generators({'calculator': 1.0, 'tic-tac-toe': 0.0})

        # Equal mix
        configure_generators({'calculator': 1.0, 'tic-tac-toe': 1.0})
    """
    for name, weight in config.items():
        if name in _registry.generators:
            _registry.weights[name] = weight
        else:
            print(f"Warning: Unknown generator '{name}'")


def list_generators() -> List[str]:
    """List all registered generator names."""
    return list(_registry.generators.keys())


# -----------------------------------------------------------------------------
# Test/Debug
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print("Available generators:", list_generators())
    print("\nGenerating sample problems...\n")

    for i in range(5):
        problem = generate_random_problem()
        print(f"{'='*60}")
        print(f"Problem {i+1}:")
        print(f"{'='*60}")
        print(f"PROMPT:\n{problem['prompt']}\n")
        print(f"RESPONSE:\n{problem['json']}\n")
