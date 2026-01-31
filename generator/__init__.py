"""
Generator package for training data generation.

Usage:
    # Use default mix (calculator=1.0, tic-tac-toe=0.5)
    from generator.unified import generate_random_problem

    # Configure custom mix
    from generator import configure_mix
    configure_mix(calculator=1.0, tictactoe=0.0)  # Only calculator

    # Add new generators in the future
    from generator.unified import _registry
    _registry.register('my_task', my_generator_fn, weight=0.5)
"""

from generator.unified import (
    generate_random_problem,
    configure_generators,
    list_generators,
)


def configure_mix(**kwargs):
    """
    Convenient way to configure generator weights.

    Args:
        calculator: Weight for calculator problems (default: 1.0)
        tictactoe: Weight for tic-tac-toe problems (default: 0.5)

    Example:
        configure_mix(calculator=1.0, tictactoe=1.0)  # Equal mix
        configure_mix(calculator=1.0, tictactoe=0.0)  # Calculator only
        configure_mix(calculator=0.0, tictactoe=1.0)  # Tic-tac-toe only
    """
    config = {}
    if 'calculator' in kwargs:
        config['calculator'] = kwargs['calculator']
    if 'tictactoe' in kwargs:
        config['tic-tac-toe'] = kwargs['tictactoe']

    if config:
        configure_generators(config)


__all__ = [
    'generate_random_problem',
    'configure_generators',
    'configure_mix',
    'list_generators',
]
