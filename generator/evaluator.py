"""
Unified evaluation system for different agent types.

Each agent defines how to:
1. Parse problems from tokenized sequences
2. Evaluate model predictions
3. Calculate correctness metrics
"""

from typing import List, Dict, Tuple, Callable
from thefuzz import fuzz
from generator.generator_calc import safe_eval_math


# -----------------------------------------------------------------------------
# Base Evaluator Interface
# -----------------------------------------------------------------------------

class AgentEvaluator:
    """Base class for agent-specific evaluation."""

    def parse_problem(self, text: str) -> Tuple[str, str]:
        """
        Parse a problem from tokenized text.

        Returns:
            (prompt, expected_answer)
        """
        raise NotImplementedError

    def evaluate(self, prompt: str, expected: str, predicted: str) -> Dict:
        """
        Evaluate a model prediction.

        Returns:
            {
                'prompt': str,
                'label': str,
                'pred': str,
                'score': float (0-1),
                'corr': int (0 or 1)
            }
        """
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Calculator Evaluator
# -----------------------------------------------------------------------------

class CalculatorEvaluator(AgentEvaluator):
    """Evaluator for calculator agent (math expressions)."""

    def parse_problem(self, text: str) -> Tuple[str, str]:
        """
        Parse calculator problem.
        Format: "Question? # Answer: \"value\""
        """
        try:
            # Expected answer is in quotes
            e = text.split('"')[-2]
            # Question ends with ?
            q = text.split('?')[0] + '?'
            return q, e
        except Exception:
            return None, None

    def evaluate(self, prompt: str, expected: str, predicted: str) -> Dict:
        """
        Evaluate calculator prediction.
        Uses numerical comparison and fuzzy string matching.
        """
        eval_score = 0
        pratio = 0
        ppred = ''

        eval_expected = safe_eval_math(expected)

        try:
            # Extract predicted answer from quotes
            parts = list(filter(lambda x: len(x) > 0, predicted.split('"')))

            if len(parts) > 1:
                pe = parts[1]
                ppred = pe

                eval_pred = safe_eval_math(pe)

                # Fuzzy string match + length penalty
                pratio = fuzz.partial_ratio(expected, pe) / 100 * min(len(pe), len(expected)) / max(len(pe), len(expected))

                # Check if correct (fuzzy match or numerical match)
                if pratio >= 0.98 or eval_expected == eval_pred or abs((eval_expected - eval_pred) / eval_expected) < 1e-5:
                    eval_score = 1
                    pratio = 1
        except Exception:
            pass

        return {
            'prompt': prompt,
            'label': expected,
            'pred': ppred,
            'score': pratio,
            'corr': eval_score
        }


# -----------------------------------------------------------------------------
# Tic-Tac-Toe Evaluator
# -----------------------------------------------------------------------------

class TicTacToeEvaluator(AgentEvaluator):
    """Evaluator for tic-tac-toe agent (game sequences)."""

    def parse_problem(self, text: str) -> Tuple[str, str]:
        """
        Parse tic-tac-toe problem.
        Format: Multi-turn conversation with tool calls.
        """
        try:
            # Split by python_start to find turns
            if '<|python_start|>' not in text:
                return None, None

            # First user message is the prompt
            parts = text.split('<|python_start|>')
            prompt = parts[0].strip()

            # Expected is the full game sequence
            expected = text[len(prompt):].strip()

            return prompt, expected
        except Exception:
            return None, None

    def evaluate(self, prompt: str, expected: str, predicted: str) -> Dict:
        """
        Evaluate tic-tac-toe prediction.
        Checks if tool calls are syntactically valid and game logic is correct.
        """
        eval_score = 0
        pratio = 0

        try:
            # Check if prediction has proper structure
            has_python_start = '<|python_start|>' in predicted
            has_python_end = '<|python_end|>' in predicted
            has_output_start = '<|output_start|>' in predicted
            has_output_end = '<|output_end|>' in predicted

            structure_score = sum([has_python_start, has_python_end, has_output_start, has_output_end]) / 4

            # Check if board updates are present
            has_board_update = 'board[' in predicted
            has_print = 'print(board)' in predicted or 'print_board' in predicted

            logic_score = sum([has_board_update, has_print]) / 2

            # Check if similar length (roughly same game length)
            len_ratio = min(len(predicted), len(expected)) / max(len(predicted), len(expected), 1)

            # Overall score
            pratio = (structure_score * 0.4 + logic_score * 0.4 + len_ratio * 0.2)

            # Consider correct if score > 0.7
            if pratio > 0.7:
                eval_score = 1
        except Exception:
            pass

        return {
            'prompt': prompt[:100],  # Truncate for display
            'label': f"<game seq len={len(expected)}>",
            'pred': f"<pred len={len(predicted)} score={pratio:.2f}>",
            'score': pratio,
            'corr': eval_score
        }


# -----------------------------------------------------------------------------
# Evaluator Registry
# -----------------------------------------------------------------------------

_evaluators = {
    'calculator': CalculatorEvaluator(),
    'tictactoe': TicTacToeEvaluator(),
}


def get_evaluator(agent_type: str) -> AgentEvaluator:
    """Get evaluator for specified agent type."""
    if agent_type not in _evaluators:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(_evaluators.keys())}")
    return _evaluators[agent_type]


def register_evaluator(agent_type: str, evaluator: AgentEvaluator):
    """Register a new evaluator for an agent type."""
    _evaluators[agent_type] = evaluator
