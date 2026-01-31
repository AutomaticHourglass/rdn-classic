import time
import random
import math
import numpy as np
from numba import njit, int32, float64

# --- CONFIGURATION ---
MAX_DEPTH = 3
MAX_NODES = 2 ** (MAX_DEPTH + 1) - 1  # 511 nodes for depth 8
PROB_UNARY = 0.3
PROB_RECURSE = 0.6

# Op Codes
OP_VAL = 0
OP_ADD, OP_SUB, OP_MUL, OP_DIV = 1, 2, 3, 4
OP_LOG, OP_SIN, OP_COS, OP_TAN, OP_EXP, OP_FACT = 5, 6, 7, 8, 9, 10
OP_ASIN, OP_ACOS, OP_ATAN = 11, 12, 13

# Mappings for String Formatting
OP_MAP = {
    OP_ADD: "+", OP_SUB: "-", OP_MUL: "*", OP_DIV: "/",
    OP_LOG: "log", OP_SIN: "sin", OP_COS: "cos", OP_TAN: "tan",
    OP_EXP: "exp", OP_FACT: "factorial", OP_ASIN: "asin", OP_ACOS: "acos", OP_ATAN: "atan"
}
NATURAL_MAP = {
    OP_ADD: "plus", OP_SUB: "minus", OP_MUL: "multiplied by", OP_DIV: "divided by",
    OP_LOG: "natural logarithm of", OP_SIN: "sine of",
    OP_COS: "cosine of", OP_TAN: "tangent of", OP_EXP: "exponential of",
    OP_FACT: "factorial of", OP_ASIN: "arc sine of", OP_ACOS: "arc cosine of", OP_ATAN: "arc tangent of"
}

# Precompute factorial table
FACT_TABLE = np.array([math.factorial(i) for i in range(171)], dtype=np.float64)


@njit(fastmath=True)
def get_factorial(val):
    if val < 0 or val > 170: return np.nan
    ival = int(val)
    if abs(val - ival) > 1e-9: return np.nan
    return FACT_TABLE[ival]


# --- CORE LOGIC (Extracted to Global Scope) ---

@njit(fastmath=True)
def _eval_recursive(ops, vals, idx):
    """
    Standalone recursive evaluation.
    Recursion is supported in Numba when types are explicit or deducible.
    """
    if idx >= len(ops): return np.nan
    op = ops[idx]

    # Base Case: Leaf
    if op == OP_VAL:
        return vals[idx]

    # Unary Ops
    if op >= 5:
        # Left child (2*idx + 1) used for unary argument
        arg = _eval_recursive(ops, vals, 2 * idx + 1)
        if np.isnan(arg) or np.isinf(arg): return np.nan

        # Domain safety checks (using abs as per your original logic)
        if op in (OP_LOG, OP_FACT):
            arg = abs(arg)

        if op == OP_LOG: return math.log(arg) if arg > 1e-9 else np.nan
        if op == OP_SIN: return math.sin(arg)
        if op == OP_COS: return math.cos(arg)
        if op == OP_TAN: return math.tan(arg)
        if op == OP_EXP:
            return math.exp(arg) if arg < 700 else np.nan
        if op == OP_FACT: return get_factorial(arg)
        return np.nan

    # Binary Ops
    left = _eval_recursive(ops, vals, 2 * idx + 1)
    right = _eval_recursive(ops, vals, 2 * idx + 2)

    if np.isnan(left) or np.isnan(right): return np.nan
    if np.isinf(left) or np.isinf(right): return np.nan  # Add infinity check

    if op == OP_ADD: return left + right
    if op == OP_SUB: return left - right
    if op == OP_MUL: return left * right
    if op == OP_DIV:
        # Ensure denominator is never zero
        denom = abs(right)
        if denom < 1e-10:  # Check before adding epsilon
            denom = 0.1
        else:
            denom = denom + 0.1
        return left / denom

    return np.nan



@njit(fastmath=True)
def evaluate_tree(ops, vals):
    return _eval_recursive(ops, vals, 0)

# ------------------------------------------------------------------
# Helper: generate a random integer with *d* digits (1‑4) uniformly
# ------------------------------------------------------------------
@njit(fastmath=True)
def rand_int_with_digits(d: int, allow_zero_first_digit: bool = False) -> int:
    """
    Return a random integer that has exactly *d* decimal digits.
    
    Parameters
    ----------
    d : int
        Desired number of digits (1‑4).
    allow_zero_first_digit : bool, default False
        If True, the leading digit may be 0 (only relevant for d == 1).
    """
    if d == 1:
        # 0‑9 inclusive
        return np.random.randint(0, 10)
    else:
        # 10^(d-1) … 10^d - 1  (e.g. d=3 → 100‑999)
        return np.random.randint(10 ** (d - 1), 10 ** d)

# ------------------------------------------------------------------
# Helper: generate a random fractional part with *d* digits
# ------------------------------------------------------------------
@njit(fastmath=True)
def rand_frac_with_digits(d: int) -> float:
    """
    Return a random fractional value with exactly *d* digits after the point.
    
    Parameters
    ----------
    d : int
        Desired number of fractional digits (1‑4).
    """
    frac_int = np.random.randint(0, 10 ** d)          # e.g. d=3 → 0‑999
    return frac_int / (10 ** d)                      # scale to [0, 1)


@njit(fastmath=True)
def generate_tree(seed):
    np.random.seed(seed)
    ops = np.zeros(MAX_NODES, dtype=np.int32)
    vals = np.zeros(MAX_NODES, dtype=np.float64)

    # Stack for generation (Iterative to avoid recursion limit during gen)
    # Stack stores: (index, depth)
    stack = np.empty((MAX_DEPTH * 2, 2), dtype=np.int32)
    stack_ptr = 0
    stack[0] = (0, 0)
    stack_ptr += 1

    while stack_ptr > 0:
        stack_ptr -= 1
        idx, depth = stack[stack_ptr]

        is_max_depth = depth >= MAX_DEPTH
        stop_recurse = (depth > 0) and (np.random.random() > PROB_RECURSE)

        if is_max_depth or stop_recurse:
            ops[idx] = OP_VAL
        
            # 1) Choose a sign
            sign = 1 if np.random.rand() > 0.5 else -1
        
            # 2) Whole part – digits 1‑4 equally probable
            whole_digits = np.random.randint(1, 5)          # 1‑4 inclusive
            whole = rand_int_with_digits(whole_digits)
        
            # 3) Fractional part – digits 1‑4 equally probable, 30 % chance of zero
            if np.random.rand() < 0.3:
                frac = 0.0
            else:
                frac_digits = np.random.randint(1, 5)      # 1‑4 inclusive
                frac = rand_frac_with_digits(frac_digits)
        
            # 4) Combine with sign
            vals[idx] = (whole + frac) * sign
        
            continue

        if np.random.random() < PROB_UNARY:
            # Unary
            ops[idx] = np.random.randint(5, 12)  # Range [5, 11]
            if (2 * idx + 1) < MAX_NODES:
                stack[stack_ptr] = (2 * idx + 1, depth + 1)
                stack_ptr += 1
        else:
            # Binary
            r = np.random.random()
            if r < 0.4:
                ops[idx] = OP_ADD
            elif r < 0.6:
                ops[idx] = OP_SUB
            elif r < 0.9:
                ops[idx] = OP_MUL
            else:
                ops[idx] = OP_DIV

            if (2 * idx + 2) < MAX_NODES:
                stack[stack_ptr] = (2 * idx + 2, depth + 1)
                stack_ptr += 1
                stack[stack_ptr] = (2 * idx + 1, depth + 1)
                stack_ptr += 1

    return ops, vals


# --- STRING FORMATTING (Python Side) ---
def format_tree(ops, vals, idx=0, natural=False):
    # # randomly chooses to have a paranthesis
    # def p():
    
    op = ops[idx]
    if op == OP_VAL:
        val = vals[idx]
        return str(int(val)) if val.is_integer() else f"{val:.4f}".rstrip('0').rstrip('.')

    left_str = format_tree(ops, vals, 2 * idx + 1, natural)

    if op >= 5:  # Unary
        # randomly decide to put or not put the parantheses
        if natural:
            if np.random.random() < 0.5:
                func_name = NATURAL_MAP[op]
            else:
                func_name = OP_MAP[op]
        else:
            func_name = OP_MAP[op]
        if not natural and op in (OP_LOG, OP_FACT):
            return f"{func_name}(abs({left_str}))"
        if np.random.random() < 0.5:
            return f"{func_name} ({left_str})"
        return f"{func_name} {left_str}"

    right_str = format_tree(ops, vals, 2 * idx + 2, natural)
    op_text = NATURAL_MAP[op] if natural else OP_MAP[op]

    if not natural and op == OP_DIV:
        right_str = f"({right_str})"

    if np.random.random() < 0.5:
        return f"{left_str} {op_text} {right_str}"
    return f"{left_str} {op_text} ({right_str})"




def safe_eval_math(expr):
    safe_dict = {
        'abs': abs, 'log': math.log, 'sin': math.sin, 'cos': math.cos, 
        'tan': math.tan, 'exp': math.exp, 'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e
    }
    
    try:
        # 1e300 is safe, 1e400 raises OverflowError
        return float(eval(expr, {"__builtins__": {}}, safe_dict))
        
    except (SyntaxError, NameError, ZeroDivisionError, ValueError, 
            TypeError, OverflowError):  # <--- Added OverflowError
        return float('nan')



def generate_random_problem():
    while True:
        # 1. Fast Generate & Evaluate (Compiled)
        seed = np.random.randint(0, 1000000000)
        ops, vals = generate_tree(seed)
        result = evaluate_tree(ops, vals)

        # 2. Check Validity
        if np.isfinite(result) and abs(result) < 1e100:
            # 3. Slow Format (Python) - Only runs on valid results
            math_str = format_tree(ops, vals, 0, natural=False)
            prompt_body = format_tree(ops, vals, 0, natural=True)

            starters = ["Calculate", "What is", "Find the result of", "Evaluate"]
            prompt = f"{random.choice(starters)} {prompt_body}?"

            return {'prompt': prompt, 'json': math_str, 'answer': float(result)}
