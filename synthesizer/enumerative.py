"""
enumerative.py
--------------
Bottom-up enumerative program synthesizer.

This is the classical baseline synthesizer, inspired by the approach
used in SKETCH (Solar-Lezama et al. 2006) and FlashFill (Gulwani 2011).

How it works:
1. Start with all terminals: variables (x, y) and constants
2. Apply all operations to existing programs to generate larger programs
3. Enumerate in order of increasing size (bottom-up)
4. Check each candidate against the input-output examples
5. Return the first program that passes all examples
6. Stop if max_depth or max_candidates is reached

This approach guarantees finding the smallest correct program
in the language if one exists within the search limits.
"""

import time
from typing import List, Optional, Tuple
from synthesizer.language import (
    Expr, Var, IntConst, StrConst,
    Add, Subtract, Multiply,
    Concat, Slice, Length,
    IfThenElse, GreaterThan, GreaterThanOrEqual, Equals,
    INT_CONSTANTS, STR_CONSTANTS, VARIABLES
)
from synthesizer.verifier import Example, verify


# Search limits to prevent infinite enumeration
MAX_DEPTH = 4
MAX_CANDIDATES = 10000


def synthesize(
    examples: List[Example],
    variables: List[str] = None,
    max_depth: int = MAX_DEPTH,
    max_candidates: int = MAX_CANDIDATES
) -> Tuple[Optional[Expr], int, float]:
    """
    Run bottom-up enumerative synthesis.
    
    Args:
        examples: list of (input_dict, expected_output) pairs
        variables: list of variable names to use (default: ["x", "y"])
        max_depth: maximum depth of programs to enumerate
        max_candidates: maximum number of candidates to try
        
    Returns:
        Tuple of:
        - Optional[Expr]: the synthesized program, or None if not found
        - int: number of candidates explored
        - float: time taken in seconds
    """
    if variables is None:
        variables = VARIABLES

    start_time = time.time()
    candidates_explored = 0

    # Determine expected output type from examples
    # This helps prune candidates of the wrong type early
    expected_type = type(examples[0][1]) if examples else None

    # Each round stores programs of a particular size
    # rounds[0] = terminals (depth 0)
    # rounds[1] = programs of depth 1
    # etc.
    rounds: List[List[Expr]] = []

    # --- Round 0: Generate terminals ---
    terminals: List[Expr] = []

    # Add variables
    for var_name in variables:
        # Only add variable if it appears in the examples
        if any(var_name in inputs for inputs, _ in examples):
            terminals.append(Var(var_name))

    # Add integer constants
    for c in INT_CONSTANTS:
        terminals.append(IntConst(c))

    # Add string constants
    for c in STR_CONSTANTS:
        terminals.append(StrConst(c))

    rounds.append(terminals)

    # Check terminals
    for program in terminals:
        candidates_explored += 1
        if candidates_explored > max_candidates:
            elapsed = time.time() - start_time
            return None, candidates_explored, elapsed

        if verify(program, examples):
            elapsed = time.time() - start_time
            return program, candidates_explored, elapsed

    # --- Rounds 1 to max_depth: Generate composite programs ---
    for depth in range(1, max_depth + 1):
        new_programs: List[Expr] = []

        # All programs generated so far (for combining)
        all_previous = []
        for r in rounds:
            all_previous.extend(r)

        # Arithmetic operations (require int operands)
        for left in all_previous:
            for right in all_previous:
                # Add
                new_programs.append(Add(left, right))
                # Subtract
                new_programs.append(Subtract(left, right))
                # Multiply
                new_programs.append(Multiply(left, right))

        # String operations
        for left in all_previous:
            for right in all_previous:
                new_programs.append(Concat(left, right))

        # Length (unary string operation)
        for expr in all_previous:
            new_programs.append(Length(expr))

        # Slice (ternary string operation)
        for string_expr in all_previous:
            for start_expr in all_previous:
                for end_expr in all_previous:
                    new_programs.append(Slice(string_expr, start_expr, end_expr))
                    if candidates_explored > max_candidates:
                        break
                if candidates_explored > max_candidates:
                    break
            if candidates_explored > max_candidates:
                break

        # Comparison operations (for use in conditionals)
        bool_exprs: List[Expr] = []
        for left in all_previous:
            for right in all_previous:
                bool_exprs.append(GreaterThan(left, right))
                bool_exprs.append(GreaterThanOrEqual(left, right))
                bool_exprs.append(Equals(left, right))

        # Conditional expressions
        for cond in bool_exprs[:50]:  # Limit bool exprs to avoid explosion
            for then_branch in all_previous:
                for else_branch in all_previous:
                    new_programs.append(IfThenElse(cond, then_branch, else_branch))
                    if candidates_explored > max_candidates:
                        break
                if candidates_explored > max_candidates:
                    break
            if candidates_explored > max_candidates:
                break

        rounds.append(new_programs)

        # Check all new programs
        for program in new_programs:
            candidates_explored += 1

            if candidates_explored > max_candidates:
                elapsed = time.time() - start_time
                return None, candidates_explored, elapsed

            if verify(program, examples):
                elapsed = time.time() - start_time
                return program, candidates_explored, elapsed

    elapsed = time.time() - start_time
    return None, candidates_explored, elapsed
