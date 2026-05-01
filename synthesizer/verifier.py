"""
verifier.py
-----------
Checks whether a program satisfies a set of input-output examples.

This verifier is used by ALL four synthesizers (enumerative, GPT-4o,
Claude, Gemini) to ensure fair comparison. A program passes verification
if and only if it produces the correct output for every example.
"""

from typing import List, Tuple, Any
from synthesizer.language import Expr, Value


# A single input-output example
# inputs is a dict mapping variable names to values e.g. {"x": 3}
# output is the expected output value
Example = Tuple[dict, Value]


def verify(program: Expr, examples: List[Example]) -> bool:
    """
    Check if a program satisfies all input-output examples.
    
    Args:
        program: the AST to evaluate
        examples: list of (input_dict, expected_output) pairs
        
    Returns:
        True if the program produces the correct output for every example
        False if any example fails or if evaluation raises an error
    """
    for inputs, expected_output in examples:
        try:
            actual_output = program.eval(inputs)
            # Use strict equality for both int and str outputs
            if actual_output != expected_output:
                return False
        except Exception:
            # Any evaluation error counts as failure
            return False
    return True


def verify_with_feedback(
    program: Expr,
    examples: List[Example]
) -> Tuple[bool, List[str]]:
    """
    Check if a program satisfies all examples and return detailed feedback.
    
    This is used by LLM synthesizers to tell the LLM which examples failed
    and why, so it can generate a better next candidate.
    
    Args:
        program: the AST to evaluate
        examples: list of (input_dict, expected_output) pairs
        
    Returns:
        Tuple of:
        - bool: True if all examples pass
        - List[str]: list of failure messages (empty if all pass)
    """
    failures = []
    
    for i, (inputs, expected_output) in enumerate(examples):
        try:
            actual_output = program.eval(inputs)
            if actual_output != expected_output:
                failures.append(
                    f"Example {i+1}: inputs={inputs}, "
                    f"expected={repr(expected_output)}, "
                    f"got={repr(actual_output)}"
                )
        except Exception as e:
            failures.append(
                f"Example {i+1}: inputs={inputs}, "
                f"expected={repr(expected_output)}, "
                f"error={str(e)}"
            )
    
    return len(failures) == 0, failures


def verify_held_out(program: Expr, held_out: Example) -> bool:
    """
    Check if a program passes the held-out test case.
    
    The held-out test case is not shown to any synthesizer during synthesis.
    It tests whether the synthesized program generalizes beyond the training
    examples, not just memorizes them.
    
    Args:
        program: the AST to evaluate
        held_out: a single (input_dict, expected_output) pair
        
    Returns:
        True if the program passes the held-out test
    """
    inputs, expected_output = held_out
    try:
        actual_output = program.eval(inputs)
        return actual_output == expected_output
    except Exception:
        return False
