"""
llm_claude.py
-------------
Claude (Anthropic) guided program synthesizer.

Uses Anthropic's claude-3-5-sonnet model to generate candidate programs.
Same approach as llm_gpt.py but using the Anthropic API.

All three LLM synthesizers (GPT-4o, Claude, Gemini) use identical
prompting strategies and the same verifier to ensure fair comparison.
"""

import os
import time
from typing import List, Optional, Tuple
import anthropic
from synthesizer.language import Expr, BNF_GRAMMAR, parse_program
from synthesizer.verifier import Example, verify, verify_with_feedback

MAX_ATTEMPTS = 20


def _build_system_prompt() -> str:
    """Build the system prompt explaining the task and grammar."""
    return f"""You are a program synthesis assistant. Your job is to find a program
in a specific expression language that satisfies given input-output examples.

The expression language grammar is:
{BNF_GRAMMAR}

Rules:
- You must only use operations and constants defined in the grammar above
- Variables are x and y (only use variables that appear in the examples)
- Return ONLY the program expression, nothing else
- No explanation, no markdown, no code blocks
- Just the raw program string like: add(x, 3) or concat(x, y)
- Start with the simplest possible program and increase complexity if it fails
"""


def _build_user_prompt(
    examples: List[Example],
    failed_attempts: List[Tuple[str, List[str]]],
    attempt_number: int
) -> str:
    """Build the user prompt for a synthesis attempt."""
    examples_str = "\n".join(
        f"  Input: {inputs} -> Output: {repr(expected)}"
        for inputs, expected in examples
    )

    prompt = f"Find a program that satisfies these examples:\n{examples_str}\n"

    if failed_attempts:
        prompt += "\nPrevious attempts that failed:\n"
        for prog_str, failures in failed_attempts[-3:]:
            prompt += f"  Program: {prog_str}\n"
            for failure in failures[:2]:
                prompt += f"    Failed: {failure}\n"
        prompt += "\nPlease try a different program that fixes these failures.\n"

    prompt += f"\nAttempt {attempt_number}: Return only the program expression."
    return prompt


def synthesize(
    examples: List[Example],
    variables: List[str] = None,
    max_attempts: int = MAX_ATTEMPTS
) -> Tuple[Optional[Expr], int, float]:
    """
    Run Claude guided synthesis.
    
    Args:
        examples: list of (input_dict, expected_output) pairs
        variables: list of variable names (unused, inferred from examples)
        max_attempts: maximum number of LLM attempts
        
    Returns:
        Tuple of:
        - Optional[Expr]: the synthesized program, or None if not found
        - int: number of candidates tried
        - float: time taken in seconds
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    start_time = time.time()
    candidates_explored = 0
    failed_attempts = []
    
    system_prompt = _build_system_prompt()
    
    for attempt in range(1, max_attempts + 1):
        user_prompt = _build_user_prompt(examples, failed_attempts, attempt)
        
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=200,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            
            program_str = response.content[0].text.strip()
            candidates_explored += 1
            
            try:
                program = parse_program(program_str)
            except ValueError:
                failed_attempts.append((program_str, [f"Could not parse: {program_str}"]))
                continue
            
            passed, failures = verify_with_feedback(program, examples)
            
            if passed:
                elapsed = time.time() - start_time
                return program, candidates_explored, elapsed
            else:
                failed_attempts.append((program_str, failures))
                
        except Exception as e:
            failed_attempts.append(("API_ERROR", [str(e)]))
            candidates_explored += 1
            time.sleep(1)
    
    elapsed = time.time() - start_time
    return None, candidates_explored, elapsed
