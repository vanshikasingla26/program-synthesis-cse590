"""
llm_gemini.py
-------------
Gemini (Google) guided program synthesizer.

Uses Google's Gemini API via the google-genai package.
Same approach as llm_gpt.py and llm_claude.py but using Google's API.
"""

import os
import time
from typing import List, Optional, Tuple
from google import genai
from google.genai import types
from synthesizer.language import Expr, BNF_GRAMMAR, parse_program
from synthesizer.verifier import Example, verify_with_feedback

MAX_ATTEMPTS = 20


def _build_prompt(examples, failed_attempts, attempt_number):
    """Build the full prompt for Gemini."""
    grammar_section = f"""You are a program synthesis assistant. Your job is to find a program
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
    examples_str = "\n".join(
        f"  Input: {inputs} -> Output: {repr(expected)}"
        for inputs, expected in examples
    )
    task_section = f"Find a program that satisfies these examples:\n{examples_str}\n"

    if failed_attempts:
        task_section += "\nPrevious attempts that failed:\n"
        for prog_str, failures in failed_attempts[-3:]:
            task_section += f"  Program: {prog_str}\n"
            for failure in failures[:2]:
                task_section += f"    Failed: {failure}\n"
        task_section += "\nPlease try a different program that fixes these failures.\n"

    task_section += f"\nAttempt {attempt_number}: Return only the program expression."
    return grammar_section + task_section


def synthesize(examples, variables=None, max_attempts=MAX_ATTEMPTS):
    """
    Run Gemini guided synthesis.

    Returns:
        Tuple of (program or None, candidates explored, time in seconds)
    """
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    start_time = time.time()
    candidates_explored = 0
    failed_attempts = []

    for attempt in range(1, max_attempts + 1):
        prompt = _build_prompt(examples, failed_attempts, attempt)

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=200
                )
            )

            program_str = response.text.strip()
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
