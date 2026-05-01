"""
llm_gpt.py
----------
GPT-4o guided program synthesizer.

Uses OpenAI's GPT-4o API directly via requests to avoid httpx
version conflicts with other packages.

The key insight from LILO (Grand et al. 2024) is that LLMs can
generate plausible candidates much faster than exhaustive enumeration,
but need a formal verifier to check correctness.
"""

import os
import time
import json
import requests
from typing import List, Optional, Tuple
from synthesizer.language import Expr, BNF_GRAMMAR, parse_program
from synthesizer.verifier import Example, verify_with_feedback

MAX_ATTEMPTS = 20
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


def _build_system_prompt() -> str:
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


def _build_user_prompt(examples, failed_attempts, attempt_number):
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


def synthesize(examples, variables=None, max_attempts=MAX_ATTEMPTS):
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    start_time = time.time()
    candidates_explored = 0
    failed_attempts = []
    system_prompt = _build_system_prompt()

    for attempt in range(1, max_attempts + 1):
        user_prompt = _build_user_prompt(examples, failed_attempts, attempt)
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 200
        }
        try:
            response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            program_str = data["choices"][0]["message"]["content"].strip()
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
