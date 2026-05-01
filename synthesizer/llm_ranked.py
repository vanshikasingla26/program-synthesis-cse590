"""
llm_ranked.py
-------------
LLM-Ranked Enumerative Synthesizer.

This is a novel hybrid approach that combines the correctness guarantees
of bottom-up enumerative synthesis with LLM intelligence for search guidance.

How it works:
1. Generate a pool of candidate programs using bottom-up enumeration
2. Ask an LLM to rank the candidates by likelihood of being correct
3. Try candidates in LLM-ranked order instead of size order
4. Verify each candidate using the formal verifier

This is inspired by DreamCoder's recognition network (Ellis et al. 2021)
which uses a neural policy to score candidates during search, but instead
of a trained neural network we use an LLM as the scoring function.

The key difference from pure LLM synthesis (llm_gpt.py etc.) is that:
- Programs are guaranteed to be syntactically valid (enumerated from grammar)
- The verifier still checks correctness formally
- LLM only guides ordering, not generation

The key difference from pure enumeration (enumerative.py) is that:
- Search order is intelligent rather than arbitrary
- Harder programs are found faster when LLM ranking is accurate
"""

import os
import time
import json
import requests
from typing import List, Optional, Tuple
from synthesizer.language import (
    Expr, Var, IntConst, StrConst,
    Add, Subtract, Multiply,
    Concat, Slice, Length,
    IfThenElse, GreaterThan, GreaterThanOrEqual, Equals,
    INT_CONSTANTS, STR_CONSTANTS, VARIABLES, BNF_GRAMMAR
)
from synthesizer.verifier import Example, verify, verify_with_feedback

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# How many candidates to generate before asking LLM to rank
POOL_SIZE = 50
MAX_CANDIDATES = 10000
MAX_DEPTH = 3  # Slightly shallower than pure enumeration to keep pool manageable


def _generate_candidate_pool(
    examples: List[Example],
    variables: List[str],
    pool_size: int = POOL_SIZE,
    max_depth: int = MAX_DEPTH
) -> List[Expr]:
    """
    Generate a pool of candidate programs using bottom-up enumeration.
    
    Instead of checking each candidate immediately, we collect a pool
    and let the LLM rank them.
    
    Args:
        examples: input-output examples (used to filter obviously wrong candidates)
        variables: variable names available
        pool_size: how many candidates to collect
        max_depth: maximum program depth
        
    Returns:
        List of candidate programs
    """
    pool = []
    
    # Round 0: terminals
    terminals = []
    for var_name in variables:
        if any(var_name in inputs for inputs, _ in examples):
            terminals.append(Var(var_name))
    for c in INT_CONSTANTS:
        terminals.append(IntConst(c))
    for c in STR_CONSTANTS:
        terminals.append(StrConst(c))
    
    pool.extend(terminals)
    rounds = [terminals]
    
    for depth in range(1, max_depth + 1):
        if len(pool) >= pool_size:
            break
            
        all_previous = []
        for r in rounds:
            all_previous.extend(r)
        
        new_programs = []
        
        # Arithmetic
        for left in all_previous[:20]:  # Limit to avoid explosion
            for right in all_previous[:20]:
                new_programs.append(Add(left, right))
                new_programs.append(Subtract(left, right))
                new_programs.append(Multiply(left, right))
        
        # String ops
        for left in all_previous[:20]:
            for right in all_previous[:20]:
                new_programs.append(Concat(left, right))
        
        for expr in all_previous[:30]:
            new_programs.append(Length(expr))
        
        # Slice
        for s in all_previous[:10]:
            for st in all_previous[:10]:
                for en in all_previous[:10]:
                    new_programs.append(Slice(s, st, en))
        
        # Conditionals
        bool_exprs = []
        for left in all_previous[:10]:
            for right in all_previous[:10]:
                bool_exprs.append(GreaterThan(left, right))
                bool_exprs.append(GreaterThanOrEqual(left, right))
                bool_exprs.append(Equals(left, right))
        
        for cond in bool_exprs[:20]:
            for then_b in all_previous[:10]:
                for else_b in all_previous[:10]:
                    new_programs.append(IfThenElse(cond, then_b, else_b))
        
        rounds.append(new_programs)
        pool.extend(new_programs)
        
        if len(pool) >= pool_size * 3:
            break
    
    return pool[:pool_size * 3]  # Return more than pool_size for LLM to rank


def _rank_candidates_with_llm(
    candidates: List[Expr],
    examples: List[Example],
    top_k: int = 20
) -> List[Expr]:
    """
    Ask GPT-4o to rank candidate programs by likelihood of being correct.
    
    Args:
        candidates: pool of candidate programs
        examples: input-output examples
        top_k: how many top candidates to return
        
    Returns:
        Reordered list of candidates (most promising first)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Format examples
    examples_str = "\n".join(
        f"  Input: {inputs} -> Output: {repr(expected)}"
        for inputs, expected in examples
    )
    
    # Format candidates (limit to avoid token overflow)
    candidate_strs = [str(c) for c in candidates[:30]]
    candidates_formatted = "\n".join(
        f"{i+1}. {prog}" for i, prog in enumerate(candidate_strs)
    )
    
    prompt = f"""You are helping with program synthesis. Given input-output examples and a list of candidate programs, rank the candidates from most likely to least likely to be the correct answer.

Examples:
{examples_str}

Candidate programs:
{candidates_formatted}

Return ONLY a JSON array of indices (1-based) in order from most to least promising.
Example format: [3, 1, 7, 2, 5, ...]
Include all {len(candidate_strs)} indices. Return only the JSON array, nothing else."""

    try:
        response = requests.post(
            OPENAI_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,  # Low temperature for ranking
                "max_tokens": 500
            },
            timeout=30
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        
        # Parse the ranking
        # Handle various formats LLM might return
        import re
        # Extract array from response
        match = re.search(r'\[[\d,\s]+\]', content)
        if match:
            ranking = json.loads(match.group())
            # Convert to 0-based indices
            ranked_indices = [i - 1 for i in ranking if 0 <= i - 1 < len(candidates)]
            # Add any missing indices at the end
            missing = [i for i in range(len(candidates)) if i not in ranked_indices]
            ranked_indices.extend(missing)
            return [candidates[i] for i in ranked_indices[:top_k]]
    except Exception:
        pass
    
    # Fallback: return original order
    return candidates[:top_k]


def synthesize(
    examples: List[Example],
    variables: List[str] = None,
    pool_size: int = POOL_SIZE,
    max_candidates: int = MAX_CANDIDATES
) -> Tuple[Optional[Expr], int, float]:
    """
    Run LLM-ranked enumerative synthesis.
    
    This hybrid approach:
    1. Generates a pool of syntactically valid candidates via enumeration
    2. Uses GPT-4o to rank them by likelihood of being correct
    3. Tries candidates in ranked order with formal verification
    
    Args:
        examples: list of (input_dict, expected_output) pairs
        variables: list of variable names
        pool_size: number of candidates to generate before ranking
        max_candidates: total candidate limit
        
    Returns:
        Tuple of:
        - Optional[Expr]: synthesized program or None
        - int: number of candidates explored
        - float: time in seconds
    """
    if variables is None:
        variables = VARIABLES
    
    start_time = time.time()
    candidates_explored = 0
    
    # Phase 1: Generate candidate pool
    pool = _generate_candidate_pool(examples, variables, pool_size)
    
    # Phase 2: LLM ranks the pool
    ranked_candidates = _rank_candidates_with_llm(pool, examples, top_k=pool_size)
    
    # Phase 3: Try candidates in ranked order
    for program in ranked_candidates:
        candidates_explored += 1
        
        if candidates_explored > max_candidates:
            break
        
        if verify(program, examples):
            elapsed = time.time() - start_time
            return program, candidates_explored, elapsed
    
    # Phase 4: If ranked candidates failed, fall back to remaining pool
    remaining = [p for p in pool if p not in ranked_candidates]
    for program in remaining:
        candidates_explored += 1
        
        if candidates_explored > max_candidates:
            break
        
        if verify(program, examples):
            elapsed = time.time() - start_time
            return program, candidates_explored, elapsed
    
    elapsed = time.time() - start_time
    return None, candidates_explored, elapsed
