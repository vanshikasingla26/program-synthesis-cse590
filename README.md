# Program Synthesis: Enumerative vs LLM-Guided Methods

**CSE 590 — Programming Languages Seminar**  
**University of Michigan**  
**Vanshika Singla**

## Overview

This project compares classical bottom-up enumerative program synthesis against three LLM-guided synthesizers (GPT-4o, Claude, Gemini) on a shared set of 20 benchmarks. It is a small-scale reproduction of the dual-system approach introduced in LILO (Grand et al., ICLR 2024), evaluated in a controlled setting with a shared expression language and verifier.

## Expression Language

Programs operate over a small language with:
- **Arithmetic**: `add(a, b)`, `subtract(a, b)`, `multiply(a, b)`
- **String**: `concat(a, b)`, `slice(s, start, end)`, `length(s)`
- **Conditionals**: `if_then_else(cond, then, else)`
- **Comparisons**: `gt(a, b)`, `gte(a, b)`, `eq(a, b)`
- **Variables**: `x`, `y`
- **Integer constants**: `-2, -1, 0, 1, 2, 3, 5, 10`
- **String constants**: `""`, `" "`, `"a"`, `"0"`

## Synthesizers

| Synthesizer | Approach | Correctness Guarantee |
|-------------|----------|----------------------|
| Enumerative | Bottom-up search, size-ordered | Yes |
| GPT-4o | LLM generation + verification | No (verified post-hoc) |
| Claude | LLM generation + verification | No (verified post-hoc) |
| Gemini | LLM generation + verification | No (verified post-hoc) |

All four synthesizers use the same verifier. LLM synthesizers receive feedback on failed attempts and can retry up to 20 times.

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/vanshikasingla26/program-synthesis-cse590.git
cd program-synthesis-cse590
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API keys
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
```

### 4. Run the evaluation
```bash
# Full evaluation (all 4 synthesizers, all 20 benchmarks)
python main.py

# Test with enumerative only (no API calls needed)
python main.py --skip-llms

# Run specific synthesizers
python main.py --synthesizers Enumerative GPT-4o

# Generate charts from existing results
python main.py --analyze-only
```

## Project Structure

```
synthesis-project/
├── synthesizer/
│   ├── language.py       # Expression language AST definition
│   ├── verifier.py       # Checks programs against examples
│   ├── enumerative.py    # Bottom-up enumerative synthesizer
│   ├── llm_gpt.py        # GPT-4o guided synthesizer
│   ├── llm_claude.py     # Claude guided synthesizer
│   └── llm_gemini.py     # Gemini guided synthesizer
├── benchmarks/
│   └── benchmarks.py     # 20 benchmark tasks
├── evaluation/
│   ├── run_eval.py       # Runs all synthesizers on all benchmarks
│   └── results.py        # Generates tables and charts
├── results/              # Generated after running (CSV + charts)
├── main.py               # Entry point
├── requirements.txt
└── README.md
```

## Benchmarks

20 tasks across four difficulty levels:

| Category | Count | Examples |
|----------|-------|---------|
| Simple Arithmetic | 5 | add inputs, multiply by constant, subtract one |
| Hard Arithmetic | 5 | absolute value, max of two, nested operations |
| Simple String | 5 | concatenate, first char, get length |
| Hard String | 5 | length plus one, slice by variable, conditional length |

Each benchmark has 3 training examples and 1 held-out test case for generalization testing.

## Results

Results are saved to `results/` after running:
- `raw_results.csv` — full results per synthesizer per benchmark
- `results_table.csv` — summary success rates
- `comparison_chart.png` — bar charts comparing synthesizers
- `benchmark_heatmap.png` — per-benchmark heatmap

## Key Findings

See the project report for full analysis. Key observations:
- Enumerative synthesis reliably solves simple benchmarks with provable correctness
- LLM-guided approaches solve harder benchmarks with fewer candidates but no guarantees
- Claude and GPT-4o outperform Gemini on complex string operations
- Results directly support LILO's finding that LLM guidance reduces search cost

## Related Work

- Solar-Lezama et al. (2006) — SKETCH: combinatorial sketching for finite programs
- Gulwani (2011) — FlashFill: synthesis for spreadsheet string transformations
- Ellis et al. (2021) — DreamCoder: wake-sleep library learning
- Chen et al. (2021) — Codex: evaluating LLMs trained on code
- Grand et al. (2024) — LILO: learning interpretable libraries
