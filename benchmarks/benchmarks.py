"""
benchmarks.py
-------------
All 20 benchmark tasks for the synthesis comparison study.

Each benchmark has:
- name: identifier
- description: what the program should do
- examples: list of (input_dict, expected_output) pairs used during synthesis
- held_out: a single test case NOT shown to any synthesizer,
  used to verify generalization
- variables: which variables are used (x, y, or both)
- difficulty: "simple_arithmetic", "hard_arithmetic",
              "simple_string", "hard_string"
- expected_program: the ground truth program (for reference)

Benchmarks are designed so that:
- Simple ones should be solvable by all four synthesizers
- Hard ones may require more search and could expose differences
  between enumeration and LLM-guided approaches
"""

from typing import List, Tuple, Dict, Any, NamedTuple
from synthesizer.verifier import Example


class Benchmark(NamedTuple):
    """A single synthesis benchmark task."""
    name: str
    description: str
    examples: List[Example]
    held_out: Example
    variables: List[str]
    difficulty: str
    expected_program: str  # string form of the ground truth program


# ---------------------------------------------------------------------------
# Simple Arithmetic Benchmarks (5)
# Expected programs are shallow, enumeration should find them quickly
# ---------------------------------------------------------------------------

BENCHMARK_1 = Benchmark(
    name="add_inputs",
    description="Add two input numbers together",
    examples=[
        ({"x": 2, "y": 3}, 5),
        ({"x": 1, "y": 4}, 5),
        ({"x": 0, "y": 7}, 7),
    ],
    held_out=({"x": 6, "y": 9}, 15),
    variables=["x", "y"],
    difficulty="simple_arithmetic",
    expected_program="add(x, y)"
)

BENCHMARK_2 = Benchmark(
    name="multiply_constant",
    description="Multiply input by 3",
    examples=[
        ({"x": 3}, 9),
        ({"x": 4}, 12),
        ({"x": 5}, 15),
    ],
    held_out=({"x": 7}, 21),
    variables=["x"],
    difficulty="simple_arithmetic",
    expected_program="multiply(x, 3)"
)

BENCHMARK_3 = Benchmark(
    name="subtract_one",
    description="Subtract 1 from input",
    examples=[
        ({"x": 5}, 4),
        ({"x": 3}, 2),
        ({"x": 10}, 9),
    ],
    held_out=({"x": 8}, 7),
    variables=["x"],
    difficulty="simple_arithmetic",
    expected_program="subtract(x, 1)"
)

BENCHMARK_4 = Benchmark(
    name="add_constant",
    description="Add 5 to input",
    examples=[
        ({"x": 2}, 7),
        ({"x": 5}, 10),
        ({"x": 0}, 5),
    ],
    held_out=({"x": 3}, 8),
    variables=["x"],
    difficulty="simple_arithmetic",
    expected_program="add(x, 5)"
)

BENCHMARK_5 = Benchmark(
    name="double",
    description="Multiply input by 2",
    examples=[
        ({"x": 3}, 6),
        ({"x": 4}, 8),
        ({"x": 5}, 10),
    ],
    held_out=({"x": 7}, 14),
    variables=["x"],
    difficulty="simple_arithmetic",
    expected_program="multiply(x, 2)"
)

# ---------------------------------------------------------------------------
# Hard Arithmetic Benchmarks (5)
# Require nested operations or conditionals
# ---------------------------------------------------------------------------

BENCHMARK_6 = Benchmark(
    name="nested_add_multiply",
    description="Multiply x by 3 then add 1",
    examples=[
        ({"x": 2}, 7),
        ({"x": 3}, 10),
        ({"x": 4}, 13),
    ],
    held_out=({"x": 5}, 16),
    variables=["x"],
    difficulty="hard_arithmetic",
    expected_program="add(multiply(x, 3), 1)"
)

BENCHMARK_7 = Benchmark(
    name="absolute_value",
    description="Return the absolute value of x",
    examples=[
        ({"x": 3}, 3),
        ({"x": -2}, 2),
        ({"x": 0}, 0),
    ],
    held_out=({"x": -5}, 5),
    variables=["x"],
    difficulty="hard_arithmetic",
    expected_program="if_then_else(gte(x, 0), x, subtract(0, x))"
)

BENCHMARK_8 = Benchmark(
    name="max_of_two",
    description="Return the maximum of x and y",
    examples=[
        ({"x": 3, "y": 5}, 5),
        ({"x": 7, "y": 2}, 7),
        ({"x": 4, "y": 4}, 4),
    ],
    held_out=({"x": 1, "y": 10}, 10),
    variables=["x", "y"],
    difficulty="hard_arithmetic",
    expected_program="if_then_else(gte(x, y), x, y)"
)

BENCHMARK_9 = Benchmark(
    name="subtract_then_multiply",
    description="Subtract 1 from x then multiply by 2",
    examples=[
        ({"x": 3}, 4),
        ({"x": 5}, 8),
        ({"x": 7}, 12),
    ],
    held_out=({"x": 4}, 6),
    variables=["x"],
    difficulty="hard_arithmetic",
    expected_program="multiply(subtract(x, 1), 2)"
)

BENCHMARK_10 = Benchmark(
    name="add_two_then_multiply",
    description="Add x and y then multiply by 2",
    examples=[
        ({"x": 1, "y": 2}, 6),
        ({"x": 3, "y": 3}, 12),
        ({"x": 0, "y": 5}, 10),
    ],
    held_out=({"x": 2, "y": 4}, 12),
    variables=["x", "y"],
    difficulty="hard_arithmetic",
    expected_program="multiply(add(x, y), 2)"
)

# ---------------------------------------------------------------------------
# Simple String Benchmarks (5)
# ---------------------------------------------------------------------------

BENCHMARK_11 = Benchmark(
    name="concat_inputs",
    description="Concatenate two string inputs",
    examples=[
        ({"x": "hi", "y": "!"}, "hi!"),
        ({"x": "hello", "y": "?"}, "hello?"),
        ({"x": "abc", "y": "def"}, "abcdef"),
    ],
    held_out=({"x": "foo", "y": "bar"}, "foobar"),
    variables=["x", "y"],
    difficulty="simple_string",
    expected_program="concat(x, y)"
)

BENCHMARK_12 = Benchmark(
    name="first_char",
    description="Return the first character of the string",
    examples=[
        ({"x": "hello"}, "h"),
        ({"x": "world"}, "w"),
        ({"x": "abc"}, "a"),
    ],
    held_out=({"x": "python"}, "p"),
    variables=["x"],
    difficulty="simple_string",
    expected_program="slice(x, 0, 1)"
)

BENCHMARK_13 = Benchmark(
    name="get_length",
    description="Return the length of the string as an integer",
    examples=[
        ({"x": "hi"}, 2),
        ({"x": "hello"}, 5),
        ({"x": "a"}, 1),
    ],
    held_out=({"x": "python"}, 6),
    variables=["x"],
    difficulty="simple_string",
    expected_program="length(x)"
)

BENCHMARK_14 = Benchmark(
    name="concat_space",
    description="Concatenate two strings with a space between them",
    examples=[
        ({"x": "hello", "y": "world"}, "hello world"),
        ({"x": "foo", "y": "bar"}, "foo bar"),
        ({"x": "a", "y": "b"}, "a b"),
    ],
    held_out=({"x": "good", "y": "morning"}, "good morning"),
    variables=["x", "y"],
    difficulty="simple_string",
    expected_program='concat(concat(x, " "), y)'
)

BENCHMARK_15 = Benchmark(
    name="double_string",
    description="Concatenate the string with itself",
    examples=[
        ({"x": "hi"}, "hihi"),
        ({"x": "ab"}, "abab"),
        ({"x": "a"}, "aa"),
    ],
    held_out=({"x": "xyz"}, "xyzxyz"),
    variables=["x"],
    difficulty="simple_string",
    expected_program="concat(x, x)"
)

# ---------------------------------------------------------------------------
# Hard String Benchmarks (5)
# Require combining string and arithmetic or using conditionals
# ---------------------------------------------------------------------------

BENCHMARK_16 = Benchmark(
    name="length_plus_one",
    description="Return the length of the string plus 1",
    examples=[
        ({"x": "hi"}, 3),
        ({"x": "hello"}, 6),
        ({"x": "a"}, 2),
    ],
    held_out=({"x": "python"}, 7),
    variables=["x"],
    difficulty="hard_string",
    expected_program="add(length(x), 1)"
)

BENCHMARK_17 = Benchmark(
    name="slice_first_two",
    description="Return the first two characters of the string",
    examples=[
        ({"x": "hello"}, "he"),
        ({"x": "world"}, "wo"),
        ({"x": "abcde"}, "ab"),
    ],
    held_out=({"x": "python"}, "py"),
    variables=["x"],
    difficulty="hard_string",
    expected_program="slice(x, 0, 2)"
)

BENCHMARK_18 = Benchmark(
    name="concat_length",
    description="Concatenate string x with string y then get the length",
    examples=[
        ({"x": "hi", "y": "!"}, 3),
        ({"x": "hello", "y": "world"}, 10),
        ({"x": "a", "y": "bc"}, 3),
    ],
    held_out=({"x": "foo", "y": "bar"}, 6),
    variables=["x", "y"],
    difficulty="hard_string",
    expected_program="length(concat(x, y))"
)

BENCHMARK_19 = Benchmark(
    name="slice_by_variable",
    description="Return the first y characters of string x",
    examples=[
        ({"x": "hello", "y": 3}, "hel"),
        ({"x": "world", "y": 2}, "wo"),
        ({"x": "abcde", "y": 4}, "abcd"),
    ],
    held_out=({"x": "python", "y": 3}, "pyt"),
    variables=["x", "y"],
    difficulty="hard_string",
    expected_program="slice(x, 0, y)"
)

BENCHMARK_20 = Benchmark(
    name="conditional_length",
    description="If string is longer than 3 chars return length else return 0",
    examples=[
        ({"x": "hello"}, 5),
        ({"x": "hi"}, 0),
        ({"x": "abcd"}, 4),
    ],
    held_out=({"x": "python"}, 6),
    variables=["x"],
    difficulty="hard_string",
    expected_program="if_then_else(gt(length(x), 3), length(x), 0)"
)

# ---------------------------------------------------------------------------
# Master list of all benchmarks
# ---------------------------------------------------------------------------

ALL_BENCHMARKS: List[Benchmark] = [
    BENCHMARK_1,
    BENCHMARK_2,
    BENCHMARK_3,
    BENCHMARK_4,
    BENCHMARK_5,
    BENCHMARK_6,
    BENCHMARK_7,
    BENCHMARK_8,
    BENCHMARK_9,
    BENCHMARK_10,
    BENCHMARK_11,
    BENCHMARK_12,
    BENCHMARK_13,
    BENCHMARK_14,
    BENCHMARK_15,
    BENCHMARK_16,
    BENCHMARK_17,
    BENCHMARK_18,
    BENCHMARK_19,
    BENCHMARK_20,
]

# Grouped by difficulty for analysis
SIMPLE_ARITHMETIC = [b for b in ALL_BENCHMARKS if b.difficulty == "simple_arithmetic"]
HARD_ARITHMETIC = [b for b in ALL_BENCHMARKS if b.difficulty == "hard_arithmetic"]
SIMPLE_STRING = [b for b in ALL_BENCHMARKS if b.difficulty == "simple_string"]
HARD_STRING = [b for b in ALL_BENCHMARKS if b.difficulty == "hard_string"]
