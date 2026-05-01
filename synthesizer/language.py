"""
language.py
-----------
Defines the expression language for program synthesis.
This is the core data structure that all four synthesizers work with.

The language supports:
- Arithmetic: add, subtract, multiply
- String: concat, slice, length
- Conditionals: if_then_else
- Variables: x, y
- Integer constants: -2, -1, 0, 1, 2, 3, 5, 10
- String constants: "", " ", "a", "0"

Programs are represented as abstract syntax trees (ASTs).
Each node in the AST is an Expr object.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union, List, Any


# ---------------------------------------------------------------------------
# Type alias for values that programs can produce
# ---------------------------------------------------------------------------
Value = Union[int, str, bool]


# ---------------------------------------------------------------------------
# Base class for all expressions
# ---------------------------------------------------------------------------
@dataclass
class Expr:
    """Base class for all expression nodes in the AST."""

    def eval(self, env: dict) -> Value:
        """Evaluate this expression given a variable environment.
        
        Args:
            env: dictionary mapping variable names to values e.g. {"x": 3, "y": 5}
            
        Returns:
            The result of evaluating this expression
            
        Raises:
            TypeError: if operands have wrong types
            ValueError: if operation is invalid
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Return a human-readable string representation of this expression."""
        raise NotImplementedError

    def depth(self) -> int:
        """Return the depth of this expression tree."""
        raise NotImplementedError

    def size(self) -> int:
        """Return the number of nodes in this expression tree."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Terminal expressions (leaves of the AST)
# ---------------------------------------------------------------------------
@dataclass
class Var(Expr):
    """A variable reference, e.g. x or y."""
    name: str

    def eval(self, env: dict) -> Value:
        if self.name not in env:
            raise ValueError(f"Variable '{self.name}' not found in environment")
        return env[self.name]

    def __str__(self) -> str:
        return self.name

    def depth(self) -> int:
        return 0

    def size(self) -> int:
        return 1


@dataclass
class IntConst(Expr):
    """An integer constant, e.g. 3 or -1."""
    value: int

    def eval(self, env: dict) -> Value:
        return self.value

    def __str__(self) -> str:
        return str(self.value)

    def depth(self) -> int:
        return 0

    def size(self) -> int:
        return 1


@dataclass
class StrConst(Expr):
    """A string constant, e.g. "" or " "."""
    value: str

    def eval(self, env: dict) -> Value:
        return self.value

    def __str__(self) -> str:
        return f'"{self.value}"'

    def depth(self) -> int:
        return 0

    def size(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# Arithmetic expressions
# ---------------------------------------------------------------------------
@dataclass
class Add(Expr):
    """Addition: add(a, b) = a + b"""
    left: Expr
    right: Expr

    def eval(self, env: dict) -> Value:
        l = self.left.eval(env)
        r = self.right.eval(env)
        if not isinstance(l, int) or not isinstance(r, int):
            raise TypeError(f"add requires integers, got {type(l)} and {type(r)}")
        return l + r

    def __str__(self) -> str:
        return f"add({self.left}, {self.right})"

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()


@dataclass
class Subtract(Expr):
    """Subtraction: subtract(a, b) = a - b"""
    left: Expr
    right: Expr

    def eval(self, env: dict) -> Value:
        l = self.left.eval(env)
        r = self.right.eval(env)
        if not isinstance(l, int) or not isinstance(r, int):
            raise TypeError(f"subtract requires integers, got {type(l)} and {type(r)}")
        return l - r

    def __str__(self) -> str:
        return f"subtract({self.left}, {self.right})"

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()


@dataclass
class Multiply(Expr):
    """Multiplication: multiply(a, b) = a * b"""
    left: Expr
    right: Expr

    def eval(self, env: dict) -> Value:
        l = self.left.eval(env)
        r = self.right.eval(env)
        if not isinstance(l, int) or not isinstance(r, int):
            raise TypeError(f"multiply requires integers, got {type(l)} and {type(r)}")
        return l * r

    def __str__(self) -> str:
        return f"multiply({self.left}, {self.right})"

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()


# ---------------------------------------------------------------------------
# String expressions
# ---------------------------------------------------------------------------
@dataclass
class Concat(Expr):
    """String concatenation: concat(a, b) = a + b"""
    left: Expr
    right: Expr

    def eval(self, env: dict) -> Value:
        l = self.left.eval(env)
        r = self.right.eval(env)
        # allow concatenating string with int by converting int to string
        if isinstance(l, int):
            l = str(l)
        if isinstance(r, int):
            r = str(r)
        if not isinstance(l, str) or not isinstance(r, str):
            raise TypeError(f"concat requires strings, got {type(l)} and {type(r)}")
        return l + r

    def __str__(self) -> str:
        return f"concat({self.left}, {self.right})"

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()


@dataclass
class Slice(Expr):
    """String slicing: slice(s, start, end) = s[start:end]"""
    string: Expr
    start: Expr
    end: Expr

    def eval(self, env: dict) -> Value:
        s = self.string.eval(env)
        start = self.start.eval(env)
        end = self.end.eval(env)
        if not isinstance(s, str):
            raise TypeError(f"slice requires a string, got {type(s)}")
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError(f"slice indices must be integers")
        return s[start:end]

    def __str__(self) -> str:
        return f"slice({self.string}, {self.start}, {self.end})"

    def depth(self) -> int:
        return 1 + max(self.string.depth(), self.start.depth(), self.end.depth())

    def size(self) -> int:
        return 1 + self.string.size() + self.start.size() + self.end.size()


@dataclass
class Length(Expr):
    """String length: length(s) = len(s)"""
    string: Expr

    def eval(self, env: dict) -> Value:
        s = self.string.eval(env)
        if not isinstance(s, str):
            raise TypeError(f"length requires a string, got {type(s)}")
        return len(s)

    def __str__(self) -> str:
        return f"length({self.string})"

    def depth(self) -> int:
        return 1 + self.string.depth()

    def size(self) -> int:
        return 1 + self.string.size()


# ---------------------------------------------------------------------------
# Conditional expressions
# ---------------------------------------------------------------------------
@dataclass
class IfThenElse(Expr):
    """Conditional: if_then_else(cond, then, else) = then if cond else else_branch"""
    condition: Expr
    then_branch: Expr
    else_branch: Expr

    def eval(self, env: dict) -> Value:
        cond = self.condition.eval(env)
        if cond:
            return self.then_branch.eval(env)
        else:
            return self.else_branch.eval(env)

    def __str__(self) -> str:
        return f"if_then_else({self.condition}, {self.then_branch}, {self.else_branch})"

    def depth(self) -> int:
        return 1 + max(
            self.condition.depth(),
            self.then_branch.depth(),
            self.else_branch.depth()
        )

    def size(self) -> int:
        return 1 + self.condition.size() + self.then_branch.size() + self.else_branch.size()


# ---------------------------------------------------------------------------
# Comparison expressions (used as conditions in if_then_else)
# ---------------------------------------------------------------------------
@dataclass
class GreaterThan(Expr):
    """Greater than comparison: a > b"""
    left: Expr
    right: Expr

    def eval(self, env: dict) -> Value:
        l = self.left.eval(env)
        r = self.right.eval(env)
        return l > r

    def __str__(self) -> str:
        return f"gt({self.left}, {self.right})"

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()


@dataclass
class GreaterThanOrEqual(Expr):
    """Greater than or equal comparison: a >= b"""
    left: Expr
    right: Expr

    def eval(self, env: dict) -> Value:
        l = self.left.eval(env)
        r = self.right.eval(env)
        return l >= r

    def __str__(self) -> str:
        return f"gte({self.left}, {self.right})"

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()


@dataclass
class Equals(Expr):
    """Equality comparison: a == b"""
    left: Expr
    right: Expr

    def eval(self, env: dict) -> Value:
        l = self.left.eval(env)
        r = self.right.eval(env)
        return l == r

    def __str__(self) -> str:
        return f"eq({self.left}, {self.right})"

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()


# ---------------------------------------------------------------------------
# Constants used by the synthesizers
# ---------------------------------------------------------------------------

# All integer constants available in the language
INT_CONSTANTS: List[int] = [-2, -1, 0, 1, 2, 3, 5, 10]

# All string constants available in the language
STR_CONSTANTS: List[str] = ["", " ", "a", "0"]

# Variable names
VARIABLES: List[str] = ["x", "y"]


# ---------------------------------------------------------------------------
# BNF Grammar (used in LLM prompts)
# ---------------------------------------------------------------------------
BNF_GRAMMAR = """
Program := Expr

Expr := Var
      | IntConst
      | StrConst
      | add(Expr, Expr)
      | subtract(Expr, Expr)
      | multiply(Expr, Expr)
      | concat(Expr, Expr)
      | slice(Expr, Expr, Expr)
      | length(Expr)
      | if_then_else(BoolExpr, Expr, Expr)

BoolExpr := gt(Expr, Expr)
          | gte(Expr, Expr)
          | eq(Expr, Expr)

Var := x | y

IntConst := -2 | -1 | 0 | 1 | 2 | 3 | 5 | 10

StrConst := "" | " " | "a" | "0"
"""


def _wrap(val):
    """Wrap a raw Python value as an Expr node if needed."""
    if isinstance(val, Expr):
        return val
    if isinstance(val, bool):
        return IntConst(int(val))
    if isinstance(val, int):
        return IntConst(val)
    if isinstance(val, str):
        return StrConst(val)
    return val


def parse_program(program_str: str) -> Expr:
    """
    Parse a program string into an Expr AST.

    This is used by LLM synthesizers to convert LLM output into
    an executable AST that can be evaluated by the verifier.

    Handles programs like:
        add(x, 3)
        if_then_else(gte(x, 0), x, subtract(0, x))
        concat(concat(x, " "), y)

    Args:
        program_str: string representation of a program

    Returns:
        Expr: the parsed AST

    Raises:
        ValueError: if the string cannot be parsed
    """
    import re

    program_str = program_str.strip()

    # Strip common LLM formatting artifacts like markdown code blocks
    program_str = re.sub(r'```[a-z]*\n?', '', program_str)
    program_str = re.sub(r'```', '', program_str)
    program_str = program_str.strip()

    # Take only the first line if LLM returned multiple lines
    program_str = program_str.split('\n')[0].strip()

    # Replace string literals with _sc() calls before eval
    def replace_str_const(match):
        s = match.group(1)
        return f'_sc("{s}")'

    processed = re.sub(r'"([^"]*)"', replace_str_const, program_str)

    # Define safe eval context
    # Each constructor wraps its arguments so bare Python ints/strs work
    safe_globals = {
        "__builtins__": {},
        "_sc":          lambda s: StrConst(s),
        "add":          lambda a, b: Add(_wrap(a), _wrap(b)),
        "subtract":     lambda a, b: Subtract(_wrap(a), _wrap(b)),
        "multiply":     lambda a, b: Multiply(_wrap(a), _wrap(b)),
        "concat":       lambda a, b: Concat(_wrap(a), _wrap(b)),
        "slice":        lambda s, st, en: Slice(_wrap(s), _wrap(st), _wrap(en)),
        "length":       lambda s: Length(_wrap(s)),
        "if_then_else": lambda c, t, e: IfThenElse(_wrap(c), _wrap(t), _wrap(e)),
        "gt":           lambda a, b: GreaterThan(_wrap(a), _wrap(b)),
        "gte":          lambda a, b: GreaterThanOrEqual(_wrap(a), _wrap(b)),
        "eq":           lambda a, b: Equals(_wrap(a), _wrap(b)),
        "x": Var("x"),
        "y": Var("y"),
    }

    try:
        result = eval(processed, safe_globals)
        return _wrap(result)
    except Exception as e:
        raise ValueError(f"Could not parse program '{program_str}': {e}")
