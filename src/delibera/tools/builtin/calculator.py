"""Safe calculator tool using AST parsing.

This tool evaluates simple arithmetic expressions safely using
Python's AST module, without using eval().
"""

import ast
import operator
from collections.abc import Callable
from typing import Any

from delibera.tools.spec import RiskLevel, ToolCapability, ToolExecutionError

# Type aliases for operator functions
BinaryOpFunc = Callable[[float | int, float | int], float | int]
UnaryOpFunc = Callable[[float | int], float | int]


class CalculatorTool:
    """A safe calculator tool that evaluates arithmetic expressions.

    Uses AST parsing to safely evaluate expressions without eval().
    Only allows basic arithmetic operations: +, -, *, /, %, **
    and parentheses for grouping.

    Input: {"expression": "1+2*3"}
    Output: {"result": 7}
    """

    # Allowed binary operators
    _BINARY_OPS: dict[type[ast.operator], BinaryOpFunc] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.FloorDiv: operator.floordiv,
    }

    # Allowed unary operators
    _UNARY_OPS: dict[type[ast.unaryop], UnaryOpFunc] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    @property
    def name(self) -> str:
        """Unique identifier for the tool."""
        return "calculator"

    @property
    def risk_level(self) -> RiskLevel:
        """Risk classification of this tool."""
        return RiskLevel.LOW

    @property
    def capability(self) -> ToolCapability:
        """Calculator is a compute tool."""
        return ToolCapability.COMPUTE

    @property
    def is_discovery(self) -> bool:
        """Whether this tool discovers new information."""
        return self.capability == ToolCapability.DISCOVERY

    def validate_input(self, tool_input: dict[str, Any]) -> None:
        """Validate the input before execution.

        Args:
            tool_input: The input dictionary to validate.

        Raises:
            ValueError: If the input is invalid.
        """
        if "expression" not in tool_input:
            raise ValueError("Missing required field: 'expression'")

        expression = tool_input["expression"]
        if not isinstance(expression, str):
            raise ValueError("Field 'expression' must be a string")

        if not expression.strip():
            raise ValueError("Field 'expression' cannot be empty")

    def execute(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Execute the calculator with the given expression.

        Args:
            tool_input: Dictionary with 'expression' key.

        Returns:
            Dictionary with 'result' key containing the computed value.

        Raises:
            ToolExecutionError: If the expression is invalid or evaluation fails.
        """
        self.validate_input(tool_input)
        expression = tool_input["expression"]

        try:
            result = self._safe_eval(expression)
            return {"result": result}
        except (ValueError, TypeError, ZeroDivisionError) as e:
            raise ToolExecutionError(self.name, str(e)) from e

    def _safe_eval(self, expression: str) -> float | int:
        """Safely evaluate an arithmetic expression using AST.

        Args:
            expression: The arithmetic expression string.

        Returns:
            The computed result.

        Raises:
            ValueError: If the expression contains disallowed constructs.
            ZeroDivisionError: If division by zero occurs.
        """
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}") from e

        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.expr) -> float | int:
        """Recursively evaluate an AST node.

        Args:
            node: The AST node to evaluate.

        Returns:
            The computed value.

        Raises:
            ValueError: If the node type is not allowed.
        """
        # Handle numeric literals
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

        # Handle binary operations
        if isinstance(node, ast.BinOp):
            bin_op_type = type(node.op)
            if bin_op_type not in self._BINARY_OPS:
                raise ValueError(f"Unsupported binary operator: {bin_op_type.__name__}")

            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_func = self._BINARY_OPS[bin_op_type]

            # Check for division by zero
            if bin_op_type in (ast.Div, ast.Mod, ast.FloorDiv) and right == 0:
                raise ZeroDivisionError("Division by zero")

            result: float | int = op_func(left, right)
            return result

        # Handle unary operations
        if isinstance(node, ast.UnaryOp):
            unary_op_type = type(node.op)
            if unary_op_type not in self._UNARY_OPS:
                raise ValueError(f"Unsupported unary operator: {unary_op_type.__name__}")

            operand = self._eval_node(node.operand)
            unary_result: float | int = self._UNARY_OPS[unary_op_type](operand)
            return unary_result

        # Reject everything else
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")
