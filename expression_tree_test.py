import unittest
import math
from hypothesis import given, strategies as st
import sys
import io

from expression_tree import parse_expression

# Disable logging during tests
# logging.disable(logging.CRITICAL)


class TestExpressionTree(unittest.TestCase):
    """Test cases for the expression tree interpreter."""

    def test_simple_addition(self):
        """Test simple addition expression."""
        tree = parse_expression("a + b")
        result = tree.evaluate({"a": 3, "b": 4})
        self.assertEqual(result, 7)

    def test_complex_expression(self):
        """Test a more complex expression."""
        tree = parse_expression("a + 2 - sin(-0.3) * (b - c)")
        result = tree.evaluate({"a": 5, "b": 10, "c": 7})
        expected = 5 + 2 - math.sin(-0.3) * (10 - 7)
        self.assertAlmostEqual(result, expected)

    def test_custom_function(self):
        """Test expression with custom function."""
        tree = parse_expression("foo(x) + y")
        result = tree.evaluate(
            {"x": 2, "y": 3},
            {"foo": lambda x: x * 42}
        )
        self.assertEqual(result, 2 * 42 + 3)

    def test_nested_expressions(self):
        """Test nested expressions with parentheses."""
        tree = parse_expression("(a + b) * (c - d)")
        result = tree.evaluate({"a": 2, "b": 3, "c": 10, "d": 5})
        self.assertEqual(result, (2 + 3) * (10 - 5))

    def test_power_operator(self):
        """Test power operator."""
        tree = parse_expression("a^b")
        result = tree.evaluate({"a": 2, "b": 3})
        self.assertEqual(result, 8)

        # Test with ** notation
        tree = parse_expression("a**b")
        result = tree.evaluate({"a": 2, "b": 3})
        self.assertEqual(result, 8)

    def test_built_in_functions(self):
        """Test built-in mathematical functions."""
        functions_to_test = [
            ("sin(x)", lambda x: math.sin(x)),
            ("cos(x)", lambda x: math.cos(x)),
            ("tan(x)", lambda x: math.tan(x)),
            ("sqrt(x)", lambda x: math.sqrt(x)),
            ("log(x)", lambda x: math.log(x)),
            ("exp(x)", lambda x: math.exp(x)),
            ("abs(x)", lambda x: abs(x))
        ]

        for expr_str, func in functions_to_test:
            tree = parse_expression(expr_str)
            x_value = 0.5
            result = tree.evaluate({"x": x_value})
            expected = func(x_value)
            self.assertAlmostEqual(result, expected)

    def test_variable_error(self):
        """Test error when variable is not defined."""
        tree = parse_expression("a + b")
        with self.assertRaises(ValueError) as context:
            tree.evaluate({"a": 1})
        self.assertIn("Variable 'b' is not defined", str(context.exception))

    def test_function_error(self):
        """Test error when function is not defined."""
        tree = parse_expression("unknown_func(x)")
        with self.assertRaises(ValueError) as context:
            tree.evaluate({"x": 1})
        self.assertIn(
            "Function 'unknown_func' is not defined", str(
                context.exception))

    def test_division_by_zero(self):
        """Test division by zero error."""
        tree = parse_expression("a / b")
        with self.assertRaises(ZeroDivisionError) as context:
            tree.evaluate({"a": 1, "b": 0})
        self.assertIn("Division by zero", str(context.exception))

    def test_invalid_expression(self):
        """Test invalid expression parsing."""
        invalid_expressions = [
            "a +",  # Incomplete
            "a + )",  # Unmatched parenthesis
            "a + (b",  # Unmatched parenthesis
            "sin()",  # Missing argument
            "1 2 3",  # Missing operators
        ]

        for expr in invalid_expressions:
            with self.assertRaises(ValueError):
                parse_expression(expr)

    def test_visualization(self):
        """Test visualization functionality."""
        tree = parse_expression("a + b * c")
        # Capture stdout to verify output
        # Remove these duplicate imports
        # import io
        # import sys
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        dot_code = tree.visualize(
            variables={
                "a": 1,
                "b": 2,
                "c": 3},
            show_trace=True)

        # Reset stdout
        sys.stdout = original_stdout

        # Print the visualization again to the actual console
        print(
            "\n=== Visualization from test_visualization ===\n",
            file=sys.stderr)
        print(dot_code, file=sys.stderr)

        # Verify the output contains expected elements
        self.assertIn("digraph G {", dot_code)
        self.assertIn("rankdir=LR;", dot_code)

        # Check that the DOT code contains node and edge definitions
        self.assertRegex(dot_code, r'\d+\[label="a\\n= 1\.00"\];')
        self.assertRegex(dot_code, r'\d+\[label="\*\\n= 6\.00"\];')
        self.assertRegex(dot_code, r'\d+ -> \d+\[label="\d+"\];')

        # Verify it was printed to console
        console_output = captured_output.getvalue()
        self.assertIn("digraph G {", console_output)

    @given(st.floats(min_value=-100, max_value=100),
           st.floats(min_value=-100, max_value=100))
    def test_addition_commutative(self, a, b):
        """Test that addition is commutative using property-based testing."""
        # Skip NaN values
        if math.isnan(a) or math.isnan(b):
            return

        tree1 = parse_expression("a + b")
        tree2 = parse_expression("b + a")

        result1 = tree1.evaluate({"a": a, "b": b})
        result2 = tree2.evaluate({"a": a, "b": b})

        self.assertAlmostEqual(result1, result2)

    @given(st.floats(min_value=-100, max_value=100),
           st.floats(min_value=-100, max_value=100),
           st.floats(min_value=-100, max_value=100))
    def test_addition_associative(self, a, b, c):
        """Test that addition is associative using property-based testing."""
        # Skip NaN values
        if math.isnan(a) or math.isnan(b) or math.isnan(c):
            return

        tree1 = parse_expression("(a + b) + c")
        tree2 = parse_expression("a + (b + c)")

        result1 = tree1.evaluate({"a": a, "b": b, "c": c})
        result2 = tree2.evaluate({"a": a, "b": b, "c": c})

        self.assertAlmostEqual(result1, result2)

    def test_complex_example(self):
        """Test a complex example with multiple operations and functions."""
        expression = "sin(a) + cos(b) * sqrt(c^2 + d^2) / (e - f)"
        tree = parse_expression(expression)

        variables = {"a": 0.5, "b": 0.3, "c": 3, "d": 4, "e": 10, "f": 5}
        result = tree.evaluate(variables)

        # Calculate expected result manually
        expected = (math.sin(0.5) +
                    math.cos(0.3) * math.sqrt(3**2 + 4**2) / (10 - 5))

        self.assertAlmostEqual(result, expected)

        # Test visualization with trace - no need to check for file
        # Remove these duplicate imports
        # import sys
        # import io
        original_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        dot_code = tree.visualize(
            variables=variables,
            show_trace=True)

        # Reset stdout
        sys.stdout = original_stdout

        # Print the visualization again to the actual console
        print(
            "\n=== Visualization from test_complex_example ===\n",
            file=sys.stderr)
        print(dot_code, file=sys.stderr)

        self.assertIn("digraph G {", dot_code)
        self.assertIn("rankdir=LR;", dot_code)


if __name__ == "__main__":
    unittest.main()
