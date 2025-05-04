# Accepted - Lab 3 - Variant 1

This project implements an expression tree parser and evaluator for mathematical expressions.
It provides functionality to parse, evaluate, and visualize expression trees with support for
variables, operators, and functions.
It maintains a tree structure for representing mathematical expressions
while providing efficient evaluation and visualization capabilities.

---

## Project Structure

- `expression_tree.py`
  Core implementation
  of the expression tree classes and parsing functionality.
- `expression_tree_test.py`
  Unit tests, Property-Based tests, and performance tests
  for the expression tree implementation.
- `README.md`
  Project documentation.

---

## Features

- **PBT: `test_simple_addition`**
  Tests simple addition expressions to ensure basic operations work correctly
  with variable substitution.
- **PBT: `test_complex_expression`**
  Validates that complex expressions with multiple operations and functions
  are evaluated correctly.
- **PBT: `test_custom_function`**
  Tests the ability to use custom functions in expressions
  and ensure they are correctly evaluated.
- **PBT: `test_nested_expressions`**
  Tests nested expressions with parentheses to verify proper order of operations
  and expression grouping.
- **PBT: `test_power_operator`**
  Verifies that the power operator (^ and **) works correctly
  for various numeric inputs.
- **PBT: `test_built_in_functions`**
  Tests built-in mathematical functions (sin, cos, tan, sqrt, log, exp, abs)
  to ensure they produce correct results.
- **PBT: `test_variable_error`**
  Verifies that appropriate errors are raised when variables
  are not defined in the evaluation context.
- **PBT: `test_function_error`**
  Tests error handling when undefined functions are used
  in expressions.
- **PBT: `test_division_by_zero`**
  Ensures proper error handling for division by zero
  operations.
- **PBT: `test_invalid_expression`**
  Tests that invalid expressions are properly detected
  and appropriate errors are raised.
- **PBT: `test_visualization`**
  Verifies the visualization functionality generates correct DOT code
  and creates visualization files.
- **PBT: `test_addition_commutative`**
  Tests the commutative property of addition using property-based testing
  to ensure a + b = b + a for various inputs.
- **PBT: `test_addition_associative`**
  Validates the associative property of addition using property-based testing
  to ensure (a + b) + c = a + (b + c).
- **PBT: `test_complex_example`**
  Tests a complex example with multiple operations and functions
  to verify the overall functionality of the expression tree.

---

## Contribution

- **Cao Xinyang**: initial version.
- **Xiong Shichi**: subsequent improvements.

---

## Changelog

- 04.05.2025 - 2  
   - Added visualization functionality
   - Implemented DOT format output for expression trees
   - Added support for tracing expression evaluation
- 03.05.2025 - 1  
   - Implemented expression parsing and evaluation
   - Added support for basic operators and functions
   - Implemented error handling for undefined variables and functions
- 02.05.2025 - 0  
   - Initial project setup

---

## Design notes

- <https://en.wikipedia.org/wiki/Abstract_syntax_tree>
- <https://en.wikipedia.org/wiki/Recursive_descent_parser>
- <https://en.wikipedia.org/wiki/DOT_(graph_description_language)>
