import re
import logging
import math
import operator
from typing import Dict, Any, Callable, Union, List, Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt

# Configure logging
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create file handler
file_handler = logging.FileHandler('expression_tree.log', mode='w')
file_handler.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Log initialization message
logger.info("Logger initialized")

# Type definitions
NodeType = Union[str, float, int]
FunctionDict = Dict[str, Callable]


class ExpressionNode:
    """Base class for all nodes in the expression tree."""
    
    def __init__(self, value: NodeType):
        """Initialize a node with a value.
        
        Args:
            value: The value stored in this node
        """
        self.value = value
        self.id = id(self)  # Unique identifier for visualization
    
    def evaluate(self, variables: Dict[str, Any] = None, 
                 functions: FunctionDict = None) -> Any:
        """Evaluate this node.
        
        Args:
            variables: Dictionary of variable names to values
            functions: Dictionary of function names to callables
            
        Returns:
            The result of evaluating this node
        """
        raise NotImplementedError("Subclasses must implement evaluate()")
    
    def __str__(self) -> str:
        return str(self.value)


class ConstantNode(ExpressionNode):
    """Node representing a constant value."""
    
    def evaluate(self, variables: Dict[str, Any] = None, 
                 functions: FunctionDict = None) -> float:
        """Return the constant value.
        
        Args:
            variables: Dictionary of variable names to values (unused)
            functions: Dictionary of function names to callables (unused)
            
        Returns:
            The constant value
        """
        logger.debug(f"Evaluating constant: {self.value}")
        return float(self.value)


class VariableNode(ExpressionNode):
    """Node representing a variable."""
    
    def evaluate(self, variables: Dict[str, Any] = None, 
                 functions: FunctionDict = None) -> float:
        """Look up the variable value.
        
        Args:
            variables: Dictionary of variable names to values
            functions: Dictionary of function names to callables (unused)
            
        Returns:
            The value of the variable
            
        Raises:
            ValueError: If the variable is not defined
        """
        if variables is None:
            variables = {}
        
        if self.value not in variables:
            raise ValueError(f"Variable '{self.value}' is not defined")
        
        logger.debug(f"Evaluating variable {self.value} = {variables[self.value]}")
        return float(variables[self.value])


class OperatorNode(ExpressionNode):
    """Node representing a binary operator."""
    
    # Dictionary mapping operator symbols to functions
    OPERATORS = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
        '^': operator.pow,
        '**': operator.pow,
    }
    
    def __init__(self, operator: str, left: ExpressionNode, right: ExpressionNode):
        """Initialize an operator node.
        
        Args:
            operator: The operator symbol
            left: The left operand node
            right: The right operand node
            
        Raises:
            ValueError: If the operator is not supported
        """
        if operator not in self.OPERATORS:
            raise ValueError(f"Unsupported operator: {operator}")
        
        super().__init__(operator)
        self.left = left
        self.right = right
    
    def evaluate(self, variables: Dict[str, Any] = None, 
                 functions: FunctionDict = None) -> float:
        """Evaluate the operation.
        
        Args:
            variables: Dictionary of variable names to values
            functions: Dictionary of function names to callables
            
        Returns:
            The result of applying the operator to the operands
            
        Raises:
            ZeroDivisionError: If division by zero occurs
            ValueError: If an invalid operation is attempted
        """
        left_val = self.left.evaluate(variables, functions)
        right_val = self.right.evaluate(variables, functions)
        
        try:
            op_func = self.OPERATORS[self.value]
            result = op_func(left_val, right_val)
            logger.debug(f"Evaluating {left_val} {self.value} {right_val} = {result}")
            return result
        except ZeroDivisionError:
            raise ZeroDivisionError(f"Division by zero: {left_val} / 0")
        except Exception as e:
            raise ValueError(f"Error evaluating {left_val} {self.value} {right_val}: {str(e)}")


class FunctionNode(ExpressionNode):
    """Node representing a function call."""
    
    # Dictionary of built-in functions
    BUILTIN_FUNCTIONS = {
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'exp': math.exp,
        'sqrt': math.sqrt,
        'abs': abs,
    }
    
    def __init__(self, function_name: str, argument: ExpressionNode):
        """Initialize a function node.
        
        Args:
            function_name: The name of the function
            argument: The argument node
        """
        super().__init__(function_name)
        self.argument = argument
    
    def evaluate(self, variables: Dict[str, Any] = None, 
                 functions: FunctionDict = None) -> float:
        """Evaluate the function.
        
        Args:
            variables: Dictionary of variable names to values
            functions: Dictionary of function names to callables
            
        Returns:
            The result of applying the function to the argument
            
        Raises:
            ValueError: If the function is not defined or an error occurs
        """
        if functions is None:
            functions = {}
        
        # Combine built-in and user-provided functions
        all_functions = {**self.BUILTIN_FUNCTIONS, **functions}
        
        if self.value not in all_functions:
            raise ValueError(f"Function '{self.value}' is not defined")
        
        arg_val = self.argument.evaluate(variables, functions)
        
        try:
            func = all_functions[self.value]
            result = func(arg_val)
            logger.debug(f"Evaluating {self.value}({arg_val}) = {result}")
            return result
        except Exception as e:
            raise ValueError(f"Error evaluating {self.value}({arg_val}): {str(e)}")


class ExpressionTree:
    """Class representing a mathematical expression tree."""
    
    def __init__(self, root: ExpressionNode):
        """Initialize an expression tree.
        
        Args:
            root: The root node of the tree
        """
        self.root = root
    
    def evaluate(self, variables: Dict[str, Any] = None, 
                 functions: FunctionDict = None) -> float:
        """Evaluate the expression tree.
        
        Args:
            variables: Dictionary of variable names to values
            functions: Dictionary of function names to callables
            
        Returns:
            The result of evaluating the expression
        """
        if variables is None:
            variables = {}
        if functions is None:
            functions = {}
        
        logger.info("Starting expression evaluation")
        # Ensure log is written to file and console
        for handler in logger.handlers:
            handler.flush()
        result = self.root.evaluate(variables, functions)
        logger.info(f"Expression evaluated to: {result}")
        # Ensure log is written again
        for handler in logger.handlers:
            handler.flush()
        return result

    def visualize(self, filename: str = "expression_tree.png", 
                  variables: Dict[str, Any] = None,
                  show_trace: bool = False) -> None:
        """Visualize the expression tree as a dataflow graph.
        
        Args:
            filename: The filename to save the visualization
            variables: Dictionary of variable names to values (for trace)
            show_trace: Whether to show evaluation trace on the graph
        """
        G = nx.DiGraph()
        
        # Build the graph
        self._build_graph(G, self.root, variables if show_trace else None)
        
        # 添加结果节点
        if variables is not None:
            result_node_id = "result"
            result_value = self.evaluate(variables)
            G.add_node(result_node_id, label=f"result\n= {result_value:.2f}")
            
            # 连接根节点到结果节点，添加edge_id
            root_id = str(self.root.id)
            # 获取当前最大的edge_id并加1
            max_edge_id = max([int(data.get('label', '0')) for _, _, data in G.edges(data=True) 
                            if data.get('label', '').isdigit()], default=0)
            G.add_edge(root_id, result_node_id, label=str(max_edge_id + 1))
        
        # 使用spring布局，但减小k值以缩短节点间距离
        pos = nx.spring_layout(G, k=0.1, iterations=50, seed=42)
        
        plt.figure(figsize=(12, 10))
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700, alpha=0.8)
        
        # 绘制边和标签，使用箭头
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, width=1.5)
        
        # 保留原始的边标签（包括edge_id）
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)
        
        # 绘制节点标签
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold')
        
        plt.title("Expression Tree Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        logger.info(f"Visualization saved to {filename}")
        # Ensure log is written
        for handler in logger.handlers:
            handler.flush()
    
    def _build_graph(self, G: nx.DiGraph, node: ExpressionNode, 
                     variables: Optional[Dict[str, Any]] = None, 
                     edge_id: int = 0) -> Tuple[str, int]:
        """Recursively build a graph representation of the expression tree.
        
        Args:
            G: The NetworkX graph
            node: The current node
            variables: Dictionary of variable names to values (for trace)
            edge_id: The current edge ID
            
        Returns:
            A tuple of (node_id, next_edge_id)
        """
        node_id = str(node.id)
        
        # Add node with appropriate label
        if variables is not None:
            try:
                value = node.evaluate(variables)
                G.add_node(node_id, label=f"{node}\n= {value:.2f}")
            except Exception:
                G.add_node(node_id, label=str(node))
        else:
            G.add_node(node_id, label=str(node))
        
        # Recursively add children
        if isinstance(node, OperatorNode):
            left_id, edge_id = self._build_graph(G, node.left, variables, edge_id)
            right_id, edge_id = self._build_graph(G, node.right, variables, edge_id)
            
            G.add_edge(left_id, node_id, label=str(edge_id))
            edge_id += 1
            G.add_edge(right_id, node_id, label=str(edge_id))
            edge_id += 1
            
        elif isinstance(node, FunctionNode):
            arg_id, edge_id = self._build_graph(G, node.argument, variables, edge_id)
            G.add_edge(arg_id, node_id, label=str(edge_id))
            edge_id += 1
        
        return node_id, edge_id


class ExpressionParser:
    """Parser for mathematical expressions."""
    
    def __init__(self):
        """Initialize the parser."""
        self.tokens = []
        self.position = 0
    
    def parse(self, expression: str) -> ExpressionTree:
        """Parse a mathematical expression into an expression tree.
        
        Args:
            expression: The expression string to parse
            
        Returns:
            An ExpressionTree representing the parsed expression
            
        Raises:
            ValueError: If the expression is invalid
        """
        logger.info(f"Parsing expression: {expression}")
        self.tokens = self._tokenize(expression)
        self.position = 0
        
        if not self.tokens:
            raise ValueError("Empty expression")
        
        root = self._parse_expression()
        
        if self.position < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.position]}")
        
        return ExpressionTree(root)
    
    def _tokenize(self, expression: str) -> List[str]:
        """Tokenize an expression string.
        
        Args:
            expression: The expression string to tokenize
            
        Returns:
            A list of tokens
        """
        # Replace ** with ^ for easier parsing
        expression = expression.replace("**", "^")
        
        # Regular expression for tokenizing
        token_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*|[0-9]+(?:\.[0-9]+)?|\+|\-|\*|\/|\^|\(|\))'
        tokens = re.findall(token_pattern, expression)
        
        logger.debug(f"Tokenized expression: {tokens}")
        return tokens
    
    def _parse_expression(self) -> ExpressionNode:
        """Parse an expression.
        
        Returns:
            An ExpressionNode representing the parsed expression
        """
        return self._parse_addition()
    
    def _parse_addition(self) -> ExpressionNode:
        """Parse addition and subtraction.
        
        Returns:
            An ExpressionNode representing the parsed expression
        """
        left = self._parse_multiplication()
        
        while self.position < len(self.tokens) and self.tokens[self.position] in ('+', '-'):
            operator = self.tokens[self.position]
            self.position += 1
            right = self._parse_multiplication()
            left = OperatorNode(operator, left, right)
        
        return left
    
    def _parse_multiplication(self) -> ExpressionNode:
        """Parse multiplication and division.
        
        Returns:
            An ExpressionNode representing the parsed expression
        """
        left = self._parse_power()
        
        while self.position < len(self.tokens) and self.tokens[self.position] in ('*', '/'):
            operator = self.tokens[self.position]
            self.position += 1
            right = self._parse_power()
            left = OperatorNode(operator, left, right)
        
        return left
    
    def _parse_power(self) -> ExpressionNode:
        """Parse exponentiation.
        
        Returns:
            An ExpressionNode representing the parsed expression
        """
        left = self._parse_factor()
        
        if self.position < len(self.tokens) and self.tokens[self.position] == '^':
            self.position += 1
            right = self._parse_power()  # Right-associative
            left = OperatorNode('**', left, right)
        
        return left
    
    def _parse_factor(self) -> ExpressionNode:
        """Parse a factor (number, variable, function call, or parenthesized expression).
        
        Returns:
            An ExpressionNode representing the parsed factor
            
        Raises:
            ValueError: If the factor is invalid
        """
        if self.position >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        
        token = self.tokens[self.position]
        self.position += 1
        
        # Number
        if re.match(r'^[0-9]+(?:\.[0-9]+)?$', token):
            return ConstantNode(float(token))
        
        # Variable or function
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
            # Check if it's a function call
            if self.position < len(self.tokens) and self.tokens[self.position] == '(':
                self.position += 1  # Skip '('
                argument = self._parse_expression()
                
                if self.position >= len(self.tokens) or self.tokens[self.position] != ')':
                    raise ValueError(f"Expected ')' after function argument")
                
                self.position += 1  # Skip ')'
                return FunctionNode(token, argument)
            else:
                # It's a variable
                return VariableNode(token)
        
        # Parenthesized expression
        elif token == '(':
            expression = self._parse_expression()
            
            if self.position >= len(self.tokens) or self.tokens[self.position] != ')':
                raise ValueError("Expected ')'")
            
            self.position += 1  # Skip ')'
            return expression
        
        # Unary minus
        elif token == '-':
            factor = self._parse_factor()
            # Represent -x as 0 - x
            return OperatorNode('-', ConstantNode(0), factor)
        
        # Unary plus (just return the factor)
        elif token == '+':
            return self._parse_factor()
        
        else:
            raise ValueError(f"Unexpected token: {token}")


def parse_expression(expression: str) -> ExpressionTree:
    """Parse a mathematical expression into an expression tree.
    
    Args:
        expression: The expression string to parse
        
    Returns:
        An ExpressionTree representing the parsed expression
    """
    parser = ExpressionParser()
    return parser.parse(expression)
