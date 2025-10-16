import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date


class CalculatorTool:
    """本地计算器工具类"""

    @staticmethod
    def safe_eval(expression: str) -> Any:
        """
        安全地评估数学表达式

        Args:
            expression: 数学表达式

        Returns:
            计算结果
        """
        import math
        import ast

        # 只允许数学表达式和函数
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow, 'len': len,
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'log': math.log, 'log10': math.log10, 'exp': math.exp,
            'pi': math.pi, 'e': math.e
        }

        try:
            # 解析表达式为AST
            tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"表达式语法错误: {str(e)}")

        # 检查允许的节点类型
        def check_node(node):
            """递归检查AST节点是否安全"""
            # 允许的基础节点类型
            if isinstance(node, (ast.Expression, ast.Module)):
                # 根节点，检查子节点
                for child in ast.iter_child_nodes(node):
                    if not check_node(child):
                        return False
                return True

            elif isinstance(node, ast.Constant):
                # 常量值（数字、字符串等）
                return True

            elif isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare)):
                # 数学运算和比较
                for child in ast.iter_child_nodes(node):
                    if not check_node(child):
                        return False
                return True

            elif isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                                   ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not,
                                   ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                # 操作符节点
                return True

            elif isinstance(node, ast.Name):
                # 变量名，必须在允许列表中
                return node.id in allowed_names

            elif isinstance(node, ast.Call):
                # 函数调用
                if isinstance(node.func, ast.Name):
                    # 直接函数名调用
                    if node.func.id in allowed_names:
                        # 检查参数
                        for arg in node.args:
                            if not check_node(arg):
                                return False
                        return True
                    else:
                        raise ValueError(f"不允许的函数调用: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    # 属性调用，如 math.sqrt
                    if (isinstance(node.func.value, ast.Name) and
                            node.func.value.id == 'math' and
                            node.func.attr in allowed_names):
                        # 检查参数
                        for arg in node.args:
                            if not check_node(arg):
                                return False
                        return True
                    else:
                        raise ValueError(f"不允许的属性调用: {node.func.attr}")
                else:
                    raise ValueError("不支持的函数调用格式")

            elif isinstance(node, ast.Attribute):
                # 属性访问，如 math.pi
                if (isinstance(node.value, ast.Name) and
                        node.value.id == 'math' and
                        node.attr in ['pi', 'e']):
                    return True
                else:
                    raise ValueError(f"不允许的属性访问: {node.attr}")

            elif isinstance(node, (ast.List, ast.Tuple)):
                # 允许列表和元组（用于min/max/sum等函数）
                for element in node.elts:
                    if not check_node(element):
                        return False
                return True

            else:
                # 其他类型的节点都不允许
                raise ValueError(f"不支持的语法节点: {type(node).__name__}")

        try:
            # 检查整个表达式树
            if not check_node(tree):
                raise ValueError("表达式包含不安全的内容")

            # 评估表达式
            result = eval(compile(tree, "<string>", "eval"), {"__builtins__": {}}, allowed_names)
            return result

        except Exception as e:
            raise ValueError(f"表达式安全检查失败: {str(e)}")

    @classmethod
    def calculate(cls, expression: str) -> Dict[str, Any]:
        """
        执行数学计算

        Args:
            expression: 数学表达式

        Returns:
            计算结果字典
        """
        try:
            # 预处理表达式
            expr = expression.strip()
            if not expr:
                return {"success": False, "result": "表达式不能为空"}

            # 替换常见数学符号
            expr = expr.replace('^', '**')  # 将 ^ 替换为 **
            expr = expr.replace('×', '*').replace('÷', '/')  # 替换乘除符号

            # 执行计算
            result = cls.safe_eval(expr)

            return {
                "success": True,
                "result": str(result),
                "expression": expression,
                "type": type(result).__name__
            }

        except SyntaxError as e:
            return {"success": False, "result": f"表达式语法错误: {str(e)}"}
        except NameError as e:
            return {"success": False, "result": f"未知的函数或变量: {str(e)}"}
        except ZeroDivisionError:
            return {"success": False, "result": "数学错误: 除零错误"}
        except ValueError as e:
            return {"success": False, "result": f"数学错误: {str(e)}"}
        except Exception as e:
            return {"success": False, "result": f"计算错误: {str(e)}"}
