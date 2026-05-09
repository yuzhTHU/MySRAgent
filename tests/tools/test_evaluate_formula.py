# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""公式评估工具的单元测试。"""

import numpy as np
import pytest

from sr_agent.tools.base_tool import BaseTool
from sr_agent.tools.evaluate_formula import EvaluateTool, SubmitFormulaTool


class TestEvaluateTool:
    """测试 EvaluateTool 的正确性。"""

    @staticmethod
    def make_tool(X, y):
        return EvaluateTool(data=X | {"y": y}, target="y")

    def test_perfect_fit(self):
        """测试完美拟合情况：y = 2*x1 + 3。"""
        X = {"x1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        y = np.array([5.0, 7.0, 9.0, 11.0, 13.0])

        tool = self.make_tool(X, y)
        result = tool.execute("x1 * 2 + 3")

        assert result["metrics"]["mse"] == 0.0
        assert result["metrics"]["r2"] == 1.0

    def test_quadratic_formula(self):
        """测试二次公式：y = x1**2。"""
        X = {"x1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])

        tool = self.make_tool(X, y)
        result = tool.execute("x1**2")

        assert result["metrics"]["mse"] == 0.0
        assert result["metrics"]["r2"] == 1.0

    def test_trigonometric_formula(self):
        """测试三角函数公式：y = sin(x1)。"""
        X = {"x1": np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])}
        y = np.array([0.0, 1.0, 0.0, -1.0])

        tool = self.make_tool(X, y)
        result = tool.execute("sin(x1)")

        assert result["metrics"]["mse"] < 1e-10

    def test_poor_fit(self):
        """测试拟合不佳的情况。"""
        X = {"x1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        y = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        tool = self.make_tool(X, y)
        result = tool.execute("x1")  # 错误的公式

        assert result["metrics"]["mse"] > 10000  # MSE 应该很大
        assert result["metrics"]["r2"] < 0  # R² 应该为负（比预测均值还差）

    def test_multiple_features(self):
        """测试多特征公式：y = x1 + x2。"""
        X = {
            "x1": np.array([1.0, 2.0, 3.0, 4.0]),
            "x2": np.array([0.5, 1.0, 1.5, 2.0]),
        }
        y = np.array([1.5, 3.0, 4.5, 6.0])

        tool = self.make_tool(X, y)
        result = tool.execute("x1 + x2")

        assert result["metrics"]["mse"] == 0.0

    def test_invalid_formula(self):
        """测试无效公式。"""
        X = {"x1": np.array([1.0, 2.0, 3.0])}
        y = np.array([1.0, 2.0, 3.0])

        tool = self.make_tool(X, y)
        result = tool("invalid_syntax!!")

        assert result.ok is False
        assert "SyntaxError" in result.result_str

    def test_with_parameter_fitting(self):
        """测试参数拟合功能。"""
        X = {"x1": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        y = np.array([2.1, 3.9, 6.2, 7.9, 10.1])  # y ≈ 2*x，带有噪声

        tool = self.make_tool(X, y)
        # 可拟合参数是 Number 节点（如 1.0），不是符号变量（如 a）
        result = tool.execute("1.0 * x1", fit=True)

        # 拟合后应该能得到较好的结果
        assert result["metrics"]["r2"] > 0.95

    def test_output_structure(self):
        """测试输出结构完整性。"""
        X = {"x1": np.array([1.0, 2.0, 3.0])}
        y = np.array([1.0, 2.0, 3.0])

        tool = self.make_tool(X, y)
        result = tool.execute("x1")

        # 检查必需的键
        assert "formula" in result
        assert "metrics" in result
        assert result["metrics"] is not None
        assert "mse" in result["metrics"]
        assert "rmse" in result["metrics"]
        assert "mae" in result["metrics"]
        assert "r2" in result["metrics"]

    def test_metadata_exists(self):
        """测试元数据存在。"""
        tool = EvaluateTool(data={"x": np.array([1.0]), "y": np.array([1.0])}, target="y")
        assert tool.metadata is not None
        assert tool.metadata.name == "evaluate_formula"

    def test_x_vars_subset(self):
        """测试 x_vars 参数可以选择子集。"""
        X = {
            "x1": np.array([1.0, 2.0, 3.0]),
            "x2": np.array([4.0, 5.0, 6.0]),
            "x3": np.array([7.0, 8.0, 9.0]),
        }
        # y = x1 * 2 + x3 * 1
        y = np.array([9.0, 12.0, 15.0])

        tool = self.make_tool(X, y)
        result = tool.execute("x1 * 2 + x3")

        assert result["metrics"]["r2"] > 0.9


class TestSubmitFormulaTool:
    """测试 SubmitFormulaTool 与 EvaluateTool 的接口一致性。"""

    def test_metadata_emphasizes_submit(self):
        tool = SubmitFormulaTool(data={"x": np.array([1.0]), "y": np.array([1.0])}, target="y")

        assert tool.metadata.name == "submit_formula"
        assert "submit" in tool.metadata.description.lower()
        assert tool.metadata.parameters == EvaluateTool.metadata.parameters

    def test_execute_matches_evaluate_formula(self):
        X = {"x1": np.array([1.0, 2.0, 3.0])}
        y = np.array([3.0, 5.0, 7.0])

        submit_result = SubmitFormulaTool(data=X | {"y": y}, target="y").execute("2 * x1 + 1")
        evaluate_result = EvaluateTool(data=X | {"y": y}, target="y").execute("2 * x1 + 1")

        assert submit_result == evaluate_result

    def test_registered_as_base_tool(self):
        assert BaseTool.create("submit_formula", create_instance=False) is SubmitFormulaTool
