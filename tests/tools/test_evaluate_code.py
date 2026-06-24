# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""代码模型评估工具的单元测试。"""

import numpy as np
import pytest

from sr_agent.tools.base_tool import BaseTool
from sr_agent.tools.evaluate_code import EvaluateCodeTool


class TestEvaluateCodeTool:
    @staticmethod
    def make_tool(X, y):
        return EvaluateCodeTool(data=X | {"y": y}, target="y")

    def test_explicit_assignments_perfect_fit(self):
        X = {"x1": np.array([1.0, 2.0, 3.0])}
        y = np.array([3.0, 5.0, 7.0])

        result = self.make_tool(X, y).execute(
            model_code="def build_model(data):\n    return {'a': 2.0, 'b': 1.0}",
            predict_code=(
                "def predict(data, model):\n"
                "    return model['a'] * data['x1'] + model['b']"
            ),
            format_code=(
                "def format_model(model):\n"
                "    return f\"{model['a']} * x1 + {model['b']}\""
            ),
        )

        assert result["formula"] == "2.0 * x1 + 1.0"
        assert result["metrics"]["mse"] == 0.0
        assert result["metrics"]["r2"] == 1.0
        assert result["is_candidate"] is True

    def test_imports_are_allowed(self):
        X = {"x1": np.array([1.0, 2.0, 3.0])}
        y = np.array([1.0, 4.0, 9.0])

        result = self.make_tool(X, y).execute(
            model_code="def build_model(data):\n    return {'power': 2}",
            predict_code=(
                "import numpy as np\n\n"
                "def predict(data, model):\n"
                "    return np.power(data['x1'], model['power'])"
            ),
            format_code="def format_model(model):\n    return 'x1**2'",
        )

        assert result["formula"] == "x1**2"
        assert result["metrics"]["mse"] == 0.0

    def test_default_model_string(self):
        X = {"x1": np.array([1.0, 2.0])}
        y = np.array([2.0, 4.0])

        result = self.make_tool(X, y).execute(
            model_code="def build_model(data):\n    return ('scale', 2.0)",
            predict_code="def predict(data, model):\n    return model[1] * data['x1']",
        )

        assert result["formula"] == "('scale', 2.0)"
        assert result["metrics"]["mse"] == 0.0

    def test_target_leakage_is_not_candidate(self):
        X = {"x1": np.array([1.0, 2.0])}
        y = np.array([2.0, 4.0])

        result = self.make_tool(X, y).execute(
            model_code="def build_model(data):\n    return None",
            predict_code="def predict(data, model):\n    return data['y']",
        )

        assert result["metrics"]["mse"] == 0.0
        assert result["is_candidate"] is False

    def test_non_default_target_is_not_candidate(self):
        data = {
            "x1": np.array([1.0, 2.0]),
            "z": np.array([3.0, 5.0]),
            "y": np.array([2.0, 4.0]),
        }
        tool = EvaluateCodeTool(data=data, target="y")

        result = tool.execute(
            model_code="def build_model(data):\n    return {'a': 2.0, 'b': 1.0}",
            predict_code=(
                "def predict(data, model):\n"
                "    return model['a'] * data['x1'] + model['b']"
            ),
            y="z",
        )

        assert result["metrics"]["mse"] == 0.0
        assert result["is_candidate"] is False

    def test_numpy_is_available(self):
        X = {"x1": np.array([0.0, np.pi / 2])}
        y = np.array([0.0, 1.0])

        result = self.make_tool(X, y).execute(
            model_code="def build_model(data):\n    return np.sin",
            predict_code="def predict(data, model):\n    return model(data['x1'])",
            format_code="def format_model(model):\n    return 'sin(x1)'",
        )

        assert result["metrics"]["mse"] < 1e-12

    def test_invalid_code_returns_tool_error(self):
        tool = self.make_tool({"x1": np.array([1.0])}, np.array([1.0]))
        result = tool(
            "import os\n\ndef build_model(data):\n    return 1",
            "def predict(data, model):\n    return data['x1']",
        )

        assert result.ok is False
        assert "Unauthorized module: os" in result.result_str

    def test_missing_required_output_raises(self):
        tool = self.make_tool({"x1": np.array([1.0])}, np.array([1.0]))

        with pytest.raises(Exception, match="signature must be"):
            tool.execute(
                "def build_model(x):\n    return x",
                "def predict(data, model):\n    return data['x1']",
            )

    def test_extra_top_level_statement_is_rejected(self):
        tool = self.make_tool({"x1": np.array([1.0])}, np.array([1.0]))

        with pytest.raises(Exception, match="top-level imports and one function"):
            tool.execute(
                "constant = 1\n\ndef build_model(data):\n    return constant",
                "def predict(data, model):\n    return data['x1']",
            )

    def test_registered_as_base_tool(self):
        assert BaseTool.create("evaluate_code", create_instance=False) is EvaluateCodeTool
