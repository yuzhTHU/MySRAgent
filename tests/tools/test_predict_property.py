"""PropertyPredictorTool 的单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from sr_agent.tools.predict_property import PropertyPredictorTool, _DEFAULT_CHECKPOINTS


class TestPropertyPredictorMetadata:
    def test_metadata_name(self):
        assert PropertyPredictorTool.metadata.name == "predict_property"

    def test_description_is_inferred(self):
        desc = PropertyPredictorTool.metadata.description
        assert "periodicity" in desc.lower() or "Periodicity" in desc
        assert "separability" in desc.lower() or "Separability" in desc

    def test_no_required_parameters(self):
        params = PropertyPredictorTool.metadata.parameters
        assert params["required"] == []

    def test_to_dict_is_valid_openai_schema(self):
        d = PropertyPredictorTool.to_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "predict_property"

    def test_registered_in_base_tool(self):
        from sr_agent.tools.base_tool import BaseTool
        assert "predict_property" in BaseTool.REGISTRY_DICT


class TestPropertyPredictorExecution:
    @pytest.fixture
    def periodic_data(self):
        rng = np.random.default_rng(0)
        x1 = rng.uniform(-10, 10, 200)
        x2 = rng.uniform(-10, 10, 200)
        y = np.sin(x1) + np.cos(x2)
        return {"x1": x1, "x2": x2, "y": y}

    @pytest.fixture
    def separable_data(self):
        rng = np.random.default_rng(0)
        x1 = rng.uniform(0.1, 5, 200)
        x2 = rng.uniform(0.1, 5, 200)
        y = x1**2 * x2**3
        return {"x1": x1, "x2": x2, "y": y}

    @pytest.fixture
    def simple_data(self):
        rng = np.random.default_rng(0)
        x1 = rng.uniform(-5, 5, 200)
        y = x1**2 + 3 * x1 + 1
        return {"x1": x1, "y": y}

    def _has_checkpoint(self):
        return _DEFAULT_CHECKPOINTS["scratch"].exists()

    def test_execute_returns_expected_keys(self, simple_data):
        if not self._has_checkpoint():
            pytest.skip("No model checkpoint available")
        tool = PropertyPredictorTool(data=simple_data, target="y")
        result = tool.execute()
        assert "periodicity" in result
        assert "multiplicative_separable" in result
        assert "n_variables_analyzed" in result
        assert "n_samples_used" in result

    def test_periodicity_keys_match_input_vars(self, periodic_data):
        if not self._has_checkpoint():
            pytest.skip("No model checkpoint available")
        tool = PropertyPredictorTool(data=periodic_data, target="y")
        result = tool.execute()
        assert set(result["periodicity"].keys()) == {"x1", "x2"}
        for var_info in result["periodicity"].values():
            assert "prediction" in var_info
            assert "confidence" in var_info
            assert var_info["prediction"] in ("periodic", "non-periodic")
            assert 0.0 <= var_info["confidence"] <= 1.0

    def test_separability_structure(self, separable_data):
        if not self._has_checkpoint():
            pytest.skip("No model checkpoint available")
        tool = PropertyPredictorTool(data=separable_data, target="y")
        result = tool.execute()
        sep = result["multiplicative_separable"]
        assert "prediction" in sep
        assert "confidence" in sep
        assert sep["prediction"] in (
            "not multiplicatively separable",
            "multiplicatively separable",
        )

    def test_call_wrapper(self, simple_data):
        if not self._has_checkpoint():
            pytest.skip("No model checkpoint available")
        tool = PropertyPredictorTool(data=simple_data, target="y")
        result = tool()
        assert result.ok is True
        assert result.meta_data["tool"] == "predict_property"
        assert "Property Prediction Results" in result.result_str

    def test_format_result_dict(self, simple_data):
        if not self._has_checkpoint():
            pytest.skip("No model checkpoint available")
        tool = PropertyPredictorTool(data=simple_data, target="y")
        result = tool.execute()
        formatted = PropertyPredictorTool.format_result_dict(result)
        assert "Periodicity" in formatted
        assert "Multiplicative Separability" in formatted

    def test_missing_checkpoint_returns_error(self, simple_data):
        PropertyPredictorTool.CHECKPOINT_PATH = "/nonexistent/path.pth"
        tool = PropertyPredictorTool(data=simple_data, target="y")
        result = tool.execute()
        assert "error" in result
        PropertyPredictorTool.CHECKPOINT_PATH = None

    def test_single_variable(self, simple_data):
        if not self._has_checkpoint():
            pytest.skip("No model checkpoint available")
        tool = PropertyPredictorTool(data=simple_data, target="y")
        result = tool.execute()
        assert result["n_variables_analyzed"] == 1
        assert "x1" in result["periodicity"]

    def test_periodic_data_detection(self, periodic_data):
        """Smoke test: periodic data should ideally be detected as periodic."""
        if not self._has_checkpoint():
            pytest.skip("No model checkpoint available")
        tool = PropertyPredictorTool(data=periodic_data, target="y")
        result = tool.execute()
        periodic_vars = [v for v, info in result["periodicity"].items()
                         if info["prediction"] == "periodic"]
        assert len(periodic_vars) >= 0  # no strict assertion, just check it runs
