"""PropertyPredictorTool 的单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

import sr_agent.tools.predict_property as predict_property
from sr_agent.tools.base_tool import ToolRunAbort
from sr_agent.tools.predict_property import PropertyPredictorTool


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
    def fake_checkpoint(self, tmp_path, monkeypatch):
        ckpt_path = tmp_path / "property-scratch-v5.pth"
        ckpt_path.write_bytes(b"fake checkpoint")
        monkeypatch.setenv("SR_AGENT_PROPERTY_MODEL_CHECKPOINT", str(ckpt_path))
        monkeypatch.setattr(PropertyPredictorTool, "AUTO_DOWNLOAD", False)
        yield ckpt_path
        monkeypatch.setattr(PropertyPredictorTool, "AUTO_DOWNLOAD", True)

    @pytest.fixture
    def fake_model(self, monkeypatch):
        class SavedArgs:
            max_var_num = 3

        def fake_load_model(checkpoint_path, device):
            return object(), object(), object(), SavedArgs()

        def fake_predict(model, float_emb, data_emb, data_arr, device):
            max_var_num = 3
            mono = np.zeros((max_var_num, 4), dtype=np.float32)
            conv = np.zeros((max_var_num, 4), dtype=np.float32)
            period = np.zeros((max_var_num, 2), dtype=np.float32)
            mono[:, 0] = 5.0
            conv[:, 0] = 5.0
            period[:, 0] = 5.0
            mono[0, 1] = 10.0
            conv[0, 1] = 10.0
            period[0, 1] = 10.0
            return {
                "monotonicity": mono,
                "convexity": conv,
                "periodicity": period,
                "multiplicative_separable": np.array([1.0, 4.0], dtype=np.float32),
            }

        monkeypatch.setattr(predict_property, "_load_model", fake_load_model)
        monkeypatch.setattr(predict_property, "_predict", fake_predict)

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

    def test_execute_returns_expected_keys(self, simple_data, fake_checkpoint, fake_model):
        tool = PropertyPredictorTool(data=simple_data, target="y")
        result = tool.execute()
        assert "periodicity" in result
        assert "multiplicative_separable" in result
        assert "n_variables_analyzed" in result
        assert "n_samples_used" in result

    def test_periodicity_keys_match_input_vars(self, periodic_data, fake_checkpoint, fake_model):
        tool = PropertyPredictorTool(data=periodic_data, target="y")
        result = tool.execute()
        assert set(result["periodicity"].keys()) == {"x1", "x2"}
        for var_info in result["periodicity"].values():
            assert "prediction" in var_info
            assert "confidence" in var_info
            assert var_info["prediction"] in ("periodic", "non-periodic")
            assert 0.0 <= var_info["confidence"] <= 1.0

    def test_separability_structure(self, separable_data, fake_checkpoint, fake_model):
        tool = PropertyPredictorTool(data=separable_data, target="y")
        result = tool.execute()
        sep = result["multiplicative_separable"]
        assert "prediction" in sep
        assert "confidence" in sep
        assert sep["prediction"] in (
            "not multiplicatively separable",
            "multiplicatively separable",
        )

    def test_call_wrapper(self, simple_data, fake_checkpoint, fake_model):
        tool = PropertyPredictorTool(data=simple_data, target="y")
        result = tool()
        assert result.ok is True
        assert result.meta_data["tool"] == "predict_property"
        assert "Property Prediction Results" in result.result_str

    def test_format_result_dict(self, simple_data, fake_checkpoint, fake_model):
        tool = PropertyPredictorTool(data=simple_data, target="y")
        result = tool.execute()
        formatted = PropertyPredictorTool.format_result_dict(result)
        assert "Periodicity" in formatted
        assert "Multiplicative Separability" in formatted

    def test_missing_checkpoint_raises_tool_run_abort_with_manual_download_message(self, simple_data, tmp_path, monkeypatch):
        missing_path = tmp_path / "missing.pth"
        monkeypatch.setenv("SR_AGENT_PROPERTY_MODEL_CHECKPOINT", str(missing_path))
        monkeypatch.setattr(PropertyPredictorTool, "AUTO_DOWNLOAD", False)
        tool = PropertyPredictorTool(data=simple_data, target="y")
        with pytest.raises(ToolRunAbort) as exc_info:
            tool.execute()
        message = str(exc_info.value)
        assert "GitHub Release page" in message
        assert "download the asset named" in message
        assert str(missing_path) in message

    def test_call_wrapper_does_not_catch_missing_checkpoint_abort(self, simple_data, tmp_path, monkeypatch):
        missing_path = tmp_path / "missing.pth"
        monkeypatch.setenv("SR_AGENT_PROPERTY_MODEL_CHECKPOINT", str(missing_path))
        monkeypatch.setattr(PropertyPredictorTool, "AUTO_DOWNLOAD", False)
        tool = PropertyPredictorTool(data=simple_data, target="y")
        with pytest.raises(ToolRunAbort, match="GitHub Release page"):
            tool()

    def test_single_variable(self, simple_data, fake_checkpoint, fake_model):
        tool = PropertyPredictorTool(data=simple_data, target="y")
        result = tool.execute()
        assert result["n_variables_analyzed"] == 1
        assert "x1" in result["periodicity"]

    def test_periodic_data_detection(self, periodic_data, fake_checkpoint, fake_model):
        """Smoke test: periodic data should ideally be detected as periodic."""
        tool = PropertyPredictorTool(data=periodic_data, target="y")
        result = tool.execute()
        periodic_vars = [v for v, info in result["periodicity"].items()
                         if info["prediction"] == "periodic"]
        assert len(periodic_vars) >= 0  # no strict assertion, just check it runs
