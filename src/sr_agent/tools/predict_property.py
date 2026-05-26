# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""数据性质预测工具。

使用预训练的 Transformer 模型从数据样本中预测公式的数学性质（周期性、乘法可分性），
为符号回归搜索提供先验知识指导。
"""
from __future__ import annotations

import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from .base_tool import BaseTool, ToolMetadata

_logger = logging.getLogger(f"sr_agent.{__name__}")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_DEFAULT_CHECKPOINTS = {
    "scratch": _PROJECT_ROOT / "logs" / "nn_tools" / "train_property" / "scratch" / "best.pth",
    "finetune": _PROJECT_ROOT / "logs" / "nn_tools" / "train_property" / "finetune" / "best.pth",
}

_PERIOD_LABELS = {0: "non-periodic", 1: "periodic"}
_SEP_LABELS = {0: "not multiplicatively separable", 1: "multiplicatively separable"}

_cached_model = {}


def _load_model(checkpoint_path: str, device: str = "cpu"):
    """Load model from checkpoint with caching to avoid repeated I/O."""
    cache_key = (checkpoint_path, device)
    if cache_key in _cached_model:
        return _cached_model[cache_key]

    import sys
    exp_path = str(_PROJECT_ROOT / "src" / "experimental")
    if exp_path not in sys.path:
        sys.path.insert(0, exp_path)

    import argparse
    from nn_tools.models import FloatEmbedder, DataEmbedder, PropertyPredictionModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = ckpt["args"]
    if isinstance(saved_args, dict):
        saved_args = argparse.Namespace(**saved_args)

    float_emb = FloatEmbedder(d_model=saved_args.d_model).to(device)
    data_emb = DataEmbedder(
        d_model=saved_args.d_model,
        pooling=saved_args.data_pooling,
        float_embedder=float_emb,
    ).to(device)
    model = PropertyPredictionModel(saved_args).to(device)

    model.load_state_dict(ckpt["model"])
    float_emb.load_state_dict(ckpt["float_embedder"])
    data_emb.load_state_dict(ckpt["data_embedder"])

    model.eval()
    float_emb.eval()
    data_emb.eval()

    result = (model, float_emb, data_emb, saved_args)
    _cached_model[cache_key] = result
    _logger.info(f"Loaded property prediction model from {checkpoint_path}")
    return result


@BaseTool.register("predict_property")
class PropertyPredictorTool(BaseTool):
    metadata = ToolMetadata(name="predict_property")

    CHECKPOINT_PATH: str = None
    MODEL_TYPE: str = "scratch"
    DEVICE: str = "cpu"

    def execute(self) -> Dict[str, Any]:
        """Predict mathematical properties (periodicity, multiplicative separability) of the data using a neural network.
        This tool analyzes the relationship between input variables and the target to detect:
        - **Periodicity**: Whether y is periodic with respect to each individual input variable.
        - **Multiplicative Separability**: Whether y can be expressed as a product of functions of individual variables, i.e., y = f(x1) * g(x2) * ...
        IMPORTANT: This tool can only detect periodicity in individual variables. Even if no single variable
        is periodic, variable COMBINATIONS (e.g., x1-x2, omega*t, n*theta) may still be periodic.
        Always consider trigonometric functions of variable combinations (sin(x1-x2), cos(omega*t), etc.)
        as candidates regardless of per-variable periodicity results.
        No arguments needed — the tool automatically uses the data provided to the agent.
        """
        data = self.context["data"]
        target_name = self.context["target"]

        ckpt_path = self.CHECKPOINT_PATH
        if ckpt_path is None:
            ckpt_path = str(_DEFAULT_CHECKPOINTS.get(self.MODEL_TYPE, _DEFAULT_CHECKPOINTS["scratch"]))

        if not Path(ckpt_path).exists():
            return {
                "error": f"Model checkpoint not found: {ckpt_path}",
                "periodicity": {},
                "multiplicative_separable": "unknown",
            }

        model, float_emb, data_emb, saved_args = _load_model(ckpt_path, self.DEVICE)
        max_var_num = saved_args.max_var_num

        input_vars = [v for v in data if v != target_name]
        if len(input_vars) > max_var_num:
            input_vars = input_vars[:max_var_num]
            _logger.warning(f"Too many variables ({len(input_vars)}), truncated to {max_var_num}")

        n_vars = len(input_vars)
        y_arr = np.asarray(data[target_name], dtype=np.float32).flatten()
        n_samples = len(y_arr)

        sample_num = min(n_samples, 200)
        rng = np.random.default_rng(42)
        idx = rng.choice(n_samples, sample_num, replace=False) if n_samples > sample_num else np.arange(n_samples)

        data_arr = np.zeros((sample_num, max_var_num + 1), dtype=np.float32)
        for i, var in enumerate(input_vars):
            data_arr[:, i] = np.asarray(data[var], dtype=np.float32).flatten()[idx]
        data_arr[:, -1] = y_arr[idx]

        data_tensor = torch.from_numpy(data_arr).unsqueeze(0)

        with torch.no_grad():
            B, S = data_tensor.shape[:2]
            val_emb = float_emb(data_tensor.to(self.DEVICE)).flatten(0, 1)
            d_emb = data_emb.pool(val_emb).reshape(B, S, -1)
            out = model(d_emb)

        period_logits = out["periodicity"][0].cpu().numpy()
        sep_logits = out["multiplicative_separable"][0].cpu().numpy()

        period_preds = period_logits.argmax(axis=-1)
        sep_pred = int(sep_logits.argmax())

        period_probs = _softmax(period_logits)
        sep_probs = _softmax(sep_logits)

        periodicity = {}
        for i, var in enumerate(input_vars):
            pred = int(period_preds[i])
            conf = float(period_probs[i, pred])
            periodicity[var] = {
                "prediction": _PERIOD_LABELS[pred],
                "confidence": round(conf, 3),
            }

        sep_conf = float(sep_probs[sep_pred])
        separability = {
            "prediction": _SEP_LABELS[sep_pred],
            "confidence": round(sep_conf, 3),
        }

        return {
            "periodicity": periodicity,
            "multiplicative_separable": separability,
            "n_variables_analyzed": n_vars,
            "n_samples_used": sample_num,
        }

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        if "error" in result:
            return f"Error: {result['error']}"

        lines = [
            f"Property Prediction Results (analyzed {result['n_variables_analyzed']} variables, "
            f"{result['n_samples_used']} samples):",
            "",
            "Periodicity (whether y is periodic w.r.t. each variable):",
        ]
        for var, info in result["periodicity"].items():
            lines.append(f"  {var}: {info['prediction']} (confidence: {info['confidence']:.1%})")

        sep = result["multiplicative_separable"]
        lines.append("")
        lines.append(f"Multiplicative Separability: {sep['prediction']} (confidence: {sep['confidence']:.1%})")

        if any(info["prediction"] == "periodic" for info in result["periodicity"].values()):
            lines.append("")
            lines.append("Hint: Periodic variables suggest the formula likely contains trigonometric functions "
                         "(sin, cos) of those variables.")

        lines.append("")
        lines.append("IMPORTANT: Per-variable periodicity analysis has a key limitation — it cannot detect "
                     "periodicity in variable COMBINATIONS. For example, sin(x1-x2) is periodic in (x1-x2) "
                     "but neither x1 nor x2 alone appears periodic; cos(omega*t) is periodic in (omega*t) "
                     "but neither omega nor t alone appears periodic. "
                     "You should still explore trigonometric functions of variable combinations "
                     "(e.g., sin(xi-xj), cos(xi*xj)) even when individual variables show non-periodic.")

        if result["multiplicative_separable"]["prediction"] == "multiplicatively separable":
            lines.append("")
            lines.append("Hint: The formula may be expressible as a product of functions of individual variables, "
                         "e.g., y = f(x1) * g(x2). Try decomposing the problem.")

        return "\n".join(lines)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
