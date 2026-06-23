# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""数据性质预测工具。

使用预训练的 Transformer 模型从数据样本中预测公式的数学性质
（单调性、凹凸性、周期性、乘法可分性），并自动分析变量组合的性质，
为符号回归搜索提供先验知识指导。

可通过下述指令将模型上传到 Github Release（需要设置 GITHUB_TOKEN 环境变量）
    python -m sr_agent.cli.upload_models --checkpoint path/to/model.pth --name property-scratch-v5 --release-tag sr-agent-models-v5
可通过下述指令从 Github Release 下载模型到本地（或者在首次调用时自动下载）
    python -m sr_agent.cli.download_models --checkpoint path/to/save/model.pth --name property-scratch-v5 --release-tag sr-agent-models-v5
"""
from __future__ import annotations

import logging
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Literal
from ..utils import download_model, get_default, tag2ansi
from .base_tool import BaseTool, ToolMetadata, ToolRunAbort

_logger = logging.getLogger(f"sr_agent.{__name__}")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODEL_CACHE_DIR = _PROJECT_ROOT / ".cache" / "nn_tools"
_MONO_LABELS = {0: "non-monotonic", 1: "increasing", 2: "decreasing", 3: "constant"}
_CONV_LABELS = {0: "non-convex/concave", 1: "convex", 2: "concave", 3: "affine"}
_PERIOD_LABELS = {0: "non-periodic", 1: "periodic"}
_SEP_LABELS = {0: "not multiplicatively separable", 1: "multiplicatively separable"}
_COMBO_OPS = {
    "*": lambda a, b: a * b,
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "/": lambda a, b: np.where(np.abs(b) > 1e-12, a / b, np.nan),
}
_cached_model = {}


def _load_model(checkpoint_path: str, device: str = "cpu"):
    """Load model from checkpoint with caching to avoid repeated I/O."""
    import torch

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


def _predict(model, float_emb, data_emb, data_arr: np.ndarray, device: str) -> dict:
    """Run model inference on a single (sample_num, max_var+1) data array.

    Returns raw logits dict with keys: monotonicity, convexity, periodicity,
    multiplicative_separable.
    """
    import torch
    data_tensor = torch.from_numpy(data_arr).unsqueeze(0)
    with torch.no_grad():
        B, S = data_tensor.shape[:2]
        val_emb = float_emb(data_tensor.to(device)).flatten(0, 1)
        d_emb = data_emb.pool(val_emb).reshape(B, S, -1)
        out = model(d_emb)
    return {k: v[0].cpu().numpy() for k, v in out.items()}


def _decode_per_var(logits: np.ndarray, label_map: dict, n_vars: int) -> list:
    """Decode per-variable logits into list of (pred_label, confidence)."""
    probs = _softmax(logits)
    preds = logits.argmax(axis=-1)
    results = []
    for i in range(n_vars):
        pred = int(preds[i])
        conf = float(probs[i, pred])
        results.append((label_map[pred], round(conf, 3)))
    return results


@BaseTool.register("predict_property")
class PropertyPredictorTool(BaseTool):
    metadata = ToolMetadata(name="predict_property")

    MODEL_TYPE: Literal[
        "property-scratch",
        "property-finetune",
        "property-scratch-v5",
    ] = "property-scratch-v5"
    MODEL_RELEASE_TAG: str = "sr-agent-models-v5"
    DEVICE: str = "cpu"
    AUTO_DOWNLOAD: bool = True

    def execute(self) -> Dict[str, Any]:
        """Predict mathematical properties of the data using a neural network.
        This tool analyzes the relationship between input variables and the target to detect:
        - **Monotonicity**: Whether y is monotonically increasing, decreasing, or constant w.r.t. each variable.
        - **Convexity**: Whether y is convex, concave, or affine w.r.t. each variable.
        - **Periodicity**: Whether y is periodic w.r.t. each individual input variable.
        - **Multiplicative Separability**: Whether y = f(x1) * g(x2) * ...
        Additionally, this tool automatically tests variable COMBINATIONS (xi*xj, xi+xj, xi-xj, xi/xj)
        to detect properties that only emerge in combinations (e.g., sin(x1*x2) is periodic in x1*x2
        but neither x1 nor x2 alone appears periodic).
        No arguments needed — the tool automatically uses the data provided to the agent.
        """
        data = self.context["data"]
        target_name = self.context["target"]
        exceptions = []

        ## 加载模型 checkpoint
        default_ckpt_path = _MODEL_CACHE_DIR / f"{self.MODEL_TYPE}.pth"
        ckpt_path = Path(os.getenv("SR_AGENT_PROPERTY_MODEL_CHECKPOINT") or default_ckpt_path).expanduser()
        if not ckpt_path.exists() and self.AUTO_DOWNLOAD:
            download_model(name=self.MODEL_TYPE, checkpoint=ckpt_path, release_tag=self.MODEL_RELEASE_TAG)
        if not ckpt_path.exists():
            repo = get_default("repo")
            release_url = f"https://github.com/{repo}/releases/tag/{self.MODEL_RELEASE_TAG}"
            raise ToolRunAbort(tag2ansi(
                f"Model checkpoint not found: {ckpt_path}.\n"
                f"Automatic download was not available. To use predict_property, please visit the GitHub Release page\n"
                f"  [blue bold]{release_url}[reset]\n"
                f"and download the asset named\n"
                f"  [blue bold]{self.MODEL_TYPE!r}[reset]\n"
                f"Then save it as\n"
                f"  [blue bold]{ckpt_path}[reset]\n"
                "Alternatively, you can ban this predict_property tool to avoid this error."
            ))

        model, float_emb, data_emb, saved_args = _load_model(ckpt_path, self.DEVICE)
        max_var_num = saved_args.max_var_num

        input_vars = [v for v in data if v != target_name]
        if len(input_vars) > max_var_num:
            input_vars = input_vars[:max_var_num]
            exceptions.append(f"Too many variables ({len(data) - 1}), only the first {max_var_num} are analyzed: {input_vars}")

        n_vars = len(input_vars)
        y_arr = np.asarray(data[target_name], dtype=np.float32).flatten()
        n_samples = len(y_arr)

        sample_num = min(n_samples, 200)
        rng = np.random.default_rng(42)
        idx = rng.choice(n_samples, sample_num, replace=False) if n_samples > sample_num else np.arange(n_samples)

        var_arrays = {}
        for var in input_vars:
            var_arrays[var] = np.asarray(data[var], dtype=np.float32).flatten()[idx]

        # --- Per-variable prediction ---
        data_arr = np.zeros((sample_num, max_var_num + 1), dtype=np.float32)
        for i, var in enumerate(input_vars):
            data_arr[:, i] = var_arrays[var]
        data_arr[:, -1] = y_arr[idx]

        out = _predict(model, float_emb, data_emb, data_arr, self.DEVICE)

        mono_decoded = _decode_per_var(out["monotonicity"], _MONO_LABELS, n_vars)
        conv_decoded = _decode_per_var(out["convexity"], _CONV_LABELS, n_vars)
        period_decoded = _decode_per_var(out["periodicity"], _PERIOD_LABELS, n_vars)

        sep_logits = out["multiplicative_separable"]
        sep_probs = _softmax(sep_logits)
        sep_pred = int(sep_logits.argmax())
        sep_conf = float(sep_probs[sep_pred])

        monotonicity = {}
        convexity = {}
        periodicity = {}
        for i, var in enumerate(input_vars):
            monotonicity[var] = {"prediction": mono_decoded[i][0], "confidence": mono_decoded[i][1]}
            convexity[var] = {"prediction": conv_decoded[i][0], "confidence": conv_decoded[i][1]}
            periodicity[var] = {"prediction": period_decoded[i][0], "confidence": period_decoded[i][1]}

        separability = {
            "prediction": _SEP_LABELS[sep_pred],
            "confidence": round(sep_conf, 3),
        }

        # --- Variable combination prediction ---
        combo_results = {}
        if n_vars >= 2:
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    vi, vj = input_vars[i], input_vars[j]
                    xi, xj = var_arrays[vi], var_arrays[vj]
                    for op_name, op_fn in _COMBO_OPS.items():
                        combo_name = f"{vi}{op_name}{vj}"
                        combo_val = op_fn(xi, xj).astype(np.float32)
                        if not np.all(np.isfinite(combo_val)):
                            valid = np.isfinite(combo_val)
                            if valid.sum() < 20:
                                continue
                            combo_val = combo_val[valid]
                            y_combo = y_arr[idx][valid]
                            combo_s = len(combo_val)
                        else:
                            y_combo = y_arr[idx]
                            combo_s = sample_num

                        remaining_vars = [input_vars[k] for k in range(n_vars) if k != i and k != j]
                        n_combo_vars = 1 + len(remaining_vars)

                        combo_arr = np.zeros((combo_s, max_var_num + 1), dtype=np.float32)
                        combo_arr[:, 0] = combo_val[:combo_s]
                        for ri, rv in enumerate(remaining_vars):
                            if not np.all(np.isfinite(combo_val)):
                                combo_arr[:, 1 + ri] = var_arrays[rv][valid][:combo_s]
                            else:
                                combo_arr[:, 1 + ri] = var_arrays[rv][:combo_s]
                        combo_arr[:, -1] = y_combo[:combo_s]

                        combo_out = _predict(model, float_emb, data_emb, combo_arr, self.DEVICE)

                        combo_mono = _decode_per_var(combo_out["monotonicity"], _MONO_LABELS, 1)
                        combo_conv = _decode_per_var(combo_out["convexity"], _CONV_LABELS, 1)
                        combo_period = _decode_per_var(combo_out["periodicity"], _PERIOD_LABELS, 1)

                        combo_results[combo_name] = {
                            "monotonicity": {"prediction": combo_mono[0][0], "confidence": combo_mono[0][1]},
                            "convexity": {"prediction": combo_conv[0][0], "confidence": combo_conv[0][1]},
                            "periodicity": {"prediction": combo_period[0][0], "confidence": combo_period[0][1]},
                        }

        result = {
            "monotonicity": monotonicity,
            "convexity": convexity,
            "periodicity": periodicity,
            "multiplicative_separable": separability,
            "n_variables_analyzed": n_vars,
            "n_samples_used": sample_num,
            "exceptions": exceptions,
        }
        if combo_results:
            result["variable_combinations"] = combo_results

        return result

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        if "error" in result:
            return f"Error: {result['error']}"

        lines = [
            f"Property Prediction Results (analyzed {result['n_variables_analyzed']} variables, "
            f"{result['n_samples_used']} samples):",
            "",
            "=== Per-Variable Properties ===",
            "",
            "Monotonicity (whether y is monotonic w.r.t. each variable):",
        ]
        for var, info in result["monotonicity"].items():
            lines.append(f"  {var}: {info['prediction']} (confidence: {info['confidence']:.1%})")

        lines.append("")
        lines.append("Convexity (whether y is convex/concave w.r.t. each variable):")
        for var, info in result["convexity"].items():
            lines.append(f"  {var}: {info['prediction']} (confidence: {info['confidence']:.1%})")

        lines.append("")
        lines.append("Periodicity (whether y is periodic w.r.t. each variable):")
        for var, info in result["periodicity"].items():
            lines.append(f"  {var}: {info['prediction']} (confidence: {info['confidence']:.1%})")

        sep = result["multiplicative_separable"]
        lines.append("")
        lines.append(f"Multiplicative Separability: {sep['prediction']} (confidence: {sep['confidence']:.1%})")

        # Per-variable hints
        if any(info["prediction"] == "periodic" for info in result["periodicity"].values()):
            lines.append("")
            lines.append("Hint: Periodic variables suggest the formula likely contains trigonometric functions "
                         "(sin, cos) of those variables.")

        if result["multiplicative_separable"]["prediction"] == "multiplicatively separable":
            lines.append("")
            lines.append("Hint: The formula may be expressible as a product of functions of individual variables, "
                         "e.g., y = f(x1) * g(x2). Try decomposing the problem.")

        # Variable combination results
        combo = result.get("variable_combinations", {})
        if combo:
            lines.append("")
            lines.append("=== Variable Combination Properties ===")
            lines.append("(Analyzing xi*xj, xi+xj, xi-xj, xi/xj to detect combination-level properties)")
            lines.append("")

            interesting = []
            for cname, cinfo in combo.items():
                flags = []
                if cinfo["periodicity"]["prediction"] == "periodic":
                    flags.append(f"periodic ({cinfo['periodicity']['confidence']:.1%})")
                if cinfo["monotonicity"]["prediction"] in ("increasing", "decreasing"):
                    flags.append(f"{cinfo['monotonicity']['prediction']} ({cinfo['monotonicity']['confidence']:.1%})")
                if cinfo["convexity"]["prediction"] in ("convex", "concave"):
                    flags.append(f"{cinfo['convexity']['prediction']} ({cinfo['convexity']['confidence']:.1%})")
                if flags:
                    interesting.append((cname, flags, cinfo))

            if interesting:
                for cname, flags, cinfo in interesting:
                    lines.append(f"  {cname}: {', '.join(flags)}")
                lines.append("")
                lines.append("Hint: Variable combinations with detected properties suggest the formula may contain "
                             "functions of those combinations. For example:")
                for cname, flags, cinfo in interesting:
                    if cinfo["periodicity"]["prediction"] == "periodic":
                        lines.append(f"  - {cname} is periodic → consider sin({cname}), cos({cname})")
                    if cinfo["monotonicity"]["prediction"] in ("increasing", "decreasing"):
                        lines.append(f"  - {cname} is {cinfo['monotonicity']['prediction']} → y may depend on {cname} directly or via a monotonic function")
                    if cinfo["convexity"]["prediction"] in ("convex", "concave"):
                        lines.append(f"  - {cname} is {cinfo['convexity']['prediction']} → consider {cname}², sqrt({cname}), exp({cname}), etc.")
            else:
                lines.append("  No notable properties detected in variable combinations.")
                lines.append("")
                lines.append("Note: Even though no combination properties were detected, complex variable "
                             "combinations (e.g., sin(xi-xj), cos(xi*xj)) should still be explored "
                             "as the model may not capture all patterns.")

        if result["exceptions"]:
            lines.append("")
            lines.append("Exceptions / Warnings:")
            for ex in result["exceptions"]:
                lines.append(f"  - {ex}")

        return "\n".join(lines)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
