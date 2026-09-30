# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""ND2 neural symbolic regression for network dynamics."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .base_tool import BaseTool, ToolMetadata


@BaseTool.register("nd2")
class ND2Tool(BaseTool):
    metadata = ToolMetadata(name="nd2")
    REPOSITORY = "https://github.com/tsinghua-fib-lab/ND2"
    DEFAULT_BINARY = ["add", "sub", "mul", "div", "pow", "regular"]
    DEFAULT_UNARY = [
        "neg", "abs", "inv", "exp", "logabs", "sin", "cos", "tan",
        "sqrtabs", "pow2", "pow3", "tanh", "sigmoid", "aggr", "sour",
        "targ",
    ]

    def execute(
        self,
        vars_node: List[str] = None,
        vars_edge: List[str] = None,
        y: str = None,
        root_type: str = "node",
        binary_operators: List[str] = None,
        unary_operators: List[str] = None,
        timeout: int = 60,
        episode_limit: int = 10000,
        beam_size: int = 10,
        max_coeff_num: int = 5,
    ) -> Dict[str, Any]:
        """Run NDformer-guided MCTS on node or edge dynamics in the current data.

        The context must contain adjacency `A` and/or edge list `G`, a target shaped
        `(time, node)` or `(time, edge)`, and node/edge variables with matching final axes.
        `ND2_HOME` must point to the official repository; `ND2_CHECKPOINT` defaults to
        `$ND2_HOME/weights/checkpoint.pth`.

        Args:
            vars_node: Node-level variable names. Inferred from the node axis when omitted.
            vars_edge: Edge-level variable names. Inferred from the edge axis when omitted.
            y: Target variable name. Defaults to the configured target.
            root_type: Output expression type, either "node" or "edge".
            binary_operators: ND2 binary tokens. Defaults to its standard binary vocabulary.
            unary_operators: ND2 unary tokens. Defaults to its standard unary vocabulary.
            timeout: Wall-time search limit in seconds, between 10 and 3600.
            episode_limit: Maximum MCTS episodes, between 1 and 1000000.
            beam_size: Number of expansions retained per MCTS step, between 1 and 100.
            max_coeff_num: Maximum fitted scalar coefficients, between 0 and 20.
        """
        status = self.backend_status()
        if not status["installed"]:
            raise RuntimeError(
                "ND2 is not configured: clone the repository and set ND2_HOME."
            )
        checkpoint = Path(
            os.environ.get(
                "ND2_CHECKPOINT",
                Path(status["root"]) / "weights" / "checkpoint.pth",
            )
        ).expanduser().resolve()
        if not checkpoint.is_file():
            raise RuntimeError(
                "ND2 checkpoint not found. Set ND2_CHECKPOINT to checkpoint.pth."
            )
        if root_type not in {"node", "edge"}:
            raise ValueError("root_type must be 'node' or 'edge'")

        timeout = max(10, min(int(timeout), 3600))
        episode_limit = max(1, min(int(episode_limit), 1_000_000))
        beam_size = max(1, min(int(beam_size), 100))
        max_coeff_num = max(0, min(int(max_coeff_num), 20))
        data = self._prepare_data(self.context["data"])
        y = (y or self.context["target"]).strip()
        if y not in data:
            raise ValueError(f"Target {y!r} is missing from the ND2 data.")

        node_count = data["A"].shape[0]
        edge_count = data["G"].shape[0]
        excluded = {"A", "G", y}
        if vars_node is None:
            vars_node = [
                name for name, value in data.items()
                if name not in excluded and np.shape(value)[-1:] == (node_count,)
            ]
        if vars_edge is None:
            vars_edge = [
                name for name, value in data.items()
                if name not in excluded and np.shape(value)[-1:] == (edge_count,)
                and name not in vars_node
            ]
        missing = [name for name in [*vars_node, *vars_edge] if name not in data]
        if missing:
            raise ValueError(f"ND2 variables are missing from data: {missing}")
        expected_width = node_count if root_type == "node" else edge_count
        if np.asarray(data[y]).ndim != 2 or np.asarray(data[y]).shape[1] != expected_width:
            raise ValueError(
                f"ND2 {root_type} target must have shape (time, {expected_width})."
            )

        search = self._run_nd2(
            root=Path(status["root"]),
            checkpoint=checkpoint,
            data=data,
            y=y,
            vars_node=vars_node,
            vars_edge=vars_edge,
            root_type=root_type,
            binary_operators=binary_operators or self.DEFAULT_BINARY,
            unary_operators=unary_operators or self.DEFAULT_UNARY,
            timeout=timeout,
            episode_limit=episode_limit,
            beam_size=beam_size,
            max_coeff_num=max_coeff_num,
        )
        split_results = {"train": {"metrics": search["train_metrics"]}}
        if search.get("validation_metrics") is not None:
            split_results["validation"] = {
                "metrics": search["validation_metrics"],
            }
        return {
            "formula": search["formula"],
            "target_expression": y,
            "is_candidate": bool(np.isfinite(search["train_metrics"]["mse"])),
            "candidate_ineligibility_reasons": [],
            "data_split_results": split_results,
            "method": "ND2-NDformer-MCTS",
            "prefix": search["prefix"],
            "backend": status,
            "config": {
                "vars_node": vars_node,
                "vars_edge": vars_edge,
                "root_type": root_type,
                "timeout": timeout,
                "episode_limit": episode_limit,
                "beam_size": beam_size,
                "max_coeff_num": max_coeff_num,
            },
        }

    def _run_nd2(
        self,
        *,
        root: Path,
        checkpoint: Path,
        data: Dict[str, np.ndarray],
        y: str,
        vars_node: List[str],
        vars_edge: List[str],
        root_type: str,
        binary_operators: List[str],
        unary_operators: List[str],
        timeout: int,
        episode_limit: int,
        beam_size: int,
        max_coeff_num: int,
    ) -> Dict[str, Any]:
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        try:
            import torch
            from ND2.GDExpr import GDExpr
            from ND2.model import NDformer
            from ND2.search import MCTS
            from ND2.search.reward_solver import RewardSolver
        except ImportError as exc:
            raise RuntimeError(
                "ND2 dependencies are unavailable in this Python environment."
            ) from exc

        device = os.environ.get(
            "ND2_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        rewarder = self._make_rewarder(
            RewardSolver, data, y, vars_node, vars_edge
        )
        model = NDformer(device=device)
        model.load(checkpoint, weights_only=False)
        model.eval()
        model.set_data(
            Xv={name: data[name] for name in vars_node},
            Xe={name: data[name] for name in vars_edge},
            A=data["A"],
            G=data["G"],
            Y=data[y],
            root_type=root_type,
            cache_data_emb=True,
        )
        estimator = MCTS(
            rewarder=rewarder,
            ndformer=model,
            vars_node=vars_node,
            vars_edge=vars_edge,
            binary=binary_operators,
            unary=unary_operators,
            log_per_episode=100,
            log_per_second=10,
            beam_size=beam_size,
            max_coeff_num=max_coeff_num,
            use_random_simulate=False,
        )
        estimator.fit(
            [root_type],
            episode_limit=episode_limit,
            time_limit=timeout,
        )
        if not estimator.best_model:
            raise RuntimeError("ND2 completed without producing a formula.")

        validation_metrics = None
        if validation_data := self.context.get("evaluation_data"):
            validation_data = self._prepare_data(validation_data)
            validator = self._make_rewarder(
                RewardSolver, validation_data, y, vars_node, vars_edge
            )
            validation_metrics = self._normalize_metrics(
                validator.evaluate(estimator.best_model)
            )
        return {
            "formula": GDExpr.prefix2str(estimator.best_model),
            "prefix": [
                value.tolist() if isinstance(value, np.ndarray) else value
                for value in estimator.best_model
            ],
            "train_metrics": self._normalize_metrics(estimator.best_metric),
            "validation_metrics": validation_metrics,
        }

    @staticmethod
    def _make_rewarder(rewarder_cls, data, y, vars_node, vars_edge):
        return rewarder_cls(
            Xv={name: data[name] for name in vars_node},
            Xe={name: data[name] for name in vars_edge},
            A=data["A"],
            G=data["G"],
            Y=data[y],
            mask=None,
        )

    @staticmethod
    def _prepare_data(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        arrays = {name: np.asarray(value) for name, value in data.items()}
        if "A" not in arrays and "G" not in arrays:
            raise ValueError("ND2 requires adjacency A or edge list G.")
        if "G" not in arrays:
            arrays["G"] = np.stack(np.nonzero(arrays["A"]), axis=-1)
        arrays["G"] = arrays["G"].astype(int, copy=False)
        if "A" not in arrays:
            node_count = int(arrays["G"].max()) + 1
            arrays["A"] = np.zeros((node_count, node_count), dtype=int)
            arrays["A"][arrays["G"][:, 0], arrays["G"][:, 1]] = 1
        arrays["A"] = arrays["A"].astype(int, copy=False)
        return arrays

    @staticmethod
    def _normalize_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
        names = {
            "RMSE": "rmse",
            "MAE": "mae",
            "MAPE": "mape",
            "R2": "r2",
            "ACC2": "acc2",
            "ACC3": "acc3",
            "ACC4": "acc4",
            "complexity": "complexity",
        }
        result = {
            output: float(metrics[input_name])
            for input_name, output in names.items()
            if input_name in metrics
        }
        result["mse"] = result["rmse"] ** 2
        return result

    @classmethod
    def backend_status(cls) -> Dict[str, Any]:
        configured = os.environ.get("ND2_HOME")
        local_root = Path.cwd() / "third-party" / "ND2"
        root = (
            Path(configured).expanduser().resolve()
            if configured
            else local_root.resolve() if (local_root / "search.py").is_file()
            else None
        )
        marker = "search.py"
        checkpoint = (
            Path(os.environ.get("ND2_CHECKPOINT", root / "weights" / "checkpoint.pth"))
            if root else None
        )
        return {
            "backend": "nd2",
            "repository": cls.REPOSITORY,
            "environment_variable": "ND2_HOME",
            "root": str(root) if root else None,
            "installed": bool(root and (root / marker).is_file()),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_available": bool(checkpoint and checkpoint.is_file()),
            "marker": marker,
        }

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        split_results = result["data_split_results"]
        train = split_results["train"]["metrics"]
        validation = split_results.get("validation", {}).get("metrics")
        lines = [
            f"ND2 discovered: {result['target_expression']} = {result['formula']}",
            f"Train RMSE={train['rmse']:.6g}, R2={train['r2']:.6g}, "
            f"complexity={train['complexity']:.0f}",
        ]
        if validation:
            lines.append(
                f"Validation RMSE={validation['rmse']:.6g}, "
                f"R2={validation['r2']:.6g}"
            )
        return "\n".join(lines)

    @classmethod
    def get_doc(cls) -> dict[str, str]:
        return {
            "name": "nd2-network-dynamics",
            "description": (
                "Use ND2 neural symbolic regression for graph and network-dynamics discovery."
            ),
            "content": (
                "ND2 expects adjacency `A` or edge list `G`, node inputs shaped `(time, node)`, "
                "optional edge inputs shaped `(time, edge)`, and a node- or edge-level target. "
                "Call `nd2` to run the actual NDformer-guided MCTS; the returned graph-dynamics "
                "formula and train/validation metrics enter SRAgent's candidate set. Configure "
                "`ND2_HOME` and `ND2_CHECKPOINT` before use. Use ordinary SRAgent tools for "
                "non-network tabular regression.\n\n"
                f"Source: {cls.REPOSITORY}"
            ),
        }


"""
Real invocation captured with the official checkpoint and Kuramoto demo data:

>>> ND2Tool.backend_status()
{'backend': 'nd2', 'repository': 'https://github.com/tsinghua-fib-lab/ND2',
 'environment_variable': 'ND2_HOME', 'root': '.../third-party/ND2',
 'installed': True, 'checkpoint': '.../third-party/ND2/weights/checkpoint.pth',
 'checkpoint_available': True, 'marker': 'search.py'}

>>> result = tool.execute(
...     vars_node=["x", "omega"], vars_edge=[], y="dx",
...     binary_operators=["add", "sub", "mul"],
...     unary_operators=["sin", "aggr", "sour", "targ"],
...     timeout=10, episode_limit=1, beam_size=2, max_coeff_num=2,
... )
>>> print(ND2Tool.format_result_dict(result))
ND2 discovered: dx = ((omega+(0.9960*x))-x)
Train RMSE=1.11731, R2=0.839655, complexity=7
"""
