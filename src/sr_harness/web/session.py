"""Single interactive run and observable hooks, without changing the search loop."""
from __future__ import annotations

import csv
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..agents.sr_agent_interactive import SRAgentInteractive
from ..core import json_value
from ..interaction import InteractionManager, WebInteractionManager
from ..runtime import InteractionController
from ..interaction.web import add_variable_descriptions
from ..runtime import ModelRouter


class InteractiveSession:
    """Own one run for the lifetime of the server; no account/session registry."""
    def __init__(self, log_dir, controller=None, agent_options=None, data=None, initial_prompt=""):
        self.controller = controller or InteractionController()
        self.lock = threading.RLock()
        self.run_id = uuid.uuid4().hex
        self.run_dir = Path(log_dir).resolve() / self.run_id
        self.workspace = self.run_dir / "workspace"
        self.workspace.mkdir(parents=True)
        self.settings = {"llm_provider": "openrouter", "llm_model": "deepseek/deepseek-v4-flash",
                         "max_refinement_depth": 50, "local_sample_size": 1,
                         "global_width": 1, "max_restart_loop": 1}
        self.settings.update(agent_options or {})
        self.pending_settings = None
        self.data = data
        self.initial_prompt = initial_prompt
        self.prompt_overrides = {}
        self.variable_descriptions = {}
        self.state = "idle"
        self.result = None
        self.run_state = None
        self.thread = None

    def snapshot(self):
        with self.lock:
            topk = (
                []
                if self.run_state is None
                else json_value([
                    record.display_dict()
                    for record in self.run_state.ranked_candidates()
                ])
            )
            return {"state": self.state, "settings": self.settings.copy(),
                    "pending_settings": self.pending_settings, "topk_records": topk,
                    "result": self.result, "run_id": self.run_id,
                    "initial_prompt": self.initial_prompt, "supplied_data": self.data is not None,
                    "workspace": str(self.workspace), **self.controller.status()}

    def start(self, payload):
        with self.lock:
            if self.state != "idle":
                raise ValueError("This server already owns a run. Restart it to begin a new task.")
            options = self.validate_settings(payload, initial=True)
            self.settings.update(options)
            description = str(payload.get(
                "problem_description",
                payload.get("prompt", "Find an interpretable formula explaining the data."),
            ))
            self.prompt_overrides = {
                role: str(payload[key])
                for role, key in (("system", "system_prompt"), ("user", "user_prompt"))
                if key in payload
            }
            self.variable_descriptions = self.validate_variable_descriptions(payload)
            if self.data is not None:
                X, y = self.data
            elif payload.get("dataset"):
                import pandas as pd
                path = self.resolve(str(payload["dataset"]))
                frame = pd.read_csv(path)
                target = str(payload.get("target", "y"))
                if target not in frame or len(frame.columns) < 2 or len(frame) < 5:
                    raise ValueError("CSV needs a target column, at least one feature, and 5 rows.")
                requested_features = payload.get("features")
                if requested_features is None:
                    features = [str(column) for column in frame if column != target]
                elif (
                    not isinstance(requested_features, list)
                    or not requested_features
                    or any(not isinstance(column, str) for column in requested_features)
                ):
                    raise ValueError("features must be a non-empty list of column names")
                else:
                    features = list(dict.fromkeys(requested_features))
                missing = [column for column in features if column not in frame]
                if missing or target in features:
                    raise ValueError("Features must exist in the CSV and must not include the target")
                try:
                    values = frame[[*features, target]].to_numpy(dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Selected target and feature columns must be numeric") from exc
                if not np.isfinite(values).all():
                    raise ValueError("Selected columns must contain finite values without missing data")
                X = {column: frame[column].to_numpy(dtype=float) for column in features}
                y = {target: frame[target].to_numpy(dtype=float)}
            else:
                rng = np.random.default_rng(42)
                x = rng.uniform(-2, 2, 100)
                X, y = {"x": x}, {"y": x*x + 2*x + 1}
                self.create_demo()
            self.state = "starting"
            self.controller.publish("activity", {"phase": "initializing"})
            self.thread = threading.Thread(target=self._run, args=(X, y, description), daemon=True)
            self.thread.start()
        return self.snapshot()

    def preview_initial_prompts(self, payload):
        description = str(payload.get(
            "problem_description",
            self.initial_prompt or "Find an interpretable formula explaining the selected target from the selected features.",
        ))
        if self.data is not None:
            X, y = self.data
            if not isinstance(y, dict):
                y = {"target": y}
        elif payload.get("dataset"):
            import pandas as pd
            frame = pd.read_csv(self.resolve(str(payload["dataset"])), nrows=1)
            target = str(payload.get("target", ""))
            requested_features = payload.get("features")
            if requested_features is None:
                features = [str(column) for column in frame if str(column) != target]
            elif (
                not isinstance(requested_features, list)
                or not requested_features
                or any(not isinstance(column, str) for column in requested_features)
            ):
                raise ValueError("features must be a non-empty list of column names")
            else:
                features = list(dict.fromkeys(requested_features))
            if (
                target not in frame
                or target in features
                or any(column not in frame for column in features)
            ):
                raise ValueError("Select a target and at least one existing feature")
            X = {str(column): np.empty(1) for column in features}
            y = {target: np.empty(1)}
        else:
            X, y = {"x": np.empty(1)}, {"y": np.empty(1)}
        settings = self.settings
        preview_agent = SimpleNamespace(
            model_router=ModelRouter(
                enabled=bool(settings.get("auto_routing", False)),
                base_provider=settings["llm_provider"],
                base_model=settings["llm_model"],
                strong_provider=settings.get("strong_llm_provider"),
                strong_model=settings.get("strong_llm_model"),
            ),
            max_refinement_depth=settings["max_refinement_depth"],
            use_workspace=True,
            tools=[],
            interaction_manager=InteractionManager(),
        )
        messages = SRAgentInteractive.build_initial_prompt(
            preview_agent, description, X, y, [],
        )
        add_variable_descriptions(
            messages, self.validate_variable_descriptions(payload), [*X, *y],
        )
        return {
            "problem_description": description,
            "system_prompt": next(message["content"] for message in messages if message["role"] == "system"),
            "user_prompt": next(message["content"] for message in messages if message["role"] == "user"),
        }

    @staticmethod
    def validate_variable_descriptions(payload):
        descriptions = payload.get("variable_descriptions", {})
        if not isinstance(descriptions, dict) or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in descriptions.items()
        ):
            raise ValueError("variable_descriptions must map column names to text")
        return {
            name: value.strip()
            for name, value in descriptions.items()
            if value.strip()
        }

    def create_demo(self):
        with self.lock:
            path = self.workspace / "demo.csv"
            if not path.exists():
                rng = np.random.default_rng(42)
                x1 = rng.uniform(-2, 2, 100)
                x2 = rng.uniform(-1, 3, 100)
                x3 = np.tile(["alpha", "beta", "gamma", "delta"], 25)
                rng.shuffle(x3)
                y = x1*x1 + 2*x2 + 1
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["x1", "x2", "x3", "y"])
                    writer.writerows(zip(x1, x2, x3, y))
            return path

    def _run(self, X, y, description):
        try:
            options = dict(self.settings)
            save_path = options.pop("save_path", str(self.run_dir))
            options.update(
                save_path=save_path,
                run_id=self.run_id,
                use_workspace=True,
            )
            options.setdefault("max_workers", 0)
            manager = WebInteractionManager(self)
            agent = SRAgentInteractive(interaction_manager=manager, **options)
            with self.lock:
                self.state = "running"
            self.controller.publish("lifecycle", {"state": "running"})
            result = agent.run(X, y, description)
        except KeyboardInterrupt as exc:
            result = getattr(exc, "partial_result", {}) | {"status": "interrupted"}
        except Exception as exc:
            result = getattr(exc, "partial_result", {}) | {"status": "failed", "error": str(exc)}
        with self.lock:
            self.result = json_value(result)
            self.state = result.get("status", "completed")
        self.controller.publish("lifecycle", {"state": self.state, "result": self.result})
        self.controller.publish("activity", {"phase": self.state})

    @staticmethod
    def validate_settings(payload, initial=False):
        allowed = {"llm_provider", "llm_model"}
        integers = {"max_refinement_depth", "local_sample_size", "global_width", "max_restart_loop"}
        if initial:
            allowed |= integers
        result = {k: v for k, v in payload.items() if k in allowed}
        for k, v in result.items():
            if k in integers:
                if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 1000:
                    raise ValueError(f"{k} must be an integer between 1 and 1000")
            elif not isinstance(v, str) or not v.strip():
                raise ValueError(f"{k} must be non-empty")
        if "llm_provider" in result and result["llm_provider"] not in {
            "openrouter", "openai", "deepseek", "siliconflow", "gemini", "lmstudio"
        }:
            raise ValueError("Unsupported web model provider")
        return result

    def configure(self, payload):
        settings = self.validate_settings(payload)
        with self.lock:
            if self.state not in {"idle", "starting", "running"}:
                raise ValueError("Run has finished")
            if self.state == "idle":
                self.settings.update(settings)
            else:
                self.pending_settings = {k: settings.get(k, (self.pending_settings or self.settings)[k])
                                         for k in ("llm_provider", "llm_model")}
        return self.snapshot()

    def resolve(self, path):
        root = self.workspace.resolve()
        candidate = (root / path).resolve()
        if Path(path).is_absolute() or not candidate.is_relative_to(root):
            raise ValueError("Path must stay inside the workspace")
        return candidate
