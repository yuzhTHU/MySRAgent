"""LLM-SRBench adapter for the upstream IGSR implementation.

The upstream checkout is used without source changes.  This module starts it in
an isolated process, injects the benchmark training data as an IGSR DataBundle,
and audits every LiteLLM call before returning control to IGSR.  Per-call usage
is flushed to ``usage.jsonl`` and relayed through ``_logger`` so completed calls
remain accountable if a later search step fails.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sr_agent._vendor.llmsr_bench.core import SEDTask, SRResult

_logger = logging.getLogger(f"sr_agent.{__name__}")
_ROOT = Path(__file__).resolve().parents[5]
_DEFAULTS = {
    # Released upstream experiment defaults, except for the deliberately
    # reduced adapter runtime budget documented below.
    "max_agent_steps": 5,
    "terms_per_round": 5,
    "first_round_n_candidates": 10,
    "keep_n_terms": 6,
    # One root batch contains n_successors=5 candidates.  Budget 5 keeps the
    # default smoke/benchmark run to that single batch (~0.5--0.75 h with the
    # current OpenRouter model); budget 6 starts another five-candidate batch.
    "total_budget": 5,
    "depth_limit": 10,
    "n_successors": 5,
    "early_stop_patience": 10,
    "retry_attempts": 10,
    "preview_n_elements": 50,
    "run_timeout": 0.0,
}


def update_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--llm_provider", choices=["openrouter"], default="openrouter")
    parser.add_argument("--llm_model", default="deepseek/deepseek-v4-flash-0731")
    parser.add_argument("--igsr_repo", type=Path, default=_ROOT / "baseline/repo/IGSR")
    parser.add_argument("--igsr_python", default=sys.executable)
    for name, default in _DEFAULTS.items():
        parser.add_argument(f"--igsr_{name}", type=type(default), default=default)
    parser.add_argument("--igsr_max_tokens", type=int, default=None,
                        help="Maximum completion tokens for each IGSR LLM call.")
    parser.add_argument("--igsr_reasoning", choices=["default", "low"], default="default",
                        help="OpenRouter reasoning effort; default leaves provider behavior unchanged.")
    parser.add_argument("--igsr_error_on_all_weights_zero",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Preserve upstream's hard failure for an all-near-zero linear fit.")
    parser.add_argument("--igsr_tree", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--igsr_early_stop", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _usage_from_response(response: Any) -> dict[str, Any]:
    """Normalize LiteLLM/OpenRouter usage like ``sr_agent.api.openrouter_api``."""
    usage = getattr(response, "usage", None)
    raw = usage.to_dict() if hasattr(usage, "to_dict") else dict(usage or {})
    prompt = int(raw.get("prompt_tokens") or 0)
    completion = int(raw.get("completion_tokens") or 0)
    total = int(raw.get("total_tokens") or prompt + completion)
    details = raw.get("completion_tokens_details") or {}
    reasoning = int(details.get("reasoning_tokens") or 0)
    tokens = {"prompt": prompt, "answer": max(completion - reasoning, 0)}
    if reasoning:
        tokens["reason"] = reasoning
    if total > prompt + completion:
        tokens["other"] = total - prompt - completion

    # OpenRouter may expose its authoritative cost in usage.cost.  LiteLLM
    # commonly moves it to response._hidden_params.response_cost instead.
    cost = raw.get("cost")
    hidden = getattr(response, "_hidden_params", None) or {}
    if cost is None and isinstance(hidden, dict):
        cost = hidden.get("response_cost")
    if cost is None:
        extra = getattr(response, "model_extra", None) or {}
        cost = extra.get("cost") if isinstance(extra, dict) else None
    return {"token": tokens, "price": {"total": float(cost or 0.0)}}


def _summarize_usage(path: Path) -> dict[str, Any]:
    token: dict[str, int] = {}
    price: dict[str, float] = {}
    calls = failed_calls = 0
    elapsed = 0.0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            calls += row.get("status") == "ok"
            failed_calls += row.get("status") != "ok"
            elapsed += float(row.get("elapsed_seconds") or 0)
            for key, value in row.get("usage", {}).get("token", {}).items():
                token[key] = token.get(key, 0) + int(value)
            for key, value in row.get("usage", {}).get("price", {}).items():
                price[key] = price.get(key, 0.0) + float(value)
    return {"calls": calls, "failed_calls": failed_calls, "elapsed_seconds": elapsed,
            "token": token, "price": price}


def _safe_term(expr: str, values: dict[str, np.ndarray]) -> np.ndarray:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            raise ValueError(f"Unsafe IGSR term: {expr!r}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError(f"Unsafe IGSR term: {expr!r}")
    result = eval(compile(tree, "<igsr-term>", "eval"),
                  {"__builtins__": {}, "np": np, "numpy": np, "abs": np.abs}, values)
    return np.asarray(result, dtype=float)


def _display_term(term: str, symbols: list[str]) -> str:
    result = term.replace("numpy.", "").replace("np.", "")
    result = re.sub(r"\bsquare\(([^()]*)\)", r"(\1)**2", result)
    for index in range(len(symbols) - 1, 0, -1):
        result = re.sub(rf"\bx{index}\b", symbols[index], result)
    return result


def _build_result(result: dict[str, Any], symbols: list[str]):
    from sr_agent._vendor.llmsr_bench.core import SRResult

    terms = list(result.get("terms") or [])
    coefficients = np.asarray(result.get("coefficients"), dtype=float).reshape(-1)
    intercept = float(result.get("intercept") or 0.0)
    if not terms or len(terms) != len(coefficients) or not np.isfinite(coefficients).all():
        raise RuntimeError("IGSR returned no finite fitted candidate.")

    pieces = [repr(intercept)] if intercept else []
    pieces.extend(f"({float(coef)!r})*({_display_term(term, symbols)})"
                  for coef, term in zip(coefficients, terms))
    expression = " + ".join(pieces) or "0.0"

    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(symbols) - 1:
            raise ValueError(f"Expected X with shape (n, {len(symbols) - 1}), got {X.shape}")
        if not len(X):
            return np.empty(0, dtype=float)
        values = {f"x{i + 1}": X[:, i] for i in range(X.shape[1])}
        columns = []
        for term in terms:
            column = _safe_term(term, values)
            columns.append(np.broadcast_to(column, (len(X),)))
        return np.column_stack(columns) @ coefficients + intercept

    return SRResult(predict=predict, expression=expression)


def run(args: argparse.Namespace, task: SEDTask) -> SRResult:
    from dotenv import dotenv_values

    X = np.asarray(task.train_X, dtype=float)
    y = np.asarray(task.train_y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y) or len(X) < 6:
        raise ValueError("IGSR requires at least six matching training samples.")
    if len(task.symbols) != X.shape[1] + 1 or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Training symbols must match columns and training data must be finite.")
    repo = Path(getattr(args, "igsr_repo", _ROOT / "baseline/repo/IGSR")).resolve()
    if not (repo / "src/igsr/method/igsr.py").is_file():
        raise FileNotFoundError(f"Clone https://github.com/DrShushen/IGSR into {repo}")

    # Deliberately read the project .env, never a shell startup file.  The
    # project value overrides a possibly inherited OPENROUTER_API_KEY.
    key = dotenv_values(_ROOT / ".env").get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is missing from the project .env.")
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = key
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"

    config = {name: getattr(args, f"igsr_{name}", default) for name, default in _DEFAULTS.items()}
    config.update(model=getattr(args, "llm_model", "deepseek/deepseek-v4-flash-0731"),
                  max_tokens=getattr(args, "igsr_max_tokens", None),
                  reasoning=getattr(args, "igsr_reasoning", "default"),
                  error_on_all_weights_zero=getattr(
                      args, "igsr_error_on_all_weights_zero", True),
                  tree=getattr(args, "igsr_tree", True),
                  early_stop=getattr(args, "igsr_early_stop", True),
                  seed=int(getattr(args, "seed", 0)))
    for name in ("max_agent_steps", "terms_per_round", "first_round_n_candidates", "keep_n_terms",
                 "total_budget", "depth_limit", "n_successors", "early_stop_patience", "retry_attempts"):
        if config[name] <= 0:
            raise ValueError(f"igsr_{name} must be positive.")
    if config["run_timeout"] < 0:
        raise ValueError("igsr_run_timeout must be nonnegative.")
    if config["max_tokens"] is not None and config["max_tokens"] <= 0:
        raise ValueError("igsr_max_tokens must be positive when provided.")

    root = Path(getattr(args, "save_path", None) or _ROOT / "logs/igsr") / "igsr"
    root.mkdir(parents=True, exist_ok=True)
    prefix = re.sub(r"[^\w.-]", "_", task.name)[:80] + "-"
    work = Path(tempfile.mkdtemp(prefix=prefix, dir=root)).resolve()
    np.savez(work / "train.npz", X=X, y=y)
    request = {"config": config, "name": task.name, "symbols": list(task.symbols),
               "descriptions": list(task.symbol_descs), "properties": list(task.symbol_properties)}
    (work / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    metadata = {"status": "running", "started_at": time.time(), "task": task.name,
                "model": config["model"], "config": config, "artifacts": str(work)}
    metadata_path = work / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _logger.info("IGSR start task=%s model=%s config=%s artifacts=%s", task.name,
                 config["model"], config, work)

    python = Path(getattr(args, "igsr_python", sys.executable))
    if not python.is_absolute():
        python = (_ROOT / python).resolve()
    if not python.is_file():
        raise FileNotFoundError(f"IGSR Python interpreter does not exist: {python}")
    # Module execution avoids this adapter's ``igsr.py`` filename shadowing
    # upstream's namespace package when the child imports ``igsr.dataset``.
    cmd = [str(python), "-m", "sr_agent._vendor.llmsr_bench.algorithms.my_igsr",
           "--worker", str(repo), str(work)]
    log_path = work / "worker.log"
    process = subprocess.Popen(cmd, cwd=repo, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)

    def relay():
        assert process.stdout is not None
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            for line in process.stdout:
                log.write(line)
                _logger.info("IGSR worker | %s", line.rstrip())

    reader = threading.Thread(target=relay, name="igsr-log-relay", daemon=True)
    reader.start()
    failure: BaseException | None = None
    try:
        returncode = process.wait(timeout=config["run_timeout"] or None)
        if returncode:
            raise RuntimeError(f"IGSR exited with status {returncode}. See {log_path}")
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        failure = RuntimeError(f"IGSR exceeded its run timeout. See {log_path}")
        failure.__cause__ = exc
    except BaseException as exc:
        failure = exc
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    finally:
        reader.join(timeout=10)
        summary = _summarize_usage(work / "usage.jsonl")
        metadata.update(status="failed" if failure else "completed", finished_at=time.time(),
                        elapsed_seconds=time.time() - metadata["started_at"], usage=summary)
        if failure:
            metadata["error"] = repr(failure)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _logger.info("IGSR usage task=%s elapsed=%.3fs calls=%d failed_calls=%d tokens=%s money_usd=%.9f artifacts=%s",
                     task.name, metadata["elapsed_seconds"], summary["calls"], summary["failed_calls"],
                     summary["token"], summary["price"].get("total", 0.0), work)
    if failure:
        raise failure
    result_path = work / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"IGSR produced no result. See {log_path}")
    return _build_result(json.loads(result_path.read_text(encoding="utf-8")), list(task.symbols))


def _worker(repo: Path, work: Path) -> None:
    sys.path.insert(0, str(repo / "src"))
    import pandas as pd
    import litellm
    from omegaconf import OmegaConf
    from sklearn.linear_model import LinearRegression
    from igsr.dataset import DataBundle
    import igsr.method.igsr as method
    from igsr.method.igsr_utils import DesignMatrix
    from igsr.agent.agent import LiteLLMAgent as UpstreamAgent

    request = json.loads((work / "request.json").read_text(encoding="utf-8"))
    cfg_in = request["config"]
    with np.load(work / "train.npz") as data:
        X, y = data["X"], data["y"]
    names = [f"x{i + 1}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=names)
    frame["y"] = y
    rng = np.random.default_rng(cfg_in["seed"])
    order = rng.permutation(len(frame))
    n_train = max(2, int(0.7 * len(frame)))
    n_val = max(2, int(0.15 * len(frame)))
    if n_train + n_val > len(frame) - 1:
        n_train, n_val = len(frame) - 3, 2
    bundle = DataBundle(
        name=request["name"], data_settings={}, dataset_train=frame.iloc[order[:n_train]].copy(),
        dataset_validation=frame.iloc[order[n_train:n_train + n_val]].copy(),
        dataset_test=frame.iloc[order[n_train + n_val:]].copy(), target_columns=["y"],
        data_dictionary="\n".join(
            [f"y (target): {request['descriptions'][0]}"]
            + [f"{name}: {desc}" for name, desc in zip(names, request["descriptions"][1:])]
        ),
    )
    method.get_dataset = lambda _cfg: bundle

    usage_path = work / "usage.jsonl"
    original_completion = litellm.completion

    def audited_completion(*args, **kwargs):
        started = time.time()
        kwargs.setdefault("extra_body", {"usage": {"include": True}})
        try:
            response = original_completion(*args, **kwargs)
            record = {"timestamp": time.time(), "status": "ok", "model": kwargs.get("model"),
                      "elapsed_seconds": time.time() - started, "usage": _usage_from_response(response)}
            return response
        except BaseException as exc:
            record = {"timestamp": time.time(), "status": "error", "model": kwargs.get("model"),
                      "elapsed_seconds": time.time() - started,
                      "usage": {"token": {}, "price": {}}, "error": repr(exc)}
            raise
        finally:
            with usage_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print("IGSR_API_USAGE " + json.dumps(record, default=str), flush=True)

    litellm.completion = audited_completion

    class OpenRouterAgent(UpstreamAgent):
        def __init__(self, *args, model="openai/gpt-4", completion_kwargs=None, **kwargs):
            if model.startswith("openai/"):
                model = "openrouter/" + model[len("openai/"):]
            completion_kwargs = dict(completion_kwargs or {})
            completion_kwargs["api_key"] = os.environ["OPENROUTER_API_KEY"]
            if cfg_in.get("max_tokens") is not None:
                completion_kwargs["max_tokens"] = int(cfg_in["max_tokens"])
            extra_body = dict(completion_kwargs.get("extra_body") or {})
            extra_body["usage"] = {"include": True}
            if cfg_in.get("reasoning") == "low":
                extra_body["reasoning"] = {"effort": "low"}
            completion_kwargs["extra_body"] = extra_body
            super().__init__(*args, model=model, completion_kwargs=completion_kwargs, **kwargs)

    method.LiteLLMAgent = OpenRouterAgent
    experiment = {
        "kind": "igsr", "tree": cfg_in["tree"], "n_iters": cfg_in["total_budget"],
        "max_agent_steps": cfg_in["max_agent_steps"], "terms_per_round": cfg_in["terms_per_round"],
        "first_round_n_candidates": cfg_in["first_round_n_candidates"],
        "keep_n_terms": cfg_in["keep_n_terms"], "total_budget": cfg_in["total_budget"],
        "depth_limit": cfg_in["depth_limit"], "n_successors": cfg_in["n_successors"],
        "c": 1.41421356, "rollout_is_just_node_reward": True, "rollout_depth": 1,
        "influence_feedback": True, "history_enabled": True, "refit_aware": False,
        "refit_aware_efficient": False, "early_stop_enabled": cfg_in["early_stop"],
        "early_stop_patience": cfg_in["early_stop_patience"], "optimization_method": "linear",
        "fallback_ridge_small_alpha_for_stability": False,
        "error_on_all_weights_zero": cfg_in["error_on_all_weights_zero"],
        "allow_error_history_length": 15, "print_prompts": False, "save_new_terms": False,
        "algorithmic_pruning": False, "additional_validation_of_terms": True,
        "retry_attempts": cfg_in["retry_attempts"], "simplified_prompts": False,
        "use_xgb_predictor": False, "keep_n_columns": None, "keep_n_columns_always_keep": None,
        "break_on_tokens": False, "break_above_N_tokens": 260000,
        "break_on_wallclock": False, "break_above_wallclock": 300,
        "preview_N_elements": cfg_in["preview_n_elements"], "collapse_seq_feats": False,
        "focus_on_seq_feats": False, "show_feature_importance": False,
        "feature_importance_max_features": 20,
    }
    cfg = OmegaConf.create({
        "logging": {"logger_name": "igsr.adapter", "mode": "w",
                    "log_file": str(work / "igsr.jsonl")},
        # `kind=openai` passes the upstream branch; OpenRouterAgent changes only
        # LiteLLM's provider prefix, leaving IGSR's prompts/search untouched.
        # Never put the real credential in cfg: upstream prints the whole cfg
        # before its logger has a chance to redact it. OpenRouterAgent injects
        # the credential directly from the child environment at request time.
        "llm": {"kind": "openai", "model_type": cfg_in["model"], "model_version": "",
                "deployment": cfg_in["model"], "api_key": "***ENV_ONLY***"},
        "experiment": experiment,
        "dataset": {"kind": "adapter", "name": request["name"], "task": "regression", "seed": None},
    })
    try:
        upstream_result = method.igsr(cfg, cfg_in["seed"])
        history = upstream_result["history_test"]
        best = min(history, key=lambda entry: float(entry["metrics"]["mse"]))
        terms = list(best["metadata"]["terms_after"])
        design = DesignMatrix.from_terms(terms, {name: X[:, i] for i, name in enumerate(names)})
        reg = LinearRegression().fit(design.phi, y)
        result = {"terms": design.term_names, "coefficients": np.asarray(reg.coef_).reshape(-1).tolist(),
                  "intercept": float(np.asarray(reg.intercept_).reshape(-1)[0]),
                  "upstream_result": upstream_result,
                  "upstream_commit": subprocess.check_output(
                      ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()}
        (work / "result.json").write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
    except BaseException:
        (work / "worker_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("Invoke this algorithm through bench_sr_agent.py --algorithm my_igsr")
    _worker(Path(sys.argv[2]), Path(sys.argv[3]))
