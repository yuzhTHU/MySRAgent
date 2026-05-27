"""
Raw SR-Scientist runner for llmsr_bench.

Unlike ``sr_scientist.py``, this module does not reimplement the SR-Scientist
search loop.  It serializes one ``SEDTask`` into the parquet format expected by
``third-party/sr_scientist/inference/infer/inference.py`` and calls
``run_inference_batch`` directly.

This keeps SR-Scientist's original assumptions: an OpenAI-compatible model API
and one or more SandboxFusion ``/run_code`` endpoints must be available.

## 安装依赖
# cd third-party/
# git clone git@github.com:GAIR-NLP/SR-Scientist.git sr_scientist
# git clone git@github.com:bytedance/SandboxFusion.git SandboxFusion
# cd SandboxFusion
# conda create -n sandbox-runtime python=3.11  # 注：sandbox-runtime 这个名称被硬编码在 SandboxFusion 中，不宜自行修改！
# conda activate sandbox-runtime
# pip install poetry
# poetry install
# mkdir -p docs/build
# cd runtime/python && pip install -r requirements.txt && cd ../..

## 启动 SandBox
# cd ./third-party/SandboxFusion
# conda activate sandbox-runtime
# make run-online HOST=127.0.0.1 PORT=8080

## 测试 SandBox
# curl 'http://127.0.0.1:8080/run_code' -H 'Content-Type: application/json' --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'

## 启动多个 SandBox 并传给 bench_sr_agent.py
# cd ./third-party/SandboxFusion
# conda activate sandbox-runtime
# make run-online HOST=127.0.0.1 PORT=9010 &
# make run-online HOST=127.0.0.1 PORT=9020 &
# make run-online HOST=127.0.0.1 PORT=9070 &
# make run-online HOST=127.0.0.1 PORT=9080 &
# cd ../../
# python bench_sr_agent.py \
# --algorithm sr_scientist \
# --datasets lsrtransform \
# --llm_provider openrouter \
# --llm_model "deepseek/deepseek-v4-flash" \
# --sr_scientist_sandbox_urls \
#     http://127.0.0.1:9010/run_code \
#     http://127.0.0.1:9020/run_code \
#     http://127.0.0.1:9030/run_code \
#     http://127.0.0.1:9040/run_code \
#     http://127.0.0.1:9050/run_code \
#     http://127.0.0.1:9060/run_code \
#     http://127.0.0.1:9070/run_code \
#     http://127.0.0.1:9080/run_code

## 关闭 SandBox
# pkill -f "sandbox.server.server:app"

By default this module checks the requested sandbox URLs once during import and
tries to auto-start missing local SandboxFusion endpoints. Set
SR_SCIENTIST_RAW_AUTO_SANDBOX=0 to disable auto-start, or set
SR_SCIENTIST_RAW_SANDBOX_URLS to choose default URLs when command-line URLs are
not available yet.
"""
from __future__ import annotations

import os
import re
import sys
import ast
import json
import time
import socket
import logging
import asyncio
import subprocess
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
from sr_agent._vendor.llmsr_bench.core import SEDTask, SRResult

_logger = logging.getLogger(f"sr_agent.{__name__}")
_ROOT = Path(__file__).resolve().parents[5]
_SANDBOX_FUSION_ROOT = _ROOT / "third-party" / "SandboxFusion"
_SR_SCIENTIST_INFERENCE = _ROOT / "third-party" / "sr_scientist" / "inference"
if _SR_SCIENTIST_INFERENCE.exists():
    sys.path.insert(0, str(_SR_SCIENTIST_INFERENCE))
else:
    raise SystemExit(
        f"SR-Scientist inference code not found at {_SR_SCIENTIST_INFERENCE!r}. Please clone the SR-Scientist repository into third-party/ and try again."
        f"Follow {__file__!r} docstring for detailed instructions."
    )

try:
    from infer.inference import run_inference_batch
except Exception as exc:
    run_inference_batch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _parse_sandbox_urls_from_argv() -> List[str]:
    urls: List[str] = []
    argv = sys.argv[1:]
    for idx, arg in enumerate(argv):
        if arg == "--sr_scientist_sandbox_urls":
            for value in argv[idx + 1:]:
                if value.startswith("-"):
                    break
                urls.append(value)
            break
        if arg.startswith("--sr_scientist_sandbox_urls="):
            urls.extend(value for value in arg.split("=", 1)[1].split(",") if value)
            break
    if not urls:
        urls = os.environ.get("SR_SCIENTIST_RAW_SANDBOX_URLS", "").split()
    return urls or ["http://127.0.0.1:8080/run_code"]


def _endpoint_host_port(url: str) -> Tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def _port_is_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _start_sandboxfusion_endpoint(host: str, port: int) -> bool:
    if not _SANDBOX_FUSION_ROOT.exists():
        _logger.warning(
            f"SandboxFusion is not installed in third-party/SandboxFusion, or conda env sandbox-runtime is missing. "
            f"Install it according to {__file__}'s module docstring."
        )
        return False

    log_dir = _ROOT / "logs" / "sr_scientist" / "sandbox"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sandboxfusion_{port}.log"
    log_file = open(log_path, "ab")
    env = os.environ.copy()
    venv_python = _SANDBOX_FUSION_ROOT / "venv" / "bin" / "python"
    if venv_python.exists():
        env.setdefault("SANDBOX_PYTHON", str(venv_python))
        cmd = [
            str(venv_python),
            "-m",
            "uvicorn",
            "sandbox.server.server:app",
            "--host",
            host,
            "--port",
            str(port),
        ]
    else:
        cmd = [
            "bash",
            "-lc",
            (
                "conda run -n sandbox-runtime python -m uvicorn "
                f"sandbox.server.server:app --host {host} --port {port}"
            ),
        ]
    try:
        subprocess.Popen(
            cmd,
            cwd=str(_SANDBOX_FUSION_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as exc:
        _logger.warning(f"Failed to auto-start SandboxFusion on {host}:{port}: {exc}. Log: {log_path}")
        log_file.close()
        return False

    for _ in range(80):
        if _port_is_open(host, port, timeout=0.1):
            _logger.info(f"Auto-started SandboxFusion at http://{host}:{port}/run_code. Log: {log_path}")
            return True
        time.sleep(0.25)
    _logger.warning(f"SandboxFusion did not become ready on {host}:{port}. Check log: {log_path}")
    return False


def _ensure_sandboxfusion_on_import():
    if os.environ.get("SR_SCIENTIST_RAW_AUTO_SANDBOX", "1").lower() in {"0", "false", "no", "off"}:
        return

    seen = set()
    for url in _parse_sandbox_urls_from_argv():
        host, port = _endpoint_host_port(url)
        key = (host, port)
        if key in seen:
            continue
        seen.add(key)
        if _port_is_open(host, port):
            continue
        _start_sandboxfusion_endpoint(host, port)


_ensure_sandboxfusion_on_import()


def update_parser(parser):
    parser.add_argument("--llm_provider", default="openrouter", help="LLM provider name used to infer model URL and API key.")
    parser.add_argument("--llm_model", default="qwen/qwen3.6-plus", help="Model name passed to SR-Scientist.")
    parser.add_argument("--sr_scientist_model_url", default=None, help="OpenAI-compatible model API base URL.")
    parser.add_argument("--sr_scientist_api_key", default=None, help="API key override.")
    parser.add_argument("--sr_scientist_sandbox_urls", nargs="+", default=["http://127.0.0.1:8080/run_code"], help="SandboxFusion /run_code endpoints used by original SR-Scientist tools.")
    parser.add_argument("--sr_scientist_mape_threshold", type=float, default=0.001, help="Initial MAPE target.")
    parser.add_argument("--sr_scientist_num_turns", type=int, default=3, help="Original SR-Scientist --num-turns.")
    parser.add_argument("--sr_scientist_top_k", type=int, default=3, help="Original SR-Scientist --top-k.")
    parser.add_argument("--sr_scientist_max_assistant_turns", type=int, default=100, help="Original SR-Scientist --max-assistant-turns.")
    return parser


def _provider_defaults(provider: str) -> Tuple[str, str]:
    provider = provider.lower()
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"
    if provider == "deepseek":
        return "https://api.deepseek.com", "DEEPSEEK_API_KEY"
    if provider == "openai":
        return "https://api.openai.com/v1", "OPENAI_API_KEY"
    if provider == "siliconflow":
        return "https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"
    raise ValueError(
        f"Unsupported llm_provider={provider!r}. Pass --sr_scientist_model_url "
        "and --sr_scientist_api_key for custom OpenAI-compatible endpoints."
    )


def _model_connection(args) -> Tuple[str, str]:
    load_dotenv()
    if args.sr_scientist_model_url:
        model_url = args.sr_scientist_model_url
        env_key = None
    else:
        model_url, env_key = _provider_defaults(args.llm_provider)
    api_key = args.sr_scientist_api_key or (os.environ.get(env_key) if env_key else None)
    if not api_key:
        _logger.warning(f"Missing API key. Set {env_key} or pass --sr_scientist_api_key.")
    return model_url, api_key


def _task_samples(task: SEDTask) -> Dict[str, Any]:
    train = np.column_stack([task.train_y, task.train_X]).astype(float).tolist()
    # The llmsr_bench algorithm interface only gives train data.  Original
    # SR-Scientist requires a test split for final aggregation, but selects the
    # submitted equation by train MSE.  Reusing train here preserves that choice;
    # bench_sr_agent will perform the real ID/OOD evaluation afterwards.
    return {"train": train, "test": train}


def _write_single_task_parquet(task: SEDTask, path: Path):
    row = {
        "dataset_identifier": "llmsr_bench",
        "expression": "",
        "symbols": task.symbols,
        "symbol_descs": task.symbol_descs,
        "symbol_properties": task.symbol_properties,
        "samples": _task_samples(task),
    }
    df = pd.DataFrame([row], index=[task.name])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _clean_code_block(raw_code: str) -> str:
    raw_code = (raw_code or "").strip()
    if "```python" in raw_code:
        return raw_code.split("```python")[-1].split("```")[0].strip()
    if "```" in raw_code:
        parts = raw_code.split("```")
        if len(parts) > 1:
            return parts[1].strip()
    return raw_code


def _compile_equation(equation_code: str):
    scope: Dict[str, Any] = {}
    exec(_clean_code_block(equation_code), {"np": np, "numpy": np}, scope)
    equation = scope.get("equation")
    if not callable(equation):
        raise ValueError("SR-Scientist result did not define equation(..., params).")
    return equation


def _expression_from_equation(equation_code: str, params: Any, task: SEDTask) -> str:
    params = params or []
    code = _clean_code_block(equation_code)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "equation"), None)
    if fn is None:
        return code

    assignments: Dict[str, ast.AST] = {}
    ret_expr = None
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            assignments[stmt.targets[0].id] = stmt.value
        elif isinstance(stmt, ast.Return):
            ret_expr = stmt.value
    if ret_expr is None:
        return code
    while isinstance(ret_expr, ast.Name) and ret_expr.id in assignments:
        ret_expr = assignments[ret_expr.id]

    expr = ast.unparse(ret_expr)
    for idx, value in enumerate(params):
        if value is not None:
            expr = re.sub(rf"\bparams\s*\[\s*{idx}\s*\]", f"({float(value):.17g})", expr)

    for original in task.symbols[1:]:
        cleaned = original.strip("$").strip("\\").replace(" ", "_").replace("text", "")
        if cleaned != original:
            expr = re.sub(rf"\b{re.escape(cleaned)}\b", lambda _: original, expr)
    return expr.replace("np.", "")


def _make_predict(equation_code: str, params: Any):
    equation = _compile_equation(equation_code)
    params_arr = np.asarray(params, dtype=float)

    def predict(X: np.ndarray) -> np.ndarray:
        try:
            y_pred = np.asarray(equation(*X.T, params_arr), dtype=float)
            if y_pred.shape == ():
                y_pred = np.full(X.shape[0], float(y_pred))
            return y_pred.reshape(-1)
        except Exception:
            return np.full(X.shape[0], np.nan)

    return predict


def _load_raw_result(output_json_path: Path, task: SEDTask) -> Dict[str, Any]:
    records = json.loads(output_json_path.read_text(encoding="utf-8"))
    if not records:
        raise RuntimeError("Original SR-Scientist produced an empty results file.")
    for record in records:
        if str(record.get("index")) == str(task.name):
            return record["submitted_results"]
    return records[0]["submitted_results"]


def run(args, task: SEDTask) -> SRResult:
    if run_inference_batch is None:
        raise RuntimeError(f"Could not import original SR-Scientist inference code from {_SR_SCIENTIST_INFERENCE}: {_IMPORT_ERROR!r}")

    work_dir = Path(tempfile.gettempdir())
    input_path = work_dir / "input.parquet"
    output_path = work_dir / "output.json"
    _logger.note(
        f"[{task.name}] Running original SR-Scientist inference: "
        f"num_turns={args.sr_scientist_num_turns}, top_k={args.sr_scientist_top_k}, "
        f"max_assistant_turns={args.sr_scientist_max_assistant_turns}"
    )
    
    try:
        model_url, api_key = _model_connection(args)
        _write_single_task_parquet(task, input_path)

        asyncio.run(
            run_inference_batch(
                model_name=args.llm_model,
                model_url=model_url,
                api_key=api_key,
                sandbox_urls=args.sr_scientist_sandbox_urls,
                parquet_file_path=str(input_path),
                source=None,
                mape_threshold=args.sr_scientist_mape_threshold,
                num_turns=args.sr_scientist_num_turns,
                top_k=args.sr_scientist_top_k,
                output_json_path=str(output_path),
                max_assistant_turns=args.sr_scientist_max_assistant_turns,
            )
        )

        submitted = _load_raw_result(output_path, task)
        equation = submitted.get("equation")
        params = submitted.get("optimized_params")
        if not equation or params is None:
            raise RuntimeError(f"Original SR-Scientist did not submit a usable equation: {submitted}")

        expression = _expression_from_equation(equation, params, task)
        _logger.note(f"[{task.name}] Original SR-Scientist submitted expression:\n{expression}")
    finally:
        if input_path.exists():
            input_path.unlink()
        if output_path.exists():
            output_path.unlink()

    return SRResult(predict=_make_predict(equation, params), expression=expression)
