"""Run the original FunctionEvolve search in an isolated subprocess.

Clone Phoinikas03/FunctionEvolve into baseline/repo/FunctionEvolve and install
its requirements in the selected Python environment. Example:

    python bench_sr_agent.py --algorithm functionevolve --datasets bio_pop_growth \
        --problem_names BPG5 --llm_model qwen/qwen3.6-27b

The default search uses 30 steps and 20 seeds. A small integration smoke test can
set --functionevolve_max_steps 0 --functionevolve_n_seeds 3. OpenRouter reasoning
is enabled by default. Defaults follow the upstream full/Qwen preset, with a
fixed training NMSE threshold (1e-11) rather than the unavailable GT baseline.

Only SEDTask training data and descriptions reach the upstream search. No GT
baseline, automatic GT threshold, or upstream symbolic verifier is invoked.
Artifacts are saved under save_path/functionevolve/<task>-<unique suffix>.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import functools
import json
import os
from pathlib import Path
import re
import re
import signal
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sr_agent._vendor.llmsr_bench.core import SEDTask, SRResult

_ROOT = Path(__file__).resolve().parents[5]
_DEFAULTS = {
    'max_steps': 30, 'n_seeds': 20, 'candidate_num': 5,
    'selector_context_size': 200, 'max_params': 10,
    'eval_workers': 16, 'timeout': 120.0, 'run_timeout': 0.0,
    'max_tokens': 110000, 'max_retries': 5, 'max_mature_nodes': 50,
    'mature_train_threshold': 1e-11, 'overfit_min_depth': 10,
    'temperature': 1.0,
}


def update_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--llm_provider', choices=['openrouter'], default='openrouter')
    parser.add_argument('--llm_model', default='qwen/qwen3.6-27b')
    parser.add_argument('--functionevolve_repo', type=Path,
                        default=_ROOT / 'baseline/repo/FunctionEvolve')
    parser.add_argument('--functionevolve_python', default=sys.executable)
    parser.add_argument('--functionevolve_optimizer', default='Structure',
                        choices=['Structure', 'DE', 'CMA-ES', 'L-BFGS-B', 'least_squares'])
    parser.add_argument('--functionevolve_reasoning', choices=['enabled', 'disabled', 'low'],
                        default='enabled', help='OpenRouter reasoning setting for all agents; low sets effort=low.')
    parser.add_argument('--functionevolve_resume', action='store_true',
                        help='Resume the newest existing task checkpoint under save_path.')
    for name, default in _DEFAULTS.items():
        parser.add_argument(f'--functionevolve_{name}', type=type(default), default=default)
    return parser


def _real_pow(base, exponent):
    """Match upstream optimizer.base._real_pow_np for signed rational powers."""
    base = np.asarray(base, dtype=float)
    numerator = int(round(float(exponent) * 3))
    value = np.power(np.abs(base), numerator / 3)
    return np.where(base < 0, -value, value) if numerator % 2 else value


def _build_result(result: dict, symbols: list[str]):
    import sympy as sp
    from sr_agent._vendor.llmsr_bench.core import SRResult

    if not result.get('expression') or not np.isfinite(result.get('train_nmse', np.inf)):
        raise RuntimeError('FunctionEvolve returned no finite fitted candidate.')
    names = [f'x{i + 1}' for i in range(len(symbols) - 1)]
    # Benchmark inputs are real-valued.  The upstream simplifier may emit
    # re(x1) when symbols lack assumptions; real symbols reduce that wrapper
    # before returning an expression to nd2py, which has no ``re`` callable.
    variables = [sp.Symbol(name, real=True) for name in names]
    local = dict(zip(names, variables))
    local['_RealPow'] = sp.Function('_RealPow')
    expr = sp.sympify(result['expression'], locals=local)
    prediction_expr = sp.sympify(result.get('prediction_expression', str(expr)), locals=local)
    unknown = (expr.free_symbols | prediction_expr.free_symbols) - set(variables)
    if unknown:
        raise RuntimeError(f'FunctionEvolve returned unresolved symbols: {unknown}')
    func = sp.lambdify(variables, prediction_expr, modules=[{'_RealPow': _real_pow}, 'numpy'])

    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(variables):
            raise ValueError(f'Expected X with shape (n, {len(variables)}), got {X.shape}')
        if not len(X):
            return np.empty(0, dtype=float)
        with np.errstate(all='ignore'):
            values = np.asarray(func(*(X[:, i] for i in range(len(variables)))), dtype=float)
        return np.broadcast_to(values, (len(X),)).copy()

    expr = expr.xreplace({var: sp.Symbol(name, real=True) for var, name in zip(variables, symbols[1:])})
    expression = str(expr)
    # nd2py uses NumPy-style inverse-trig names and lower-case abs/min/max.
    for source, target in {'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan',
                           'Abs': 'abs', 'Min': 'min', 'Max': 'max'}.items():
        expression = re.sub(rf'\b{source}(?=\()', target, expression)
    return SRResult(predict=predict, expression=expression)


def run(args: argparse.Namespace, task: SEDTask) -> SRResult:
    from dotenv import dotenv_values

    X = np.asarray(task.train_X, dtype=float)
    y = np.asarray(task.train_y, dtype=float).reshape(-1)
    if X.ndim != 2 or len(X) != len(y) or not len(y):
        raise ValueError('FunctionEvolve requires a nonempty training matrix and matching targets.')
    if len(task.symbols) != X.shape[1] + 1 or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Training symbols must match columns and training data must be finite.')
    repo = Path(getattr(args, 'functionevolve_repo', _ROOT / 'baseline/repo/FunctionEvolve')).resolve()
    if not (repo / 'src/search.py').is_file():
        raise FileNotFoundError(f'Clone https://github.com/Phoinikas03/FunctionEvolve into {repo}')
    env = os.environ.copy()
    # The project .env key takes precedence over shell startup configuration.
    key = dotenv_values(_ROOT / '.env').get('OPENROUTER_API_KEY') or env.get('OPENROUTER_API_KEY')
    if not key:
        raise ValueError('OPENROUTER_API_KEY is missing from the project .env/environment.')
    env['OPENROUTER_API_KEY'] = key
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    env.setdefault('LLM_REQUEST_TIMEOUT', '300')
    config = {name: getattr(args, f'functionevolve_{name}', default)
              for name, default in _DEFAULTS.items()}
    for name in ('n_seeds', 'candidate_num', 'eval_workers', 'max_params', 'max_tokens',
                 'max_retries', 'max_mature_nodes', 'selector_context_size', 'timeout'):
        if config[name] <= 0:
            raise ValueError(f'functionevolve_{name} must be positive.')
    if config['max_steps'] < 0 or config['mature_train_threshold'] < 0 or config['run_timeout'] < 0:
        raise ValueError('Search steps and maturity threshold must be nonnegative.')
    config.update(model=getattr(args, 'llm_model', 'qwen/qwen3.6-27b'),
                  optimizer=getattr(args, 'functionevolve_optimizer', 'Structure'),
                  reasoning=getattr(args, 'functionevolve_reasoning', 'enabled'),
                  resume=getattr(args, 'functionevolve_resume', False),
                  seed=getattr(args, 'seed', -1), verbose=getattr(args, 'verbose', False))
    if config['seed'] >= 0:
        env['PYTHONHASHSEED'] = str(config['seed'])
    root = Path(getattr(args, 'save_path', None) or _ROOT / 'logs/functionevolve') / 'functionevolve'
    root.mkdir(parents=True, exist_ok=True)
    prefix = re.sub(r'[^\w.-]', '_', task.name)[:80] + '-'
    existing = sorted(root.glob(prefix + '*'), key=lambda path: path.stat().st_mtime, reverse=True)
    resumable = next((path for path in existing if (path / 'checkpoint.json').is_file()), None)
    work = resumable.resolve() if config['resume'] and resumable else Path(
        tempfile.mkdtemp(prefix=prefix, dir=root)).resolve()
    np.savez(work / 'train.npz', X=X, y=y)
    (work / 'request.json').write_text(json.dumps(dict(config=config, name=task.name,
        symbols=list(task.symbols), descriptions=list(task.symbol_descs),
        properties=list(task.symbol_properties)), indent=2), encoding='utf-8')
    cmd = [str(getattr(args, 'functionevolve_python', sys.executable)), str(Path(__file__).resolve()),
           '--worker', str(repo), str(work)]
    log_path = work / 'worker.log'
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=config['run_timeout'] or None)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise RuntimeError(f'FunctionEvolve exceeded its run timeout. See {log_path}') from exc
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if returncode:
        raise RuntimeError(f'FunctionEvolve exited with status {returncode}. See {log_path}')
    result = json.loads((work / 'result.json').read_text(encoding='utf-8'))
    return _build_result(result, list(task.symbols))


def _worker(repo: Path, work: Path):
    # This script runs outside sr_agent imports; upstream owns the generic src namespace.
    sys.path.insert(0, str(repo))
    from src.dataset import SRDataset
    from src.evaluator import Evaluator
    from src.evolution_tree import EvolutionTree
    from src.generator import create_generator
    from src.selector import create_selector
    from src.mutator import LLMMutator
    from src.llm_client import build_openai_client, LLMUsageLogger
    from src.search import TreeSearch
    from src.checkpoint import load_checkpoint
    from src.optimizer.base import parse_expr, detect_rational_constrained, make_safe_expr
    import random
    import sympy as sp

    request = json.loads((work / 'request.json').read_text(encoding='utf-8'))
    os.environ['FUNCTIONEVOLVE_API_AUDIT'] = str(work / 'api_calls.jsonl')
    cfg = request['config']
    if cfg['seed'] >= 0:
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
    with np.load(work / 'train.npz') as data:
        X, y = data['X'], data['y']
    names = [f'x{i + 1}' for i in range(X.shape[1])]
    # Canonical names avoid collisions with SymPy functions and parameter names.
    ds = SRDataset.from_arrays(X, y, symbols=['y'] + names,
        symbol_descs=request['descriptions'], equation_name=request['name'])
    ds.symbol_properties = request['properties']
    usage = LLMUsageLogger(str(work / 'usage.csv'))
    llm = dict(model=cfg['model'], base_url='https://openrouter.ai/api/v1',
               api_key=os.environ['OPENROUTER_API_KEY'], max_tokens=cfg['max_tokens'],
               max_retries=cfg['max_retries'], usage_logger=usage,
               temperature=cfg['temperature'])
    search = None
    try:
        if cfg.get('resume') and (work / 'search.txt').is_file():
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            for name in ('search.txt', 'diagnostics.log'):
                source = work / name
                if source.is_file():
                    source.replace(work / f'{source.stem}.pre_resume_{stamp}{source.suffix}')
        generator = create_generator(**llm)
        selector = create_selector(**llm)
        mutator = LLMMutator(api_client=build_openai_client(llm['model'], llm['base_url'],
            api_key=llm['api_key']), model=llm['model'], max_tokens=cfg['max_tokens'],
            max_retries=cfg['max_retries'], usage_logger=usage, temperature=cfg['temperature'])
        reasoning = ({'effort': 'low'} if cfg['reasoning'] == 'low'
                     else {'enabled': cfg['reasoning'] == 'enabled'})
        for agent in (generator, selector, mutator):
            agent.api.chat.completions.create = functools.partial(agent.api.chat.completions.create,
                extra_body={'reasoning': reasoning,
                            'usage': {'include': True}})
            # The upstream budget is total tokens. OpenRouter caps completion tokens.
            original = agent.api.chat.completions.create
            def capped_create(_original=original, **kwargs):
                if kwargs.get('max_tokens') is not None:
                    kwargs['max_tokens'] = min(kwargs['max_tokens'], 65536)
                return _original(**kwargs)
            agent.api.chat.completions.create = capped_create
        search = TreeSearch(dataset=ds,
            evaluator=Evaluator(feature_names=names, X_train=X, y_train=y,
                                optimizer=cfg['optimizer'], timeout=cfg['timeout']),
            tree=EvolutionTree(), generator=generator, selector=selector, llm_mutator=mutator,
            max_steps=cfg['max_steps'], n_seeds=cfg['n_seeds'], candidate_num=cfg['candidate_num'],
            selector_context_size=cfg['selector_context_size'], max_params=cfg['max_params'],
            n_parent_workers=cfg['candidate_num'], n_eval_workers=cfg['eval_workers'],
            timeout=cfg['timeout'], max_mature_nodes=cfg['max_mature_nodes'],
            mature_train_threshold=cfg['mature_train_threshold'], mature_anneal_budget=0,
            overfit_min_depth=cfg['overfit_min_depth'], log_path=str(work / 'search.txt'), verbose=cfg['verbose'],
            checkpoint_path=str(work / 'checkpoint.json'),
            diagnostic_log_path=str(work / 'diagnostics.log'))
        if cfg.get('resume') and (work / 'checkpoint.json').is_file():
            search.load_checkpoint_state(load_checkpoint(work / 'checkpoint.json'))
        search.initialize_seeds()
        search.run()
        result = search.get_best_result()
        if not result.get('expression') or not np.isfinite(result.get('train_nmse', np.inf)):
            raise RuntimeError('Search produced no finite fitted candidate.')
        params = result['params']
        values = result['best_params']
        if len(params) != len(values):
            raise RuntimeError('Fitted parameter count does not match the skeleton.')
        expr = parse_expr(result['expression'], params, names)
        safe = make_safe_expr(expr, params, detect_rational_constrained(expr, params, names, X), names)
        replacements = {sp.Symbol(name): sp.Float(repr(float(value)), 17) for name, value in zip(params, values)}
        result['skeleton'] = result['expression']
        result['expression'] = str(expr.subs(replacements))
        result['prediction_expression'] = str(safe.subs(replacements))
        result['model'] = cfg['model']
        result['reasoning'] = cfg['reasoning']
        result['parameter_float_hex'] = {name: float(value).hex() for name, value in zip(params, values)}
        result['key_source'] = 'project .env/environment'
        result['upstream_commit'] = subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        (work / 'result.json').write_text(json.dumps(result, default=str, indent=2), encoding='utf-8')
    finally:
        if search is not None:
            search.close_log()
        usage.close()


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--worker':
        raise SystemExit('Invoke this algorithm through bench_sr_agent.py --algorithm functionevolve')
    _worker(Path(sys.argv[2]), Path(sys.argv[3]))
