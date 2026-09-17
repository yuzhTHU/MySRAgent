"""Four-process FunctionEvolve benchmark launcher and billing/precision receipts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASKS = {
    'bio_pop_growth': ['BPG0', 'BPG4', 'BPG5', 'BPG6', 'BPG9', 'BPG10'],
    'chem_react': ['CRK6', 'CRK12', 'CRK13', 'CRK16', 'CRK20', 'CRK35'],
    'matsci': ['MatSci0', 'MatSci6', 'MatSci12', 'MatSci14', 'MatSci21', 'MatSci22'],
    'phys_osc': ['PO2', 'PO10', 'PO22', 'PO27', 'PO37', 'PO40'],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def summarize(output):
    records = []
    for path in sorted(output.glob('tasks/*/timing.json')):
        timing = json.loads(path.read_text())
        directory = path.parent
        receipts = []
        for audit in list(directory.glob('functionevolve/*/api_calls.jsonl')) + list(directory.glob('judge_api_calls.jsonl')):
            receipts.extend(json.loads(line) for line in audit.read_text().splitlines() if line.strip())
        search = [r for r in receipts if r['component'] != 'benchmark_judge']
        successful = [r for r in receipts if r.get('generation_id')]
        missing = [r['generation_id'] for r in successful if r.get('cost_usd') is None]
        result_paths = list(directory.glob('results/*.jsonl'))
        result = json.loads(result_paths[0].read_text().splitlines()[-1]) if result_paths else {}
        record = dict(timing, prompt_tokens=sum(r.get('usage', {}).get('prompt_tokens', 0) or 0 for r in receipts),
                      completion_tokens=sum(r.get('usage', {}).get('completion_tokens', 0) or 0 for r in receipts),
                      total_tokens=sum(r.get('usage', {}).get('total_tokens', 0) or 0 for r in receipts),
                      search_tokens=sum(r.get('usage', {}).get('total_tokens', 0) or 0 for r in search),
                      search_cost_usd=sum(r.get('cost_usd') or 0 for r in search),
                      total_recorded_cost_usd=sum(r.get('cost_usd') or 0 for r in receipts),
                      missing_cost_generation_ids=missing,
                      api_errors=sum(bool(r.get('error')) for r in receipts),
                      expression=result.get('discovered_expression'),
                      id_metrics=result.get('id_metrics'), ood_metrics=result.get('ood_metrics'),
                      symbolic_acc=result.get('symbolic_acc'), benchmark_error=result.get('error'))
        records.append(record)
    count = len(records)
    summary = dict(updated_at=now(), completed_tasks=count, tasks=records,
                   sum_task_wall_seconds=sum(r['wall_seconds'] for r in records),
                   sum_search_seconds=sum(r.get('search_seconds', 0) for r in records),
                   total_tokens=sum(r['total_tokens'] for r in records),
                   total_recorded_cost_usd=sum(r['total_recorded_cost_usd'] for r in records),
                   average_tokens_per_task=sum(r['total_tokens'] for r in records) / count if count else None,
                   average_search_tokens_per_task=sum(r['search_tokens'] for r in records) / count if count else None,
                   cost_complete=bool(count) and all(not r['missing_cost_generation_ids'] and not r['api_errors'] for r in records))
    controller = output / 'controller/timing.json'
    if controller.exists():
        summary['controller_timing'] = json.loads(controller.read_text())
    dump(output / 'summary.json', summary)
    return summary


def task(dataset, problem, output, pilot=False):
    import bench_sr_agent as bench
    from openai.resources.chat.completions import Completions
    from sr_agent._vendor.llmsr_bench.algorithms import functionevolve as fe

    directory = output / 'tasks' / problem
    directory.mkdir(parents=True, exist_ok=True)
    benchmark_args = ['bench_sr_agent.py', '--algorithm', 'functionevolve', '--datasets', dataset,
        '--problem_names', problem, '--llm_model', 'qwen/qwen3.6-27b', '--seed', '260917',
        '--save_dir', str(output / 'tasks'), '--exp_name', problem]
    if pilot:
        benchmark_args += ['--functionevolve_max_steps', '0', '--functionevolve_n_seeds', '2',
            '--functionevolve_timeout', '5', '--functionevolve_eval_workers', '2',
            '--functionevolve_max_tokens', '8192', '--functionevolve_reasoning', 'disabled']
    sys.argv = benchmark_args
    args = bench.build_argparser().parse_args()
    args.save_path = str(directory)
    args.command = shlex.join([sys.executable, *benchmark_args])
    bench.seed_all(args.seed)
    bench.setup_logging(info_level='info', exp_name=problem, save_path=directory / 'info.log', force=True)
    bench.save_args(args, directory / 'args.json')
    lock = threading.Lock()
    original_create = Completions.create

    def audited_judge(client, *positional, **kwargs):
        # Scope: this benchmark process only. Search runs in a separate subprocess.
        kwargs['extra_body'] = dict(kwargs.get('extra_body') or {},
                                    reasoning={'enabled': False}, usage={'include': True})
        started = time.monotonic()
        row = dict(timestamp=now(), component='benchmark_judge', model=kwargs.get('model'))
        try:
            response = original_create(client, *positional, **kwargs)
            usage = response.usage.model_dump() if response.usage else {}
            row.update(generation_id=response.id, usage=usage, cost_usd=usage.get('cost'), error=None)
            return response
        except Exception as exc:
            row.update(generation_id=None, usage={}, cost_usd=None, error=type(exc).__name__)
            raise
        finally:
            row['duration_s'] = time.monotonic() - started
            with lock, (directory / 'judge_api_calls.jsonl').open('a') as out:
                out.write(json.dumps(row) + '\n')

    Completions.create = audited_judge
    original_evaluate = bench.evaluate_problem
    search_times = []

    def archived_evaluate(evaluate_args, problem_data, sr_fn, exp_path):
        arrays = {split: values for split, values in problem_data.samples.items()}
        np.savez_compressed(directory / 'benchmark_samples.npz', **arrays)
        dump(directory / 'problem.json', dict(name=problem_data.equation_idx,
            symbols=problem_data.symbols, descriptions=problem_data.symbol_descs,
            gt_expression=problem_data.gt_expression.to_str(), dataset=dataset))

        def archived_run(run_args, sr_task):
            start = time.monotonic()
            result = sr_fn(run_args, sr_task)
            search_times.append(time.monotonic() - start)
            predictions = {split: result.predict(values[:, 1:]) for split, values in arrays.items()}
            np.savez_compressed(directory / 'predictions.npz', **predictions)
            raw_path = next(directory.glob('functionevolve/*/result.json'))
            raw = json.loads(raw_path.read_text())
            restored = fe._build_result(raw, list(problem_data.symbols))
            checks = {split: bool(np.array_equal(restored.predict(values[:, 1:]), predictions[split], equal_nan=True))
                      for split, values in arrays.items()}
            if not all(checks.values()):
                raise RuntimeError('Saved formula does not reproduce predictions exactly')
            dump(directory / 'precision_check.json', dict(predictions_roundtrip_exact=checks,
                parameter_float_hex=raw.get('parameter_float_hex'),
                expression=result.expression, canonical_expression=raw['expression'],
                prediction_expression=raw['prediction_expression']))
            return result

        return original_evaluate(evaluate_args, problem_data, archived_run, exp_path)

    bench.evaluate_problem = archived_evaluate
    started_at, started = now(), time.monotonic()
    status = 'completed'
    try:
        bench.main(args)
        result_paths = list(directory.glob('results/*.jsonl'))
        if not result_paths or 'error' in json.loads(result_paths[0].read_text().splitlines()[-1]):
            status = 'benchmark_error'
    except BaseException:
        status = 'failed'
        raise
    finally:
        dump(directory / 'timing.json', dict(problem=problem, dataset=dataset, status=status,
            started_at=started_at, finished_at=now(), wall_seconds=time.monotonic() - started,
            search_seconds=sum(search_times), command=benchmark_args))


def controller(output, workers):
    output.mkdir(parents=True, exist_ok=True)
    directory = output / 'controller'
    directory.mkdir(exist_ok=True)
    # Validate all exact IDs before submitting any paid search.
    import pandas as pd
    for dataset, names in TASKS.items():
        split = {'bio_pop_growth':'lsr_synth_bio_pop_growth', 'chem_react':'lsr_synth_chem_react',
                 'matsci':'lsr_synth_matsci', 'phys_osc':'lsr_synth_phys_osc'}[dataset]
        parquet = next((ROOT / 'data/llm-srbench-data/data').glob(split + '-*.parquet'))
        available = set(pd.read_parquet(parquet)['name'])
        if not set(names) <= available:
            raise ValueError(f'Missing IDs: {set(names) - available}')
    started_at, started = now(), time.monotonic()
    manifest = dict(started_at=started_at, workers=workers, tasks=TASKS, model='qwen/qwen3.6-27b',
        seed=260917, key_source='project .env loaded before Python',
        search_preset='upstream run.sh full + published Qwen temperature/token settings',
        deviations=['OpenRouter completion limit capped at 65536',
            'No GT baseline: upstream TreeSearch fixed train NMSE threshold 1e-11',
            'Judge reasoning disabled (search reasoning enabled)',
            'No per-equation total timeout; evaluation workers=16 per equation'],
        algorithm_defaults=__import__('sr_agent._vendor.llmsr_bench.algorithms.functionevolve',fromlist=['_DEFAULTS'])._DEFAULTS)
    dump(directory / 'manifest.json', manifest)
    sources = ['bench_sr_agent.py', 'scripts/functionevolve_experiment.py',
               'src/sr_agent/_vendor/llmsr_bench/algorithms/functionevolve.py',
               'baseline/repo/FunctionEvolve/src/llm_client.py']
    snapshot = {}
    for name in sources:
        data = (ROOT / name).read_bytes()
        destination = directory / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        snapshot[name] = hashlib.sha256(data).hexdigest()
    dump(directory / 'source_sha256.json', snapshot)
    dump(directory / 'upstream_version.json', dict(commit=subprocess.check_output(
        ['git', '-C', str(ROOT / 'baseline/repo/FunctionEvolve'), 'rev-parse', 'HEAD'], text=True).strip()))
    (directory / 'upstream_logging.patch').write_text(subprocess.check_output(
        ['git', '-C', str(ROOT / 'baseline/repo/FunctionEvolve'), 'diff'], text=True))
    import requests
    models = requests.get('https://openrouter.ai/api/v1/models', timeout=30)
    models.raise_for_status()
    dump(directory / 'model_pricing_snapshot.json', next(m for m in models.json()['data'] if m['id']==manifest['model']))

    def run_one(dataset, problem):
        task_started_at, task_started = now(), time.monotonic()
        log_dir = output / 'tasks' / problem
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), 'task', '--output', str(output),
               '--dataset', dataset, '--problem', problem]
        with (log_dir / 'console.log').open('w') as log:
            code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        return dict(problem=problem, dataset=dataset, returncode=code, started_at=task_started_at,
                    finished_at=now(), subprocess_wall_seconds=time.monotonic() - task_started)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, dataset, problem) for dataset, names in TASKS.items() for problem in names]
        for future in as_completed(futures):
            record = future.result()
            with (directory / 'progress.jsonl').open('a') as out:
                out.write(json.dumps(record) + '\n')
            summarize(output)
    dump(directory / 'timing.json', dict(started_at=started_at, finished_at=now(),
        wall_seconds=time.monotonic() - started, status='completed'))
    summarize(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['task', 'controller', 'summarize'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dataset')
    parser.add_argument('--problem')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode == 'controller':
        controller(output, args.workers)
    elif args.mode == 'task':
        task(args.dataset, args.problem, output, args.pilot)
    else:
        print(json.dumps(summarize(output), indent=2))


if __name__ == '__main__':
    main()
