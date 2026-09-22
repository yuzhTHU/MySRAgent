"""PySR benchmark adapter with persistent, full precision artifacts."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import traceback
from pathlib import Path

import numpy as np

from .pysr import best_expression
from ..core import SEDTask, SRResult

logger = logging.getLogger(__name__)


def update_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--pysr_timeout', type=float, default=1500)
    parser.add_argument('--pysr_niterations', type=int, default=1000000)
    parser.add_argument('--pysr_maxsize', type=int, default=30)
    parser.add_argument('--pysr_populations', type=int, default=31)
    parser.add_argument('--pysr_max_samples', type=int, default=0)
    parser.add_argument('--pysr_binary_operators', nargs='+', default=['+', '-', '*', '/'])
    parser.add_argument('--pysr_unary_operators', nargs='*', default=['sin', 'cos', 'exp', 'log', 'sqrt', 'square'])
    parser.add_argument('--pysr_model_selection', choices=['best', 'accuracy', 'score'], default='best')
    parser.add_argument('--pysr_parallelism', choices=['serial', 'multithreading', 'multiprocessing'], default='multithreading')
    parser.add_argument('--pysr_procs', type=int, default=4)
    return parser


def run(args: argparse.Namespace, task: SEDTask) -> SRResult:
    started = time.time()
    directory = Path(args.save_path).resolve() / 'pysr' / re.sub(r'[^\w.-]', '_', task.name)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {'task': task.name, 'symbols': task.symbols, 'seed': args.seed,
                'started_at': started, 'status': 'running', 'pid': os.getpid()}

    def save_metadata():
        temporary = directory / 'metadata.json.tmp'
        temporary.write_text(json.dumps(metadata, indent=2, default=str))
        temporary.replace(directory / 'metadata.json')

    save_metadata()
    try:
        os.environ.setdefault('PYTHON_JULIACALL_HANDLE_SIGNALS', 'yes')
        os.environ.setdefault('PYTHON_JULIACALL_THREADS', str(args.pysr_procs))
        os.environ.setdefault('JULIA_NUM_THREADS', str(args.pysr_procs))
        os.environ.setdefault('JULIA_NUM_GC_THREADS', '1')
        from pysr import PySRRegressor
        import pysr
        X, y = task.train_X, task.train_y
        indices = np.arange(len(y))
        if 0 < args.pysr_max_samples < len(y):
            indices = np.random.default_rng(args.seed).choice(len(y), args.pysr_max_samples, replace=False)
            X, y = X[indices], y[indices]
        np.save(directory / 'fit_indices.npy', indices)
        names = [f'x{i + 1}' for i in range(X.shape[1])]
        model = PySRRegressor(
            niterations=args.pysr_niterations, timeout_in_seconds=args.pysr_timeout,
            maxsize=args.pysr_maxsize, populations=args.pysr_populations,
            binary_operators=args.pysr_binary_operators, unary_operators=args.pysr_unary_operators,
            model_selection=args.pysr_model_selection, precision=64, print_precision=17,
            parallelism=args.pysr_parallelism, procs=args.pysr_procs, random_state=args.seed,
            output_directory=str(directory), run_id='search', temp_equation_file=False,
            progress=False, verbosity=1,
        )
        metadata.update(pysr_version=pysr.__version__, parameters=model.get_params())
        save_metadata()
        fit_started = time.time()
        model.fit(X, y, variable_names=names)
        fit_seconds = time.time() - fit_started
        expression = best_expression(model, names, task.symbols[1:])
        (directory / 'expression.txt').write_text(expression + '\n')
        model.equations_.drop(columns=['lambda_format'], errors='ignore').to_csv(
            directory / 'equations.csv', index=False, float_format='%.17g')
        selected = model.get_best()
        metadata.update(status='completed', expression=expression,
                        internal_expression=str(selected.equation), selected_index=int(selected.name),
                        fit_seconds=fit_seconds, finished_at=time.time(),
                        total_seconds=time.time() - started)
        save_metadata()
        logger.info('PySR %s: %s (fit %.3fs); artifacts: %s', task.name, expression, fit_seconds, directory)
    except BaseException:
        metadata.update(status='failed', finished_at=time.time(), total_seconds=time.time() - started,
                        error=traceback.format_exc())
        save_metadata()
        raise

    def predict(X):
        prediction = np.asarray(model.predict(X), dtype=np.float64).reshape(-1)
        # Benchmark calls ID then OOD; persist exact predictions for metric audits.
        path = directory / f'predictions_{predict.calls}.npy'
        np.save(path, prediction)
        predict.calls += 1
        return prediction

    predict.calls = 0
    return SRResult(predict=predict, expression=expression)
