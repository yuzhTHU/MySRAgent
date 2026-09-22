import argparse
import json
import sys
import subprocess

import numpy as np
import pytest

from sr_agent._vendor.llmsr_bench.algorithms import get_algorithm
from sr_agent._vendor.llmsr_bench.algorithms import functionevolve as fe
from sr_agent._vendor.llmsr_bench.core import SEDTask


def test_result_restores_symbols_and_predicts_unseen_points():
    result = fe._build_result({'expression': '2*x1**2 + 3*x2 + 4', 'train_nmse': 0},
                             ['y', 'P', 't'])
    assert 'P' in result.expression and 't' in result.expression
    np.testing.assert_allclose(result.predict(np.array([[2, 5], [-3, 1]])), [27, 25])
    assert result.predict(np.empty((0, 2))).shape == (0,)
    with pytest.raises(ValueError):
        result.predict(np.ones((3, 1)))


def test_constant_result_broadcasts_to_batch():
    result = fe._build_result({'expression': '4.25', 'train_nmse': 0.1}, ['y', 'x'])
    np.testing.assert_allclose(result.predict(np.zeros((3, 1))), [4.25] * 3)


def test_real_part_wrapper_is_removed_for_real_benchmark_inputs():
    result = fe._build_result({'expression': '2*exp(re(x1))', 'train_nmse': 0}, ['y', 'x'])
    assert 're(' not in result.expression
    np.testing.assert_allclose(result.predict(np.array([[0], [1]])), [2, 2*np.e])


def test_sympy_function_names_are_normalized_for_nd2py():
    result = fe._build_result({'expression': 'atan(x1) + Abs(x1) + Max(0, x1)',
                               'train_nmse': 0}, ['y', 'x'])
    assert 'arctan(' in result.expression
    assert 'abs(' in result.expression
    assert 'max(' in result.expression
    assert not any(name in result.expression for name in ('atan(', 'Abs(', 'Max('))


def test_signed_rational_power_matches_upstream_semantics():
    result = fe._build_result({'expression': 'x1**(1/3)',
        'prediction_expression': '_RealPow(x1, 1/3)', 'train_nmse': 0}, ['y', 'x'])
    np.testing.assert_allclose(result.predict(np.array([[-8], [0], [27]])), [-2, 0, 3])


@pytest.mark.parametrize('raw', [
    {}, {'expression': 'x1', 'train_nmse': float('inf')},
    {'expression': 'c0*x1', 'train_nmse': 0.1},
])
def test_invalid_candidates_are_rejected(raw):
    with pytest.raises(RuntimeError):
        fe._build_result(raw, ['y', 'x'])


def test_benchmark_dispatch_and_worker_contract(tmp_path, monkeypatch):
    parser = fe.update_parser(argparse.ArgumentParser())
    assert get_algorithm('functionevolve') is fe.run
    args = parser.parse_args([])
    args.save_path = str(tmp_path / 'artifacts')
    args.seed = 42
    args.functionevolve_repo = tmp_path / 'repo'
    (args.functionevolve_repo / 'src').mkdir(parents=True)
    (args.functionevolve_repo / 'src/search.py').touch()
    monkeypatch.setattr(fe, '_ROOT', tmp_path)
    (tmp_path / '.env').write_text('OPENROUTER_API_KEY=test-project-key\n')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-shell-key')

    class FakeProcess:
        def __init__(self, cmd, **kwargs):
            assert cmd[0] == sys.executable
            assert kwargs['env']['OPENROUTER_API_KEY'] == 'test-project-key'
            assert kwargs['start_new_session']
            from pathlib import Path
            work = Path(cmd[-1])
            request = json.loads((work / 'request.json').read_text())
            assert request['config']['model'] == 'qwen/qwen3.6-27b'
            assert not {'test_X', 'test_y', 'expression'} & request.keys()
            assert 'test-project-key' not in (work / 'request.json').read_text()
            with np.load(work / 'train.npz') as data:
                np.testing.assert_allclose(data['X'], [[1], [2]])
                assert set(data.files) == {'X', 'y'}
            (work / 'result.json').write_text(json.dumps(
                {'expression': '2*x1+3', 'train_nmse': 0}))

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(fe.subprocess, 'Popen', FakeProcess)
    task = SEDTask('test', ['y', 'P'], ['response', 'population'], ['O', 'V'],
                  np.array([[1], [2]]), np.array([5, 7]))
    result = fe.run(args, task)
    np.testing.assert_allclose(result.predict(np.array([[10]])), [23])


def test_timeout_kills_worker_process_group(tmp_path, monkeypatch):
    args = fe.update_parser(argparse.ArgumentParser()).parse_args([])
    args.save_path = str(tmp_path / 'artifacts')
    args.functionevolve_run_timeout = 1.0
    args.functionevolve_repo = tmp_path / 'repo'
    (args.functionevolve_repo / 'src').mkdir(parents=True)
    (args.functionevolve_repo / 'src/search.py').touch()
    monkeypatch.setattr(fe, '_ROOT', tmp_path)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    kills = []

    class TimedOutProcess:
        pid = 12345

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired('worker', timeout)
            return -9

    monkeypatch.setattr(fe.subprocess, 'Popen', lambda *a, **kw: TimedOutProcess())
    monkeypatch.setattr(fe.os, 'killpg', lambda pid, sig: kills.append((pid, sig)))
    task = SEDTask('timeout', ['y', 'x'], ['', ''], ['O', 'V'],
                  np.array([[1], [2]]), np.array([1, 2]))
    with pytest.raises(RuntimeError, match='run timeout'):
        fe.run(args, task)
    assert kills == [(12345, fe.signal.SIGKILL)]
