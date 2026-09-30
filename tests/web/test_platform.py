import threading
import time

import pytest

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient

from sr_harness.api.core import LLMResult, ToolCall
from sr_harness.api.llm_api import LLMAPI
from sr_harness.web.app import create_app
from sr_harness.web.interaction import InteractionController
from sr_harness.web.session import InteractiveSession


@pytest.fixture
def platform(tmp_path):
    session = InteractiveSession(tmp_path, agent_options={'tools': ['evaluate_formula', 'workspace_shell']})
    with TestClient(create_app(tmp_path, controller=session.controller, session=session)) as client:
        yield client, session


def test_workspace_roundtrip_and_boundaries(platform, tmp_path):
    client, session = platform
    assert client.get('/').status_code == 200
    assert client.get('/viewer').status_code == 200
    assert client.put('/api/workspace/upload?path=data/sample.csv', content=b'x,y\n1,2').status_code == 200
    assert client.get('/api/workspace/download?path=data/sample.csv').content == b'x,y\n1,2'
    assert client.get('/api/workspace?path=data').json()['entries'][0]['name'] == 'sample.csv'
    demo = client.post('/api/data/demo').json()
    assert demo['path'] == 'demo.csv'
    assert 'demo.csv' in client.get('/api/data/csv-files').json()['files']
    preview = client.get('/api/data/preview', params={'path': 'demo.csv', 'rows': 5}).json()
    assert preview['columns'] == ['x1', 'x2', 'x3', 'y']
    assert preview['numeric'] == {'x1': True, 'x2': True, 'x3': False, 'y': True}
    full_preview = client.get('/api/data/preview', params={'path': 'demo.csv', 'rows': 100}).json()
    assert {row['x3'] for row in full_preview['rows']} == {'alpha', 'beta', 'gamma', 'delta'}
    assert len(preview['rows']) == 5
    assert preview['truncated']
    prompts = client.post('/api/data/prompts', json={
        'dataset': 'demo.csv', 'target': 'y', 'features': ['x1', 'x2'],
        'problem_description': 'Explain the curve.',
        'variable_descriptions': {'x1': 'first input', 'y': 'measured response'},
    })
    assert prompts.status_code == 200
    assert 'Symbolic Regression Agent' in prompts.json()['system_prompt']
    assert 'Explain the curve.' in prompts.json()['user_prompt']
    assert "Feature names: ['x1', 'x2']" in prompts.json()['user_prompt']
    assert '- x1: first input' in prompts.json()['user_prompt']
    assert '- y: measured response' in prompts.json()['user_prompt']
    assert client.post('/api/data/prompts', json={
        'dataset': 'demo.csv', 'target': 'y', 'features': 'x',
    }).status_code == 400
    assert client.put('/api/workspace/upload?path=data/sample.csv', content=b'overwrite').status_code == 409
    for path in ['../outside', '/tmp/outside']:
        assert client.put('/api/workspace/upload', params={'path': path}, content=b'no').status_code == 400
    outside = tmp_path / 'outside'
    outside.write_text('private')
    (session.workspace / 'link').symlink_to(outside)
    assert client.get('/api/workspace/download?path=link').status_code == 400
    assert client.post('/api/session/start', json={'max_refinement_depth': 0}).status_code == 400
    assert session.state == 'idle'
    assert client.post('/api/session/start', json={'dataset': 'missing.csv'}).status_code == 400


def test_question_reconnect_and_stop():
    controller = InteractionController()
    results = []
    thread = threading.Thread(target=lambda: results.append(controller.ask('Continue?')))
    thread.start()
    for _ in range(100):
        if controller.status()['questions']:
            break
        time.sleep(.01)
    question = next(iter(controller.status()['questions']))
    # Pending questions must survive event buffer eviction/browser reconnect.
    for _ in range(1001):
        controller.publish('test', {})
    controller.reply(question, 'yes')
    thread.join(2)
    assert results == ['yes']
    with pytest.raises(ValueError):
        controller.reply(question, 'duplicate')
    controller.command('message', 'guidance')
    controller.wait_until_running()
    assert controller.checkpoint() == ['guidance']
    controller.command('stop')
    with pytest.raises(KeyboardInterrupt):
        controller.checkpoint()


def test_real_search_loop_with_fake_llm(platform, monkeypatch):
    client, session = platform
    prompts = []
    models = []

    class FakeAPI:
        tool_description_json = []
        def __call__(self, prompt, **kwargs):
            prompts.append(prompt.copy())
            if len(prompts) == 1:
                session.configure({'llm_provider': 'openai', 'llm_model': 'test-next-model'})
            def generate():
                call = ToolCall('evaluate_formula', {'f': 'x**2 + 2*x + 1'}, id='test')
                message = {'role': 'assistant', 'content': 'Evaluate a quadratic.', 'reasoning': 'Model reasoning',
                           'tool_calls': [{'id': 'test', 'type': 'function', 'function': {
                               'name': call.name, 'arguments': '{"f":"x**2 + 2*x + 1"}'}}]}
                yield {'content': message['content'], 'tool_call': [call], 'message': message}
                return {'usage': {'token': {}, 'price': {}}, 'responses': []}
            return LLMResult(generate())

    def create(provider, model, **kwargs):
        models.append((provider, model))
        return FakeAPI()

    monkeypatch.setattr(LLMAPI, 'create', create)
    client.put('/api/workspace/upload?path=notes.txt', content=b'research')
    client.post('/api/control/command', json={'action': 'message', 'message': 'Prefer simple formulas'})
    response = client.post('/api/session/start', json={
        'max_refinement_depth': 2, 'problem_description': 'Find the formula',
        'system_prompt': 'Custom system prompt', 'user_prompt': 'Custom user prompt',
    })
    assert response.status_code == 200, response.text
    session.thread.join(20)
    assert not session.thread.is_alive()
    assert session.state == 'completed', session.result
    assert session.topk
    assert session.topk[0]['mse'] < 1e-20
    assert client.get('/api/workspace/download?path=notes.txt').content == b'research'
    assert len(prompts) == 2
    assert models[-1] == ('openai', 'test-next-model')
    assert session.settings['llm_model'] == 'test-next-model'
    assert prompts[0][:2] == [
        {'role': 'system', 'content': 'Custom system prompt'},
        {'role': 'user', 'content': 'Custom user prompt'},
    ]
    assert all(any(m.get('content') == 'Prefer simple formulas' for m in p) for p in prompts)
    kinds = [e['kind'] for e in session.controller.events()]
    assert all(k in kinds for k in ['context', 'assistant', 'tool_start', 'tool_result', 'topk', 'lifecycle'])
    runs = client.get('/api/runs').json()['runs']
    assert runs[0]['record_count'] == 2
    assert client.post('/api/session/start', json={}).status_code == 400


def test_csv_input_and_failure_state(platform, monkeypatch):
    from sr_harness.web.session import WebInteractiveAgent
    client, session = platform
    client.put('/api/workspace/upload?path=measurements.csv',
               content=(b'temperature,humidity,station,pressure\n'
                        b'1,10,a,2\n2,20,b,4\n3,30,c,6\n4,40,d,8\n5,50,e,10\n'))
    seen = {}
    def fail_run(self, X, y, description):
        seen.update(X=X, y=y, description=description)
        raise RuntimeError('test provider unavailable')
    monkeypatch.setattr(WebInteractiveAgent, 'run', fail_run)
    response = client.post('/api/session/start', json={
        'dataset': 'measurements.csv', 'target': 'pressure',
        'features': ['humidity'], 'prompt': 'Fit pressure'})
    assert response.status_code == 200
    session.thread.join(10)
    assert list(seen['X']) == ['humidity']
    assert list(seen['y']) == ['pressure']
    assert seen['description'] == 'Fit pressure'
    assert session.state == 'failed'
    assert client.get('/api/session').json()['result']['error'] == 'test provider unavailable'
    assert (session.run_dir / 'web_result.json').exists()


def test_current_tree_never_scans_history(platform, monkeypatch):
    import json
    from sr_harness.web import app as web_app
    client, session = platform
    def no_scan(*args):
        raise AssertionError('current run must not traverse historical logs')
    monkeypatch.setattr(web_app, '_iter_run_dirs', no_scan)
    # Before the first complete iteration, respond promptly with 404.
    assert client.get(f'/api/runs/{session.run_dir.name}/records').status_code == 404
    (session.run_dir / 'manifest.json').write_text('{}')
    record = {'seq': 1, 'node_id': 'node1', 'core': {'tool_names': []}, 'detail': {'prompt': []}}
    (session.run_dir / 'records.jsonl').write_text(json.dumps(record)+'\n')
    for key in [session.run_dir.name, web_app._run_key(session.run_dir.parent, session.run_dir)]:
        response = client.get(f'/api/runs/{key}/records')
        assert response.status_code == 200
        assert response.json()['records'][0]['node_id'] == 'node1'
        assert client.get(f'/api/runs/{key}/records?after_seq=1').json()['records'] == []


def test_discovery_skips_workspace(tmp_path):
    from sr_harness.web.app import _iter_run_dirs, _resolve_run_dir, _run_key
    run = tmp_path / 'experiment' / 'run'
    run.mkdir(parents=True)
    (run / 'manifest.json').write_text('{}')
    (run / 'records.jsonl').touch()
    nested = run / 'sr_workspace_test' / 'nested'
    nested.mkdir(parents=True)
    (nested / 'manifest.json').write_text('{}')
    (nested / 'records.jsonl').touch()
    assert list(_iter_run_dirs(tmp_path)) == [run]
    assert _resolve_run_dir(tmp_path, _run_key(tmp_path, run)) == run
    external = tmp_path.parent / 'external-run'
    external.mkdir(exist_ok=True)
    (external / 'manifest.json').write_text('{}')
    (external / 'records.jsonl').touch()
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        _resolve_run_dir(tmp_path, _run_key(tmp_path, external))


def test_model_wait_pause_ack_and_stop(platform, monkeypatch):
    client, session = platform
    entered, release = threading.Event(), threading.Event()
    class WaitingAPI:
        tool_description_json = []
        def __call__(self, prompt, **kwargs):
            def generate():
                entered.set()
                assert release.wait(5)
                yield from []
                return {'usage': {'token': {}, 'price': {}}, 'responses': []}
            return LLMResult(generate())
    monkeypatch.setattr(LLMAPI, 'create', lambda *args, **kwargs: WaitingAPI())
    try:
        client.post('/api/session/start', json={'max_refinement_depth': 2})
        assert entered.wait(5)
        status = client.get('/api/session').json()
        assert status['activity']['phase'] == 'model'
        assert status['activity']['coord']['L'] == 1
        assert status['activity']['model'] == session.settings['llm_model']
        assert status['activity']['since'] <= status['server_time']
        client.post('/api/control/command', json={'action': 'pause'})
        assert not client.get('/api/session').json()['waiting_at_boundary']
        release.set()
        deadline = time.monotonic() + 5
        while not session.controller.status()['waiting_at_boundary'] and time.monotonic() < deadline:
            time.sleep(.01)
        assert client.get('/api/session').json()['waiting_at_boundary']
    finally:
        release.set()
        session.controller.command('stop')
        if session.thread:
            session.thread.join(5)
    assert session.state == 'interrupted'
