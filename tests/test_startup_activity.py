import ast
import io
import runpy
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).parents[1] / 'src/unirobosim_isaaclab'
Activity = runpy.run_path(str(ROOT / 'startup_activity.py'))['KitStartupActivity']
def line(ms, seconds):
    return f'[Info] [{ms:,}ms] [omni.rtx] *** Waiting for RtPso async group async compilation: {seconds} seconds so far\n'

def test_only_new_increasing_activity(tmp_path):
    p = tmp_path / 'worker.log';p.write_text(line(5000, 5));a = Activity(p)
    assert not a.poll()
    with p.open('a') as f:f.write(line(10000, 10))
    assert a.poll()
    assert not a.poll()
    with p.open('a') as f:f.write(line(10000, 10) + line(11000, 9) + 'unrelated warning\n')
    assert not a.poll()
    with p.open('a') as f:f.write(line(15000, 15))
    assert a.poll()

def test_other_worker_and_partial_line(tmp_path):
    p = tmp_path / 'one';p.touch();q = tmp_path / 'two';q.touch();a = Activity(p)
    q.write_text(line(10000, 10));assert not a.poll()
    text = line(10000, 10);p.write_text(text[:-1]);assert not a.poll()
    with p.open('a') as f:f.write('\n')
    assert a.poll()

def test_missing_log_no_activity(tmp_path):
    p=tmp_path/'log';p.touch();a=Activity(p);p.unlink();assert not a.poll()

class Timeout(Exception): pass
class Error(Exception): pass
class Progress:
    def __init__(self, phase): self.phase=phase

@pytest.mark.parametrize('mode, expected', [('silent','idle'), ('active','hard'), ('ready','ready')])
def test_real_startup_supervisor_idle_and_hard_deadlines(mode, expected):
    tree=ast.parse((ROOT/'worker.py').read_text());method=next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name=='_receive_startup')
    method.returns=None
    method.args.args[1].annotation=None
    clock=SimpleNamespace(now=0.0);phases=('kit_launching','kit_ready')
    class Probe:
        def poll(self):return mode != 'silent'
    class Fake:
        _process=SimpleNamespace(startup_activity=Probe())
        phase=0
        def _receive(self, operation, *, timeout_seconds, allow_startup_progress):
            if self.phase==0:self.phase=1;return Progress('kit_launching')
            if mode=='ready' and clock.now>=4:
                if self.phase==1:self.phase=2;return Progress('kit_ready')
                return 'ready'
            clock.now += timeout_seconds
            raise Timeout()
    ns={'Any':object,'IsaacLabAdapterConfig':object,'time':SimpleNamespace(monotonic=lambda:clock.now),'_STARTUP_PHASE_IDLE_TIMEOUT_SECONDS':{'process_spawned':2,'kit_ready':2},'_STARTUP_PHASES':phases,'_WorkerStartupProgress':Progress,'NativeWorkerError':Error,'_NativeWorkerTimeout':Timeout}
    exec(compile(ast.Module(body=[method],type_ignores=[]), 'worker.py', 'exec'),ns)
    config=SimpleNamespace(worker_startup_hard_timeout_s=6,worker_kit_launch_idle_timeout_s=2)
    if expected=='ready':
        assert ns['_receive_startup'](Fake(),config)=='ready';assert clock.now==4
    else:
        with pytest.raises(Timeout,match=expected+' limit'):ns['_receive_startup'](Fake(),config)
        assert clock.now == (2 if expected=='idle' else 6)
