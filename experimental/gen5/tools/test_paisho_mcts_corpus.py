import argparse
import json
from pathlib import Path
import tempfile
import os
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
import paisho_mcts_corpus as corpus


class CorpusTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.binary = self.root / 'paisho-mcts-corpus'
        self.binary.write_text('''#!/usr/bin/env python3
import json,sys,time
from pathlib import Path
out=Path(sys.argv[5]);out.mkdir()
if int(sys.argv[3]) == 21: time.sleep(5)
(out/'game.psr').write_text('test fixture')
(out/'result.json').write_text(json.dumps({'terminal':True,'decisions':30,'cuts':[0,10,20]}))
''')
        self.binary.chmod(0o755)
        self.args = argparse.Namespace(output=self.root/'corpus', deadline=1.0,
            games=3, workers=2, simulations=32, batch=64, wait_us=0,
            source_limit=100, max_attempts=8, first_id=20, backend='cpu')

    def tearDown(self):
        self.temp.cleanup()

    def run_corpus(self):
        with patch.object(corpus, 'BINARY', self.binary):
            corpus.generate(self.args)

    def test_deadline_replacement_exact_quota_and_resume_repairs_summary(self):
        self.run_corpus()
        output = self.args.output
        rows = corpus.load_receipts(output)
        self.assertEqual(sum(r['status']=='accepted' for r in rows),3)
        self.assertEqual(sum(r['status']=='timeout' for r in rows),1)
        before = {p: p.read_bytes() for p in output.glob('attempts/*/receipt.json')}
        (output/'summary.json').unlink()
        self.run_corpus()
        self.assertTrue(json.loads((output/'summary.json').read_text())['complete'])
        self.assertEqual(before,{p:p.read_bytes() for p in output.glob('attempts/*/receipt.json')})

    def test_corruption_and_plan_changes_refused(self):
        self.run_corpus()
        self.args.simulations=8
        with self.assertRaisesRegex(ValueError,'settings differ'):
            self.run_corpus()
        self.args.simulations=32
        game=next(self.args.output.glob('attempts/*/game.psr'))
        game.write_text('corrupt')
        with self.assertRaisesRegex(ValueError,'corrupt corpus'):
            self.run_corpus()

    def test_interrupted_uncommitted_attempt_not_reused(self):
        self.run_corpus()
        path=self.args.output/'attempts'/'00000027'
        path.mkdir()
        self.run_corpus()
        self.assertFalse((path/'receipt.json').exists())

    def test_default_cpu32_deadline_preserves_legacy_resume_and_explicit_override(self):
        self.args.deadline=None
        self.assertEqual(corpus.game_deadline(self.args),8)
        self.args.output.mkdir()
        (self.args.output/'plan.json').write_text(json.dumps({'settings':{'deadline_seconds':50}}))
        self.assertEqual(corpus.game_deadline(self.args),50)
        self.args.deadline=12
        self.assertEqual(corpus.game_deadline(self.args),12)

    def test_sigterm_reaps_running_game_processes(self):
        self.binary.write_text("#!/usr/bin/env python3\nimport os,sys,time\nfrom pathlib import Path\nPath(sys.argv[5]).parent.joinpath('worker.pid').write_text(str(os.getpid()))\ntime.sleep(30)\n")
        self.args.deadline=50
        args_repr=repr(vars(self.args)).replace("PosixPath(","Path(")
        code=(f"import sys;sys.path.insert(0,{str(Path(corpus.__file__).parent)!r});"
              "from pathlib import Path;import argparse;import paisho_mcts_corpus as c;"
              f"c.BINARY=Path({str(self.binary)!r});c.generate(argparse.Namespace(**{args_repr}))")
        with (self.root/'parent.log').open('w') as log:
            parent=subprocess.Popen([sys.executable,'-c',code],stdout=log,stderr=log)
            try:
                end=time.monotonic()+10
                while not list(self.args.output.glob('attempts/*/worker.pid')):
                    if parent.poll() is not None or time.monotonic()>end:
                        self.fail('worker did not start: '+(self.root/'parent.log').read_text())
                    time.sleep(0.05)
                pids=[int(p.read_text()) for p in self.args.output.glob('attempts/*/worker.pid')]
                parent.send_signal(signal.SIGTERM)
                self.assertNotEqual(parent.wait(timeout=5),0)
                for pid in pids:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid,0)
            finally:
                if parent.poll() is None:
                    parent.kill();parent.wait()


if __name__=='__main__':
    unittest.main()
