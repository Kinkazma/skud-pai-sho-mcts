import tempfile
import unittest
import sys
import json
from pathlib import Path
from paisho_micro_campaign import command, execute, write
class CampaignTests(unittest.TestCase):
    def test_command_is_bounded_with_one_compute_pool_and_history(self):
        c=command(Path('/tmp/run'),36000,10)
        self.assertEqual(c[c.index('--seconds')+1],'36000')
        self.assertEqual(c[c.index('--threads')+1],'10')
        self.assertEqual(c[c.index('--workers')+1],'10')
        self.assertIn('--history-interval',c)
        self.assertNotIn('--replay',c)
        c = command(Path('/tmp/run'), 36000, 10, with_replay=True)
        self.assertEqual(c[c.index('--replay')+1], '/tmp/run/initial-replay.json')
    def test_changed_binary_refused_before_start(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'binary').write_text('changed')
            write(out/'plan.json',{'hashes':{'binary':'wrong'}})
            with self.assertRaisesRegex(ValueError,'frozen input changed'):
                execute(out)
            self.assertFalse((out/'started.json').exists())
    def test_exit_status_and_duplicate_start(self):
        for code in [0, 7]:
            with tempfile.TemporaryDirectory() as d:
                out=Path(d)
                write(out/'plan.json',{'hashes':{},'seconds':1,'command':[sys.executable,'-c',f'raise SystemExit({code})']})
                self.assertEqual(execute(out),code)
                status=json.loads((out/'status.json').read_text())
                self.assertEqual(status['state'],'completed' if code==0 else 'failed')
                with self.assertRaises(FileExistsError): execute(out)
if __name__=='__main__':unittest.main()
