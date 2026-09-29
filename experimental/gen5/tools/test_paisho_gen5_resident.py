import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import tempfile
import unittest
from unittest.mock import patch
import paisho_gen5_resident as r
from paisho_gen5_actions import read

class ResidentTests(unittest.TestCase):
    def fixture(self,d):
        p=Path(d);(p/'bin').mkdir();(p/'training').mkdir()
        (p/'bin/paisho-gen5').write_bytes(b'validated native')
        (p/'config.json').write_text('{}')
        hashes={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ('bin/paisho-gen5','config.json')}
        (p/'plan.json').write_text(json.dumps(dict(end_unix_seconds=500,hashes=hashes)))
        (p/'status.json').write_text(json.dumps(dict(training_pid=42,end_unix_seconds=500,state='running')))
        clock=p/'training.pause-clock.bin';clock.write_bytes(struct.pack('<Q',0))
        (p/'training/resident-clock.json').write_text(json.dumps(dict(schema='paisho-resident-clock-v1',pid=42,path=str(clock))))
        state=dict(paused=True,remaining_seconds=400,resident_pause=r.pause_ticket(p,42,read,100_000_000_000))
        return p,state,clock

    @patch.object(r,'stopped',return_value=True)
    @patch.object(r.time,'time',return_value=150)
    @patch.object(r.time,'monotonic_ns',return_value=150_000_000_000)
    def test_preserves_pid_ram_and_remaining_time(self,*_):
        with tempfile.TemporaryDirectory() as d:
            p,s,clock=self.fixture(d);saved=[]
            with patch.object(r.os,'kill') as kill:
                self.assertTrue(r.resume(s,p,None,read,lambda _:42,lambda v:saved.append(v.copy())))
                kill.assert_called_once_with(42,signal.SIGCONT)
            self.assertEqual(struct.unpack('<Q',clock.read_bytes())[0],50_000_000_000)
            self.assertEqual(s['remaining_seconds'],400)
            self.assertEqual(read(p/'status.json')['end_unix_seconds'],550)
            self.assertFalse(s['last_resume']['replay_reload']);self.assertTrue(saved)

    @patch.object(r,'stopped',return_value=True)
    @patch.object(r.time,'time',return_value=150)
    @patch.object(r.time,'monotonic_ns',return_value=150_000_000_000)
    def test_failed_signal_retry_does_not_double_count_pause(self,*_):
        with tempfile.TemporaryDirectory() as d:
            p,s,clock=self.fixture(d)
            with patch.object(r.os,'kill',side_effect=OSError('signal failed')):
                with self.assertRaises(OSError):r.resume(s,p,None,read,lambda _:42,lambda _:None)
            self.assertTrue(s['paused'])
            with patch.object(r.os,'kill'):r.resume(s,p,None,read,lambda _:42,lambda _:None)
            self.assertEqual(struct.unpack('<Q',clock.read_bytes())[0],50_000_000_000)
            self.assertEqual(s['remaining_seconds'],400)

    @patch.object(r,'stopped',return_value=True)
    def test_changed_runtime_preparation_or_process_requires_handoff(self,*_):
        with tempfile.TemporaryDirectory() as d:
            p,s,clock=self.fixture(d)
            for change in (dict(prepared_next={'binary':'next'}),dict(runtime={'sha256':'different'}),dict(resident_pause=None)):
                with patch.object(r.os,'kill') as kill:
                    self.assertFalse(r.resume(dict(s,**change),p,None,read,lambda _:42,lambda _:None))
                    kill.assert_not_called()
            self.assertIsNone(r.eligible(s,p,None,read,lambda _:43))
            self.assertEqual(clock.read_bytes(),bytes(8))
