import json
import os
from pathlib import Path
import signal
import struct
import tempfile
import unittest
from unittest.mock import patch
from paisho_gen5_actions import Actions,read
from paisho_gen5_campaign import write
from paisho_gen5_capture import drain


class CaptureTests(unittest.TestCase):
    def setUp(self):
        stopped=patch('paisho_gen5_resident.stopped',return_value=True)
        stopped.start();self.addCleanup(stopped.stop)

    def clock_fixture(self,root):
        from paisho_gen5_resident import pause_ticket
        clock=root/'training.pause-clock.bin';clock.write_bytes(struct.pack('<Q',7_000_000_000))
        write(root/'training/resident-clock.json',{'schema':'paisho-resident-clock-v1','pid':42,'path':str(clock)})
        return clock,pause_ticket(root,42,read,100_000_000_000)

    def test_drain_excludes_pause_before_first_signal(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'training').mkdir();clock,ticket=self.clock_fixture(root)
            write(root/'status.json',{'state':'completed','returncode':0})
            write(root/'training/durable-progress.json',{'version':13,'updates':60,'completed':6})
            def signal_sent(pid,sig):
                self.assertEqual((pid,sig),(42,signal.SIGCONT))
                self.assertEqual(struct.unpack('<Q',clock.read_bytes())[0],57_000_000_000)
            with patch('paisho_gen5_resident.time.monotonic_ns',return_value=150_000_000_000),patch('paisho_gen5_capture.os.kill',side_effect=signal_sent):
                result=drain(root,42,read,Mock(side_effect=[42,42,None]),write,resident_pause=ticket,save_pause=lambda _:None)
            self.assertTrue(result['paused_time_preserved'])

    def test_timeout_rebases_ticket_and_retains_active_drain_time_on_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'training').mkdir();clock,ticket=self.clock_fixture(root);saved=[]
            with patch('paisho_gen5_resident.time.monotonic_ns',side_effect=[150_000_000_000,160_000_000_000]),patch('paisho_gen5_capture.os.kill') as kill:
                with self.assertRaisesRegex(ValueError,'still pending'):
                    drain(root,42,read,lambda _:42,write,timeout=0,resident_pause=ticket,save_pause=saved.append)
                self.assertEqual([c.args for c in kill.call_args_list],[(42,signal.SIGCONT),(42,signal.SIGSTOP)])
            self.assertEqual(saved[0]['monotonic_ns'],160_000_000_000)
            self.assertEqual(saved[0]['offset_ns'],57_000_000_000)
            from paisho_gen5_resident import exclude_pause
            with patch('paisho_gen5_resident.time.monotonic_ns',return_value=200_000_000_000):
                exclude_pause(root,42,saved[0],read,lambda _:42)
            # 7s prior + 50s first pause + 40s second pause. The 10s drain remains active.
            self.assertEqual(struct.unpack('<Q',clock.read_bytes())[0],97_000_000_000)

    def test_invalid_clock_pid_or_state_never_signals(self):
        from unittest.mock import Mock
        for fault in ['inode','pid','running','missing-ticket']:
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as d:
                root=Path(d);(root/'training').mkdir();clock,ticket=self.clock_fixture(root)
                if fault=='inode':clock.rename(root/'old-clock');clock.write_bytes(struct.pack('<Q',7_000_000_000))
                current=Mock(side_effect=[42,43]) if fault=='pid' else lambda _:42
                with patch('paisho_gen5_resident.time.monotonic_ns',return_value=150_000_000_000),patch('paisho_gen5_resident.stopped',return_value=fault!='running'),patch('paisho_gen5_capture.os.kill') as kill:
                    with self.assertRaises(ValueError):
                        drain(root,42,read,current,write,resident_pause=None if fault=='missing-ticket' else ticket,save_pause=lambda _:None)
                    kill.assert_not_called()
                self.assertFalse((root/'training/stop-request.json').exists())

    def test_failed_signal_retry_is_idempotent(self):
        from paisho_gen5_resident import exclude_pause
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'training').mkdir();clock,ticket=self.clock_fixture(root);saved=[]
            with patch('paisho_gen5_resident.time.monotonic_ns',return_value=150_000_000_000),patch('paisho_gen5_capture.os.kill',side_effect=OSError('signal failed')):
                with self.assertRaises(OSError):
                    drain(root,42,read,lambda _:42,write,resident_pause=ticket,save_pause=saved.append)
                exclude_pause(root,42,ticket,read,lambda _:42)
            self.assertEqual(struct.unpack('<Q',clock.read_bytes())[0],57_000_000_000)
            self.assertEqual(saved,[])

    def test_final_checkpoint_must_cover_progress_observed_in_ram(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'training').mkdir()
            write(root/'status.json',{'state':'completed','returncode':0})
            write(root/'training/progress.json',{'version':13,'updates':60,'completed':6})
            write(root/'training/durable-progress.json',{'version':12,'updates':40,'completed':5})
            from unittest.mock import Mock
            with patch('paisho_gen5_capture.os.kill') as kill:
                with self.assertRaisesRegex(ValueError,'older than resident'):
                    drain(root,42,read,Mock(side_effect=[42,None,None]),write)
            kill.assert_called_once_with(42,signal.SIGCONT)

    def test_drain_uses_final_native_commit_and_never_kills(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'training').mkdir()
            write(root/'status.json',{'state':'completed','returncode':0})
            final={'version':13,'updates':60,'completed':6};write(root/'training/durable-progress.json',final)
            with patch('paisho_gen5_capture.os.kill') as kill:
                from unittest.mock import Mock
                result=drain(root,42,read,Mock(side_effect=[42,None]),write)
            self.assertEqual(result['updates'],60);kill.assert_called_once_with(42,signal.SIGCONT)

    def test_timeout_suspends_same_native_without_discarding_memory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'training').mkdir()
            with patch('paisho_gen5_capture.os.kill') as kill:
                with self.assertRaisesRegex(ValueError,'still pending'):
                    drain(root,42,read,lambda _:42,write,timeout=0)
            self.assertEqual([c.args for c in kill.call_args_list],[(42,signal.SIGCONT),(42,signal.SIGSTOP)])

    def test_resume_prepares_after_capturing_the_twenty_new_updates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'campaign';(root/'training').mkdir(parents=True)
            write(root/'status.json',{'state':'running','training_pid':42})
            old={'version':12,'updates':40,'completed':5};new={'version':13,'updates':60,'completed':6}
            write(root/'training/durable-progress.json',old);write(root/'training/progress.json',new)
            write(root/'config.json',{'control_stop':True,'model':'old'})
            a=Actions(root);s=read(a.path);s.update(job_pid=os.getpid(),busy=True,paused=True,remaining_seconds=7200);a.save(s)
            order=[]
            def capture(*_,**kwargs):
                order.append('capture');write(root/'training/durable-progress.json',new)
                return {'pid':42,'kind':'cooperative-native-drain',**new}
            def prepare(source,out):
                order.append('prepare');self.assertEqual(read(source/'training/durable-progress.json'),new)
                out.mkdir();write(out/'receipt.json',{'model':'latest-60'});write(out/'resume-progress.json',new);write(out/'replay.index.json',{})
            def start(config,binary,out,*args,**kwargs):
                order.append('start');self.assertEqual(read(config)['model'],'latest-60');out.mkdir();return {}
            with patch('paisho_gen5_actions.native',return_value=42),patch('paisho_gen5_actions.subprocess.check_output',return_value='T'),patch('paisho_gen5_actions.capture_native',side_effect=capture),patch('paisho_gen5_actions.prepare',side_effect=prepare),patch('paisho_gen5_actions.start',side_effect=start),patch('paisho_gen5_actions.os.kill') as kill:
                a.continue_run(2);kill.assert_not_called()
            self.assertEqual(order,['capture','prepare','start'])
            self.assertEqual(read(a.path)['remaining_seconds'],7200)

    def test_failed_capture_preserves_latest_pause_ticket_and_never_starts(self):
        for renewed in [False,True]:
            with self.subTest(renewed=renewed),tempfile.TemporaryDirectory() as d:
                root=Path(d)/'campaign';(root/'training').mkdir(parents=True)
                write(root/'status.json',{'state':'running','training_pid':42})
                write(root/'config.json',{'control_stop':True,'model':'old'})
                a=Actions(root);state=read(a.path);ticket={'pid':42,'offset_ns':7,'monotonic_ns':100}
                state.update(job_pid=os.getpid(),busy=True,paused=True,resident_pause=ticket);a.save(state)
                new_ticket={**ticket,'offset_ns':57,'monotonic_ns':160}
                def capture(*_,**kwargs):
                    self.assertEqual(kwargs['resident_pause'],ticket)
                    if renewed:kwargs['save_pause'](new_ticket)
                    raise ValueError('capture failed')
                with patch('paisho_gen5_actions.native',return_value=42),patch('paisho_gen5_actions.subprocess.check_output',return_value='T'),patch('paisho_gen5_actions.capture_native',side_effect=capture),patch('paisho_gen5_actions.start') as start,patch('paisho_gen5_actions.os.kill') as kill:
                    with self.assertRaisesRegex(ValueError,'capture failed'):a.continue_run(2)
                    start.assert_not_called();kill.assert_not_called()
                final=read(a.path)
                self.assertTrue(final['paused']);self.assertFalse(final['busy'])
                self.assertEqual(final['resident_pause'],new_ticket if renewed else ticket)
