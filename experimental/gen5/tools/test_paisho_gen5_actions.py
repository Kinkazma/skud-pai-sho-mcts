import json
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import Mock,patch
from paisho_gen5_actions import Actions,duration,read,native


class ActionTests(unittest.TestCase):
    def root(self,d):
        root=(Path(d)/'run').resolve();root.mkdir()
        (root/'status.json').write_text(json.dumps({'training_pid':42,'end_unix_seconds':time.time()+21600}))
        return root

    def test_prepared_curriculum_stages_without_start_and_is_used_once(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d);(root/'training').mkdir()
            progress={'completed':5,'version':12,'updates':40}
            (root/'status.json').write_text('{"state":"completed","returncode":0}')
            (root/'training/progress.json').write_text(json.dumps(progress))
            resume=Path(d)/'resume.json';resume.write_text(json.dumps(progress))
            replay=Path(d)/'replay.json';replay.write_text('{}')
            config=Path(d)/'next.json';config.write_text(json.dumps({'model':'latest','resume_progress':str(resume),'replay_index':str(replay),'case_curriculum':{'losses':10}}))
            binary=Path(d)/'new-binary';binary.write_bytes(b'new runtime')
            a=Actions(root)
            with patch('paisho_gen5_actions.native',return_value=None),patch('paisho_gen5_actions.subprocess.Popen') as spawn:
                a.stage(config,binary);spawn.assert_not_called()
            state=read(a.path);self.assertIn('prepared_next',state)
            state['paused']=True;a.save(state)
            (root/'status.json').write_text('{"state":"running"}')
            with patch('paisho_gen5_actions.native',return_value=42),patch('paisho_gen5_actions.subprocess.Popen') as spawn:
                a.stage(config,binary);spawn.assert_not_called()
            state.update(job_pid=os.getpid(),busy=True);a.save(state)
            # A live receipt beyond the durable point must not overwrite the
            # explicitly staged, function-preserving model migration.
            (root/'training/progress.json').write_text(json.dumps({**progress,'completed':6}))
            (root/'training/durable-progress.json').write_text(json.dumps(progress))
            def launch(config,selected,out,end,history_source=None):
                self.assertEqual(selected,binary.resolve())
                self.assertEqual(read(config)['case_curriculum']['losses'],10)
                self.assertEqual(read(config)['model'],'latest')
                out.mkdir();return {'output':str(out)}
            with patch('paisho_gen5_actions.native',return_value=None),patch('paisho_gen5_actions.start',side_effect=launch):a.continue_run(2,'gen3-replay')
            self.assertNotIn('prepared_next',read(a.path))
            self.assertEqual(read(a.path)['remaining_seconds'],7200)

    def test_stop_clears_remaining_without_starting_or_failing(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d);a=Actions(root)
            state=read(a.path);state.update(job_pid=os.getpid(),busy=True,paused=True,remaining_seconds=36000);a.save(state)
            with patch('paisho_gen5_actions.native',return_value=None),patch('paisho_gen5_actions.start') as launch:
                a.stop_run();launch.assert_not_called()
            state=read(a.path);self.assertEqual(state['remaining_seconds'],0)
            self.assertTrue(state['stopped']);self.assertFalse(state['paused']);self.assertIsNone(state['error'])
            with self.assertRaises(ValueError):a.request('resume')

    def test_cooperative_stop_resumes_paused_native_and_does_not_kill_it(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d);(root/'training').mkdir();(root/'config.json').write_text('{"control_stop":true}')
            a=Actions(root);state=read(a.path);state.update(job_pid=os.getpid(),busy=True,paused=True);a.save(state)
            (root/'status.json').write_text('{"state":"completed","returncode":0}')
            (root/'training/durable-progress.json').write_text('{"version":13,"updates":60,"completed":6}')
            with patch('paisho_gen5_actions.native',side_effect=[42,42,None]),patch('paisho_gen5_actions.os.kill') as kill:
                a.stop_run();kill.assert_called_once_with(42,signal.SIGCONT)
            self.assertTrue((root/'training/stop-request.json').exists());self.assertTrue(read(a.path)['stopped'])

    def test_modes_preserve_training_and_reference_budgets_separately(self):
        from paisho_gen5_modes import apply
        options={'case_curriculum':{'losses':10},'threads':10,'reference':'fixed-gen3.1'}
        apply(options,'gen3-replay')
        self.assertTrue(options['legacy_replay']);self.assertFalse(options['historical'])
        self.assertFalse(options['reuse_idle_secondary']);self.assertEqual(options['history_interval'],0)
        self.assertEqual(options['budgets'],[[256,.5],[512,.5]])
        self.assertEqual(options['minimum_search_depth'],5)
        apply(options,'selfplay');self.assertTrue(options['historical']);self.assertFalse(options['legacy_replay'])
        self.assertEqual(options['reference'],'fixed-gen3.1')
        options['minimum_search_depth']=8
        apply(options,'gen3-replay')
        self.assertEqual(options['minimum_search_depth'],8)

    def test_staged_runtime_is_hash_checked_and_mode_change_keeps_ladder(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d);(root/'training').mkdir()
            (root/'status.json').write_text('{"state":"completed","returncode":0}')
            (root/'training/progress.json').write_text('{"completed":5}')
            (root/'config.json').write_text(json.dumps({'case_curriculum':{'losses':10},'model':'old'}))
            binary=Path(d)/'binary';binary.write_bytes(b'new')
            a=Actions(root);a.install_runtime(binary,2)
            state=read(a.path);state.update(job_pid=os.getpid(),busy=True);a.save(state)
            def capture(source,output):
                output.mkdir();(output/'receipt.json').write_text('{"model":"latest"}')
                (output/'resume-progress.json').write_text('{"case_states":{"0":{}},"legacy_ladder":{"stage":2}}')
                (output/'replay.index.json').write_text('{}')
            def launch(config,selected,out,end,history_source=None):
                options=read(config);self.assertTrue(options['legacy_replay']);self.assertEqual(options['archive_workers'],2)
                resumed=read(options['resume_progress']);self.assertEqual(resumed['case_states'],{})
                self.assertEqual(resumed['legacy_ladder']['stage'],2);self.assertEqual(selected,binary.resolve())
                out.mkdir();return {}
            with patch('paisho_gen5_actions.native',return_value=None),patch('paisho_gen5_actions.prepare',side_effect=capture),patch('paisho_gen5_actions.start',side_effect=launch):a.continue_run(2,'gen3-replay')
            self.assertFalse(read(a.path)['stopped'])

    def test_duration_limits(self):
        for hours in (2,5,6,10,12,15,20,24,.5):self.assertEqual(duration(hours),hours)
        for hours in (0,-1,25,float('nan'),float('inf')):
            with self.assertRaises(ValueError):duration(hours)

    def test_pause_is_idempotent_and_preserves_remaining_time(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d);a=Actions(root)
            with patch('paisho_gen5_actions.native',return_value=42),patch('paisho_gen5_resident.stopped',return_value=True),patch('paisho_gen5_actions.os.kill') as kill:
                value=a.request('pause')
                self.assertTrue(value['paused']);self.assertFalse(value['running'])
                self.assertGreater(value['remaining_seconds'],21590)
                a.request('pause');kill.assert_called_once_with(42,signal.SIGSTOP)
            self.assertTrue(Actions(root).snapshot()['paused'])

    def test_running_and_busy_runs_cannot_be_duplicated(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d);a=Actions(root)
            with patch('paisho_gen5_actions.native',return_value=42),patch('paisho_gen5_actions.subprocess.Popen') as spawn:
                with self.assertRaises(ValueError):a.request('start',6)
                spawn.assert_not_called()
            state=read(a.path);state.update(busy=True,job_pid=55);a.save(state)
            with patch('paisho_gen5_actions.command',return_value=f'{Path(__import__("paisho_gen5_actions").__file__).resolve()} --worker {a.path} --hours 6'),patch('paisho_gen5_actions.native',return_value=None):
                with self.assertRaises(ValueError):a.request('start',6)

    def test_macos_discarded_argv_is_not_an_actionable_native(self):
        from paisho_gen5_actions import command
        with patch('paisho_gen5_actions.subprocess.run',return_value=Mock(stdout='R (paisho-gen5)')):
            self.assertEqual(command(42),'')

    def test_exiting_process_label_is_rechecked_before_identity_error(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d)
            with patch('paisho_gen5_actions.command',side_effect=['(paisho-gen5)','']):
                self.assertIsNone(native(root))

    def test_pid_reuse_never_signals_unrelated_process(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d)
            with patch('paisho_gen5_actions.command',return_value='unrelated process'):
                with self.assertRaises(ValueError):native(root)
                (root/'status.json').write_text('{"training_pid":42,"state":"completed"}')
                self.assertIsNone(native(root))

    def test_resume_passes_remaining_duration_to_detached_worker(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d);a=Actions(root);state=read(a.path)
            state.update(paused=True,remaining_seconds=7200);a.save(state)
            with patch('paisho_gen5_actions.native',return_value=None),patch('paisho_gen5_actions.subprocess.Popen',return_value=Mock(pid=55)) as spawn,patch('paisho_gen5_actions.command',return_value=f'{Path(__import__("paisho_gen5_actions").__file__).resolve()} --worker {a.path}'):
                value=a.request('resume');self.assertTrue(value['busy'])
                self.assertEqual(spawn.call_args.args[0][-1],'2.0')
                self.assertTrue(spawn.call_args.kwargs['start_new_session'])

    def test_completed_run_recovers_durable_inputs_and_preserves_history(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d);a=Actions(root);state=read(a.path);state.update(job_pid=os.getpid(),busy=True);a.save(state)
            (root/'training').mkdir();(root/'training/progress.json').write_text('{"completed":5}')
            (root/'config.json').write_text(json.dumps({'model':'old','budgets':[[512,.8],[2048,.2]],'evaluation_anchor':'same-anchor'}))
            def capture(source,output):
                output.mkdir();(output/'receipt.json').write_text('{"model":"latest-durable"}')
                (output/'resume-progress.json').write_text('{}');(output/'replay.index.json').write_text('{}')
            def launch(config,binary,out,end,history_source=None):
                value=read(config);self.assertEqual(value['model'],'latest-durable')
                self.assertEqual(value['evaluation_anchor'],'same-anchor')
                self.assertEqual(value['budgets'],[[512,.8],[2048,.2]])
                self.assertEqual(history_source,root/'training/history')
                out.mkdir();return {'output':str(out)}
            with patch('paisho_gen5_actions.native',return_value=None),patch('paisho_gen5_actions.prepare',side_effect=capture),patch('paisho_gen5_actions.start',side_effect=launch):a.continue_run(24)
            saved=read(a.path);self.assertFalse(saved['busy']);self.assertFalse(saved['paused'])
            self.assertNotEqual(saved['campaign'],str(root));self.assertEqual(saved['remaining_seconds'],86400)
            self.assertEqual(read(Path(saved['campaign'])/'dashboard-control-root.json')['state'],str(a.path))

    def test_failed_recovery_leaves_paused_native_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            root=self.root(d);a=Actions(root);state=read(a.path);state.update(job_pid=os.getpid(),busy=True,paused=True);a.save(state)
            (root/'training').mkdir();(root/'training/progress.json').write_text('{"completed":5}')
            (root/'config.json').write_text('{}')
            with patch('paisho_gen5_actions.native',return_value=42),patch('paisho_gen5_actions.subprocess.check_output',return_value='T'),patch('paisho_gen5_actions.prepare',side_effect=ValueError('bad hash')),patch('paisho_gen5_actions.os.kill') as kill:
                with self.assertRaisesRegex(ValueError,'bad hash'):a.continue_run(6)
                kill.assert_not_called()
            self.assertEqual(read(a.path)['error'],'bad hash');self.assertTrue(read(a.path)['paused'])

class HttpActionTests(unittest.TestCase):
    def test_local_token_required_and_valid_post_dispatched(self):
        from http.server import ThreadingHTTPServer
        import threading,urllib.request,urllib.error
        from types import SimpleNamespace
        from paisho_gen5_dashboard import make_handler
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'control.json';path.write_text('{"token":"test-token"}')
            actions=SimpleNamespace(path=path,request=Mock(return_value={'paused':True}))
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(SimpleNamespace(actions=actions)))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f'http://127.0.0.1:{server.server_port}/api/control'
            try:
                for headers in ({},{'X-Paisho-Control':'test-token','Origin':'https://unrelated.example'}):
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(urllib.request.Request(url,data=b'{"action":"pause"}',headers=headers))
                    self.assertEqual(error.exception.code,403)
                actions.request.assert_not_called()
                response=urllib.request.urlopen(urllib.request.Request(url,data=b'{"action":"pause"}',headers={'X-Paisho-Control':'test-token'}))
                self.assertTrue(json.load(response)['paused']);actions.request.assert_called_once_with('pause',None)
            finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
