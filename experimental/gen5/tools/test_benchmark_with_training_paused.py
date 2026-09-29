import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_with_training_paused import run_offline_paused, run_paused


class PauseTests(unittest.TestCase):
    def exercise(self, desired, fail=False, user_pause=False):
        state = dict(desired=desired, observed=desired, completed=False, revision=0, pid=123)
        actions = []

        def request(action):
            if action != "status":
                actions.append(action)
                state.update(desired="paused" if action == "pause" else "running")
                state.update(observed=state["desired"], revision=state["revision"] + 1)
            return dict(state)

        def command():
            self.assertEqual(state["observed"], "paused")
            if user_pause:
                state["revision"] += 1
            if fail:
                raise RuntimeError("benchmark failed")
            return 0

        with patch("benchmark_with_training_paused.time.sleep"):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "benchmark failed"):
                    run_paused(request, command)
            else:
                self.assertEqual(run_paused(request, command), 0)
        return actions

    def test_resume_after_success_and_failure(self):
        for fail in (False, True):
            self.assertEqual(self.exercise("running", fail), ["pause", "resume"])

    def test_preserve_existing_or_new_user_pause(self):
        self.assertEqual(self.exercise("paused"), [])
        self.assertEqual(self.exercise("running", user_pause=True), ["pause"])

    def test_waits_for_in_progress_example_export_before_benchmark(self):
        statuses = iter([
            dict(desired="paused", observed="paused", completed=False, pid=123),
            dict(observed="paused", completed=False, examples_busy=True),
            dict(observed="paused", completed=False, examples_busy=False),
        ])
        calls = []
        def request(action):
            self.assertEqual(action, "status")
            calls.append(action)
            return next(statuses)
        with patch("benchmark_with_training_paused.time.sleep"):
            self.assertEqual(run_paused(request, lambda: len(calls)), 3)


class OfflinePauseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = SimpleNamespace(
            state_path=Path(self.directory.name) / "control-state.json",
            fingerprint="frozen-campaign",
        )
        self.state = dict(job_fingerprint=self.config.fingerprint,
                          desired="paused", observed="paused", process_pid=None)

    def write_state(self):
        self.config.state_path.write_text(json.dumps(self.state) + "\n")
        return self.config.state_path.read_bytes()

    def test_preserves_pause_on_success_failure_and_exception(self):
        before = self.write_state()
        for returncode in (0, 7):
            self.assertEqual(run_offline_paused(self.config, lambda: returncode), returncode)
            self.assertEqual(self.config.state_path.read_bytes(), before)
        def failure():
            raise RuntimeError("benchmark failed")
        with self.assertRaisesRegex(RuntimeError, "benchmark failed"):
            run_offline_paused(self.config, failure)
        self.assertEqual(self.config.state_path.read_bytes(), before)

    def test_rejects_unsettled_or_wrong_campaign_without_running_command(self):
        for key, value in (("job_fingerprint", "different"), ("desired", "running"),
                           ("observed", "pausing"), ("process_pid", 123)):
            with self.subTest(key=key):
                original = self.state[key]
                self.state[key] = value
                before = self.write_state()
                with self.assertRaisesRegex(ValueError, "matching persisted pause"):
                    run_offline_paused(self.config, lambda: self.fail("command ran"))
                self.assertEqual(self.config.state_path.read_bytes(), before)
                self.state[key] = original

    def test_missing_state_is_not_an_implicit_pause(self):
        with self.assertRaises(FileNotFoundError):
            run_offline_paused(self.config, lambda: self.fail("command ran"))


if __name__ == "__main__":
    unittest.main()

class Gen3PauseTests(unittest.TestCase):
    def exercise(self, initial='running', fail=False, changed=False):
        from benchmark_with_training_paused import run_gen3_paused
        state=dict(state=initial,action_id='same');actions=[]
        def request(action):
            actions.append(action);state['state']='paused' if action=='pause' else 'starting';return dict(state)
        def command():
            self.assertIn(state['state'],('paused','completed'))
            if changed:state['action_id']='user-changed'
            if fail:raise RuntimeError('trial failed')
            return 7
        with patch('paisho_gen3_controls.read',side_effect=lambda p:dict(state)),patch('paisho_gen3_controls.Gen3Controls') as control:
            control.return_value.request.side_effect=request
            if fail:
                with self.assertRaises(RuntimeError):run_gen3_paused(Path('dummy'),command)
            else:self.assertEqual(run_gen3_paused(Path('dummy'),command),7)
        return actions
    def test_restore_only_our_running_campaign_after_success_and_failure(self):
        self.assertEqual(self.exercise(),['pause','resume'])
        self.assertEqual(self.exercise(fail=True),['pause','resume'])
        self.assertEqual(self.exercise(initial='paused'),[])
        self.assertEqual(self.exercise(initial='completed'),[])
        self.assertEqual(self.exercise(changed=True),['pause'])
