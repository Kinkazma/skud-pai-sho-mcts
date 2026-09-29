import signal
import unittest
from unittest.mock import patch

import stop_compact_history_at_deadline as stop


class StopHistoryTests(unittest.TestCase):
    def test_changed_process_is_never_signalled(self):
        with patch.object(stop, "process_identity", return_value="reused-pid"), \
                patch.object(stop.os, "kill") as kill:
            self.assertEqual(stop.stop_expected(123, "original"), "already-exited-or-identity-changed")
            kill.assert_not_called()

    def test_only_the_exact_companion_receives_graceful_signal(self):
        with patch.object(stop, "process_identity", return_value="original"), \
                patch.object(stop.os, "kill") as kill:
            self.assertEqual(stop.stop_expected(123, "original"), "stop-signal-sent")
            kill.assert_called_once_with(123, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
