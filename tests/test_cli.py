"""Capture/replay behavior without network traffic or a desktop window."""

import contextlib
import io
import pathlib
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mesh_vis import __main__ as cli
from mesh_vis import frame, view, zenoh_source
from tests.test_frame import _frame


class TestCapture(unittest.TestCase):
    def test_live_save_uses_keypress_selection_before_new_frame(self):
        selected = _frame(seq=1)
        incoming = _frame(seq=2)
        viewer = Mock()
        viewer.paused = False
        viewer.frame = selected
        viewer.step.side_effect = [True, False]
        viewer.take_save_requests.return_value = [selected]
        viewer.show.side_effect = lambda f: setattr(viewer, "frame", f)
        client = Mock()
        client.lidar.get_latest.return_value = incoming
        client.lidar.rejected = 0
        client.lidar.frame_count = 2
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(key="test", out=directory, expected_records=0, dense_points="unknown")
            with patch.object(zenoh_source, "ZenohClient", return_value=client), \
                 patch.object(view, "LiveCloudViewer", return_value=viewer), \
                 patch.object(cli.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
                cli.run_live(args)
            saved = list(pathlib.Path(directory).glob("*.npz"))
            self.assertEqual(len(saved), 1)
            restored = frame.load_npz(saved[0])
            self.assertEqual(restored.seq, selected.seq)
            self.assertEqual(restored.xyz.tobytes(), selected.xyz.tobytes())
        self.assertIs(viewer.frame, incoming)
        viewer.close.assert_called_once()
        client.stop.assert_called_once()

    def test_disabled_count_check_applies_to_save_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            for expected in (0, 115200):
                out = io.StringIO()
                args = SimpleNamespace(out=directory, expected_records=expected)
                with contextlib.redirect_stdout(out):
                    cli.save_shown(_frame(), args)
                self.assertEqual("unexpected record count" in out.getvalue(), bool(expected))
                self.assertNotIn("PARTIAL", out.getvalue())

    def test_save_before_first_frame_writes_nothing(self):
        with patch.object(frame, "save_npz") as save, contextlib.redirect_stdout(io.StringIO()):
            cli.save_shown(None, SimpleNamespace())
            save.assert_not_called()


class TestCommands(unittest.TestCase):
    def test_old_count_flag_alias_and_new_flag_agree(self):
        for flag in ("--full-records", "--expected-records"):
            args = cli.build_parser().parse_args(["capture", flag, "0"])
            self.assertEqual(args.expected_records, 0)

    def test_negative_expected_count_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["capture", "--expected-records", "-1"])

    def test_inspect_without_window_or_zenoh(self):
        with tempfile.TemporaryDirectory() as directory:
            path = frame.save_npz(_frame(), directory)
            out = io.StringIO()
            with patch.object(view, "LiveCloudViewer") as window, \
                 patch.object(zenoh_source, "ZenohClient") as client, \
                 contextlib.redirect_stdout(out):
                cli.main(["inspect", str(path), "--no-show", "--expected-records", "0"])
            window.assert_not_called()
            client.assert_not_called()
            self.assertIn("32 records, 27 finite", out.getvalue())
            self.assertIn("dense_points=unknown", out.getvalue())
            self.assertNotIn("unexpected record count", out.getvalue())


if __name__ == "__main__":
    unittest.main()
