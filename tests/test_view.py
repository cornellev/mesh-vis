"""Viewer controls are tested without opening a desktop window."""

from collections import deque
from types import SimpleNamespace
import unittest

import numpy as np

from mesh_vis import view


class TestClusterColors(unittest.TestCase):
    """Runs everywhere -- pure colour mapping, no window."""

    def test_unassigned_is_gray_and_clusters_are_not(self):
        colors = view.cluster_colors(np.array([-1, 5, -1, 7], dtype=np.int32))
        np.testing.assert_allclose(colors[[0, 2]], [view.UNASSIGNED_GRAY] * 2)
        for c in colors[[1, 3]]:
            self.assertGreater(float(c.max() - c.min()), 0.5)  # saturated, not gray

    def test_same_id_same_colour_and_neighbouring_ids_differ(self):
        ids = np.arange(1, 301, dtype=np.int32)
        colors = view.cluster_colors(np.concatenate([ids, ids]))
        np.testing.assert_array_equal(colors[:300], colors[300:])
        step = np.linalg.norm(np.diff(colors[:300], axis=0), axis=1)
        self.assertGreater(float(step.min()), 0.3)

    def test_shape_range_and_edge_cases(self):
        colors = view.cluster_colors(np.array([0, 1, 2**24 + 1, 2**31 - 1], dtype=np.int32))
        self.assertEqual(colors.shape, (4, 3))
        self.assertTrue(np.all((colors >= 0) & (colors <= 1)))
        self.assertEqual(view.cluster_colors(np.zeros(0, dtype=np.int32)).shape, (0, 3))


class TestSaveSelection(unittest.TestCase):
    def setUp(self):
        self.viewer = view.LiveCloudViewer.__new__(view.LiveCloudViewer)
        self.viewer._save_requests = deque()
        self.viewer.frame = SimpleNamespace(seq=1)
        self.viewer.paused = False

    def test_save_retains_selected_frame_when_live_frame_advances(self):
        selected = self.viewer.frame
        self.viewer._on_s(None, view._GLFW_PRESS, 0)
        self.viewer.frame = SimpleNamespace(seq=2)
        self.assertEqual(self.viewer.take_save_requests(), [selected])
        self.assertEqual(self.viewer.take_save_requests(), [])

    def test_repeated_presses_preserve_each_selection_and_ignore_key_release(self):
        first = self.viewer.frame
        self.viewer._on_s(None, view._GLFW_PRESS, 0)
        second = self.viewer.frame = SimpleNamespace(seq=2)
        self.viewer._on_s(None, 0, 0)
        self.viewer._on_s(None, view._GLFW_PRESS, 0)
        self.assertEqual(self.viewer.take_save_requests(), [first, second])

    def test_save_before_first_frame_is_explicit(self):
        self.viewer.frame = None
        self.viewer._on_s(None, view._GLFW_PRESS, 0)
        self.assertEqual(self.viewer.take_save_requests(), [None])

    def test_pause_toggles_on_press_only(self):
        self.viewer._on_space(None, view._GLFW_PRESS, 0)
        self.assertTrue(self.viewer.paused)
        self.viewer._on_space(None, 0, 0)
        self.assertTrue(self.viewer.paused)
        self.viewer._on_space(None, view._GLFW_PRESS, 0)
        self.assertFalse(self.viewer.paused)


if __name__ == "__main__":
    unittest.main()
