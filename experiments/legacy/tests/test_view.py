"""Tests for the rendering helper.

The render tests are opt-in: offscreen `create_window(visible=False)` blocks
forever without an attached window server, which hangs the whole suite. Run
them from a normal desktop session with:

    CLOUDLAB_RENDER_TESTS=1 python3 -m unittest discover -s tests -t .
"""

import os
import pathlib
import tempfile
import unittest

import numpy as np
import open3d as o3d

from cloudlab import surface, view

RENDER = unittest.skipUnless(
    os.environ.get("CLOUDLAB_RENDER_TESTS"),
    "needs a window server; set CLOUDLAB_RENDER_TESTS=1",
)


class TestArgumentHandling(unittest.TestCase):
    """Runs everywhere -- no window is opened."""

    def test_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            view.show([], "t", save="x.png", backend="raytracer")

    def test_save_path_parent_is_created_before_rendering(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "nested" / "deep" / "shot.png"
            view.prepare_save_path(out)
            self.assertTrue(out.parent.is_dir())


@RENDER
class TestSave(unittest.TestCase):
    def setUp(self):
        self.mesh = surface.make_shape("sphere", radius=1.0, resolution=20)

    def test_save_writes_a_readable_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "shot.png"
            view.show([self.mesh], "test", save=out, width=320, height=240)
            self.assertEqual(out.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            img = o3d.io.read_image(str(out))
            self.assertEqual(tuple(np.asarray(img).shape[:2]), (240, 320))

    def test_save_renders_the_geometry_not_a_blank_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty, full = pathlib.Path(tmp) / "e.png", pathlib.Path(tmp) / "f.png"
            view.show([], "empty", save=empty, width=320, height=240)
            view.show([self.mesh], "full", save=full, width=320, height=240)
            a = np.asarray(o3d.io.read_image(str(empty))).astype(float)
            b = np.asarray(o3d.io.read_image(str(full))).astype(float)
            self.assertGreater(float(np.abs(a - b).mean()), 1.0)


if __name__ == "__main__":
    unittest.main()
