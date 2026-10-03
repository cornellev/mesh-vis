"""Interactive viewer for live frames and saved snapshots; GUI runs on the main thread."""

from __future__ import annotations

from collections import deque
import pathlib

import numpy as np
import open3d as o3d

UNASSIGNED_GRAY = (0.45, 0.45, 0.45)
_GOLDEN = 0.618033988749895


def cluster_colors(cluster_id: np.ndarray, unassigned=UNASSIGNED_GRAY) -> np.ndarray:
    """One hue per cluster id, gray for negative ids (-1 = unassigned).

    Hues step by the golden ratio, so consecutive ids -- which the segmenter
    hands out to neighbouring segments -- land far apart on the colour wheel.
    Ids are local to a frame: the same object can change colour every frame.
    """
    ids = np.asarray(cluster_id).astype(np.int64)
    hue = (ids * _GOLDEN) % 1.0
    rgb = np.clip(np.abs((hue[:, None] * 6 + np.array([0, 4, 2])) % 6 - 3) - 1, 0, 1)
    colors = 0.25 + 0.75 * rgb
    colors[ids < 0] = unassigned
    return colors


def prepare_save_path(path: str | pathlib.Path) -> pathlib.Path:
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


_GLFW_PRESS = 1
_KEY_SPACE = 32
_KEY_S = ord("S")


class LiveCloudViewer:
    """A window that keeps one point cloud and swaps its contents per frame.

    Keys: Space pauses/resumes, S asks to save the frame on screen (it
    shadows the legacy viewer's mesh-shading toggle, which does nothing for
    points), Esc/Q closes. Saving is only *requested* here -- the caller
    decides where the frame goes -- so this class needs nothing from Zenoh or
    the snapshot format.

    The camera is fitted once, to the first frame with any finite points.
    After that the cloud is updated in place, so the view stays where you put
    it while frames stream in.
    """

    def __init__(
        self,
        title: str = "mesh-vis",
        width: int = 1280,
        height: int = 960,
        point_size: float = 2.0,
        background=(0.05, 0.05, 0.08),
        axes_size: float = 1.0,
    ):
        self.paused = False
        self.frame = None          # the CloudFrame currently on screen
        self._save_requests = deque()
        self._fitted = False

        self._vis = o3d.visualization.VisualizerWithKeyCallback()
        if not self._vis.create_window(window_name=title, width=width, height=height):
            raise RuntimeError("could not open a window; this needs a desktop session")
        opt = self._vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.asarray(background, dtype=float)
        self._vis.register_key_action_callback(_KEY_SPACE, self._on_space)
        self._vis.register_key_action_callback(_KEY_S, self._on_s)
        # Axes at the sensor origin: red x, green y, blue z, `axes_size` metres.
        self._vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size))
        self._pcd = o3d.geometry.PointCloud()

    def show(self, frame) -> None:
        """Put `frame` on screen: finite points only, coloured by cluster."""
        finite = frame.finite_mask()
        self._pcd.points = o3d.utility.Vector3dVector(frame.xyz[finite].astype(np.float64))
        self._pcd.colors = o3d.utility.Vector3dVector(cluster_colors(frame.cluster_id[finite]))
        if self._fitted:
            self._vis.update_geometry(self._pcd)
        elif finite.any():
            self._vis.add_geometry(self._pcd, reset_bounding_box=True)
            self._fitted = True
        self.frame = frame

    def step(self) -> bool:
        """Handle window events and redraw once. False once the window closes."""
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def take_save_requests(self) -> list:
        """Frames selected at each S keypress, even if show() has since advanced."""
        requested = list(self._save_requests)
        self._save_requests.clear()
        return requested

    def screenshot(self, path: str | pathlib.Path) -> pathlib.Path:
        out = prepare_save_path(path)
        self._vis.capture_screen_image(str(out), do_render=True)
        return out

    def close(self) -> None:
        self._vis.destroy_window()

    def _on_space(self, vis, action, mods) -> bool:
        if action == _GLFW_PRESS:
            self.paused = not self.paused
            print("[paused]" if self.paused else "[resumed]", flush=True)
        return False

    def _on_s(self, vis, action, mods) -> bool:
        if action == _GLFW_PRESS:
            self._save_requests.append(self.frame)
        return False
