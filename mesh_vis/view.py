"""Interactive viewer for live frames and saved snapshots; GUI runs on the main thread."""

from __future__ import annotations

from collections import deque
import math
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


# Open3D's scroll zoom. ViewControl::Scale adds dy * STEP and refuses to go
# below MIN, which is where this viewer's wheel stops. MAX matches that cap
# so zooming back out still ends on the whole cloud.
_OPEN3D_ZOOM_STEP = 0.02
_OPEN3D_ZOOM_DEFAULT = 0.7
_OPEN3D_ZOOM_MIN = 0.02
_OPEN3D_ZOOM_MAX = 2.0
_ZOOM_FLOOR = 1e-8
_GLFW_RELEASE = 0


class _ScrollZoom:
    """The stock scroll zoom, continued past Open3D's floor of 0.02.

    Above that floor a wheel notch moves zoom by exactly 0.02, as
    ViewControl::Scale does. Below it, the same notch scales the zoom, so the
    camera can keep closing in and the way back out lands on the same values.
    """

    def __init__(self, extent: float):
        self.extent = max(float(extent), 1e-9)
        self.zoom = _OPEN3D_ZOOM_DEFAULT

    def scroll(self, dy: float) -> None:
        if not dy:
            return
        step = dy * _OPEN3D_ZOOM_STEP
        if self.zoom > _OPEN3D_ZOOM_MIN:
            nxt = self.zoom + step
            if nxt >= _OPEN3D_ZOOM_MIN:
                self.zoom = min(nxt, _OPEN3D_ZOOM_MAX)
                return
            used = (self.zoom - _OPEN3D_ZOOM_MIN) / _OPEN3D_ZOOM_STEP
            self.zoom = _OPEN3D_ZOOM_MIN
            self._scale_below(dy + used)
            return
        if self._scale_below(dy) and self.zoom > _OPEN3D_ZOOM_MIN:
            extra = math.log(self.zoom / _OPEN3D_ZOOM_MIN) * _OPEN3D_ZOOM_MIN / _OPEN3D_ZOOM_STEP
            self.zoom = min(_OPEN3D_ZOOM_MIN + extra * _OPEN3D_ZOOM_STEP, _OPEN3D_ZOOM_MAX)

    def _scale_below(self, dy: float) -> bool:
        """Scale zoom beneath the floor. False once a guard has clamped it."""
        scaled = dy * _OPEN3D_ZOOM_STEP / _OPEN3D_ZOOM_MIN
        if scaled >= 40.0:
            self.zoom = _OPEN3D_ZOOM_MAX
            return False
        if scaled <= -40.0:
            self.zoom = _ZOOM_FLOOR
            return False
        self.zoom *= math.exp(scaled)
        if self.zoom < _ZOOM_FLOOR:
            self.zoom = _ZOOM_FLOOR
            return False
        return True

    def clip(self, fov_deg: float) -> tuple[float, float]:
        """Near and far planes. Open3D's near plane is 1% of the scene and
        starts cutting the cloud off as the camera approaches it."""
        half = math.radians(max(float(fov_deg), 1e-3) * 0.5)
        distance = self.zoom * self.extent / math.tan(half)
        near = min(0.01 * self.extent, distance * 0.25)
        near = max(near, 1e-9)
        far = distance + 3.0 * self.extent
        far = min(far, near * 1.0e5)
        if far < distance * 2.0:
            far = distance * 2.0
        return near, far

    def apply(self, ctr) -> None:
        near, far = self.clip(ctr.get_field_of_view())
        ctr.set_constant_z_near(near)
        ctr.set_constant_z_far(far)
        ctr.set_zoom(self.zoom)


def _scene_extent(pcd: o3d.geometry.PointCloud, normal_length: float) -> float:
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return 1.0
    if normal_length > 0:
        pts = np.vstack([pts, pts + normal_length * np.asarray(pcd.normals)])
    extent = float((pts.max(axis=0) - pts.min(axis=0)).max())
    return extent if extent > 0.0 else 1.0


def show_points(pcd: o3d.geometry.PointCloud, title: str, point_size: float = 2.0,
                background=(0.05, 0.05, 0.08), normal_length: float = 0.0,
                mesh: o3d.geometry.TriangleMesh | None = None) -> None:
    """One window with one cloud in one colour, fitted to it; returns when closed.
    A positive `normal_length` also draws each point's normal as a cyan line.
    With `mesh`, the shaded mesh is shown instead of the points (W: wireframe).
    Scroll zoom matches Open3D until its floor, then keeps going."""
    if mesh is not None:
        shown = o3d.geometry.TriangleMesh(mesh).paint_uniform_color((0.9, 0.75, 0.3))
    else:
        shown = o3d.geometry.PointCloud(pcd).paint_uniform_color((0.9, 0.75, 0.3))
    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(window_name=title, width=1280, height=960):
        raise RuntimeError("could not open a window; this needs a desktop session")
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.asarray(background, dtype=float)
    # Points stay flat-coloured; a mesh needs lighting to show its shape, and
    # its back faces too, since rotating past the sensor's side shows them.
    opt.light_on = mesh is not None
    opt.mesh_show_back_face = True
    vis.add_geometry(shown)
    if normal_length > 0:
        start = np.asarray(pcd.points)
        end = start + normal_length * np.asarray(pcd.normals)
        n = len(start)
        lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.vstack([start, end])),
                                     o3d.utility.Vector2iVector(np.column_stack([np.arange(n), np.arange(n) + n])))
        vis.add_geometry(lines.paint_uniform_color((0.3, 0.8, 1.0)))
    # Scrolling moves the camera toward the lookat point. Open3D puts it at the
    # bounding-box centre, which can be empty space; use the cluster point
    # nearest the median instead, inside the densest part.
    pts = np.asarray(pcd.points)
    lookat = pts[np.argmin(np.linalg.norm(pts - np.median(pts, axis=0), axis=1))]
    vis.get_view_control().set_lookat(lookat)
    zoom = _ScrollZoom(_scene_extent(pcd, normal_length))
    vis.register_mouse_scroll_callback(lambda vis, x, y: _on_scroll(zoom, vis, y))
    vis.register_key_action_callback(ord("["), lambda vis, action, mods: _on_fov(zoom, vis, action, -1.0))
    vis.register_key_action_callback(ord("]"), lambda vis, action, mods: _on_fov(zoom, vis, action, 1.0))
    vis.register_key_action_callback(ord("R"), lambda vis, action, mods: _on_reset_view(zoom, vis, action, lookat))
    vis.run()
    vis.destroy_window()


def _on_scroll(zoom: _ScrollZoom, vis, dy: float) -> bool:
    zoom.scroll(dy)
    zoom.apply(vis.get_view_control())
    return False


def _on_fov(zoom: _ScrollZoom, vis, action: int, step: float) -> bool:
    if action == _GLFW_RELEASE:
        return False
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step)
    zoom.apply(ctr)
    return False


def _on_reset_view(zoom: _ScrollZoom, vis, action: int, lookat) -> bool:
    if action == _GLFW_RELEASE:
        return False
    vis.reset_view_point()
    vis.get_view_control().set_lookat(lookat)
    zoom.zoom = _OPEN3D_ZOOM_DEFAULT
    zoom.apply(vis.get_view_control())
    return False


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

    The camera is fitted once, to the first frame with any finite points:
    centred on the sensor and showing `view_range` metres above and below it, or
    the whole frame when `view_range` is 0. After that the cloud is updated in
    place, so the view stays where you put it while frames stream in.
    """

    def __init__(
        self,
        title: str = "mesh-vis",
        width: int = 1280,
        height: int = 960,
        point_size: float = 2.0,
        background=(0.05, 0.05, 0.08),
        axes_size: float = 1.0,
        view_range: float = 0.0,
    ):
        self.paused = False
        self.frame = None          # the CloudFrame currently on screen
        self._save_requests = deque()
        self._fitted = False
        self._view_range = view_range

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
            if self._view_range > 0:
                self._zoom_to_sensor(frame.sensor_origin)
            self._fitted = True
        self.frame = frame

    def _zoom_to_sensor(self, origin) -> None:
        # Open3D's visible half-height is zoom * the scene's largest
        # bounding-box extent; the axes at the origin are part of the scene.
        pts = np.vstack([np.asarray(self._pcd.points), np.zeros((1, 3))])
        extent = float((pts.max(axis=0) - pts.min(axis=0)).max())
        ctr = self._vis.get_view_control()
        ctr.set_lookat(np.asarray(origin, dtype=float))
        ctr.set_zoom(self._view_range / extent)

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
