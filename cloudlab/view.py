"""Rendering helper.

Rendering on this machine is interactive-only. Both offscreen paths are broken
on this Open3D build:

  * `rendering.OffscreenRenderer` aborts the process outright -- a Filament
    Metal `readPixels` precondition panic, not a catchable exception.
  * the legacy `Visualizer.create_window(visible=False)` blocks forever when no
    window server is attached.

So `show()` defaults to opening a real window, and `save=` is best-effort: it
works from a desktop session and will hang without one. Nothing in the library
or the test suite depends on it.
"""

from __future__ import annotations

import pathlib

import open3d as o3d

BACKENDS = ("legacy",)


def prepare_save_path(path: str | pathlib.Path) -> pathlib.Path:
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def show(
    geometries,
    title: str = "cloudlab",
    save: str | pathlib.Path | None = None,
    width: int = 1280,
    height: int = 960,
    point_size: float = 2.0,
    background=(0.05, 0.05, 0.08),
    backend: str = "legacy",
) -> None:
    """Open an interactive window. With `save` set, render to a PNG instead --
    only from a desktop session (see module docstring)."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {list(BACKENDS)}")

    geometries = list(geometries)
    if save is None:
        o3d.visualization.draw_geometries(
            geometries, window_name=title, width=width, height=height
        )
        return

    out = prepare_save_path(save)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=width, height=height, visible=False)
    try:
        for geometry in geometries:
            vis.add_geometry(geometry)
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = background
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(out), do_render=True)
    finally:
        vis.destroy_window()
