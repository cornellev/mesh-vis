"""Tier 2 test data: a simulated spinning lidar.

A rotating lidar fires a fixed set of beams at different elevations while
spinning in azimuth. Ray-casting that pattern into a scene reproduces the three
properties that make real lidar hard to mesh, each on its own knob:

  occlusion   -- rays stop at the first hit, so there is nothing behind an
                 obstacle and nothing on the far side of anything. One sweep is
                 a 2.5D shell, not a closed surface.
  anisotropy  -- samples are tight along a ring and wide across rings, so no
                 single radius suits both. `ring_spacing` measures the gap.
  falloff     -- rings land further apart the further out they go, so density
                 drops with range and one global parameter cannot fit the scene.

Turn `n_beams` down and watch a reconstruction pipeline fall apart; that is the
intuition this module exists to give.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class LidarSpec:
    """Beam geometry. Defaults are roughly a 16-beam +/-15 degree scanner."""

    n_beams: int = 16
    fov_up_deg: float = 15.0
    fov_down_deg: float = -15.0
    n_azimuth: int = 1024
    max_range: float = 100.0

    def __post_init__(self) -> None:
        if self.n_beams < 1:
            raise ValueError(f"n_beams must be >= 1, got {self.n_beams}")
        if self.n_azimuth < 1:
            raise ValueError(f"n_azimuth must be >= 1, got {self.n_azimuth}")
        if self.fov_up_deg < self.fov_down_deg:
            raise ValueError(
                f"fov_up_deg ({self.fov_up_deg}) must be >= fov_down_deg ({self.fov_down_deg})"
            )
        if self.max_range <= 0:
            raise ValueError(f"max_range must be positive, got {self.max_range}")

    @property
    def n_rays(self) -> int:
        return self.n_beams * self.n_azimuth

    def elevations(self) -> np.ndarray:
        """Beam elevations in radians, ascending. A single beam sits mid-FOV."""
        if self.n_beams == 1:
            return np.radians([(self.fov_up_deg + self.fov_down_deg) / 2.0])
        return np.radians(
            np.linspace(self.fov_down_deg, self.fov_up_deg, self.n_beams)
        )

    def azimuths(self) -> np.ndarray:
        """Azimuths in radians over one full turn, excluding the repeat at 2pi."""
        return np.linspace(0.0, 2 * np.pi, self.n_azimuth, endpoint=False)


@dataclass(frozen=True)
class Scan:
    """One sweep. `ring` and `column` keep the sensor's grid structure, which
    is thrown away by a plain point cloud but is exactly what tells you the
    along-ring vs across-ring spacing."""

    points: np.ndarray
    ring: np.ndarray
    column: np.ndarray
    range: np.ndarray
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __len__(self) -> int:
        return len(self.points)

    def to_point_cloud(self, color_by: str | None = None) -> o3d.geometry.PointCloud:
        """Convert to an Open3D cloud, optionally coloured to expose structure."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        if color_by is None:
            return pcd
        if color_by == "ring":
            pcd.colors = o3d.utility.Vector3dVector(_ring_colors(self.ring))
        elif color_by == "range":
            pcd.colors = o3d.utility.Vector3dVector(_range_colors(self.range))
        else:
            raise ValueError(f"unknown color_by {color_by!r}; use 'ring' or 'range'")
        return pcd


def ray_directions(spec: LidarSpec) -> np.ndarray:
    """Unit directions shaped (n_beams, n_azimuth, 3). +z is up, azimuth
    advances from +x toward +y."""
    el = spec.elevations()[:, None]
    az = spec.azimuths()[None, :]
    cos_el = np.cos(el)
    return np.stack(
        np.broadcast_arrays(cos_el * np.cos(az), cos_el * np.sin(az), np.sin(el) * np.ones_like(az)),
        axis=-1,
    )


def scan_scene(
    meshes,
    origin,
    spec: LidarSpec | None = None,
    range_sigma: float = 0.0,
    dropout: float = 0.0,
    seed: int | None = None,
) -> Scan:
    """Cast the beam pattern into `meshes` from `origin` and keep what it hits.

    `range_sigma` perturbs the measured distance *along the ray*, which is how
    a range sensor actually errs. `dropout` deletes returns at random, standing
    in for dark or specular surfaces that send nothing back.
    """
    spec = spec or LidarSpec()
    if range_sigma < 0:
        raise ValueError(f"range_sigma must be non-negative, got {range_sigma}")
    if not 0.0 <= dropout <= 1.0:
        raise ValueError(f"dropout must be in [0, 1], got {dropout}")

    origin = np.asarray(origin, dtype=float).reshape(3)
    if isinstance(meshes, o3d.geometry.TriangleMesh):
        meshes = [meshes]
    meshes = [m for m in meshes if len(m.triangles) > 0]
    if not meshes:
        return _empty_scan(origin)

    scene = o3d.t.geometry.RaycastingScene()
    for mesh in meshes:
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    dirs = ray_directions(spec).reshape(-1, 3)
    rays = np.empty((len(dirs), 6), dtype=np.float32)
    rays[:, :3] = origin
    rays[:, 3:] = dirs
    t = scene.cast_rays(o3d.core.Tensor(rays)).get("t_hit").numpy().astype(float)

    hit = np.isfinite(t) & (t <= spec.max_range)
    ring = np.repeat(np.arange(spec.n_beams), spec.n_azimuth)[hit]
    column = np.tile(np.arange(spec.n_azimuth), spec.n_beams)[hit]
    dirs, t = dirs[hit], t[hit]

    if dropout > 0 or range_sigma > 0:
        rng = np.random.default_rng(seed)
        if dropout > 0:
            keep = rng.random(len(t)) >= dropout
            dirs, t, ring, column = dirs[keep], t[keep], ring[keep], column[keep]
        if range_sigma > 0:
            t = t + rng.normal(0.0, range_sigma, len(t))

    return Scan(
        points=origin + dirs * t[:, None],
        ring=ring,
        column=column,
        range=t,
        origin=origin,
    )


def ring_spacing(scan: Scan) -> tuple[float, float]:
    """(along-ring, across-ring) median neighbour distance.

    The ratio between them is the anisotropy that breaks ball pivoting on raw
    lidar, and the reason the CLEAN stage voxel-downsamples at roughly the
    across-ring figure -- that is the real resolution of the data.
    """
    if len(scan) < 2:
        raise ValueError("ring_spacing needs at least 2 points")
    order = np.lexsort((scan.column, scan.ring))
    along = _adjacent_gaps(scan.points[order], scan.ring[order], scan.column[order])

    order = np.lexsort((scan.ring, scan.column))
    across = _adjacent_gaps(scan.points[order], scan.column[order], scan.ring[order])

    if along.size == 0 or across.size == 0:
        raise ValueError("not enough adjacent samples to measure spacing")
    return float(np.median(along)), float(np.median(across))


def make_test_scene() -> list[o3d.geometry.TriangleMesh]:
    """Ground plane plus a few obstacles: enough structure to exercise every
    pipeline stage (a plane to segment, clusters to separate, shadows to see).
    Ground top sits at z=0, so put the sensor at a realistic mast height."""
    ground = o3d.geometry.TriangleMesh.create_box(40.0, 40.0, 0.2)
    ground.translate((-20.0, -20.0, -0.2))

    scene = [ground]
    for w, d, h, pos in [
        (1.5, 1.5, 1.5, (5.0, -0.75, 0.0)),
        (2.0, 1.0, 2.5, (-7.0, 3.0, 0.0)),
        (1.0, 4.0, 1.0, (3.0, -8.0, 0.0)),
    ]:
        box = o3d.geometry.TriangleMesh.create_box(w, d, h)
        box.translate(pos)
        scene.append(box)

    pole = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3, height=3.0)
    pole.translate((-4.0, -6.0, 1.5))
    scene.append(pole)

    for mesh in scene:
        mesh.compute_vertex_normals()
    return scene


def _empty_scan(origin: np.ndarray) -> Scan:
    return Scan(
        points=np.zeros((0, 3)),
        ring=np.zeros(0, dtype=int),
        column=np.zeros(0, dtype=int),
        range=np.zeros(0),
        origin=origin,
    )


def _adjacent_gaps(points: np.ndarray, group: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Distances between samples that are neighbours in the sensor grid: same
    `group`, consecutive `index`. Skips gaps left by missing returns."""
    adjacent = (np.diff(group) == 0) & (np.diff(index) == 1)
    if not np.any(adjacent):
        return np.zeros(0)
    return np.linalg.norm(np.diff(points, axis=0)[adjacent], axis=1)


def _ring_colors(ring: np.ndarray) -> np.ndarray:
    """One distinct hue per ring index, so the scan-line structure is visible."""
    if len(ring) == 0:
        return np.zeros((0, 3))
    span = max(int(ring.max()) + 1, 1)
    return _hsv_to_rgb(ring.astype(float) / span, 0.85, 1.0)


def _range_colors(rng_: np.ndarray) -> np.ndarray:
    if len(rng_) == 0:
        return np.zeros((0, 3))
    lo, hi = float(rng_.min()), float(rng_.max())
    t = np.zeros_like(rng_) if hi <= lo else (rng_ - lo) / (hi - lo)
    return _hsv_to_rgb(0.66 * (1.0 - t), 0.85, 1.0)


def _hsv_to_rgb(h: np.ndarray, s: float, v: float) -> np.ndarray:
    """Vectorised HSV->RGB for hue arrays in [0, 1). Avoids a matplotlib dep."""
    h = np.asarray(h, dtype=float) % 1.0
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    full = np.full_like(h, v)
    table = np.array([[full, t, np.full_like(h, p)],
                      [q, full, np.full_like(h, p)],
                      [np.full_like(h, p), full, t],
                      [np.full_like(h, p), q, full],
                      [t, np.full_like(h, p), full],
                      [full, np.full_like(h, p), q]])
    return table[i % 6, :, np.arange(len(h))]
