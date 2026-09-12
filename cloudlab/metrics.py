"""MEASURE and CHECK -- the two numbers that keep the rest of the pipeline honest.

`spacing` is the first thing to run on any new cloud: nearly every parameter
downstream (voxel size, normal radius, BPA radii, alpha, DBSCAN eps) is a
length, and they should all be derived from this one measurement rather than
guessed separately.

`distance_to_mesh` is the last thing to run: it says how far the reconstructed
surface strayed from the points it was built from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class Spacing:
    """Nearest-neighbour distance statistics for a cloud."""

    median: float
    mean: float
    p05: float
    p95: float
    count: int

    def __str__(self) -> str:
        return (
            f"{self.count} pts | nn-dist median {self.median:.4f} "
            f"mean {self.mean:.4f} p05 {self.p05:.4f} p95 {self.p95:.4f}"
        )


def spacing(pcd: o3d.geometry.PointCloud) -> Spacing:
    """Nearest-neighbour distance stats. The p95/p05 ratio is the tell for
    anisotropy: uniform clouds sit near 2, raw lidar runs an order higher."""
    if len(pcd.points) < 2:
        raise ValueError("spacing needs at least 2 points")
    d = np.asarray(pcd.compute_nearest_neighbor_distance())
    return Spacing(
        median=float(np.median(d)),
        mean=float(d.mean()),
        p05=float(np.percentile(d, 5)),
        p95=float(np.percentile(d, 95)),
        count=len(d),
    )


def distance_to_mesh(
    pcd: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh
) -> np.ndarray:
    """Unsigned distance from each point to the nearest triangle.

    Unsigned, so it cannot tell you which side of the surface you are on --
    only how far off it you are.
    """
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros(0)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
    return scene.compute_distance(query).numpy().astype(float)
