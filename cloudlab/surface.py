"""Tier 1 test data: sample a point cloud from a surface you already know.

Why this and not `np.random.rand(n, 3)`: that fills a *volume* uniformly, and
meshing algorithms reconstruct *surfaces*. There is no surface inside a cube of
noise to find, so every reconstructor returns junk. Sampling a known mesh gives
you a cloud that has a surface in it AND the ground-truth mesh it came from, so
`metrics.distance_to_mesh` can score your reconstruction exactly.
"""

from __future__ import annotations

import copy

import numpy as np
import open3d as o3d

SHAPES = {
    "sphere": o3d.geometry.TriangleMesh.create_sphere,
    "box": o3d.geometry.TriangleMesh.create_box,
    "cylinder": o3d.geometry.TriangleMesh.create_cylinder,
    "torus": o3d.geometry.TriangleMesh.create_torus,
}

METHODS = ("uniform", "poisson_disk")


def make_shape(name: str, **kwargs) -> o3d.geometry.TriangleMesh:
    """Build one of the known primitives. kwargs go to the Open3D factory."""
    try:
        factory = SHAPES[name]
    except KeyError:
        raise ValueError(
            f"unknown shape {name!r}; choose from {sorted(SHAPES)}"
        ) from None
    mesh = factory(**kwargs)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def sample_surface(
    mesh: o3d.geometry.TriangleMesh,
    n_points: int,
    method: str = "poisson_disk",
    seed: int | None = None,
    with_normals: bool = False,
) -> o3d.geometry.PointCloud:
    """Draw `n_points` from the surface of `mesh`.

    "uniform" is area-weighted and independent per point, so it clumps --
    spacing varies a lot. "poisson_disk" enforces a minimum separation, giving
    the even coverage that ball pivoting and normal estimation prefer. It costs
    more because it oversamples then thins.
    """
    if n_points <= 0:
        raise ValueError(f"n_points must be positive, got {n_points}")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {list(METHODS)}")

    src = mesh
    if with_normals and not mesh.has_triangle_normals():
        src = copy.deepcopy(mesh)
        src.compute_triangle_normals()

    if seed is not None:
        o3d.utility.random.seed(seed)

    if method == "uniform":
        return src.sample_points_uniformly(
            number_of_points=n_points, use_triangle_normal=with_normals
        )
    return src.sample_points_poisson_disk(
        number_of_points=n_points, use_triangle_normal=with_normals
    )


def add_gaussian_noise(
    pcd: o3d.geometry.PointCloud, sigma: float, seed: int | None = None
) -> o3d.geometry.PointCloud:
    """Isotropic jitter, as a stand-in for sensor noise.

    Note this displaces points in all directions; a real range sensor errs
    along the ray. `lidar.scan_scene(range_sigma=...)` does the latter.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    out = copy.deepcopy(pcd)
    if sigma == 0:
        return out
    pts = np.asarray(out.points)
    rng = np.random.default_rng(seed)
    out.points = o3d.utility.Vector3dVector(pts + rng.normal(0.0, sigma, pts.shape))
    return out


def local_flatness(pcd: o3d.geometry.PointCloud, k: int = 30) -> float:
    """Mean surface *variation*: smallest PCA eigenvalue over their sum, per
    neighbourhood.

    Near 0 means every neighbourhood is locally planar -- the points lie on a
    surface. Near 1/3 means neighbourhoods are isotropic blobs -- the points
    fill a volume and there is no surface to reconstruct. This is the numeric
    difference between tier-1 data and the random-cube starter cloud.
    """
    pts = np.asarray(pcd.points)
    if len(pts) <= k:
        raise ValueError(f"need more than k={k} points, got {len(pts)}")
    tree = o3d.geometry.KDTreeFlann(pcd)
    # Legacy KDTreeFlann has no batch query, so the search loops; the linear
    # algebra does not have to. Gather indices, then do one batched eigensolve.
    idx = np.empty((len(pts), k), dtype=np.int64)
    for i, p in enumerate(pts):
        _, nn, _ = tree.search_knn_vector_3d(p, k)
        idx[i] = np.asarray(nn)[:k]
    nbrs = pts[idx]
    centred = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = centred.transpose(0, 2, 1) @ centred / (k - 1)
    eig = np.linalg.eigvalsh(cov)
    total = eig.sum(axis=1)
    ratios = np.divide(eig[:, 0], total, out=np.zeros(len(pts)), where=total > 0)
    return float(ratios.mean())
