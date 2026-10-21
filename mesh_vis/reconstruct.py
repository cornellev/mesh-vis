"""Offline reconstruction of one cluster from a saved frame. No drawing here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

# Voxel size, when not given, as a multiple of the cluster's median
# nearest-neighbour spacing, so near and far clusters get comparable treatment.
VOXEL_SPACINGS = 3.0
# Normal search radius, when not given, as a multiple of the downsampled
# spacing. Smaller radii leave many sparse RoboSense neighbourhoods on a
# single beam, where the fitted plane's direction is arbitrary.
NORMAL_SPACINGS = 6.0
MAX_NN = 50
# Ball radii, when not given, as multiples of the downsampled spacing. The
# largest equals the default normal radius, enough to bridge the gaps
# between RoboSense beams.
BALL_SPACINGS = (1.5, 3.0, 6.0)
# Mesh cleanup defaults. Triangles with an edge longer than MAX_EDGE_SPACINGS
# x the spacing (the largest ball radius) mostly bridge gaps between sparse
# points; pieces under MIN_PIECE_TRIANGLES are floating fragments.
MAX_EDGE_SPACINGS = 6.0
MIN_PIECE_TRIANGLES = 10
SMOOTH_ITERATIONS = 10


@dataclass
class Prepared:
    points: o3d.geometry.PointCloud        # the cluster's finite points, as saved
    deduplicated: o3d.geometry.PointCloud
    downsampled: o3d.geometry.PointCloud
    duplicates: int
    spacing: float                         # median nearest-neighbour distance, metres
    voxel_size: float                      # 0 = not downsampled


def median_spacing(pcd: o3d.geometry.PointCloud) -> float:
    if len(pcd.points) < 2:
        return 0.0
    return float(np.median(pcd.compute_nearest_neighbor_distance()))


def prepare_cluster(frame, cluster_id: int, voxel_size: float | None = None) -> Prepared:
    """Select one cluster, remove exact duplicates, then voxel-downsample.

    `voxel_size=None` derives it from the cluster's spacing; 0 skips downsampling.
    """
    selected = frame.finite_mask() & (frame.cluster_id == cluster_id)
    if not selected.any():
        raise ValueError(f"cluster {cluster_id} has no finite points in this frame")
    points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame.xyz[selected].astype(np.float64)))

    deduplicated = o3d.geometry.PointCloud(points).remove_duplicated_points()
    spacing = median_spacing(deduplicated)
    if voxel_size is None:
        voxel_size = VOXEL_SPACINGS * spacing
    downsampled = deduplicated.voxel_down_sample(voxel_size) if voxel_size > 0 else deduplicated

    return Prepared(
        points=points,
        deduplicated=deduplicated,
        downsampled=downsampled,
        duplicates=len(points.points) - len(deduplicated.points),
        spacing=spacing,
        voxel_size=voxel_size,
    )


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float | None = None,
                     max_nn: int = MAX_NN, sensor_origin=(0.0, 0.0, 0.0)) -> float:
    """Fit a plane to each point's neighbours (PCA) and point it at the sensor.

    Normals are written into `pcd`. `radius=None` derives it from the spacing.
    Returns the radius used.
    """
    if radius is None:
        radius = NORMAL_SPACINGS * median_spacing(pcd)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_towards_camera_location(np.asarray(sensor_origin, dtype=float))
    return radius


def ball_pivot(pcd: o3d.geometry.PointCloud, radii=None) -> tuple[o3d.geometry.TriangleMesh, list[float]]:
    """Ball-pivoting mesh from points with normals, then basic cleanup.

    `radii=None` derives them from the spacing. Returns the mesh and the radii used;
    the mesh has no triangles when the balls could not span the points.
    """
    if radii is None:
        spacing = median_spacing(pcd)
        radii = [k * spacing for k in BALL_SPACINGS]
    radii = [float(r) for r in radii]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh, radii


@dataclass
class Cleaned:
    mesh: o3d.geometry.TriangleMesh
    max_edge: float          # 0 = no cutoff
    long_removed: int        # triangles removed for a long edge
    pieces_removed: int      # small pieces removed
    small_removed: int       # triangles in those pieces


def clean_mesh(mesh: o3d.geometry.TriangleMesh, spacing: float, max_edge: float | None = None,
               min_triangles: int = MIN_PIECE_TRIANGLES,
               smooth_iterations: int = SMOOTH_ITERATIONS) -> Cleaned:
    """Remove long-edged triangles, then small pieces, then Taubin-smooth.

    `max_edge=None` derives it from `spacing`. A value of 0 skips that step.
    Long edges go first so the bridges they formed do not hold pieces together.
    """
    mesh = o3d.geometry.TriangleMesh(mesh)
    if max_edge is None:
        max_edge = MAX_EDGE_SPACINGS * spacing

    long_removed = 0
    if max_edge > 0 and len(mesh.triangles):
        corners = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]          # (T, 3 corners, xyz)
        edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)    # (T, 3 edges)
        too_long = edges.max(axis=1) > max_edge
        mesh.remove_triangles_by_mask(too_long)
        long_removed = int(too_long.sum())

    pieces_removed = small_removed = 0
    if min_triangles > 0 and len(mesh.triangles):
        piece, piece_size, _ = mesh.cluster_connected_triangles()
        piece_size = np.asarray(piece_size)
        small = piece_size[np.asarray(piece)] < min_triangles
        mesh.remove_triangles_by_mask(small)
        pieces_removed = int((piece_size < min_triangles).sum())
        small_removed = int(small.sum())

    mesh.remove_unreferenced_vertices()
    if smooth_iterations > 0 and len(mesh.triangles):
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
    mesh.compute_vertex_normals()
    return Cleaned(mesh, max_edge, long_removed, pieces_removed, small_removed)
