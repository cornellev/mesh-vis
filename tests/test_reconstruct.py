"""Cluster selection, deduplication and downsampling on made-up frames."""

import unittest

import numpy as np
import open3d as o3d

from mesh_vis import frame as fr
from mesh_vis import reconstruct


def _grid_frame(step=0.01):
    """Cluster 5: a 20x20 grid in the z=2 plane, one point copied, one NaN row.
    Everything else is unassigned."""
    u, v = np.meshgrid(np.arange(20) * step, np.arange(20) * step)
    grid = np.column_stack([u.ravel(), v.ravel(), np.full(u.size, 2.0)])
    xyz = np.vstack([grid, grid[:1], [[np.nan] * 3], [[9.0, 9.0, 9.0]]]).astype(np.float32)
    cluster_id = np.full(len(xyz), fr.UNASSIGNED, dtype=np.int32)
    cluster_id[:len(grid) + 2] = 5
    return fr.CloudFrame(xyz=xyz, intensity=np.zeros(len(xyz), np.float32),
                         cluster_id=cluster_id, stamp_sec=0, stamp_nanosec=0)


class TestPrepareCluster(unittest.TestCase):
    def test_selects_finite_cluster_points_and_removes_duplicate(self):
        prepared = reconstruct.prepare_cluster(_grid_frame(), 5, voxel_size=0)
        self.assertEqual(len(prepared.points.points), 401)
        self.assertEqual(prepared.duplicates, 1)
        self.assertEqual(len(prepared.downsampled.points), 400)

    def test_default_voxel_follows_spacing(self):
        for step in (0.01, 0.05):
            prepared = reconstruct.prepare_cluster(_grid_frame(step), 5)
            self.assertAlmostEqual(prepared.spacing, step, places=5)
            self.assertAlmostEqual(prepared.voxel_size, reconstruct.VOXEL_SPACINGS * step, places=5)
            self.assertLess(len(prepared.downsampled.points), 400)

    def test_missing_cluster_is_an_error(self):
        with self.assertRaises(ValueError):
            reconstruct.prepare_cluster(_grid_frame(), 99)


class TestEstimateNormals(unittest.TestCase):
    def test_plane_normals_point_at_sensor(self):
        prepared = reconstruct.prepare_cluster(_grid_frame(), 5)
        radius = reconstruct.estimate_normals(prepared.downsampled)
        self.assertAlmostEqual(radius, reconstruct.NORMAL_SPACINGS * reconstruct.median_spacing(prepared.downsampled))
        # The grid lies in z=2 and the sensor is at the origin, below it.
        np.testing.assert_allclose(np.asarray(prepared.downsampled.normals),
                                   np.tile([0.0, 0.0, -1.0], (len(prepared.downsampled.points), 1)), atol=1e-6)


class TestBallPivot(unittest.TestCase):
    def test_plane_becomes_a_mesh_on_its_points(self):
        prepared = reconstruct.prepare_cluster(_grid_frame(), 5, voxel_size=0)
        reconstruct.estimate_normals(prepared.downsampled)
        mesh, radii = reconstruct.ball_pivot(prepared.downsampled)
        self.assertEqual(len(radii), len(reconstruct.BALL_SPACINGS))
        # A 20x20 grid has 19x19 squares, two triangles each.
        self.assertEqual(len(mesh.triangles), 2 * 19 * 19)
        np.testing.assert_allclose(np.asarray(mesh.vertices)[:, 2], 2.0, atol=1e-6)

    def test_radii_too_small_give_no_triangles(self):
        prepared = reconstruct.prepare_cluster(_grid_frame(), 5, voxel_size=0)
        reconstruct.estimate_normals(prepared.downsampled)
        mesh, _ = reconstruct.ball_pivot(prepared.downsampled, [0.001])
        self.assertEqual(len(mesh.triangles), 0)


def _grid_mesh(n=10, step=0.01):
    """An n x n grid of points in the z=0 plane, two triangles per square."""
    u, v = np.meshgrid(np.arange(n) * step, np.arange(n) * step)
    vertices = np.column_stack([u.ravel(), v.ravel(), np.zeros(u.size)])
    i = np.arange(n - 1)
    corner = (i[:, None] * n + i[None, :]).ravel()
    triangles = np.vstack([np.column_stack([corner, corner + 1, corner + n]),
                           np.column_stack([corner + 1, corner + n + 1, corner + n])])
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(triangles))


class TestCleanMesh(unittest.TestCase):
    def test_long_edge_triangle_is_removed(self):
        mesh = _grid_mesh()
        n = len(mesh.vertices)
        mesh.vertices.append([0.5, 0.0, 0.0])                 # far from the grid
        mesh.triangles.append([0, 1, n])
        cleaned = reconstruct.clean_mesh(mesh, 0.01, min_triangles=0, smooth_iterations=0)
        self.assertAlmostEqual(cleaned.max_edge, reconstruct.MAX_EDGE_SPACINGS * 0.01)
        self.assertEqual(cleaned.long_removed, 1)
        self.assertEqual(len(cleaned.mesh.triangles), 2 * 9 * 9)
        self.assertEqual(len(cleaned.mesh.vertices), 100)     # the far vertex is gone too

    def test_small_piece_is_removed(self):
        mesh = _grid_mesh()
        n = len(mesh.vertices)
        for corner in ([1, 1, 0], [1.01, 1, 0], [1, 1.01, 0]):  # one separate triangle
            mesh.vertices.append(corner)
        mesh.triangles.append([n, n + 1, n + 2])
        cleaned = reconstruct.clean_mesh(mesh, 0.01, max_edge=0, smooth_iterations=0)
        self.assertEqual((cleaned.pieces_removed, cleaned.small_removed), (1, 1))
        self.assertEqual(len(cleaned.mesh.triangles), 2 * 9 * 9)

    def test_smoothing_flattens_a_bump_and_keeps_triangles(self):
        mesh = _grid_mesh()
        vertices = np.asarray(mesh.vertices)
        vertices[55, 2] = 0.01                                 # one raised interior vertex
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        cleaned = reconstruct.clean_mesh(mesh, 0.01, max_edge=0, min_triangles=0)
        self.assertEqual(len(cleaned.mesh.triangles), 2 * 9 * 9)
        self.assertLess(np.abs(np.asarray(cleaned.mesh.vertices)[:, 2]).max(), 0.005)

    def test_all_steps_off_changes_nothing(self):
        cleaned = reconstruct.clean_mesh(_grid_mesh(), 0.01, max_edge=0, min_triangles=0, smooth_iterations=0)
        self.assertEqual(len(cleaned.mesh.triangles), 2 * 9 * 9)
        np.testing.assert_allclose(np.asarray(cleaned.mesh.vertices), np.asarray(_grid_mesh().vertices))


if __name__ == "__main__":
    unittest.main()
