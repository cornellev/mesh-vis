"""Tests for the MEASURE and CHECK primitives."""

import unittest

import numpy as np
import open3d as o3d

from cloudlab import metrics


def _cloud(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=float))
    return pcd


class TestDistanceToMesh(unittest.TestCase):
    def setUp(self):
        # Unit sphere, finely tessellated so the polyhedral error stays tiny.
        self.sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=80)

    def test_centre_of_sphere_is_one_radius_away(self):
        d = metrics.distance_to_mesh(_cloud([[0, 0, 0]]), self.sphere)
        self.assertAlmostEqual(float(d[0]), 1.0, places=2)

    def test_point_on_surface_has_zero_distance(self):
        d = metrics.distance_to_mesh(_cloud([[0, 0, 1.0]]), self.sphere)
        self.assertLess(float(d[0]), 1e-3)

    def test_distance_is_unsigned(self):
        inside = metrics.distance_to_mesh(_cloud([[0, 0, 0.5]]), self.sphere)
        outside = metrics.distance_to_mesh(_cloud([[0, 0, 1.5]]), self.sphere)
        self.assertGreater(float(inside[0]), 0.0)
        self.assertGreater(float(outside[0]), 0.0)

    def test_returns_one_distance_per_point(self):
        d = metrics.distance_to_mesh(_cloud([[0, 0, 0], [0, 0, 2], [3, 0, 0]]), self.sphere)
        self.assertEqual(d.shape, (3,))

    def test_empty_cloud_returns_empty_array(self):
        d = metrics.distance_to_mesh(_cloud(np.zeros((0, 3))), self.sphere)
        self.assertEqual(d.shape, (0,))


class TestSpacing(unittest.TestCase):
    def test_regular_grid_spacing_is_the_grid_pitch(self):
        # A 10x10 lattice with 0.25 pitch: every point's nearest neighbour is 0.25 away.
        g = np.arange(10) * 0.25
        xx, yy = np.meshgrid(g, g)
        pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
        s = metrics.spacing(_cloud(pts))
        self.assertAlmostEqual(s.median, 0.25, places=6)
        self.assertAlmostEqual(s.mean, 0.25, places=6)
        self.assertEqual(s.count, 100)

    def test_percentiles_are_ordered(self):
        rng = np.random.default_rng(0)
        s = metrics.spacing(_cloud(rng.random((500, 3))))
        self.assertLessEqual(s.p05, s.median)
        self.assertLessEqual(s.median, s.p95)

    def test_denser_cloud_has_smaller_spacing(self):
        rng = np.random.default_rng(1)
        sparse = metrics.spacing(_cloud(rng.random((200, 3))))
        dense = metrics.spacing(_cloud(rng.random((2000, 3))))
        self.assertLess(dense.median, sparse.median)

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            metrics.spacing(_cloud([[0, 0, 0]]))


if __name__ == "__main__":
    unittest.main()
