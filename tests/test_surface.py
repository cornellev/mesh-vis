"""Tier 1 tests: sampling a point cloud from a known surface.

The defining property is that every sampled point lies ON the source mesh.
That is what makes the cloud meshable and what the random-cube cloud lacks.
"""

import unittest

import numpy as np
import open3d as o3d

from cloudlab import metrics, surface


class TestMakeShape(unittest.TestCase):
    def test_known_shapes_build_non_empty_meshes(self):
        for name in surface.SHAPES:
            with self.subTest(shape=name):
                mesh = surface.make_shape(name)
                self.assertGreater(len(mesh.vertices), 0)
                self.assertGreater(len(mesh.triangles), 0)

    def test_unknown_shape_raises_with_helpful_message(self):
        with self.assertRaises(ValueError) as ctx:
            surface.make_shape("dodecahedron")
        self.assertIn("dodecahedron", str(ctx.exception))

    def test_kwargs_reach_the_factory(self):
        small = surface.make_shape("sphere", radius=0.5)
        big = surface.make_shape("sphere", radius=5.0)
        self.assertLess(small.get_max_bound()[0], big.get_max_bound()[0])


class TestSampleSurface(unittest.TestCase):
    def setUp(self):
        self.mesh = surface.make_shape("sphere", radius=2.0, resolution=60)

    def test_returns_the_requested_number_of_points(self):
        for method in surface.METHODS:
            with self.subTest(method=method):
                pcd = surface.sample_surface(self.mesh, 500, method=method, seed=0)
                self.assertEqual(len(pcd.points), 500)

    def test_every_point_lies_on_the_source_surface(self):
        # This is the whole point of tier 1 -- the ground truth is exact.
        for method in surface.METHODS:
            with self.subTest(method=method):
                pcd = surface.sample_surface(self.mesh, 2000, method=method, seed=0)
                d = metrics.distance_to_mesh(pcd, self.mesh)
                self.assertLess(float(d.max()), 1e-4)

    def test_poisson_disk_is_more_evenly_spaced_than_uniform(self):
        n = 1500
        uni = surface.sample_surface(self.mesh, n, method="uniform", seed=0)
        pds = surface.sample_surface(self.mesh, n, method="poisson_disk", seed=0)
        spread_uni = metrics.spacing(uni).p95 / metrics.spacing(uni).p05
        spread_pds = metrics.spacing(pds).p95 / metrics.spacing(pds).p05
        self.assertLess(spread_pds, spread_uni)

    def test_same_seed_reproduces_the_same_cloud(self):
        a = surface.sample_surface(self.mesh, 300, seed=7)
        b = surface.sample_surface(self.mesh, 300, seed=7)
        np.testing.assert_allclose(np.asarray(a.points), np.asarray(b.points))

    def test_different_seeds_give_different_clouds(self):
        a = surface.sample_surface(self.mesh, 300, seed=1)
        b = surface.sample_surface(self.mesh, 300, seed=2)
        self.assertFalse(np.allclose(np.asarray(a.points), np.asarray(b.points)))

    def test_normals_are_available_and_unit_length(self):
        pcd = surface.sample_surface(self.mesh, 400, seed=0, with_normals=True)
        self.assertTrue(pcd.has_normals())
        norms = np.linalg.norm(np.asarray(pcd.normals), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            surface.sample_surface(self.mesh, 100, method="magic")

    def test_non_positive_count_raises(self):
        with self.assertRaises(ValueError):
            surface.sample_surface(self.mesh, 0)


class TestAddNoise(unittest.TestCase):
    def setUp(self):
        self.mesh = surface.make_shape("sphere", radius=2.0, resolution=60)
        self.clean = surface.sample_surface(self.mesh, 3000, seed=0)

    def test_zero_sigma_leaves_the_cloud_untouched(self):
        noisy = surface.add_gaussian_noise(self.clean, sigma=0.0, seed=0)
        np.testing.assert_allclose(np.asarray(noisy.points), np.asarray(self.clean.points))

    def test_noise_pushes_points_off_the_surface_by_about_sigma(self):
        sigma = 0.02
        noisy = surface.add_gaussian_noise(self.clean, sigma=sigma, seed=0)
        d = metrics.distance_to_mesh(noisy, self.mesh)
        # Offsets are isotropic; only the surface-normal component shows up as
        # distance, so the mean lands near sigma*sqrt(2/pi), not sigma.
        self.assertAlmostEqual(float(d.mean()), sigma * np.sqrt(2 / np.pi), delta=0.2 * sigma)

    def test_noise_does_not_change_the_point_count(self):
        noisy = surface.add_gaussian_noise(self.clean, sigma=0.01, seed=0)
        self.assertEqual(len(noisy.points), len(self.clean.points))

    def test_input_cloud_is_not_mutated(self):
        before = np.asarray(self.clean.points).copy()
        surface.add_gaussian_noise(self.clean, sigma=0.05, seed=0)
        np.testing.assert_allclose(np.asarray(self.clean.points), before)

    def test_same_seed_reproduces_the_same_noise(self):
        a = surface.add_gaussian_noise(self.clean, sigma=0.01, seed=3)
        b = surface.add_gaussian_noise(self.clean, sigma=0.01, seed=3)
        np.testing.assert_allclose(np.asarray(a.points), np.asarray(b.points))

    def test_negative_sigma_raises(self):
        with self.assertRaises(ValueError):
            surface.add_gaussian_noise(self.clean, sigma=-0.1)


class TestWhyTheRandomCubeFails(unittest.TestCase):
    """Documents the reason the starter cloud cannot be meshed."""

    def test_volume_noise_has_no_surface_but_sampled_points_do(self):
        rng = np.random.default_rng(0)
        cube = o3d.geometry.PointCloud()
        cube.points = o3d.utility.Vector3dVector(rng.random((2000, 3)))
        cube.estimate_normals()

        mesh = surface.make_shape("sphere", radius=0.5, resolution=60)
        shell = surface.sample_surface(mesh, 2000, seed=0, with_normals=True)

        # A surface cloud is locally flat: the smallest PCA eigenvalue of a
        # neighbourhood is near zero. Volume noise is locally isotropic.
        self.assertLess(surface.local_flatness(shell), 0.1)
        self.assertGreater(surface.local_flatness(cube), 0.1)


if __name__ == "__main__":
    unittest.main()
