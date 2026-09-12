"""Tier 2 tests: a simulated spinning lidar.

The three properties worth simulating are occlusion (nothing behind an
obstacle), ring anisotropy (tight along a ring, wide across rings) and
range falloff. Each gets a test.
"""

import unittest

import numpy as np
import open3d as o3d

from cloudlab import lidar, metrics


class TestLidarSpec(unittest.TestCase):
    def setUp(self):
        self.spec = lidar.LidarSpec(n_beams=16, fov_up_deg=15.0, fov_down_deg=-15.0, n_azimuth=1024)

    def test_elevations_span_the_declared_fov(self):
        el = np.degrees(self.spec.elevations())
        self.assertEqual(el.shape, (16,))
        self.assertAlmostEqual(float(el.min()), -15.0, places=6)
        self.assertAlmostEqual(float(el.max()), 15.0, places=6)

    def test_elevations_are_evenly_spaced_and_ascending(self):
        el = self.spec.elevations()
        gaps = np.diff(el)
        self.assertTrue(np.all(gaps > 0))
        np.testing.assert_allclose(gaps, gaps[0])

    def test_single_beam_sits_at_the_middle_of_the_fov(self):
        spec = lidar.LidarSpec(n_beams=1, fov_up_deg=10.0, fov_down_deg=-30.0)
        self.assertAlmostEqual(float(np.degrees(spec.elevations())[0]), -10.0, places=6)

    def test_azimuths_cover_one_full_turn_without_repeating_it(self):
        az = self.spec.azimuths()
        self.assertEqual(az.shape, (1024,))
        self.assertAlmostEqual(float(az.min()), 0.0, places=6)
        self.assertLess(float(az.max()), 2 * np.pi)
        np.testing.assert_allclose(np.diff(az), 2 * np.pi / 1024)

    def test_ray_count_is_beams_times_azimuth(self):
        self.assertEqual(self.spec.n_rays, 16 * 1024)

    def test_invalid_geometry_raises(self):
        with self.assertRaises(ValueError):
            lidar.LidarSpec(n_beams=0)
        with self.assertRaises(ValueError):
            lidar.LidarSpec(n_azimuth=0)
        with self.assertRaises(ValueError):
            lidar.LidarSpec(fov_up_deg=-10.0, fov_down_deg=10.0)
        with self.assertRaises(ValueError):
            lidar.LidarSpec(max_range=0.0)


class TestRayDirections(unittest.TestCase):
    def setUp(self):
        self.spec = lidar.LidarSpec(n_beams=8, n_azimuth=64)

    def test_shape_is_beams_by_azimuth_by_three(self):
        self.assertEqual(lidar.ray_directions(self.spec).shape, (8, 64, 3))

    def test_every_direction_is_a_unit_vector(self):
        d = lidar.ray_directions(self.spec)
        np.testing.assert_allclose(np.linalg.norm(d, axis=-1), 1.0, atol=1e-12)

    def test_row_elevation_matches_the_spec(self):
        d = lidar.ray_directions(self.spec)
        got = np.arcsin(d[:, :, 2])
        want = np.repeat(self.spec.elevations()[:, None], 64, axis=1)
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_azimuth_advances_around_z(self):
        d = lidar.ray_directions(lidar.LidarSpec(n_beams=1, fov_up_deg=0.0, fov_down_deg=0.0, n_azimuth=4))
        np.testing.assert_allclose(d[0, 0], [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(d[0, 1], [0, 1, 0], atol=1e-12)


class TestScanGeometry(unittest.TestCase):
    def test_sensor_at_sphere_centre_sees_every_ray_at_one_radius(self):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5.0, resolution=120)
        spec = lidar.LidarSpec(n_beams=8, n_azimuth=256)
        scan = lidar.scan_scene(sphere, origin=(0, 0, 0), spec=spec)
        self.assertEqual(len(scan), spec.n_rays)
        self.assertTrue(np.all(scan.range < 5.0 + 1e-3))
        self.assertTrue(np.all(scan.range > 5.0 * 0.995))

    def test_hit_points_lie_on_the_scene_surface(self):
        scene = lidar.make_test_scene()
        scan = lidar.scan_scene(scene, origin=(0, 0, 1.8), spec=lidar.LidarSpec(n_beams=16, n_azimuth=512))
        merged = o3d.geometry.TriangleMesh()
        for m in scene:
            merged += m
        d = metrics.distance_to_mesh(scan.to_point_cloud(), merged)
        self.assertLess(float(d.max()), 1e-3)

    def test_rays_that_hit_nothing_are_dropped(self):
        empty = o3d.geometry.TriangleMesh()
        scan = lidar.scan_scene(empty, origin=(0, 0, 0), spec=lidar.LidarSpec(n_beams=4, n_azimuth=16))
        self.assertEqual(len(scan), 0)
        self.assertEqual(scan.points.shape, (0, 3))

    def test_targets_beyond_max_range_are_not_returned(self):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=50.0, resolution=40)
        spec = lidar.LidarSpec(n_beams=4, n_azimuth=32, max_range=10.0)
        self.assertEqual(len(lidar.scan_scene(sphere, (0, 0, 0), spec)), 0)

    def test_ring_ids_and_ranges_agree_with_the_points(self):
        scene = lidar.make_test_scene()
        spec = lidar.LidarSpec(n_beams=16, n_azimuth=256)
        scan = lidar.scan_scene(scene, origin=(0, 0, 1.8), spec=spec)
        self.assertTrue(np.all(scan.ring >= 0))
        self.assertTrue(np.all(scan.ring < spec.n_beams))
        measured = np.linalg.norm(scan.points - scan.origin, axis=1)
        np.testing.assert_allclose(measured, scan.range, rtol=1e-4)


class TestOcclusion(unittest.TestCase):
    """The property that makes a single sweep 2.5D rather than a surface."""

    def test_a_near_wall_hides_the_wall_behind_it(self):
        near = o3d.geometry.TriangleMesh.create_box(0.2, 20, 20)
        near.translate((5.0, -10, -10))
        far = o3d.geometry.TriangleMesh.create_box(0.2, 20, 20)
        far.translate((9.0, -10, -10))
        spec = lidar.LidarSpec(n_beams=8, fov_up_deg=10, fov_down_deg=-10, n_azimuth=512)
        scan = lidar.scan_scene([near, far], origin=(0, 0, 0), spec=spec)
        self.assertGreater(len(scan), 0)
        # Everything returned in front is the near wall; nothing from behind it.
        forward = scan.points[scan.points[:, 0] > 0]
        self.assertGreater(len(forward), 0)
        self.assertLess(float(forward[:, 0].max()), 5.3)

    def test_a_box_casts_a_shadow_on_the_ground(self):
        ground = o3d.geometry.TriangleMesh.create_box(60, 60, 0.2)
        ground.translate((-30, -30, -0.2))
        box = o3d.geometry.TriangleMesh.create_box(1.5, 1.5, 1.5)
        box.translate((4.25, -0.75, 0.0))
        spec = lidar.LidarSpec(n_beams=48, fov_up_deg=2.0, fov_down_deg=-25.0, n_azimuth=4096)
        scan = lidar.scan_scene([ground, box], origin=(0, 0, 2.0), spec=spec)

        p = scan.points
        on_ground = p[p[:, 2] < 0.05]
        band = (on_ground[:, 0] > 7.0) & (on_ground[:, 0] < 15.0)
        shadow = band & (np.abs(on_ground[:, 1]) < 0.4)
        lit = band & (np.abs(on_ground[:, 1]) > 2.0) & (np.abs(on_ground[:, 1]) < 5.0)

        self.assertEqual(int(shadow.sum()), 0, "ground behind the box should be empty")
        self.assertGreater(int(lit.sum()), 0, "ground beside the box should be sampled")


class TestSamplingCharacter(unittest.TestCase):
    def test_spacing_along_a_ring_is_tighter_than_across_rings(self):
        ground = o3d.geometry.TriangleMesh.create_box(80, 80, 0.2)
        ground.translate((-40, -40, -0.2))
        spec = lidar.LidarSpec(n_beams=16, fov_up_deg=-2.0, fov_down_deg=-20.0, n_azimuth=1024)
        scan = lidar.scan_scene(ground, origin=(0, 0, 2.0), spec=spec)
        along, across = lidar.ring_spacing(scan)
        self.assertLess(along, across)
        self.assertGreater(across / along, 3.0)

    def test_rings_spread_out_with_range(self):
        ground = o3d.geometry.TriangleMesh.create_box(200, 200, 0.2)
        ground.translate((-100, -100, -0.2))
        spec = lidar.LidarSpec(n_beams=32, fov_up_deg=-1.0, fov_down_deg=-25.0, n_azimuth=512)
        scan = lidar.scan_scene(ground, origin=(0, 0, 2.0), spec=spec)
        # Ring radius on flat ground grows faster than linearly with ring index;
        # consecutive-ring gaps must grow as the rings get further out.
        radius = np.array([
            np.median(np.linalg.norm(scan.points[scan.ring == r][:, :2], axis=1))
            for r in range(spec.n_beams) if np.any(scan.ring == r)
        ])
        radius.sort()
        gaps = np.diff(radius)
        self.assertGreater(gaps[-1], gaps[0])


class TestNoiseAndDropout(unittest.TestCase):
    def setUp(self):
        self.sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5.0, resolution=120)
        self.spec = lidar.LidarSpec(n_beams=8, n_azimuth=256)

    def test_range_noise_moves_points_along_the_ray(self):
        sigma = 0.05
        scan = lidar.scan_scene(self.sphere, (0, 0, 0), self.spec, range_sigma=sigma, seed=0)
        offset = scan.range - 5.0
        self.assertAlmostEqual(float(offset.std()), sigma, delta=0.3 * sigma)
        direction = scan.points / np.linalg.norm(scan.points, axis=1, keepdims=True)
        clean = lidar.scan_scene(self.sphere, (0, 0, 0), self.spec)
        clean_dir = clean.points / np.linalg.norm(clean.points, axis=1, keepdims=True)
        np.testing.assert_allclose(direction, clean_dir, atol=1e-5)

    def test_dropout_removes_points(self):
        full = lidar.scan_scene(self.sphere, (0, 0, 0), self.spec)
        thin = lidar.scan_scene(self.sphere, (0, 0, 0), self.spec, dropout=0.5, seed=0)
        self.assertLess(len(thin), len(full))
        self.assertGreater(len(thin), 0)

    def test_total_dropout_returns_nothing(self):
        scan = lidar.scan_scene(self.sphere, (0, 0, 0), self.spec, dropout=1.0, seed=0)
        self.assertEqual(len(scan), 0)

    def test_same_seed_reproduces_the_same_scan(self):
        a = lidar.scan_scene(self.sphere, (0, 0, 0), self.spec, range_sigma=0.05, dropout=0.3, seed=11)
        b = lidar.scan_scene(self.sphere, (0, 0, 0), self.spec, range_sigma=0.05, dropout=0.3, seed=11)
        np.testing.assert_allclose(a.points, b.points)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            lidar.scan_scene(self.sphere, (0, 0, 0), self.spec, range_sigma=-1.0)
        with self.assertRaises(ValueError):
            lidar.scan_scene(self.sphere, (0, 0, 0), self.spec, dropout=1.5)


class TestScanConversion(unittest.TestCase):
    def test_point_cloud_round_trips_the_points(self):
        scene = lidar.make_test_scene()
        scan = lidar.scan_scene(scene, (0, 0, 1.8), lidar.LidarSpec(n_beams=8, n_azimuth=128))
        pcd = scan.to_point_cloud()
        np.testing.assert_allclose(np.asarray(pcd.points), scan.points)

    def test_ring_colouring_gives_one_colour_per_ring(self):
        scene = lidar.make_test_scene()
        spec = lidar.LidarSpec(n_beams=8, n_azimuth=128)
        scan = lidar.scan_scene(scene, (0, 0, 1.8), spec)
        pcd = scan.to_point_cloud(color_by="ring")
        colors = np.asarray(pcd.colors)
        self.assertEqual(len(colors), len(scan))
        self.assertEqual(len(np.unique(colors, axis=0)), len(np.unique(scan.ring)))

    def test_unknown_colouring_raises(self):
        scan = lidar.scan_scene(lidar.make_test_scene(), (0, 0, 1.8), lidar.LidarSpec(n_beams=4, n_azimuth=64))
        with self.assertRaises(ValueError):
            scan.to_point_cloud(color_by="mood")


class TestTestScene(unittest.TestCase):
    def test_scene_has_a_ground_plane_and_obstacles(self):
        scene = lidar.make_test_scene()
        self.assertGreater(len(scene), 1)
        for m in scene:
            self.assertGreater(len(m.triangles), 0)

    def test_scene_is_wide_and_mostly_flat(self):
        merged = o3d.geometry.TriangleMesh()
        for m in lidar.make_test_scene():
            merged += m
        extent = merged.get_axis_aligned_bounding_box().get_extent()
        self.assertGreater(extent[0], 20.0)
        self.assertGreater(extent[1], 20.0)
        self.assertLess(extent[2], extent[0] / 4)


if __name__ == "__main__":
    unittest.main()
