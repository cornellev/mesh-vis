"""Tests for the frame type and its .npz snapshots.

Decoding from wire bytes is tested in test_zenoh_source.py, next to the
publisher-compatible encoders it shares with the Zenoh integration tests.
"""

import math
import pathlib
import tempfile
import unittest

import numpy as np

from mesh_vis import frame as fr


def _frame(n_blocks=1, seq=7, **kwargs):
    """A frame shaped like real output: 32 records per block, a few NaN
    placeholders, one inf, a large label float32 cannot hold exactly."""
    n = n_blocks * 32
    rng = np.random.default_rng(0)
    xyz = rng.normal(0.0, 5.0, (n, 3)).astype(np.float32)
    intensity = rng.uniform(0, 255, n).astype(np.float32)
    cluster_id = np.full(n, fr.UNASSIGNED, dtype=np.int32)
    cluster_id[:10] = 3
    cluster_id[10:12] = 2**24 + 1
    xyz[20:24] = np.nan           # missing returns: NaN rows, label -1
    xyz[24, 2] = np.inf
    return fr.CloudFrame(
        xyz=xyz, intensity=intensity, cluster_id=cluster_id,
        stamp_sec=1_690_000_000, stamp_nanosec=5, seq=seq, **kwargs,
    )


class TestCloudFrame(unittest.TestCase):
    def test_rejects_float_labels(self):
        f = _frame()
        with self.assertRaises(ValueError):
            fr.CloudFrame(xyz=f.xyz, intensity=f.intensity,
                          cluster_id=f.cluster_id.astype(np.float32),
                          stamp_sec=0, stamp_nanosec=0)

    def test_rejects_float64_points_and_mismatched_lengths(self):
        f = _frame()
        with self.assertRaises(ValueError):
            fr.CloudFrame(xyz=f.xyz.astype(np.float64), intensity=f.intensity,
                          cluster_id=f.cluster_id, stamp_sec=0, stamp_nanosec=0)
        with self.assertRaises(ValueError):
            fr.CloudFrame(xyz=f.xyz, intensity=f.intensity[:-1],
                          cluster_id=f.cluster_id, stamp_sec=0, stamp_nanosec=0)

    def test_arrays_are_read_only_but_the_callers_are_not(self):
        xyz = np.zeros((32, 3), dtype=np.float32)
        f = fr.CloudFrame(xyz=xyz, intensity=np.zeros(32, np.float32),
                          cluster_id=np.zeros(32, np.int32), stamp_sec=0, stamp_nanosec=0)
        with self.assertRaises(ValueError):
            f.xyz[0, 0] = 1.0
        xyz[0, 0] = 1.0  # caller's own array untouched by the freeze

    def test_finite_mask_drops_nan_and_inf_rows(self):
        mask = _frame().finite_mask()
        self.assertFalse(mask[20:25].any())
        self.assertEqual(int(mask.sum()), 32 - 5)

    def test_summary_counts_only_finite_points(self):
        line = _frame().summary(expected_records=32)
        # 2 clusters (3 and 2**24+1); the 5 non-finite rows are not "unassigned"
        self.assertIn("32 records, 27 finite", line)
        self.assertIn("2 clusters, 15 unassigned", line)
        self.assertNotIn("unexpected record count", line)

    def test_record_count_is_only_an_optional_diagnostic(self):
        self.assertIn("unexpected record count", _frame().summary())
        self.assertEqual(_frame(n_blocks=3600).record_count_note(), "")
        self.assertEqual(_frame().record_count_note(0), "")
        self.assertNotIn("unexpected record count", _frame().summary(0))
        with self.assertRaises(ValueError):
            _frame().record_count_note(-1)


class TestSnapshot(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = pathlib.Path(tmp.name)

    def test_round_trip_is_bit_identical(self):
        f = _frame(sensor_origin=(0.1, 0.0, 1.8))
        g = fr.load_npz(fr.save_npz(f, self.dir))
        for name in ("xyz", "intensity", "cluster_id"):
            a, b = getattr(f, name), getattr(g, name)
            self.assertEqual(a.dtype, b.dtype, name)
            self.assertEqual(a.shape, b.shape, name)
            self.assertEqual(a.tobytes(), b.tobytes(), name)  # NaN rows included
        self.assertEqual(int(g.cluster_id[10]), 2**24 + 1)
        for name in ("stamp_sec", "stamp_nanosec", "seq", "source_key",
                     "frame_id", "units", "sensor_origin", "dense_points"):
            self.assertEqual(getattr(f, name), getattr(g, name), name)
        self.assertTrue(math.isnan(g.received_monotonic))

    def test_same_stamp_and_seq_never_overwrites(self):
        f = _frame()
        paths = {fr.save_npz(f, self.dir) for _ in range(3)}
        self.assertEqual(len(paths), 3)
        for p in paths:
            self.assertEqual(fr.load_npz(p).cluster_id.tobytes(), f.cluster_id.tobytes())

    def test_unknown_and_configured_density_settings_round_trip(self):
        for setting in (None, False, True):
            f = _frame(dense_points=setting)
            self.assertIs(fr.load_npz(fr.save_npz(f, self.dir)).dense_points, setting)

    def test_version_one_snapshots_still_load(self):
        f = _frame(dense_points=False)
        path = fr.save_npz(f, self.dir)
        with np.load(path, allow_pickle=False) as data:
            contents = dict(data)
        contents["format_version"] = np.int64(1)
        contents["dense_points"] = np.bool_(False)
        old = self.dir / "version-one.npz"
        np.savez(old, **contents)
        loaded = fr.load_npz(old)
        self.assertIs(loaded.dense_points, False)
        self.assertEqual(loaded.xyz.tobytes(), f.xyz.tobytes())

    def test_unknown_format_version_is_refused(self):
        path = fr.save_npz(_frame(), self.dir)
        with np.load(path) as data:
            contents = dict(data)
        contents["format_version"] = np.int64(99)
        newer = self.dir / "newer.npz"
        np.savez(newer, **contents)
        with self.assertRaises(ValueError):
            fr.load_npz(newer)

    def test_other_npz_files_are_refused(self):
        other = self.dir / "other.npz"
        np.savez(other, points=np.zeros((4, 3)))
        with self.assertRaises(ValueError):
            fr.load_npz(other)


if __name__ == "__main__":
    unittest.main()
