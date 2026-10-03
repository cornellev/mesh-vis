"""Checks for the Zenoh lidar path: wire bytes -> CloudFrame, and
ZenohClient -> LidarSubscriber.

Adapted from the Autonomy Dashboard's tests/test_lidar_zenoh.py. Unit tests
publish frames encoded exactly like `encode_points` in virtual-rgbd-sensor's
rslidar_sdk_node.rs on an isolated session (no multicast discovery), so they
never touch the LAN and need no lidar. DecodeTests need no Zenoh at all:

    python3 -m unittest tests.test_zenoh_source

Live mode subscribes with the default Zenoh config (or ZENOH_CONFIG) and
prints the frames the real `rslidar_viz` publisher sends (start it first):

    python3 tests/test_zenoh_source.py --live [seconds]
"""

import importlib.util
from pathlib import Path
import struct
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh_vis import frame as fr

HAS_ZENOH = importlib.util.find_spec("zenoh") is not None
NAN = float("nan")


def encode_points(stamp_sec, stamp_nanosec, data, point_step):
    """Mirror of `encode_points` in rslidar_sdk_node.rs."""
    width = len(data) // point_step
    return struct.pack("<iIII", stamp_sec, stamp_nanosec, width, point_step) + data


def segmented_data(points):
    """XYZI f32 + trailing i32 cluster id per point, like `build_segmented_data`."""
    return b"".join(struct.pack("<ffffi", *p) for p in points)


def pad_to_blocks(points):
    """Pad to a whole number of 32-channel blocks with NaN placeholders, the
    way the publisher fills missing returns when dense_points is false."""
    points = list(points)
    return points + [(NAN, NAN, NAN, 0.0, -1)] * (-len(points) % 32)


def segmented(points, stamp_sec=1, stamp_nanosec=0):
    return encode_points(stamp_sec, stamp_nanosec, segmented_data(pad_to_blocks(points)), 20)


class DecodeTests(unittest.TestCase):
    """frame.decode_segmented on publisher-compatible bytes. No Zenoh needed."""

    def test_segmented_frame_is_decoded(self):
        points = [
            (1.0, 2.0, 3.0, 10.0, 0),
            (NAN, 0.0, 0.0, 0.0, -1),   # invalid return, kept in place
            (-4.5, 0.5, -1.25, 99.0, 7),
        ]
        f = fr.decode_segmented(encode_points(12, 500_000_000, segmented_data(pad_to_blocks(points)), 20))

        self.assertEqual((f.stamp_sec, f.stamp_nanosec), (12, 500_000_000))
        self.assertEqual(len(f), 32)
        self.assertEqual(f.cluster_id.dtype, np.int32)
        self.assertEqual(f.cluster_id[:3].tolist(), [0, -1, 7])
        self.assertTrue(np.isnan(f.xyz[1, 0]))
        mask = f.finite_mask()
        self.assertEqual(np.flatnonzero(mask).tolist(), [0, 2])
        np.testing.assert_array_equal(f.xyz[mask], [[1, 2, 3], [-4.5, 0.5, -1.25]])
        np.testing.assert_array_equal(f.intensity[mask], [10, 99])

    def test_labels_are_decoded_as_integers_not_floats(self):
        big = 2**24 + 1   # float32 rounds this to 2**24
        f = fr.decode_segmented(segmented([(0, 0, 0, 0, big), (0, 0, 0, 0, 2**31 - 1),
                                           (0, 0, 0, 0, -1)]))
        self.assertEqual(f.cluster_id[:3].tolist(), [big, 2**31 - 1, -1])
        self.assertNotEqual(int(np.float32(big)), big)

    def test_infinite_coordinates_are_kept_but_not_finite(self):
        f = fr.decode_segmented(segmented([(1, 1, float("inf"), 0, 4), (1, 1, 1, 0, 4)]))
        self.assertTrue(np.isinf(f.xyz[0, 2]))
        self.assertEqual(f.finite_mask()[:2].tolist(), [False, True])

    def test_empty_frame_is_well_formed(self):
        self.assertEqual(len(fr.decode_segmented(encode_points(5, 0, b"", 20))), 0)

    def test_arbitrary_record_counts_are_valid_without_layout_inference(self):
        for count in (1, 32, 33):
            payload = encode_points(1, 0, segmented_data([(1, 1, 1, 0, 1)] * count), 20)
            f = fr.decode_segmented(payload)
            self.assertEqual(len(f), count)
            self.assertIsNone(f.dense_points)
            self.assertIs(fr.decode_segmented(payload, dense_points=True).dense_points, True)

    def test_malformed_payloads_are_rejected(self):
        good = segmented([(1, 1, 1, 1, 1)])
        # (payload, words the error must contain): the reason matters, not
        # just the rejection -- a raw frame also fails the length check.
        cases = {
            "shorter than header": (b"short", "header"),
            "raw 16-byte records": (encode_points(2, 0, struct.pack("<ffff", 0.5, -0.5, 1.5, 42.0) * 32, 16),
                                    "point_step is 16"),
            "unknown point_step": (struct.pack("<iIII", 2, 0, 1, 12) + b"\0" * 12, "point_step is 12"),
            "truncated": (good[:-4], "header says"),
            "trailing bytes": (good + b"\0" * 4, "header says"),
        }
        for name, (payload, reason) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, reason):
                fr.decode_segmented(payload)


@unittest.skipUnless(HAS_ZENOH, "pip install eclipse-zenoh")
class LidarSubscriberTests(unittest.TestCase):
    def setUp(self):
        import zenoh
        from mesh_vis import zenoh_source

        # Isolated session: no multicast discovery and no listening socket, so
        # the test only ever sees its own puts.
        config = zenoh.Config()
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("listen/endpoints", "[]")

        self.key = zenoh_source.LIDAR_KEY
        self.client = zenoh_source.ZenohClient(config)
        self.client.start()
        self.addCleanup(self.client.stop)

    def publish_and_wait(self, payload):
        before = self.client.lidar.get_latest()
        self.client.session.put(self.key, payload)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            latest = self.client.lidar.get_latest()
            if latest is not before:
                return latest
            time.sleep(0.01)
        return self.client.lidar.get_latest()

    def publish_and_wait_for_reject(self, payload):
        """Waits on the reject counter instead of a timeout, so a malformed
        payload is known to have been handled before the test checks it."""
        before = self.client.lidar.rejected
        self.client.session.put(self.key, payload)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if self.client.lidar.rejected > before:
                return
            time.sleep(0.01)
        self.fail("subscriber never handled the malformed payload")

    def test_segmented_frame_is_decoded(self):
        points = [
            (1.0, 2.0, 3.0, 10.0, 0),
            (float("nan"), 0.0, 0.0, 0.0, -1),   # invalid return, kept in place
            (-4.5, 0.5, -1.25, 99.0, 7),
        ]
        f = self.publish_and_wait(encode_points(12, 500_000_000, segmented_data(pad_to_blocks(points)), 20))

        self.assertIsInstance(f, fr.CloudFrame)
        self.assertEqual((f.stamp_sec, f.stamp_nanosec, f.seq), (12, 500_000_000, 1))
        self.assertEqual(f.source_key, self.key)
        self.assertEqual(len(f), 32)
        self.assertEqual(f.cluster_id[:3].tolist(), [0, -1, 7])
        np.testing.assert_array_equal(f.xyz[f.finite_mask()], [[1, 2, 3], [-4.5, 0.5, -1.25]])
        self.assertTrue(np.isfinite(f.received_monotonic))

    def test_raw_frame_is_rejected(self):
        data = struct.pack("<ffff", 0.5, -0.5, 1.5, 42.0) * 32
        self.publish_and_wait_for_reject(encode_points(3, 0, data, 16))

        self.assertIsNone(self.client.lidar.get_latest())
        self.assertIn("point_step", self.client.lidar.last_error)

    def test_malformed_payloads_keep_previous_frame(self):
        good = self.publish_and_wait(segmented([(1, 1, 1, 1, 1)]))
        self.assertIsNotNone(good)

        truncated = segmented([(2, 2, 2, 2, 2)], stamp_sec=2)[:-4]
        trailing = segmented([(2, 2, 2, 2, 2)], stamp_sec=2) + b"\0" * 4
        bad_step = struct.pack("<iIII", 2, 0, 1, 12) + b"\0" * 12
        for payload in (b"short", truncated, trailing, bad_step):
            self.publish_and_wait_for_reject(payload)
            self.assertIs(self.client.lidar.get_latest(), good)
        self.assertEqual(self.client.lidar.rejected, 4)

    def test_compacted_frame_is_accepted(self):
        payload = encode_points(1, 0, segmented_data([(1, 2, 3, 4, 5)] * 33), 20)
        latest = self.publish_and_wait(payload)
        self.assertEqual(len(latest), 33)
        self.assertIsNone(latest.dense_points)
        self.assertEqual(self.client.lidar.rejected, 0)

    def test_seq_advances_when_replay_repeats_a_stamp(self):
        payload = segmented([(1, 1, 1, 1, 1)], stamp_sec=9)
        first = self.publish_and_wait(payload)
        second = self.publish_and_wait(payload)

        self.assertEqual((first.seq, second.seq), (1, 2))
        self.assertEqual(first.stamp, second.stamp)


def live(seconds):
    from mesh_vis import zenoh_source

    client = zenoh_source.ZenohClient()
    client.start()
    print(f"listening on {client.key} for {seconds:.0f}s...")
    last_seq = 0
    frames = 0
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            latest = client.lidar.get_latest()
            if latest is not None and latest.seq != last_seq:
                last_seq = latest.seq
                frames += 1
                line = latest.summary()
                pts = latest.xyz[latest.finite_mask()]
                if len(pts):
                    mn, mx = pts.min(0), pts.max(0)
                    line += (f" | bbox x[{mn[0]:.2f},{mx[0]:.2f}] y[{mn[1]:.2f},{mx[1]:.2f}]"
                             f" z[{mn[2]:.2f},{mx[2]:.2f}]")
                line += f" | rejected {client.lidar.rejected}"
                print(line, flush=True)
            time.sleep(0.01)
    finally:
        client.stop()
    print(f"received {frames} frames, rejected {client.lidar.rejected}")
    return frames


if __name__ == "__main__":
    if "--live" in sys.argv:
        args = sys.argv[sys.argv.index("--live") + 1:]
        sys.exit(0 if live(float(args[0]) if args else 10) else 1)
    unittest.main()
