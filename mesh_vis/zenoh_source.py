"""Live frames from the rslidar publisher over Zenoh.

Adapted from the Autonomy Dashboard: `ZenohClient` from
backend/zenoh_client.py and `LidarSubscriber` from
backend/subscribers/perception.py, without the ROS, camera and dashboard
parts. The session lifecycle and the single lock-protected latest-frame slot
are unchanged. Parsing moved to `frame.decode_segmented`, which is stricter
than the dashboard's decoder: see that module for the wire format.

The subscriber callback runs on a Zenoh thread. It only decodes and swaps the
slot; anything slower (drawing, reconstruction) belongs on the main thread,
which polls `get_latest()`.

`zenoh` is imported inside `start()`, so this module and everything else in
mesh_vis import without eclipse-zenoh installed.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from typing import TYPE_CHECKING

from mesh_vis.frame import CloudFrame, decode_segmented

if TYPE_CHECKING:
    import zenoh

LIDAR_KEY = os.getenv("LIDAR_ZENOH_KEY", "rslidar/points/segmented")

class ZenohClient:
    """Own the shared Zenoh session and the subscribers declared on it."""

    def __init__(self, config: zenoh.Config | None = None, key: str = LIDAR_KEY,
                 dense_points: bool | None = None):
        self._config = config
        self.key = key
        self.dense_points = dense_points
        self.session = None
        self.lidar = None
        self._subscribers = []

    def start(self):
        import zenoh

        try:
            # ZENOH_CONFIG may point at a json5 file, e.g. to connect to a
            # router when multicast discovery can't reach the publisher.
            config = self._config
            if config is None:
                config = zenoh.Config.from_env() if os.getenv("ZENOH_CONFIG") else zenoh.Config()
            self.session = zenoh.open(config)
            print("[ZENOH] session opened", flush=True)

            # All subscribers go here
            self.lidar = LidarSubscriber(self.session, self.key, self.dense_points)
            self._subscribers.append(self.lidar)
            print(f"[ZENOH] subscribed to {self.key}", flush=True)
        except Exception:
            self.stop()
            raise

    def stop(self):
        for subscriber in self._subscribers:
            subscriber.close()
        self._subscribers.clear()

        if self.session is not None:
            self.session.close()
            self.session = None


class LidarSubscriber:
    def __init__(self, session: zenoh.Session, key: str = "rslidar/points/segmented",
                 dense_points: bool | None = None):
        self.key = key
        self.dense_points = dense_points
        self._lock = threading.Lock()
        self._latest: CloudFrame | None = None
        self._frame_count = 0        # accepted frames; also each frame's seq
        self._rejected = 0
        self._last_error: str | None = None
        self._sub = session.declare_subscriber(key, self._callback)

    def _callback(self, sample):
        received = time.monotonic()
        try:
            frame = decode_segmented(sample.payload.to_bytes(), str(sample.key_expr),
                                     received_monotonic=received, dense_points=self.dense_points)
        except ValueError as err:
            # Malformed: count it and keep the last good frame.
            with self._lock:
                self._rejected += 1
                self._last_error = str(err)
            return
        with self._lock:
            self._frame_count += 1
            self._latest = dataclasses.replace(frame, seq=self._frame_count)

    def get_latest(self) -> CloudFrame | None:
        with self._lock:
            return self._latest

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def rejected(self) -> int:
        with self._lock:
            return self._rejected

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def close(self):
        self._sub.undeclare()
