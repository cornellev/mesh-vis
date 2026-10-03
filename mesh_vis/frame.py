"""One segmented lidar frame: decoding it off the wire and saving it to disk.

The publisher (virtual-rgbd-sensor's `rslidar_sdk_node.rs`, `encode_points` +
`build_segmented_data`) sends a 16-byte little-endian header

    stamp_sec: i32, stamp_nanosec: u32, count: u32, point_step: u32

followed by `count` 20-byte records `x, y, z, intensity: f32, cluster_id: i32`.

Every record is kept, including the NaN placeholders the decoder writes for
missing returns (`dense_points: false` in its config.yaml). Filter with
`finite_mask()` for viewing and reconstruction, never before saving.

Record order only identifies firing channels/blocks when the publisher is
known to preserve placeholders. It does not provide calibrated ring indices,
azimuths, or missing-packet information. No layout is inferred from point count.

`frame_id`, `units`, `sensor_origin` and `dense_points` are not transmitted.
They are configured assumptions. `dense_points=None` means the publisher's
setting is unknown; count divisibility cannot establish it.
"""

from __future__ import annotations

import math
import pathlib
import struct
from dataclasses import dataclass

import numpy as np

HEADER_FORMAT = "<iIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
RECORD = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("intensity", "<f4"), ("cluster_id", "<i4"),
])
UNASSIGNED = -1

# A diagnostic preset for the current recording, not a wire-format constraint
# or proof that a sweep covers a complete revolution.
DEFAULT_EXPECTED_RECORDS = 115_200

FORMAT_VERSION = 2
_ARRAYS = ("xyz", "intensity", "cluster_id")
_FIELDS = (
    "stamp_sec", "stamp_nanosec", "seq", "source_key",
    "frame_id", "units", "sensor_origin", "dense_points",
)


@dataclass(frozen=True, eq=False, repr=False)
class CloudFrame:
    """One received message. Array views prohibit accidental writes through
    this object; callers supplying arrays must not mutate their backing storage."""

    xyz: np.ndarray            # (N, 3) float32, NaN rows kept
    intensity: np.ndarray      # (N,) float32
    cluster_id: np.ndarray     # (N,) int32, -1 = unassigned, ids local to this frame
    stamp_sec: int
    stamp_nanosec: int
    seq: int = 0               # local receive counter; replay repeats stamps
    received_monotonic: float = math.nan   # this process only, never saved
    source_key: str = "rslidar/points/segmented"
    frame_id: str = "rslidar"
    units: str = "m"
    sensor_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dense_points: bool | None = None

    def __post_init__(self) -> None:
        n = len(self.xyz)
        expected = {
            "xyz": ((n, 3), np.float32),
            "intensity": ((n,), np.float32),
            "cluster_id": ((n,), np.int32),
        }
        for name, (shape, dtype) in expected.items():
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy array, got {type(arr).__name__}")
            if arr.shape != shape or arr.dtype != dtype:
                raise ValueError(
                    f"{name} must be {np.dtype(dtype).name} with shape {shape}, "
                    f"got {arr.dtype.name} with shape {arr.shape}"
                )
            view = arr.view()
            view.flags.writeable = False
            object.__setattr__(self, name, view)
        if len(self.sensor_origin) != 3:
            raise ValueError(f"sensor_origin must have 3 values, got {self.sensor_origin!r}")
        if self.dense_points is not None and not isinstance(self.dense_points, bool):
            raise ValueError("dense_points must be True, False, or None (unknown)")

    def __len__(self) -> int:
        return len(self.xyz)

    def __repr__(self) -> str:
        return f"CloudFrame(seq={self.seq}, stamp={self.stamp}, records={len(self)})"

    @property
    def stamp(self) -> str:
        """Source timestamp as text. Kept as two integers, as on the wire."""
        return f"{self.stamp_sec}.{self.stamp_nanosec:09d}"

    def finite_mask(self) -> np.ndarray:
        """True where all of x, y, z are finite. Excludes NaN and +/-inf."""
        return np.isfinite(self.xyz).all(axis=1)

    def record_count_note(self, expected_records: int = DEFAULT_EXPECTED_RECORDS) -> str:
        """Optional diagnostic; zero disables it. This does not establish coverage."""
        if expected_records < 0:
            raise ValueError("expected_records must be nonnegative")
        if expected_records and len(self) != expected_records:
            return f"unexpected record count (got {len(self)}, expected {expected_records})"
        return ""

    def summary(self, expected_records: int = DEFAULT_EXPECTED_RECORDS) -> str:
        """One line for the terminal. Counts are over finite points only;
        NaN placeholders always carry -1 and would swamp `unassigned`."""
        finite = self.finite_mask()
        labels = self.cluster_id[finite]
        n_clusters = len(np.unique(labels[labels >= 0]))
        unassigned = int(np.count_nonzero(labels == UNASSIGNED))
        line = (
            f"seq {self.seq} | stamp {self.stamp} | {len(self)} records, "
            f"{int(finite.sum())} finite | {n_clusters} clusters, {unassigned} unassigned"
        )
        note = self.record_count_note(expected_records)
        if note:
            line += f" | {note}"
        return line


def decode_segmented(
    payload: bytes,
    source_key: str = "rslidar/points/segmented",
    seq: int = 0,
    received_monotonic: float = math.nan,
    dense_points: bool | None = None,
) -> CloudFrame:
    """Parse one segmented message. Raises ValueError rather than guessing,
    so a malformed message can never replace the last good frame."""
    if len(payload) < HEADER_SIZE:
        raise ValueError(f"payload is {len(payload)} bytes, shorter than the {HEADER_SIZE}-byte header")
    stamp_sec, stamp_nanosec, count, point_step = struct.unpack_from(HEADER_FORMAT, payload, 0)
    if point_step != RECORD.itemsize:
        raise ValueError(
            f"point_step is {point_step}, expected {RECORD.itemsize} "
            f"(x, y, z, intensity, cluster_id); 16 is the raw, unsegmented key"
        )
    expected_len = HEADER_SIZE + count * RECORD.itemsize
    if len(payload) != expected_len:
        raise ValueError(
            f"payload is {len(payload)} bytes, header says {count} records = {expected_len} bytes"
        )
    # frombuffer is a read-only view of the message bytes; build owned copies.
    records = np.frombuffer(payload, dtype=RECORD, count=count, offset=HEADER_SIZE)
    return CloudFrame(
        xyz=np.stack([records["x"], records["y"], records["z"]], axis=1).astype(np.float32, copy=False),
        intensity=records["intensity"].astype(np.float32),
        cluster_id=records["cluster_id"].astype(np.int32),
        stamp_sec=stamp_sec,
        stamp_nanosec=stamp_nanosec,
        seq=seq,
        received_monotonic=received_monotonic,
        source_key=source_key,
        dense_points=dense_points,
    )


def save_npz(frame: CloudFrame, out_dir: str | pathlib.Path) -> pathlib.Path:
    """Write the whole frame, NaN rows included. Never overwrites: replay
    repeats source stamps and `seq` restarts every run, so a clash gets a
    numeric suffix instead."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{frame.stamp_sec}_{frame.stamp_nanosec:09d}_seq{frame.seq:06d}"
    arrays = {
        "format_version": np.int64(FORMAT_VERSION),
        "xyz": frame.xyz,
        "intensity": frame.intensity,
        "cluster_id": frame.cluster_id,
        "stamp_sec": np.int64(frame.stamp_sec),
        "stamp_nanosec": np.int64(frame.stamp_nanosec),
        "seq": np.int64(frame.seq),
        "source_key": np.str_(frame.source_key),
        "frame_id": np.str_(frame.frame_id),
        "units": np.str_(frame.units),
        "sensor_origin": np.asarray(frame.sensor_origin, dtype=np.float64),
        "dense_points": np.int8(-1 if frame.dense_points is None else frame.dense_points),
    }
    for attempt in range(1000):
        path = out_dir / (stem + (f"_{attempt}" if attempt else "") + ".npz")
        try:
            with open(path, "xb") as f:
                np.savez(f, **arrays)
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"could not find a free filename for {stem} in {out_dir}")


def load_npz(path: str | pathlib.Path) -> CloudFrame:
    """Reopen a snapshot without Zenoh. `received_monotonic` is not restored:
    it only meant something inside the process that received the frame."""
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in ("format_version", *_ARRAYS, *_FIELDS) if k not in data.files]
        if missing:
            raise ValueError(f"{path} is not a mesh-vis frame snapshot; missing {missing}")
        version = int(data["format_version"])
        if version not in (1, FORMAT_VERSION):
            raise ValueError(f"{path} has format_version {version}, this code reads {FORMAT_VERSION}")
        dense_value = int(data["dense_points"])
        if dense_value not in ((0, 1) if version == 1 else (-1, 0, 1)):
            raise ValueError(f"invalid dense_points metadata: {dense_value}")
        return CloudFrame(
            xyz=data["xyz"],
            intensity=data["intensity"],
            cluster_id=data["cluster_id"],
            stamp_sec=int(data["stamp_sec"]),
            stamp_nanosec=int(data["stamp_nanosec"]),
            seq=int(data["seq"]),
            source_key=str(data["source_key"]),
            frame_id=str(data["frame_id"]),
            units=str(data["units"]),
            sensor_origin=tuple(float(v) for v in data["sensor_origin"]),
            dense_points=None if dense_value == -1 else bool(dense_value),
        )
