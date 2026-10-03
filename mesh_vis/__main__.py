"""Capture segmented LiDAR frames and inspect saved snapshots."""

from __future__ import annotations

import argparse
import pathlib
import time

from mesh_vis import frame

NO_FRAMES_HINT_S = 5.0


def run_live(args) -> None:
    from mesh_vis import view, zenoh_source

    dense_points = {"unknown": None, "true": True, "false": False}[args.dense_points]
    client = zenoh_source.ZenohClient(key=args.key, dense_points=dense_points)
    client.start()
    viewer = None
    try:
        viewer = view.LiveCloudViewer(title=f"live: {args.key}")
        print("Space pause/resume | S save selected frame | Esc/Q quit", flush=True)
        started = time.monotonic()
        hinted = False
        shown_seq = 0
        rejected_seen = 0
        while viewer.step():
            # Requests retain the frame selected at the keypress, independently
            # of incoming frames or whether playback is paused.
            for selected in viewer.take_save_requests():
                save_shown(selected, args)
            if not viewer.paused:
                latest = client.lidar.get_latest()
                if latest is not None and latest.seq != shown_seq:
                    viewer.show(latest)
                    shown_seq = latest.seq
                    print(latest.summary(args.expected_records), flush=True)
            if shown_seq == 0 and not hinted and time.monotonic() - started > NO_FRAMES_HINT_S:
                print(f"no frames on {args.key} after {NO_FRAMES_HINT_S:.0f}s -- is rslidar_viz running?",
                      flush=True)
                hinted = True

            rejected = client.lidar.rejected
            if rejected != rejected_seen:
                print(f"rejected {rejected - rejected_seen} malformed message(s): {client.lidar.last_error}",
                      flush=True)
                rejected_seen = rejected
            time.sleep(0.01)
    finally:
        try:
            if viewer is not None:
                viewer.close()
        finally:
            client.stop()
        print(f"received {client.lidar.frame_count} frames, rejected {client.lidar.rejected}")


def save_shown(shown, args) -> None:
    if shown is None:
        print("nothing on screen to save yet", flush=True)
        return
    path = frame.save_npz(shown, args.out)
    note = shown.record_count_note(args.expected_records)
    suffix = f" | {note}" if note else ""
    print(f"saved seq {shown.seq} -> {path}{suffix}", flush=True)


def run_offline(args) -> None:
    snapshot = frame.load_npz(args.path)
    print(f"{args.path}\n  {snapshot.summary(args.expected_records)}")
    dense = "unknown" if snapshot.dense_points is None else str(snapshot.dense_points).lower()
    print(f"  configured assumptions: frame_id={snapshot.frame_id} units={snapshot.units} "
          f"sensor_origin={snapshot.sensor_origin} dense_points={dense}")
    if args.no_show:
        return

    from mesh_vis import view

    viewer = view.LiveCloudViewer(title=f"snapshot: {pathlib.Path(args.path).name}")
    try:
        viewer.show(snapshot)
        while viewer.step():
            if viewer.take_save_requests():
                print(f"already saved: {args.path}", flush=True)
            time.sleep(0.01)
    finally:
        viewer.close()


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="view live frames and save with S")
    capture.add_argument("--out", default="captures", help="snapshot folder (default: captures/)")
    capture.add_argument("--key", default=None, help="Zenoh key (default: $LIDAR_ZENOH_KEY or rslidar/points/segmented)")
    capture.add_argument("--dense-points", choices=("unknown", "true", "false"), default="unknown",
                         help="record a known publisher setting; does not infer or change sensor layout")
    capture.set_defaults(run=run_live)
    inspect = commands.add_parser("inspect", help="reopen an NPZ snapshot without Zenoh")
    inspect.add_argument("path", help="saved .npz snapshot")
    inspect.add_argument("--no-show", action="store_true", help="print metadata/counts without a window")
    inspect.set_defaults(run=run_offline)
    for command in (capture, inspect):
        command.add_argument("--expected-records", "--full-records", dest="expected_records",
                             type=nonnegative_int, default=frame.DEFAULT_EXPECTED_RECORDS,
                             help="optional count diagnostic (default: 115200); 0 disables it, not a completeness check")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "capture" and args.key is None:
        from mesh_vis.zenoh_source import LIDAR_KEY
        args.key = LIDAR_KEY
    try:
        args.run(args)
    except KeyboardInterrupt:
        pass  # Resource cleanup is handled by the command's finally blocks.


if __name__ == "__main__":
    main()
