# mesh-vis

Segmented LiDAR point clouds to mesh visualization. The current application
captures RoboSense frames over Zenoh and reopens lossless snapshots in Open3D.
Offline mesh reconstruction is the next stage; it is not implemented yet.

## Organization

```text
mesh_vis/               active application: frame, zenoh_source, view, CLI
tests/                  active application tests
docs/v1overview.md      pipeline plan and completion criteria
captures/               generated NPZ snapshots (gitignored)
outputs/                reserved for generated meshes/reports (gitignored)
experiments/legacy/     preserved playground, examples, tests, and KITTI data
```

The application has no dependency on the legacy playground. The earlier
`examples/05_capture.py` is now the `python3 -m mesh_vis` entry point.

## Environment

Run application commands from this repository's root. Use a Python environment
with the dependencies in `requirements.txt` (`python3 -m pip install -r
requirements.txt` in your chosen environment). The existing local environment
uses Python 3.14.2, NumPy 2.4.1, Open3D 0.19.0+1a9eb99, and eclipse-zenoh 1.10.1.
The Open3D development build is recorded for reference, not pinned as a portable
PyPI dependency. Viewing needs a desktop session; text inspection does not.

## How a sweep gets here

mesh-vis receives already segmented frames. It does not read the PCAP and it does not segment points.

1. Start with a PCAP file on your machine. Recordings are not in this repo.
2. Segmentation is done by `rslidar_viz`, a compiled program from the neighboring `virtual-rgbd-sensor` repository. You point it at the PCAP. It decodes the RoboSense packets into points, runs the two-layer graph clustering in that repository, and publishes each frame.
3. `python3 -m mesh_vis capture` subscribes over Zenoh to `rslidar/points/segmented` and displays those frames.
4. Press **S** to write the frame on screen to `captures/*.npz`. That file is the segmented cloud (`x, y, z, intensity, cluster_id`) used from here. `inspect` reopens it with this repo alone.

A live lidar can take the place of the PCAP. Until `rslidar_viz` is publishing, capture has nothing to show.

## What a fresh clone still needs

This repository plus a PCAP is not enough to produce a snapshot.

- This repo, with the dependencies in `requirements.txt` installed. Viewing needs a desktop session.
- The PCAP on your machine.
- `data/rslidar-local.yaml` with `pcap_path` set to that file. When `pcap_path` is set, `rslidar_viz` replays the file. The yaml and the PCAP are gitignored, so a fresh clone has neither. See [data/README.md](data/README.md).
- `virtual-rgbd-sensor` checked out beside this repo and already built, so `../virtual-rgbd-sensor/target/release/rslidar_viz` exists. mesh-vis does not contain that program.  TO BE LOOKED AT AND FIXED

Someone who already has a `captures/*.npz` file can inspect it without the PCAP or `rslidar_viz`.

## Capture

From this repository's root, start the publisher, then the subscriber:

```bash
"../virtual-rgbd-sensor/target/release/rslidar_viz" "data/rslidar-local.yaml"
```

```bash
python3 -m mesh_vis capture
```

- **Space:** pause/resume display; the subscriber keeps receiving.
- **S:** save the frame selected at the keypress, even during playback.
- **Esc / Q:** close the window and subscription.
- Orbit/zoom using the Open3D mouse controls; frame updates preserve your camera.

Snapshots go to `captures/`; `--out PATH` selects another directory. Save messages
include the selected sequence and filename. Repeated saves create unique files.
Gray points are unassigned; other colors identify clusters within this frame.
Object colors can change between frames because labels are not tracked over time.

The default key is `rslidar/points/segmented`. Override it with `--key` or
`LIDAR_ZENOH_KEY`. `ZENOH_CONFIG` accepts the existing Zenoh configuration-file
mechanism when explicit endpoints are needed; choose addresses for your setup.

## Inspect a snapshot

Stop the publisher if desired, then reopen the exact path printed when saving:

```bash
python3 -m mesh_vis inspect captures/YOUR_SAVED_FILE.npz
python3 -m mesh_vis inspect captures/YOUR_SAVED_FILE.npz --no-show
```

Inspection does not open a Zenoh session. It reports the saved timestamp, sequence,
point/cluster counts, and configured assumptions. `--no-show` prints these without
creating a window. Both earlier version-1 and new version-2 snapshots are readable.

## Sensor assumptions and count diagnostics

Any point count is accepted when the segmented payload header/length are valid.
Original records, including NaN/inf coordinates, are saved; the viewer displays
only finite XYZ. Cluster labels remain int32.

The publisher does not transmit its `dense_points` setting. Snapshots record it
as unknown unless you explicitly provide `--dense-points false` or
`--dense-points true` to match the publisher. This records an assumption; it does
not change the sensor or infer scan layout. Firing channels are not necessarily
elevation-ordered rings, and record order does not supply calibrated azimuth.

`--expected-records N` prints an optional count mismatch diagnostic. Its default,
115200, is a preset for the current recording, not proof of a complete revolution.
Different sensor settings or missing packets can change the count. Disable this
diagnostic in both display and save messages with:

```bash
python3 -m mesh_vis capture --expected-records 0
```

`--full-records` remains an alias for compatibility. Neither flag filters frames.
Source timestamps may repeat during replay; sequence numbers identify arrivals.

## Tests and desktop acceptance

```bash
python3 -m unittest discover -v
```

The active tests cover decoding, persistence, isolated Zenoh delivery, CLI behavior,
and viewer control logic without opening windows. Zenoh integration tests need
permission to initialize local shared memory and skip if Zenoh is not installed.

On your desktop, verify:

1. Start capture: points appear, camera position persists, and rejected-message
   count stays zero with the current publisher.
2. Pause, orbit, save, resume, and save again during playback. Save twice while
   paused to confirm filenames differ without changing the selected sequence.
3. Close capture, stop the publisher, and inspect the saved file. Geometry, colors,
   source timestamp, and counts should match the captured frame. Camera pose is
   not saved. Run text-only inspection too.
4. Run capture with `--expected-records 0`; neither frame summaries nor save
   messages should contain a count-mismatch diagnostic.
5. Keep a few sweeps containing a dense nearby surface, a corner/curved object,
   and sparse distant returns for reconstruction experiments.

Legacy checks are separate and still runnable:

```bash
cd experiments/legacy
python3 -m unittest discover -s tests -t . -v
python3 examples/03_tier2_fake_lidar.py --no-show
```

See [the v1 plan](docs/v1overview.md) for the remaining reconstruction steps and
[the preserved playground](experiments/legacy/README.md) for earlier experiments.
