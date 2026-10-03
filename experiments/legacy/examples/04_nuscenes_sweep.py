"""Tier 3 first look: one real nuScenes lidar sweep, no cleaning, sensor frame.

Shows: 32 rings around the sensor. Colour is height by default (blue = ground,
red = high). The dense blob at the centre is the recording car's own roof --
remove it before stacking sweeps.

Nothing here is transformed: x = right, y = forward, z = up, metres, relative
to the lidar. The ground sits about 1.84 m below the sensor.

    python3 examples/04_nuscenes_sweep.py
    python3 examples/04_nuscenes_sweep.py --index 200 --color intensity
    python3 examples/04_nuscenes_sweep.py --color ring
"""

import argparse
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from cloudlab import metrics, view

DEFAULT_DATA = __import__("pathlib").Path(__file__).resolve().parents[4] / "v1.0-mini"


def blue_to_red(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return np.stack([t, np.full_like(t, 0.3), 1.0 - t], axis=1).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="nuScenes v1.0-mini root")
    ap.add_argument("--index", type=int, default=0, help="which keyframe file, 0-403")
    ap.add_argument("--color", default="height", choices=["height", "intensity", "ring"])
    ap.add_argument("--save", default=None, help="render to PNG (needs a desktop session)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    files = sorted((__import__("pathlib").Path(args.data) / "samples" / "LIDAR_TOP").glob("*.pcd.bin"))
    if not files:
        sys.exit(f"no lidar files under {args.data}/samples/LIDAR_TOP")
    path = files[args.index]

    p = np.fromfile(path, dtype=np.float32).reshape(-1, 5)  # x, y, z, intensity, ring
    xyz, intensity, ring = p[:, :3].astype(np.float64), p[:, 3], p[:, 4]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if args.color == "height":
        pcd.colors = o3d.utility.Vector3dVector(blue_to_red(xyz[:, 2], -2.0, 4.0))
    elif args.color == "intensity":
        pcd.colors = o3d.utility.Vector3dVector(blue_to_red(intensity, 0.0, 100.0))
    else:
        pcd.colors = o3d.utility.Vector3dVector(blue_to_red(ring, 0.0, 31.0))

    dist = np.linalg.norm(xyz, axis=1)
    print(f"{path.name}  (file {args.index} of {len(files)})")
    print(f"  {len(p)} points")
    print(f"  x {xyz[:, 0].min():.1f} to {xyz[:, 0].max():.1f} m   y {xyz[:, 1].min():.1f} to {xyz[:, 1].max():.1f} m"
          f"   z {xyz[:, 2].min():.1f} to {xyz[:, 2].max():.1f} m")
    print(f"  distance from sensor: median {np.median(dist):.1f} m, 99% within {np.percentile(dist, 99):.1f} m")
    print(f"  {metrics.spacing(pcd)}")

    if args.no_show and not args.save:
        return
    view.show([pcd], f"nuScenes sweep ({args.color}): {path.name}", save=args.save)


if __name__ == "__main__":
    main()
