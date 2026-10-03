"""Tier 2 test cloud: one fake spinning-lidar sweep of a toy scene.

Shows: front-only hits (occlusion) and ring anisotropy -- tight along a ring,
wide between rings. --beams down makes that gap explode (breaks BPA / normals).

Run the same CLEAN -> mesh on this sweep, then compare:
  * your voxel / radii to the printed across-ring spacing (not along-ring)
  * the meshed scene to the viewer: boxes stay boxes, shadows stay empty

    python3 examples/03_tier2_fake_lidar.py
    python3 examples/03_tier2_fake_lidar.py --beams 8
    python3 examples/03_tier2_fake_lidar.py --beams 64 --range-sigma 0.02 --dropout 0.05
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from cloudlab import lidar, metrics, view


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beams", type=int, default=16)
    ap.add_argument("--azimuth", type=int, default=1024)
    ap.add_argument("--fov-up", type=float, default=2.0)
    ap.add_argument("--fov-down", type=float, default=-24.0)
    ap.add_argument("--height", type=float, default=1.8, help="sensor height, metres")
    ap.add_argument("--range-sigma", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--color", default="ring", choices=["ring", "range"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="render to PNG (needs a desktop session)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    spec = lidar.LidarSpec(
        n_beams=args.beams,
        fov_up_deg=args.fov_up,
        fov_down_deg=args.fov_down,
        n_azimuth=args.azimuth,
    )
    scene = lidar.make_test_scene()
    scan = lidar.scan_scene(
        scene,
        origin=(0.0, 0.0, args.height),
        spec=spec,
        range_sigma=args.range_sigma,
        dropout=args.dropout,
        seed=args.seed,
    )

    along, across = lidar.ring_spacing(scan)
    pcd = scan.to_point_cloud()
    print(f"{spec.n_rays} rays fired, {len(scan)} returns ({100*len(scan)/spec.n_rays:.1f}% hit)")
    print(f"  {metrics.spacing(pcd)}")
    print(f"  along-ring spacing  {along:.3f} m")
    print(f"  across-ring spacing {across:.3f} m   -> {across/along:.1f}x anisotropy")
    print(f"  range {scan.range.min():.1f} - {scan.range.max():.1f} m")
    print("\nThat ratio is the whole problem. A radius small enough to follow a ring")
    print("cannot bridge to the next one; a radius big enough to bridge rings smears")
    print("detail along them. Voxel-downsampling near the across-ring figure is the fix.")

    if args.no_show and not args.save:
        return
    view.show(
        [scan.to_point_cloud(color_by=args.color)],
        f"tier 2: {args.beams}-beam lidar",
        save=args.save,
    )


if __name__ == "__main__":
    main()
