"""Tier 1 test cloud: points on a mesh you still have (the answer key).

Shows: the orange cloud lies on a surface (distance ~0, variation ~0). The blue
cube is 01's volume noise -- no surface, here only so you can see the contrast.

Run CLEAN -> mesh on the orange cloud, then compare:
  * your mesh to `mesh` with metrics.distance_to_mesh (stay near 0, or --noise)
  * post-CLEAN spacing / variation to the numbers this script prints

    python3 examples/02_tier1_surface_cloud.py
    python3 examples/02_tier1_surface_cloud.py --shape torus --method uniform
    python3 examples/02_tier1_surface_cloud.py --noise 0.02
"""

import argparse
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from cloudlab import metrics, surface, view


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="sphere", choices=sorted(surface.SHAPES))
    ap.add_argument("--points", type=int, default=5000)
    ap.add_argument("--method", default="poisson_disk", choices=list(surface.METHODS))
    ap.add_argument("--noise", type=float, default=0.0, help="isotropic jitter, metres")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="render to PNG (needs a desktop session)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    mesh = surface.make_shape(args.shape, radius=2.0) if args.shape in ("sphere", "cylinder") \
        else surface.make_shape(args.shape)
    pcd = surface.sample_surface(
        mesh, args.points, method=args.method, seed=args.seed, with_normals=True
    )
    if args.noise > 0:
        pcd = surface.add_gaussian_noise(pcd, args.noise, seed=args.seed)

    rng = np.random.default_rng(args.seed)
    cube = o3d.geometry.PointCloud()
    cube.points = o3d.utility.Vector3dVector(rng.random((args.points, 3)))

    print(f"surface cloud ({args.shape}, {args.method})")
    print(f"  {metrics.spacing(pcd)}")
    print(f"  distance to source mesh: max {metrics.distance_to_mesh(pcd, mesh).max():.6f} m")
    print(f"  surface variation:       {surface.local_flatness(pcd):.4f}  (0 = lies on a surface)")
    print("random cube cloud (what 01_random_cloud.py builds)")
    print(f"  {metrics.spacing(cube)}")
    print(f"  surface variation:       {surface.local_flatness(cube):.4f}  (1/3 = fills a volume)")
    print("\nThe second number is why the cube cannot be meshed: every neighbourhood")
    print("is an isotropic blob, so there is no local plane for a mesher to follow.")

    if args.no_show and not args.save:
        return
    pcd.paint_uniform_color([1.0, 0.45, 0.1])
    cube.paint_uniform_color([0.35, 0.45, 0.9])
    cube.translate(-cube.get_center() + mesh.get_center() + np.array([6.0, 0, 0]))
    view.show([pcd, cube], "tier 1: surface cloud vs volume noise", save=args.save)


if __name__ == "__main__":
    main()
