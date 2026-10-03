"""Build a random point cloud and open the Open3D viewer."""

import numpy as np
import open3d as o3d


def main() -> None:
    xyz = np.random.rand(2000, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color([1.0, 0.4, 0.0])
    o3d.visualization.draw_geometries([pcd], window_name="open3d-playground")


if __name__ == "__main__":
    main()
