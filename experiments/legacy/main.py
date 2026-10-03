import numpy as np
import open3d as o3d
from pathlib import Path

# Input: .bin raw data
points = np.fromfile(Path(__file__).resolve().parent / 'kitti_sample_bin/0000000032.bin', dtype=np.float32).reshape(-1, 4)
xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)

# Point Cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.visualization.draw_geometries([pcd])
