import numpy as np
import open3d as o3d

# Input: .bin raw data
points = np.fromfile('kitti_sample_bin/0000000032.bin', dtype=np.float32).reshape(-1, 4)
xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)

# Point Cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.visualization.draw_geometries([pcd])

