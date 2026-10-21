# nuScenes frames

Three LIDAR_TOP sweeps from nuScenes v1.0-mini in the mesh-vis snapshot format. Use them to check each pipeline stage on clean, recognizable objects. They are not from our lidar or our segmenter: set and judge parameters on the RoboSense frames in `frames/`.

Clusters come from the nuScenes annotations, not from `rslidar_viz`. A point belongs to an object if it is inside the object's 3D box and its lidarseg class matches the object's category. All other points are -1. The recording car's points are removed.

| File | Object to mesh |
| --- | --- |
| `frame_1535657121_649287000_seq000000.npz` | cluster 2, truck side, 10,019 pts, 2.5 m |
| `frame_1535489307_946546000_seq000000.npz` | cluster 51, car, 2,625 pts, 3.0 m |
| `frame_1535385108_451734000_seq000000.npz` | cluster 41, bendy bus, 5,354 pts, 4.0 m |
