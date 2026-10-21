# Frames

Six segmented frames used as the fixed input for reconstruction. Captured 2026-10-20 from `data/test_cloud.pcap` with `rslidar_viz` built from `virtual-rgbd-sensor` commit `813206e`, with `dense_points` recorded as false.

The sensor does not move during the recording, so these are repeated scans of one scene. Cluster ids are assigned per frame, so the same object has a different id in each file.

| File | Stamp (s) | Finite points | Clusters |
| --- | --- | --- | --- |
| `frame_1483235723_999802351_seq000139.npz` | 1483235723.999 | 84,490 | 286 |
| `frame_1483235729_799750566_seq000009.npz` | 1483235729.800 | 84,598 | 281 |
| `frame_1483235735_599822521_seq000038.npz` | 1483235735.600 | 84,610 | 274 |
| `frame_1483235741_399788380_seq000067.npz` | 1483235741.399 | 84,508 | 327 |
| `frame_1483235747_199832439_seq000096.npz` | 1483235747.200 | 84,654 | 289 |
| `frame_1483235752_999730349_seq000125.npz` | 1483235753.000 | 84,501 | 275 |

All six have 115,200 records.

## Large clusters

Cluster id (point count) of each object over 1,000 points, in the file order above.

| Median range | Extent x × y × z (m) | 723.999 | 729.800 | 735.600 | 741.399 | 747.200 | 753.000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.35 m | 1.52 × 1.22 × 0.41 | 1 (42,707) | 2 (42,519) | 1 (43,158) | 1 (42,437) | 2 (42,946) | 1 (42,148) |
| 4.86 m | 8.24 × 12.88 × 3.24 | 21 (19,521) | 21 (19,363) | 24 (19,375) | 17 (19,420) | 23 (19,434) | 18 (19,241) |
| 1.28 m | 1.80 × 1.16 × 0.74 | 20 (14,029) | 20 (10,530) | 23 (10,474) | 16 (11,501) | 22 (14,713) | 17 (10,442) |
| 2.60 m | 2.33 × 2.04 × 0.78 | 15 (1,262) | 14 (1,266) | 17 (1,255) | 11 (1,270) | 16 (1,262) | 11 (1,248) |

- The 0.35 m cluster is within about 0.8 m of the sensor, likely its mount or the surface it sits on.
- The 1.28 m cluster's point count changes between frames because the segmenter splits that region differently.
