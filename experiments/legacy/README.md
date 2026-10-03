# mesh-vis

This directory preserves the earlier playground, its tests, and sample data.
The active capture application is now in `../../mesh_vis`; start with the
[repository README](../../README.md). Run the commands below from this directory.

```bash
python3 -m unittest discover -s tests -t . -v
python3 examples/02_tier1_surface_cloud.py --no-show
python3 examples/03_tier2_fake_lidar.py --no-show
python3 main.py
```

nuScenes remains an external dataset at the workspace's `v1.0-mini` directory;
`examples/04_nuscenes_sweep.py --data PATH` overrides that location.

LIDAR point cloud to mesh visualization

Utilizing Open3D library

`cloudlab/` builds test clouds: sample a known mesh (tier 1) or fake a lidar sweep (tier 2). `metrics` measures point spacing and distance to a mesh.
`examples/` runs those two tiers so you can see spacing, anisotropy, and why a random cube cannot be meshed.
`tests/` is stdlib unittest for `cloudlab`; rendering tests stay off unless `CLOUDLAB_RENDER_TESTS=1`.
