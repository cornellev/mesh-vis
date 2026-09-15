# mesh-vis

LIDAR point cloud to mesh visualization

Utilizing Open3D library

`cloudlab/` builds test clouds: sample a known mesh (tier 1) or fake a lidar sweep (tier 2). `metrics` measures point spacing and distance to a mesh.
`examples/` runs those two tiers so you can see spacing, anisotropy, and why a random cube cannot be meshed.
`tests/` is stdlib unittest for `cloudlab`; rendering tests stay off unless `CLOUDLAB_RENDER_TESTS=1`.
