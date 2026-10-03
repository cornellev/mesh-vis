# Local input data

This directory holds machine-local producer inputs and configuration. Large PCAP
files and `*-local.yaml` files are gitignored.

From the `mesh-vis` repository root, start the existing producer binary with:

```bash
"../virtual-rgbd-sensor/target/release/rslidar_viz" "data/rslidar-local.yaml"
```

`rslidar-local.yaml` points to `data/test_cloud.pcap`. Neither file is required
after another machine or service is already publishing `rslidar/points/segmented`.
