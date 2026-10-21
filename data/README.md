# Local input data

This directory is for your PCAP and your local yaml. Both are gitignored. Set `pcap_path` to your PCAP before starting the publisher. That publisher, `rslidar_viz` from `virtual-rgbd-sensor`, is what segments the PCAP. See the README for the full path from PCAP to a saved snapshot.

From the `mesh-vis` repository root, start the existing producer binary with:

```bash
"../virtual-rgbd-sensor/target/release/rslidar_viz" "data/rslidar-local.yaml"
```
