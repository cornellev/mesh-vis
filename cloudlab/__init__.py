"""Scratch library for the point-cloud -> mesh playground.

Modules mirror the pipeline stages:
    metrics  -- MEASURE (point spacing) and CHECK (point-to-mesh distance)
    surface  -- tier 1 test data: sample a cloud from a known surface
    lidar    -- tier 2 test data: simulate a spinning lidar with occlusion
"""

from cloudlab import lidar, metrics, surface

__all__ = ["lidar", "metrics", "surface"]
