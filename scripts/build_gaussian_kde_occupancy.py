#!/usr/bin/env python3
"""从压缩 3DGS 的地面/竖向表面密度构建三态二维占据图。

这是一条独立、可复现的离线链路：Gaussian 几何与人工巡视位姿是唯一输入。
不会读取 occupancy、navmesh、碰撞体或语义标签。白=free，黑=occupied，灰=unknown。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from navmem3d.world.supersplat import load_supersplat_compressed_ply


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--patrol", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resolution", type=float, default=0.10)
    p.add_argument("--margin", type=float, default=0.65)
    p.add_argument("--floor-band", nargs=2, type=float, default=(-0.25, 0.20))
    p.add_argument("--vertical-band", nargs=2, type=float, default=(0.30, 2.20))
    p.add_argument("--floor-sigma-m", type=float, default=0.25)
    p.add_argument("--vertical-sigma-m", type=float, default=0.15)
    p.add_argument("--floor-quantile", type=float, default=0.42)
    p.add_argument("--vertical-quantile", type=float, default=0.72)
    p.add_argument("--minimum-floor-component-m2", type=float, default=0.50)
    p.add_argument("--patrol-free-radius", type=float, default=0.20)
    return p.parse_args()


def odd_kernel(sigma_px: float) -> int:
    return max(3, int(2 * np.ceil(3 * sigma_px) + 1) | 1)


def raster_line(x0: int, y0: int, x1: int, y1: int):
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield x0, y0
        if (x0, y0) == (x1, y1):
            return
        twice = 2 * err
        if twice >= dy:
            err += dy
            x0 += sx
        if twice <= dx:
            err += dx
            y0 += sy


def weighted_density(x: np.ndarray, y: np.ndarray, weights: np.ndarray, shape: tuple[int, int], mask: np.ndarray, sigma_px: float) -> np.ndarray:
    grid = np.zeros(shape, dtype=np.float32)
    np.add.at(grid, (y[mask], x[mask]), weights[mask])
    return cv2.GaussianBlur(grid, (odd_kernel(sigma_px), odd_kernel(sigma_px)), sigmaX=sigma_px, sigmaY=sigma_px)


def build(a: argparse.Namespace) -> dict:
    patrol = json.loads(a.patrol.read_text(encoding="utf-8"))["frames"]
    poses = np.asarray([frame["camera_position"] for frame in patrol], dtype=np.float32)
    splats = load_supersplat_compressed_ply(a.asset)
    points, opacity = splats.means, splats.opacities

    origin = poses[:, :2].min(0) - a.margin
    ceiling = poses[:, :2].max(0) + a.margin
    width = int(np.ceil((ceiling[0] - origin[0]) / a.resolution)) + 1
    height = int(np.ceil((ceiling[1] - origin[1]) / a.resolution)) + 1
    x = np.floor((points[:, 0] - origin[0]) / a.resolution).astype(np.int32)
    y = np.floor((points[:, 1] - origin[1]) / a.resolution).astype(np.int32)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    # Clip invalid indices before using np.add.at.
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    shape = (height, width)

    floor_mask = inside & (points[:, 2] >= a.floor_band[0]) & (points[:, 2] <= a.floor_band[1])
    vertical_mask = inside & (points[:, 2] >= a.vertical_band[0]) & (points[:, 2] <= a.vertical_band[1])
    floor_density = weighted_density(x, y, opacity, shape, floor_mask, a.floor_sigma_m / a.resolution)
    vertical_density = weighted_density(x, y, opacity, shape, vertical_mask, a.vertical_sigma_m / a.resolution)
    floor_positive = floor_density[floor_density > 0]
    vertical_positive = vertical_density[vertical_density > 0]
    if not len(floor_positive) or not len(vertical_positive):
        raise RuntimeError("Gaussian 表面密度为空；检查高度带与坐标系")
    floor_threshold = float(np.quantile(floor_positive, a.floor_quantile))
    vertical_threshold = float(np.quantile(vertical_positive, a.vertical_quantile))

    free = floor_density >= floor_threshold
    n, labels, stats, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=8)
    min_cells = int(np.ceil(a.minimum_floor_component_m2 / (a.resolution * a.resolution)))
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_cells)
    keep = keep[keep != 0]
    free = np.isin(labels, keep)
    occupied = vertical_density >= vertical_threshold
    occupied = cv2.morphologyEx(occupied.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    occupied = cv2.morphologyEx(occupied.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    occupied = cv2.dilate(occupied.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    occupied &= cv2.dilate(free.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    free &= ~occupied

    # The recorded patrol is physical free-space evidence.  It only clears the
    # swept tube, never synthesizes an edge or an unvisited free area.
    radius = int(np.ceil(a.patrol_free_radius / a.resolution))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = xx * xx + yy * yy <= radius * radius
    path = np.floor((poses[:, :2] - origin) / a.resolution).astype(np.int32)
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        for px, py in raster_line(int(x0), int(y0), int(x1), int(y1)):
            left, right = max(0, px - radius), min(width, px + radius + 1)
            top, bottom = max(0, py - radius), min(height, py + radius + 1)
            local = disk[top - (py - radius) : bottom - (py - radius), left - (px - radius) : right - (px - radius)]
            free[top:bottom, left:right][local] = True
            occupied[top:bottom, left:right][local] = False

    state = np.full(shape, 127, dtype=np.uint8)
    state[free] = 255
    state[occupied] = 0
    a.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(state).save(a.output_dir / "observed_occupancy.png")
    Image.fromarray(state).resize((width * 2, height * 2), Image.Resampling.NEAREST).save(a.output_dir / "observed_occupancy_preview.png")
    metadata = {
        "schema_version": "1.0",
        "type": "gaussian_kde_occupancy",
        "method": "opacity_weighted_2d_kde_of_floor_and_vertical_gaussian_surface_bands",
        "asset": str(a.asset.resolve()), "patrol": str(a.patrol.resolve()),
        "state_encoding": {"occupied": 0, "unknown": 127, "free": 255},
        "origin_xy": origin.tolist(), "scale_m": a.resolution, "shape_hw": [height, width],
        "floor_z_band": list(a.floor_band), "vertical_z_band": list(a.vertical_band),
        "floor_sigma_m": a.floor_sigma_m, "vertical_sigma_m": a.vertical_sigma_m,
        "floor_quantile": a.floor_quantile, "vertical_quantile": a.vertical_quantile,
        "resolved_thresholds": {"floor": floor_threshold, "vertical": vertical_threshold},
        "minimum_floor_component_m2": a.minimum_floor_component_m2,
        "patrol_free_radius_m": a.patrol_free_radius,
        "counts": {"free": int(free.sum()), "occupied": int(occupied.sum()), "unknown": int((state == 127).sum())},
        "hidden_occupancy_consumed": False, "simulator_navmesh_consumed": False,
        "simulator_collider_consumed": False, "semantic_labels_consumed": False,
    }
    (a.output_dir / "occupancy_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result = build(arguments())
    print(json.dumps(result["counts"], ensure_ascii=False))
