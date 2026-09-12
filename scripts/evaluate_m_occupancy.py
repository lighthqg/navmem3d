#!/usr/bin/env python3
"""Evaluate a robot-observed M occupancy against hidden A occupancy.

The A grid is evaluator-only: it is read exclusively here after M generation
and is never written into the robot map.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def inflate(image: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return image
    result = image.copy()
    offsets = [(dx, dy) for dy in range(-radius_px, radius_px + 1) for dx in range(-radius_px, radius_px + 1) if dx * dx + dy * dy <= radius_px * radius_px]
    for y, x in np.argwhere(image == 0):
        for dx, dy in offsets:
            xx, yy = x + dx, y + dy
            if 0 <= yy < image.shape[0] and 0 <= xx < image.shape[1]:
                result[yy, xx] = 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-occupancy", type=Path, required=True)
    parser.add_argument("--m-metadata", type=Path, required=True)
    parser.add_argument("--a-occupancy", type=Path, required=True)
    parser.add_argument("--a-metadata", type=Path, required=True)
    parser.add_argument("--robot-radius", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    m = np.asarray(Image.open(args.m_occupancy).convert("L"))
    a = np.asarray(Image.open(args.a_occupancy).convert("L"))
    mm = json.loads(args.m_metadata.read_text())
    am = json.loads(args.a_metadata.read_text())
    a_scale = float(am["scale"])
    a = inflate(a, math.ceil(args.robot_radius / a_scale))
    ox, oy = mm["origin_xy"]
    scale = float(mm["scale_m"])
    upper_x, lower_y = float(am["upper"][0]), float(am["lower"][1])

    yy, xx = np.indices(m.shape)
    wx, wy = ox + xx * scale, oy + yy * scale
    ac = np.rint((upper_x - wx) / a_scale).astype(int)
    ar = np.rint((wy - lower_y) / a_scale).astype(int)
    inside = (ar >= 0) & (ar < a.shape[0]) & (ac >= 0) & (ac < a.shape[1])
    truth = np.full(m.shape, 127, np.uint8)
    truth[inside] = a[ar[inside], ac[inside]]
    scored = (m != 127) & (truth != 127)
    m_free, m_occ = m == 255, m == 0
    t_free, t_occ = truth == 255, truth == 0
    counts = {
        "scored_cells": int(scored.sum()),
        "m_observed_coverage_of_truth_known": float(scored.sum() / max(1, (truth != 127).sum())),
        "agreement": int((scored & (m == truth)).sum()),
        "dangerous_false_free": int((scored & m_free & t_occ).sum()),
        "conservative_false_occupied": int((scored & m_occ & t_free).sum()),
        "correct_free": int((scored & m_free & t_free).sum()),
        "correct_occupied": int((scored & m_occ & t_occ).sum()),
    }
    counts["agreement_rate"] = counts["agreement"] / max(1, counts["scored_cells"])
    counts["dangerous_false_free_rate"] = counts["dangerous_false_free"] / max(1, counts["correct_free"] + counts["dangerous_false_free"])
    counts["occupied_precision"] = counts["correct_occupied"] / max(1, counts["correct_occupied"] + counts["conservative_false_occupied"])

    # Gray=unscored, white=correct free, black=correct occupied,
    # red=dangerous false-free, blue=conservative false-occupied.
    visual = np.full((*m.shape, 3), (145, 145, 145), dtype=np.uint8)
    visual[scored & m_free & t_free] = (242, 242, 242)
    visual[scored & m_occ & t_occ] = (25, 25, 25)
    visual[scored & m_free & t_occ] = (220, 65, 55)
    visual[scored & m_occ & t_free] = (70, 135, 220)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visual).resize((m.shape[1] * 4, m.shape[0] * 4), Image.Resampling.NEAREST).save(args.output.with_suffix(".png"))
    result = {"schema_version": "0.1", "protocol": "evaluator_only_A_occupancy_comparison", "m_occupancy": str(args.m_occupancy.resolve()), "a_occupancy": str(args.a_occupancy.resolve()), "a_inflated_for_robot_radius_m": args.robot_radius, "hidden_A_used_only_for_evaluation": True, **counts}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
