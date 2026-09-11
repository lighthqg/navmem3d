#!/usr/bin/env python3
"""Select information-rich robot observations for reconstructing world B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw


def quality(path: Path) -> dict[str, float]:
    image = np.asarray(Image.open(path).convert("RGB").resize((128, 96))).astype(float)
    quadrants = [
        image[y : y + 48, x : x + 64].std(axis=(0, 1)).mean()
        for y in (0, 48)
        for x in (0, 64)
    ]
    return {
        "rgb_std": float(image.std(axis=(0, 1)).mean()),
        "min_quadrant_std": float(min(quadrants)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--search-radius", type=int, default=18)
    args = parser.parse_args()

    trajectory = json.loads(args.trajectory.read_text())
    frames = trajectory["frames"]
    targets = np.linspace(0, len(frames) - 1, args.count).round().astype(int)
    selected: list[int] = []
    records = []
    args.output.mkdir(parents=True, exist_ok=True)
    for old in args.output.glob("*.png"):
        old.unlink()

    for order, target in enumerate(targets):
        low = max(0, target - args.search_radius)
        high = min(len(frames), target + args.search_radius + 1)
        candidates = []
        for index in range(low, high):
            if index in selected:
                continue
            metrics = quality(args.renders / f"frame_{index:04d}.png")
            # Prefer information across the whole image while staying near the
            # uniform temporal target. This rejects close-up blank walls.
            score = metrics["min_quadrant_std"] - 0.12 * abs(index - target)
            candidates.append((score, index, metrics))
        _, index, metrics = max(candidates)
        selected.append(index)
        source = args.renders / f"frame_{index:04d}.png"
        destination = args.output / f"{order:02d}_frame_{index:04d}.png"
        shutil.copy2(source, destination)
        frame = frames[index]
        records.append(
            {
                "order": order,
                "uniform_target": int(target),
                "frame_index": index,
                **metrics,
                "frame_id": frame["frame_id"],
                "image": destination.name,
                "camera_position": frame["camera_position"],
                "look_at": frame["look_at"],
            }
        )

    manifest = {
        "source": "occupancy_constrained_robot_patrol",
        "selection": "uniform_temporal_targets_with_quadrant_quality_gate",
        "world_role": "A_ground_truth",
        "destination": "Marble_world_B_input",
        "uses_evaluator_labels": False,
        "frames": records,
    }
    (args.output / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")

    thumbs = []
    for record in records:
        image = Image.open(args.output / record["image"]).convert("RGB")
        image.thumbnail((320, 240))
        canvas = Image.new("RGB", (320, 266), "white")
        canvas.paste(image, ((320 - image.width) // 2, 0))
        ImageDraw.Draw(canvas).text(
            (8, 244),
            f'{record["frame_id"]}  q={record["min_quadrant_std"]:.1f}',
            fill="black",
        )
        thumbs.append(canvas)
    sheet = Image.new("RGB", (640, 266 * 4), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 2) * 320, (index // 2) * 266))
    sheet.save(args.output / "contact_sheet.png")
    print(json.dumps({"selected": selected, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
