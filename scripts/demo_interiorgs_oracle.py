#!/usr/bin/env python3
"""Evaluator-only smoke test: InteriorGS label query -> reachable A* goal."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

from PIL import Image, ImageDraw

from navmem3d.navigation.grid_runtime import OccupancyGrid, astar_path


def largest_component(grid: OccupancyGrid, blocked: set[tuple[int, int]]):
    remaining = {
        (col, row)
        for row in range(grid.height)
        for col in range(grid.width)
        if (col, row) not in blocked
    }
    largest: list[tuple[int, int]] = []
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        component = [start]
        while queue:
            col, row = queue.popleft()
            for neighbor in (
                (col - 1, row),
                (col + 1, row),
                (col, row - 1),
                (col, row + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        if len(component) > len(largest):
            largest = component
    return largest


def bbox_center(instance: dict[str, object]) -> tuple[float, float, float]:
    points = instance["bounding_box"]
    return tuple(
        sum(float(point[axis]) for point in points) / len(points)
        for axis in ("x", "y", "z")
    )


def run(scene: Path, term: str, output: Path, robot_radius_m: float) -> dict[str, object]:
    grid = OccupancyGrid.from_interiorgs_scene(scene)
    scale = abs(grid.transform.pixel_to_world[0])
    blocked = grid.inflated_blocked(round(robot_radius_m / scale))
    component = largest_component(grid, blocked)
    component_set = set(component)
    if not component:
        raise RuntimeError("no navigable component remains after obstacle inflation")
    start = min(component, key=lambda point: point[0] + point[1])

    labels = json.loads((scene / "labels.json").read_text())
    matches = [
        instance
        for instance in labels
        if str(instance.get("label", "")).strip().casefold() == term.casefold()
        and instance.get("bounding_box")
    ]
    candidates = []
    for instance in matches:
        center = bbox_center(instance)
        center_pixel = grid.transform.to_pixel(center[0], center[1])
        goal = min(
            component_set,
            key=lambda point: (point[0] - center_pixel[0]) ** 2
            + (point[1] - center_pixel[1]) ** 2,
        )
        path = astar_path(grid, start, goal, blocked=blocked)
        if path:
            candidates.append((len(path), instance, center, goal, path))
    if not candidates:
        raise RuntimeError(f"no reachable {term!r} instance")
    _, selected, center, goal, path = min(candidates, key=lambda item: item[0])

    output.mkdir(parents=True, exist_ok=True)
    image = Image.open(scene / "occupancy.png").convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.line(path, fill=(255, 40, 40), width=2)
    for point, color in ((start, (30, 200, 30)), (goal, (30, 90, 255))):
        draw.ellipse(
            (point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3),
            fill=color,
        )
    image.resize((grid.width * 4, grid.height * 4), Image.Resampling.NEAREST).save(
        output / f"oracle_{term}_path.png"
    )
    result = {
        "evaluator_only": True,
        "warning": "Uses InteriorGS ground-truth labels; not a system-under-test result.",
        "scene_id": scene.name,
        "query": term,
        "matched_instances": len(matches),
        "reachable_instances": len(candidates),
        "selected_instance_id": str(selected["ins_id"]),
        "selected_label": selected["label"],
        "entity_center_world": center,
        "start_pixel": start,
        "goal_pixel": goal,
        "start_world": grid.transform.to_world(*start),
        "goal_world": grid.transform.to_world(*goal),
        "path_nodes": len(path),
        "robot_radius_m": robot_radius_m,
    }
    (output / f"oracle_{term}_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--term", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-radius-m", type=float, default=0.20)
    args = parser.parse_args()
    print(json.dumps(run(args.scene, args.term, args.output, args.robot_radius_m), indent=2))
