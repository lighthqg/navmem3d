#!/usr/bin/env python3
"""Resolve B query goals to safe, reachable approach cells in world A.

This validator does not certify B→A registration.  It reports the distance
between the registered target projection and the selected free approach cell;
large projection errors remain explicitly unverified.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from navmem3d.navigation.grid_runtime import OccupancyGrid, astar_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--robot-radius-px", type=int, default=4)
    ap.add_argument("--max-projection-error-m", type=float, default=1.0)
    args = ap.parse_args()
    if args.max_projection_error_m <= 0:
        raise ValueError("max-projection-error-m must be positive")
    goals = json.loads(args.goals.read_text())
    grid = OccupancyGrid.from_interiorgs_scene(args.scene)
    free = [(c, r) for r in range(grid.height) for c in range(grid.width) if grid.is_free(c, r)]
    start = min(free, key=lambda p: (p[0] - grid.width / 2) ** 2 + (p[1] - grid.height / 2) ** 2)
    blocked = grid.inflated_blocked(args.robot_radius_px)
    safe_free = [p for p in free if p not in blocked]
    entities, reachable, accepted, projection_errors = [], 0, 0, []
    for entity in goals.get("entities", []):
        out = {**entity, "goal_candidates": []}
        for candidate in entity.get("goal_candidates", []):
            point = candidate.get("position_a", candidate.get("position"))
            if point is None:
                continue
            col, row = grid.transform.to_pixel(float(point[0]), float(point[1]))
            desired = (round(col), round(row))
            approach = min(safe_free, key=lambda p: (p[0] - desired[0]) ** 2 + (p[1] - desired[1]) ** 2)
            approach_world = grid.transform.to_world(*approach)
            projection_error = math.dist((float(point[0]), float(point[1])), approach_world)
            path = astar_path(grid, start, approach, blocked=blocked)
            path_found = path is not None
            path_world = [list(grid.transform.to_world(*node)) for node in path] if path else []
            if path and len(path) >= 2:
                prev_world = grid.transform.to_world(*path[-2])
                yaw_rad = math.atan2(approach_world[1] - prev_world[1], approach_world[0] - prev_world[0])
            else:
                yaw_rad = math.atan2(float(point[1]) - approach_world[1], float(point[0]) - approach_world[0])
            path_length_m = sum(math.dist(left, right) for left, right in zip(path_world, path_world[1:]))
            projection_accepted = projection_error <= args.max_projection_error_m
            status = "reachable" if path_found and projection_accepted else (
                "registration_projection_too_far" if path_found else "unreachable"
            )
            item = {
                **candidate,
                "desired_pixel": list(desired),
                "approach_pixel": list(approach),
                "approach_world": list(approach_world),
                "projection_error_m": projection_error,
                "projection_accepted": projection_accepted,
                "path_found": path_found,
                "path_nodes": len(path) if path else None,
                "path_length_m": path_length_m if path else None,
                "approach_yaw_rad": yaw_rad,
                "approach_pose_a_xy_yaw": [approach_world[0], approach_world[1], yaw_rad],
                "reachability_status": status,
            }
            out["goal_candidates"].append(item)
            projection_errors.append(projection_error)
            reachable += int(path_found)
            accepted += int(path_found and projection_accepted)
        entities.append(out)
    result = {
        "schema_version": "0.2",
        "world_a": str(args.scene.resolve()),
        "source_goals": str(args.goals.resolve()),
        "registration_status": goals.get("registration", {}).get("status", "unregistered"),
        "start_pixel": list(start),
        "robot_radius_px": args.robot_radius_px,
        "max_projection_error_m": args.max_projection_error_m,
        "candidate_count": sum(len(e["goal_candidates"]) for e in entities),
        "path_found_candidate_count": reachable,
        "projection_accepted_candidate_count": accepted,
        "median_projection_error_m": sorted(projection_errors)[len(projection_errors) // 2] if projection_errors else None,
        "max_projection_error_observed_m": max(projection_errors) if projection_errors else None,
        "entities": entities,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "candidates": result["candidate_count"], "path_found": reachable, "projection_accepted": accepted, "median_projection_error_m": result["median_projection_error_m"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
