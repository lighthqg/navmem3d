from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path

from navmem3d.navigation.grid_runtime import OccupancyGrid, astar_path


def _inside_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        crosses = (y1 > y) != (y2 > y)
        if crosses and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
        previous = current
    return inside


def _largest_component(
    grid: OccupancyGrid, blocked: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    remaining = {
        (col, row)
        for row in range(grid.height)
        for col in range(grid.width)
        if (col, row) not in blocked
    }
    largest: list[tuple[int, int]] = []
    while remaining:
        seed = remaining.pop()
        queue = deque([seed])
        component = [seed]
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


def _coverage_waypoints(
    component: list[tuple[int, int]], count: int
) -> list[tuple[int, int]]:
    center = (
        sum(point[0] for point in component) / len(component),
        sum(point[1] for point in component) / len(component),
    )
    waypoints = [
        min(
            component,
            key=lambda point: (point[0] - center[0]) ** 2
            + (point[1] - center[1]) ** 2,
        )
    ]
    while len(waypoints) < count:
        next_point = max(
            component,
            key=lambda point: min(
                (point[0] - chosen[0]) ** 2 + (point[1] - chosen[1]) ** 2
                for chosen in waypoints
            ),
        )
        if next_point in waypoints:
            break
        waypoints.append(next_point)
    return waypoints


def build_interiorgs_patrol(
    scene_dir: str | Path,
    output_path: str | Path,
    *,
    frame_count: int = 300,
    waypoint_count: int = 8,
    robot_radius_m: float = 0.20,
    sensor_height_m: float = 1.25,
    fps: float = 10.0,
) -> dict[str, object]:
    """Create a deterministic closed robot patrol inside InteriorGS occupancy."""
    if frame_count < 2 or waypoint_count < 2 or fps <= 0:
        raise ValueError("frame_count, waypoint_count and fps must be positive")
    scene_dir = Path(scene_dir)
    grid = OccupancyGrid.from_interiorgs_scene(scene_dir)
    scale = abs(grid.transform.pixel_to_world[0])
    blocked = grid.inflated_blocked(math.ceil(robot_radius_m / scale))
    structure_path = scene_dir / "structure.json"
    if structure_path.exists():
        room_polygons = [
            room["profile"]
            for room in json.loads(structure_path.read_text()).get("rooms", [])
            if room.get("profile")
        ]
        if room_polygons:
            blocked.update(
                (col, row)
                for row in range(grid.height)
                for col in range(grid.width)
                if not any(
                    _inside_polygon(*grid.transform.to_world(col, row), polygon)
                    for polygon in room_polygons
                )
            )
    component = _largest_component(grid, blocked)
    if not component:
        raise RuntimeError("no navigable component remains after obstacle inflation")
    unordered = _coverage_waypoints(component, waypoint_count)

    ordered = [unordered.pop(0)]
    route: list[tuple[int, int]] = [ordered[0]]
    while unordered:
        options = []
        for waypoint in unordered:
            path = astar_path(grid, ordered[-1], waypoint, blocked=blocked)
            if path:
                options.append((len(path), waypoint, path))
        if not options:
            raise RuntimeError("coverage waypoint is disconnected")
        _, waypoint, path = min(options, key=lambda item: item[0])
        ordered.append(waypoint)
        unordered.remove(waypoint)
        route.extend(path[1:])
    closing = astar_path(grid, ordered[-1], ordered[0], blocked=blocked)
    if closing:
        route.extend(closing[1:])

    sample_indices = [
        round(index * (len(route) - 1) / (frame_count - 1))
        for index in range(frame_count)
    ]
    frames = []
    for index, route_index in enumerate(sample_indices):
        pixel = route[route_index]
        next_pixel = route[sample_indices[(index + 1) % frame_count]]
        x, y = grid.transform.to_world(*pixel)
        next_x, next_y = grid.transform.to_world(*next_pixel)
        if math.hypot(next_x - x, next_y - y) < 1e-6 and index:
            look_at = frames[-1]["look_at"]
        else:
            length = max(math.hypot(next_x - x, next_y - y), 1e-9)
            look_at = [
                x + (next_x - x) / length,
                y + (next_y - y) / length,
                sensor_height_m,
            ]
        frames.append(
            {
                "frame_id": f"frame_{index:04d}",
                "timestamp": index / fps,
                "pixel": list(pixel),
                "camera_position": [x, y, sensor_height_m],
                "look_at": look_at,
                "up": [0.0, 0.0, 1.0],
            }
        )

    result = {
        "schema_version": "0.1",
        "sequence_id": f"{scene_dir.name}_patrol",
        "world_role": "A_ground_truth",
        "evaluation_access": "system",
        "purpose": "limited_robot_observations_for_world_B_reconstruction",
        "scene_dir": str(scene_dir.resolve()),
        "coordinate_frame": "interiorgs_xyz_right_back_up",
        "fps": fps,
        "duration_s": (frame_count - 1) / fps,
        "robot_radius_m": robot_radius_m,
        "sensor_height_m": sensor_height_m,
        "occupancy_scale_m": scale,
        "waypoints_pixel": [list(point) for point in ordered],
        "route_nodes": len(route),
        "frames": frames,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
