from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


FREE = 255
OCCUPIED = 0
UNKNOWN = 127


@dataclass(frozen=True)
class GridTransform:
    """Explicit affine transform between occupancy pixels and metric world XY."""

    pixel_to_world: tuple[float, ...]

    @classmethod
    def from_interiorgs_metadata(
        cls, metadata: dict[str, object]
    ) -> "GridTransform":
        """Build the transform used by InteriorGS ``visualize_meta.py``."""
        scale = float(metadata["scale"])
        upper = metadata["upper"]
        lower = metadata["lower"]
        if scale <= 0 or not isinstance(upper, list) or not isinstance(lower, list):
            raise ValueError("invalid InteriorGS occupancy metadata")
        return cls(
            (
                -scale,
                0.0,
                float(upper[0]),
                0.0,
                scale,
                float(lower[1]),
                0.0,
                0.0,
                1.0,
            )
        )

    def __post_init__(self) -> None:
        if len(self.pixel_to_world) != 9:
            raise ValueError("pixel_to_world must be a row-major 3x3 matrix")
        a, b, _, d, e, _, _, _, w = self.pixel_to_world
        if abs(a * e - b * d) < 1e-12 or abs(w) < 1e-12:
            raise ValueError("pixel_to_world must be invertible")

    def to_world(self, col: float, row: float) -> tuple[float, float]:
        m = self.pixel_to_world
        x = m[0] * col + m[1] * row + m[2]
        y = m[3] * col + m[4] * row + m[5]
        w = m[6] * col + m[7] * row + m[8]
        return x / w, y / w

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        m = self.pixel_to_world
        if abs(m[6]) > 1e-12 or abs(m[7]) > 1e-12 or abs(m[8] - 1.0) > 1e-12:
            raise ValueError("projective transforms are not supported for world_to_pixel")
        determinant = m[0] * m[4] - m[1] * m[3]
        dx, dy = x - m[2], y - m[5]
        col = (m[4] * dx - m[1] * dy) / determinant
        row = (-m[3] * dx + m[0] * dy) / determinant
        return col, row


@dataclass(frozen=True)
class OccupancyGrid:
    cells: tuple[tuple[int, ...], ...]
    transform: GridTransform

    def __post_init__(self) -> None:
        if not self.cells or not self.cells[0]:
            raise ValueError("occupancy grid must not be empty")
        width = len(self.cells[0])
        if any(len(row) != width for row in self.cells):
            raise ValueError("occupancy rows must have equal length")

    @classmethod
    def from_png(cls, path: str | Path, transform: GridTransform) -> "OccupancyGrid":
        from PIL import Image

        image = Image.open(path).convert("L")
        width, height = image.size
        values = tuple(image.getdata())
        rows = tuple(
            tuple(values[row * width : (row + 1) * width]) for row in range(height)
        )
        return cls(rows, transform)

    @classmethod
    def from_interiorgs_scene(cls, scene_dir: str | Path) -> "OccupancyGrid":
        scene_dir = Path(scene_dir)
        metadata = json.loads((scene_dir / "occupancy.json").read_text())
        transform = GridTransform.from_interiorgs_metadata(metadata)
        return cls.from_png(scene_dir / "occupancy.png", transform)

    @property
    def width(self) -> int:
        return len(self.cells[0])

    @property
    def height(self) -> int:
        return len(self.cells)

    def contains(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def is_free(self, col: int, row: int) -> bool:
        return self.contains(col, row) and self.cells[row][col] == FREE

    def world_is_free(self, x: float, y: float) -> bool:
        col, row = self.transform.to_pixel(x, y)
        return self.is_free(round(col), round(row))

    def inflated_blocked(self, radius_pixels: int) -> set[tuple[int, int]]:
        blocked = {
            (col, row)
            for row in range(self.height)
            for col in range(self.width)
            if not self.is_free(col, row)
        }
        if radius_pixels <= 0:
            return blocked
        offsets = [
            (dx, dy)
            for dy in range(-radius_pixels, radius_pixels + 1)
            for dx in range(-radius_pixels, radius_pixels + 1)
            if dx * dx + dy * dy <= radius_pixels * radius_pixels
        ]
        return {
            (col + dx, row + dy)
            for col, row in blocked
            for dx, dy in offsets
            if self.contains(col + dx, row + dy)
        }


def astar_path(
    grid: OccupancyGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    blocked: Iterable[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    blocked_cells = set(blocked) if blocked is not None else grid.inflated_blocked(0)
    if start in blocked_cells or goal in blocked_cells:
        return None
    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    queue: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
    costs = {start: 0.0}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost != costs.get(current):
            continue
        if current == goal:
            path = [current]
            while current in previous:
                current = previous[current]
                path.append(current)
            return list(reversed(path))
        for dx, dy, step_cost in moves:
            neighbor = current[0] + dx, current[1] + dy
            if not grid.contains(*neighbor) or neighbor in blocked_cells:
                continue
            if dx and dy:
                if (current[0] + dx, current[1]) in blocked_cells or (
                    current[0], current[1] + dy
                ) in blocked_cells:
                    continue
            new_cost = cost + step_cost
            if new_cost >= costs.get(neighbor, math.inf):
                continue
            costs[neighbor] = new_cost
            previous[neighbor] = current
            heuristic = math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
            heapq.heappush(queue, (new_cost + heuristic, new_cost, neighbor))
    return None


def differential_drive_step(
    pose_xy_yaw: Sequence[float], linear_mps: float, angular_rps: float, dt_s: float
) -> tuple[float, float, float]:
    if len(pose_xy_yaw) != 3 or dt_s <= 0:
        raise ValueError("pose must be (x, y, yaw) and dt_s must be positive")
    x, y, yaw = map(float, pose_xy_yaw)
    if abs(angular_rps) < 1e-8:
        return (
            x + linear_mps * dt_s * math.cos(yaw),
            y + linear_mps * dt_s * math.sin(yaw),
            yaw,
        )
    next_yaw = yaw + angular_rps * dt_s
    radius = linear_mps / angular_rps
    return (
        x + radius * (math.sin(next_yaw) - math.sin(yaw)),
        y - radius * (math.cos(next_yaw) - math.cos(yaw)),
        math.atan2(math.sin(next_yaw), math.cos(next_yaw)),
    )
