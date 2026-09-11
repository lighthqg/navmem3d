from __future__ import annotations

import math

from navmem3d.schemas import FrameRecord


def select_uniform(frames: tuple[FrameRecord, ...], count: int) -> list[FrameRecord]:
    _validate_count(frames, count)
    if count >= len(frames):
        return list(frames)
    if count == 1:
        return [frames[len(frames) // 2]]
    indices = [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)]
    return [frames[index] for index in indices]


def select_pose_coverage(
    frames: tuple[FrameRecord, ...], count: int, rotation_weight_m: float = 0.5
) -> list[FrameRecord]:
    """Greedy farthest-point sampling in translation and camera rotation space."""
    _validate_count(frames, count)
    if count >= len(frames):
        return list(frames)
    selected = [0]
    while len(selected) < count:
        remaining = (index for index in range(len(frames)) if index not in selected)
        next_index = max(
            remaining,
            key=lambda index: min(
                _pose_distance(frames[index], frames[chosen], rotation_weight_m)
                for chosen in selected
            ),
        )
        selected.append(next_index)
    return [frames[index] for index in sorted(selected)]


def _validate_count(frames: tuple[FrameRecord, ...], count: int) -> None:
    if not frames:
        raise ValueError("frames must not be empty")
    if count < 1:
        raise ValueError("count must be positive")


def _pose_distance(a: FrameRecord, b: FrameRecord, rotation_weight_m: float) -> float:
    ta = (a.t_world_camera[3], a.t_world_camera[7], a.t_world_camera[11])
    tb = (b.t_world_camera[3], b.t_world_camera[7], b.t_world_camera[11])
    translation = math.sqrt(sum((x - y) ** 2 for x, y in zip(ta, tb)))

    ra = _rotation(a.t_world_camera)
    rb = _rotation(b.t_world_camera)
    trace_relative = sum(ra[row][col] * rb[row][col] for row in range(3) for col in range(3))
    cosine = max(-1.0, min(1.0, (trace_relative - 1.0) / 2.0))
    angle = math.acos(cosine)
    return translation + rotation_weight_m * angle


def _rotation(matrix: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(matrix[row * 4 + col] for col in range(3)) for row in range(3))
