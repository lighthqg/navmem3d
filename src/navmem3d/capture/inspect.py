from __future__ import annotations

from pathlib import Path

from navmem3d.schemas import CaptureSequence


def inspect_sequence(sequence_path: str | Path) -> dict[str, object]:
    path = Path(sequence_path)
    sequence = CaptureSequence.load(path)
    missing_rgb = []
    missing_depth = []
    for frame in sequence.frames:
        if not (path.parent / frame.rgb_uri).exists():
            missing_rgb.append(frame.frame_id)
        if frame.depth_uri and not (path.parent / frame.depth_uri).exists():
            missing_depth.append(frame.frame_id)
    return {
        "sequence_id": sequence.sequence_id,
        "frame_count": len(sequence.frames),
        "duration_s": sequence.frames[-1].timestamp_s - sequence.frames[0].timestamp_s,
        "coordinate_frame": sequence.coordinate_frame,
        "rgb_complete": not missing_rgb,
        "depth_complete": not missing_depth,
        "missing_rgb_frame_ids": missing_rgb,
        "missing_depth_frame_ids": missing_depth,
    }

