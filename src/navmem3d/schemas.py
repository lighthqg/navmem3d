from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


class SchemaError(ValueError):
    """Raised when an interchange document violates the NavMem3D contract."""


def _require(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise SchemaError(f"missing required field: {key}")
    return data[key]


def _matrix(values: Any, size: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(values, list) or len(values) != size:
        raise SchemaError(f"{field_name} must contain {size} numbers")
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{field_name} must contain only numbers") from exc


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    timestamp_s: float
    rgb_uri: str
    intrinsics: tuple[float, ...]
    t_world_camera: tuple[float, ...]
    depth_uri: str | None = None
    quality: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameRecord":
        return cls(
            frame_id=str(_require(data, "frame_id")),
            timestamp_s=float(_require(data, "timestamp_s")),
            rgb_uri=str(_require(data, "rgb_uri")),
            depth_uri=data.get("depth_uri"),
            intrinsics=_matrix(_require(data, "intrinsics"), 9, "intrinsics"),
            t_world_camera=_matrix(
                _require(data, "T_world_camera"), 16, "T_world_camera"
            ),
            quality={str(k): float(v) for k, v in data.get("quality", {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["intrinsics"] = list(self.intrinsics)
        data["T_world_camera"] = list(self.t_world_camera)
        data.pop("t_world_camera")
        return data


@dataclass(frozen=True)
class CaptureSequence:
    sequence_id: str
    coordinate_frame: str
    camera_model: str
    frames: tuple[FrameRecord, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaptureSequence":
        frames_data = _require(data, "frames")
        if not isinstance(frames_data, list) or not frames_data:
            raise SchemaError("frames must be a non-empty list")
        frames = tuple(FrameRecord.from_dict(item) for item in frames_data)
        ids = [frame.frame_id for frame in frames]
        if len(ids) != len(set(ids)):
            raise SchemaError("frame_id values must be unique")
        return cls(
            sequence_id=str(_require(data, "sequence_id")),
            coordinate_frame=str(_require(data, "coordinate_frame")),
            camera_model=str(_require(data, "camera_model")),
            frames=frames,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CaptureSequence":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "coordinate_frame": self.coordinate_frame,
            "camera_model": self.camera_model,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class WorldManifest:
    schema_version: str
    world_id: str
    builder: str
    asset_uri: str
    asset_format: str
    coordinate_frame: str
    metric_scale: float | None = None
    source_sequence_id: str | None = None
    collider_uri: str | None = None
    asset_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldManifest":
        metric_scale = data.get("metric_scale")
        if metric_scale is not None and float(metric_scale) <= 0:
            raise SchemaError("metric_scale must be positive")
        return cls(
            schema_version=str(_require(data, "schema_version")),
            world_id=str(_require(data, "world_id")),
            builder=str(_require(data, "builder")),
            asset_uri=str(_require(data, "asset_uri")),
            asset_format=str(_require(data, "asset_format")).lower(),
            coordinate_frame=str(_require(data, "coordinate_frame")),
            metric_scale=None if metric_scale is None else float(metric_scale),
            source_sequence_id=data.get("source_sequence_id"),
            collider_uri=data.get("collider_uri"),
            asset_sha256=data.get("asset_sha256"),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "WorldManifest":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
