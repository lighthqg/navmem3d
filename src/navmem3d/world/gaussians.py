from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import struct


SPLAT_RECORD_SIZE = 32
_SPLAT_RECORD = struct.Struct("<6f8B")


@dataclass(frozen=True)
class GaussianAssetSummary:
    path: str
    format: str
    file_size_bytes: int
    gaussian_count: int
    position_min: tuple[float, float, float]
    position_max: tuple[float, float, float]
    position_p05: tuple[float, float, float]
    position_p95: tuple[float, float, float]
    scale_min: tuple[float, float, float]
    scale_max: tuple[float, float, float]
    mean_opacity: float
    sampled_gaussians: int
    statistics_are_sampled: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def inspect_gaussian_asset(
    path: str | Path, *, max_samples: int = 200_000
) -> GaussianAssetSummary:
    asset = Path(path)
    if not asset.is_file():
        raise FileNotFoundError(asset)
    suffix = asset.suffix.lower()
    if suffix == ".splat":
        return inspect_antimatter_splat(asset, max_samples=max_samples)
    raise ValueError(f"unsupported Gaussian asset format: {suffix or '<none>'}")


def inspect_antimatter_splat(
    path: str | Path, *, max_samples: int = 200_000
) -> GaussianAssetSummary:
    """Inspect the common headerless antimatter15 .splat format.

    Each 32-byte record stores xyz and linear scale as six little-endian
    float32 values, followed by RGBA and a quantized wxyz quaternion.
    Statistics are sampled deterministically for large assets so inspection
    stays fast while the exact record count is always reported.
    """
    asset = Path(path)
    size = asset.stat().st_size
    if size == 0 or size % SPLAT_RECORD_SIZE:
        raise ValueError(
            f"invalid .splat size {size}: expected a positive multiple of {SPLAT_RECORD_SIZE}"
        )
    count = size // SPLAT_RECORD_SIZE
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    sample_count = min(count, max_samples)
    stride = max(1, count // sample_count)

    position_min = [math.inf] * 3
    position_max = [-math.inf] * 3
    scale_min = [math.inf] * 3
    scale_max = [-math.inf] * 3
    opacity_sum = 0
    actual_samples = 0
    sampled_positions = ([], [], [])

    with asset.open("rb") as handle:
        index = 0
        while index < count and actual_samples < sample_count:
            handle.seek(index * SPLAT_RECORD_SIZE)
            raw = handle.read(SPLAT_RECORD_SIZE)
            if len(raw) != SPLAT_RECORD_SIZE:
                raise ValueError(f"truncated .splat record at index {index}")
            values = _SPLAT_RECORD.unpack(raw)
            position = values[0:3]
            scale = values[3:6]
            if not all(math.isfinite(value) for value in position + scale):
                raise ValueError(f"non-finite Gaussian data at index {index}")
            for axis in range(3):
                position_min[axis] = min(position_min[axis], position[axis])
                position_max[axis] = max(position_max[axis], position[axis])
                scale_min[axis] = min(scale_min[axis], scale[axis])
                scale_max[axis] = max(scale_max[axis], scale[axis])
                sampled_positions[axis].append(position[axis])
            opacity_sum += values[9]
            actual_samples += 1
            index += stride

    def percentile(values: list[float], fraction: float) -> float:
        values.sort()
        offset = fraction * (len(values) - 1)
        lower = int(offset)
        upper = min(lower + 1, len(values) - 1)
        weight = offset - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return GaussianAssetSummary(
        path=str(asset.resolve()),
        format="antimatter15-splat",
        file_size_bytes=size,
        gaussian_count=count,
        position_min=tuple(position_min),
        position_max=tuple(position_max),
        position_p05=tuple(percentile(axis, 0.05) for axis in sampled_positions),
        position_p95=tuple(percentile(axis, 0.95) for axis in sampled_positions),
        scale_min=tuple(scale_min),
        scale_max=tuple(scale_max),
        mean_opacity=opacity_sum / (255.0 * actual_samples),
        sampled_gaussians=actual_samples,
        statistics_are_sampled=actual_samples < count,
    )
