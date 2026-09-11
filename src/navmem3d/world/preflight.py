from __future__ import annotations

from pathlib import Path
import json
import math


def _dependencies():
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("input preflight requires NumPy and Pillow") from exc
    return np, Image


def inspect_world_builder_input(selection_path: str | Path) -> dict[str, object]:
    """Check a selected multi-view set before spending a world-builder call."""
    np, Image = _dependencies()
    source = Path(selection_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    frames = document.get("frames", [])
    if not frames:
        raise ValueError("selection has no frames")
    results = []
    hashes = []
    positions = []
    blocking = []
    warnings = []

    for frame in frames:
        image_path = (source.parent / frame["rgb_uri"]).resolve()
        if not image_path.is_file():
            blocking.append(f"missing image: {image_path}")
            continue
        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        gray = rgb.mean(axis=2)
        laplacian = (
            -4.0 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        thumbnail = np.asarray(image.resize((16, 16)).convert("L"), dtype=np.float32)
        perceptual_hash = thumbnail > thumbnail.mean()
        hashes.append(perceptual_hash)
        transform = frame.get("T_world_camera", [])
        if len(transform) == 16:
            positions.append([float(transform[3]), float(transform[7]), float(transform[11])])
        metrics = {
            "frame_id": str(frame["frame_id"]),
            "image_uri": str(image_path),
            "width": image.width,
            "height": image.height,
            "mean_brightness": float(gray.mean()),
            "dark_fraction": float((gray < 0.03).mean()),
            "bright_fraction": float((gray > 0.97).mean()),
            "laplacian_variance": float(laplacian.var()),
        }
        if min(image.size) < 384:
            blocking.append(f"{frame['frame_id']}: shortest side below 384 px")
        if metrics["laplacian_variance"] < 0.00015:
            warnings.append(f"{frame['frame_id']}: possibly blurred or low-texture")
        if metrics["mean_brightness"] < 0.08 or metrics["mean_brightness"] > 0.92:
            warnings.append(f"{frame['frame_id']}: extreme mean brightness")
        results.append(metrics)

    closest_hash_pair = None
    min_hash_distance = math.inf
    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            distance = int(np.count_nonzero(hashes[left] != hashes[right]))
            if distance < min_hash_distance:
                min_hash_distance = distance
                closest_hash_pair = [results[left]["frame_id"], results[right]["frame_id"]]
    if min_hash_distance < 8:
        warnings.append(f"near-duplicate views: {closest_hash_pair}")

    coverage = None
    if positions:
        position_array = np.asarray(positions, dtype=np.float64)
        span = position_array.max(axis=0) - position_array.min(axis=0)
        pairwise = np.linalg.norm(
            position_array[:, None, :] - position_array[None, :, :], axis=2
        )
        nonzero = pairwise[pairwise > 1e-9]
        coverage = {
            "translation_span_xyz": span.tolist(),
            "minimum_pair_distance": None if nonzero.size == 0 else float(nonzero.min()),
            "maximum_pair_distance": float(pairwise.max()),
        }
        if float(np.linalg.norm(span)) < 0.5:
            warnings.append("camera translation coverage is small")

    return {
        "sequence_id": document.get("sequence_id"),
        "frame_count": len(results),
        "ready_for_paid_generation": not blocking,
        "blocking_issues": blocking,
        "warnings": warnings,
        "closest_perceptual_hash_pair": closest_hash_pair,
        "minimum_hash_distance_bits": None if math.isinf(min_hash_distance) else min_hash_distance,
        "pose_coverage": coverage,
        "frames": results,
    }
