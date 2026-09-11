from __future__ import annotations

from pathlib import Path
import json


def derive_entity_geometry(
    fused_index_path: str | Path,
    asset_path: str | Path,
    output_path: str | Path,
    *,
    near_threshold_ratio: float = 0.06,
) -> dict[str, object]:
    """Derive recomputable entity geometry and relations from Gaussian membership."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("entity geometry requires NumPy") from exc
    source = Path(fused_index_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    asset = Path(asset_path)
    if asset.suffix.lower() == ".ply":
        header = asset.read_bytes()[:4096]
        if b"element chunk " in header:
            from navmem3d.world.supersplat import load_supersplat_compressed_ply
            positions = np.asarray(load_supersplat_compressed_ply(asset).means)
        else:
            from plyfile import PlyData
            vertex = PlyData.read(str(asset)).elements[0]
            required = {"x", "y", "z"}
            names = {prop.name for prop in vertex.properties}
            missing = required - names
            if missing:
                raise ValueError(f"PLY asset is missing position properties: {sorted(missing)}")
            positions = np.stack([vertex[name] for name in ("x", "y", "z")], axis=1)
    elif asset.suffix.lower() == ".splat":
        dtype = np.dtype(
            [
                ("xyz", "<f4", (3,)),
                ("scale", "<f4", (3,)),
                ("rgba", "u1", (4,)),
                ("quat", "u1", (4,)),
            ]
        )
        positions = np.asarray(np.memmap(asset, dtype=dtype, mode="r")["xyz"])
    else:
        raise ValueError(f"unsupported Gaussian geometry asset: {asset.suffix}")
    positions = positions[np.isfinite(positions).all(axis=1)]
    if positions.size == 0:
        raise ValueError("Gaussian asset has no finite positions")
    scene_low = np.quantile(positions, 0.05, axis=0)
    scene_high = np.quantile(positions, 0.95, axis=0)
    scene_diagonal = float(np.linalg.norm(scene_high - scene_low))
    near_threshold = scene_diagonal * near_threshold_ratio
    entities = []
    for entity in document.get("entities", []):
        ids = np.load(source.parent / entity["membership_uri"], allow_pickle=False)
        if ids.size == 0 or int(ids.max()) >= len(positions):
            raise ValueError(f"invalid membership for {entity['entity_id']}")
        xyz = np.asarray(positions[ids], dtype=np.float64)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        if xyz.size == 0:
            raise ValueError(f"membership has no finite Gaussian positions for {entity['entity_id']}")
        low = np.quantile(xyz, 0.05, axis=0)
        high = np.quantile(xyz, 0.95, axis=0)
        center = np.median(xyz, axis=0)
        entities.append(
            {
                **entity,
                "derived_geometry": {
                    "center": center.tolist(),
                    "bounds_p05_p95": [low.tolist(), high.tolist()],
                    "extent": (high - low).tolist(),
                },
                "relations": [],
            }
        )

    def bbox_gap(left, right):
        return np.maximum(0.0, np.maximum(left[0] - right[1], right[0] - left[1]))

    for left_index, left in enumerate(entities):
        left_bounds = np.asarray(left["derived_geometry"]["bounds_p05_p95"])
        left_center = np.asarray(left["derived_geometry"]["center"])
        for right_index, right in enumerate(entities):
            if left_index == right_index:
                continue
            right_bounds = np.asarray(right["derived_geometry"]["bounds_p05_p95"])
            right_center = np.asarray(right["derived_geometry"]["center"])
            gap = float(np.linalg.norm(bbox_gap(left_bounds, right_bounds)))
            if gap <= near_threshold:
                left["relations"].append(
                    {
                        "predicate": "near",
                        "object_id": right["entity_id"],
                        "distance": gap,
                    }
                )
            delta = right_center - left_center
            separation = np.abs(delta)
            dominant = int(np.argmax(separation))
            if dominant == 0:
                predicate = "left_of" if delta[0] > 0 else "right_of"
            elif dominant == 1:
                # This source .splat uses y-down coordinates.
                predicate = "above" if delta[1] > 0 else "below"
            else:
                predicate = "behind" if delta[2] > 0 else "in_front_of"
            left["relations"].append(
                {
                    "predicate": predicate,
                    "object_id": right["entity_id"],
                    "center_axis_distance": float(separation[dominant]),
                }
            )
    result = {
        "schema_version": "0.2",
        "world_id": document.get("world_id"),
        "world_role": "B_internal_twin",
        "evaluation_access": "system",
        "world_geometry_source": str(Path(asset_path).resolve()),
        "coordinate_frame": "source_native_y_down",
        "geometry_is_recomputable_cache": True,
        "scene_robust_bounds": [scene_low.tolist(), scene_high.tolist()],
        "near_threshold": near_threshold,
        "near_threshold_ratio": near_threshold_ratio,
        "entities": entities,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
