from __future__ import annotations

from collections import Counter
from pathlib import Path
import json


def _dependencies():
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("semantic mask compilation requires NumPy and Pillow") from exc
    return np, Image


def compile_mask_annotations(
    annotation_path: str | Path,
    output_dir: str | Path,
    *,
    min_view_votes: int = 1,
) -> dict[str, object]:
    """Lift 2D instance masks to stable Gaussian membership arrays.

    Gaussian IDs are counted at most once per observation, so the threshold is
    measured in supporting views rather than pixels. Input paths are resolved
    relative to the annotation JSON.
    """
    np, Image = _dependencies()
    if min_view_votes <= 0:
        raise ValueError("min_view_votes must be positive")
    source = Path(annotation_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    instances = document.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("annotations must contain a non-empty instances list")
    destination = Path(output_dir)
    membership_dir = destination / "memberships"
    membership_dir.mkdir(parents=True, exist_ok=True)
    compiled = []

    for instance in instances:
        entity_id = str(instance["entity_id"])
        observations = instance.get("observations", [])
        if not observations:
            raise ValueError(f"entity {entity_id} has no observations")
        votes: Counter[int] = Counter()
        source_views = []
        for observation in observations:
            mask_path = source.parent / observation["mask_uri"]
            id_path = source.parent / observation["gaussian_id_uri"]
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            id_map = np.load(id_path, allow_pickle=False)
            if mask.shape != id_map.shape:
                raise ValueError(
                    f"mask/ID shape mismatch for {entity_id}: {mask.shape} != {id_map.shape}"
                )
            ids = np.unique(id_map[mask])
            ids = ids[ids != np.iinfo(np.uint32).max]
            votes.update(int(value) for value in ids)
            source_views.append(str(observation.get("view_id", mask_path.stem)))
        kept = np.array(
            sorted(key for key, count in votes.items() if count >= min_view_votes),
            dtype=np.uint32,
        )
        membership_path = membership_dir / f"{entity_id}.npy"
        np.save(membership_path, kept)
        compiled.append(
            {
                "entity_id": entity_id,
                "labels": [str(value).strip().lower() for value in instance.get("labels", [])],
                "attributes": [
                    str(value).strip().lower() for value in instance.get("attributes", [])
                ],
                "membership_uri": str(membership_path.relative_to(destination)),
                "gaussian_count": int(kept.size),
                "source_views": source_views,
                "relations": list(instance.get("relations", [])),
            }
        )
    result = {
        "schema_version": "0.1",
        "world_id": str(document["world_id"]),
        "asset_uri": str(document["asset_uri"]),
        "entities": compiled,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "entity_index.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def lift_mask_set(
    mask_manifest_path: str | Path,
    gaussian_id_path: str | Path,
    output_dir: str | Path,
    *,
    min_gaussians: int = 32,
    gaussian_weight_path: str | Path | None = None,
    min_weight: float = 0.0,
    min_pixel_hits: int = 1,
) -> dict[str, object]:
    """Lift every SAM mask in one virtual view into an unlabeled 3D proposal."""
    np, Image = _dependencies()
    source = Path(mask_manifest_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    id_map = np.load(gaussian_id_path, allow_pickle=False)
    weight_map = (
        np.load(gaussian_weight_path, allow_pickle=False)
        if gaussian_weight_path is not None
        else None
    )
    if id_map.ndim not in (2, 3):
        raise ValueError(f"Gaussian ID map must be HxW or HxWxK, got {id_map.shape}")
    if weight_map is not None and weight_map.shape != id_map.shape:
        raise ValueError(f"ID/weight shape mismatch: {id_map.shape} != {weight_map.shape}")
    if min_pixel_hits <= 0:
        raise ValueError("min_pixel_hits must be positive")
    destination = Path(output_dir)
    membership_dir = destination / "memberships"
    membership_dir.mkdir(parents=True, exist_ok=True)
    proposals = []
    for record in manifest.get("masks", []):
        mask_path = source.parent / record["mask_uri"]
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape != id_map.shape[:2]:
            raise ValueError(f"mask/ID shape mismatch: {mask.shape} != {id_map.shape[:2]}")
        sampled_ids = id_map[mask].reshape(-1)
        valid = sampled_ids != np.iinfo(np.uint32).max
        if weight_map is not None:
            sampled_weights = weight_map[mask].reshape(-1)
            valid &= sampled_weights >= min_weight
        sampled_ids = sampled_ids[valid]
        ids, hit_counts = np.unique(sampled_ids, return_counts=True)
        ids = ids[hit_counts >= min_pixel_hits].astype(np.uint32)
        if ids.size < min_gaussians:
            continue
        proposal_id = str(record["mask_id"])
        membership_path = membership_dir / f"{proposal_id}.npy"
        np.save(membership_path, ids)
        proposals.append(
            {
                "proposal_id": proposal_id,
                "membership_uri": str(membership_path.relative_to(destination)),
                "gaussian_count": int(ids.size),
                "source_mask_uri": str(mask_path.resolve()),
                "predicted_iou": float(record["predicted_iou"]),
                "stability_score": float(record["stability_score"]),
                "bbox_xywh": record["bbox_xywh"],
                "category_id": record.get("category_id"),
                "labels": record.get("labels", []),
                "detector_score": record.get("detector_score"),
            }
        )
    result = {
        "schema_version": "0.1",
        "source_mask_manifest": str(source.resolve()),
        "gaussian_id_map": str(Path(gaussian_id_path).resolve()),
        "gaussian_weight_map": str(Path(gaussian_weight_path).resolve())
        if gaussian_weight_path is not None
        else None,
        "min_weight": min_weight,
        "min_pixel_hits": min_pixel_hits,
        "proposal_count": len(proposals),
        "proposals": proposals,
    }
    (destination / "proposal_index.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
