from __future__ import annotations

from pathlib import Path
import json


def deduplicate_entity_index(
    entity_index_path: str | Path,
    output_path: str | Path,
    *,
    minimum_containment: float = 0.75,
    minimum_jaccard: float = 0.45,
) -> dict[str, object]:
    """Suppress category-consistent duplicate entities by 3D membership overlap."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("entity deduplication requires NumPy") from exc
    source = Path(entity_index_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    entities = document.get("entities", [])
    def resolve_membership(entity: dict[str, object]) -> Path:
        raw = Path(str(entity["membership_uri"]))
        candidates = [
            source.parent / raw,
            source.parent / "entities.json" / raw,
            source.parent.parent / raw,
        ]
        for ancestor in (source.parent, *source.parent.parents):
            candidates.append(ancestor / raw)
            candidates.append(ancestor / "entities" / raw)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise ValueError(
            f"membership file for {entity['entity_id']} is not reachable from {source}: {raw}"
        )

    memberships = {
        entity["entity_id"]: np.load(resolve_membership(entity), allow_pickle=False)
        for entity in entities
    }
    ranked = sorted(
        entities,
        key=lambda entity: (int(entity.get("view_count", 1)), int(entity["gaussian_count"])),
        reverse=True,
    )
    kept = []
    duplicate_of: dict[str, str] = {}
    evidence = []
    for candidate in ranked:
        candidate_ids = memberships[candidate["entity_id"]]
        duplicate = None
        for canonical in kept:
            if canonical.get("category_id") != candidate.get("category_id"):
                continue
            canonical_ids = memberships[canonical["entity_id"]]
            intersection = np.intersect1d(candidate_ids, canonical_ids, assume_unique=True).size
            if intersection == 0:
                continue
            containment = intersection / min(candidate_ids.size, canonical_ids.size)
            jaccard = intersection / (
                candidate_ids.size + canonical_ids.size - intersection
            )
            if containment >= minimum_containment or jaccard >= minimum_jaccard:
                duplicate = canonical
                duplicate_of[candidate["entity_id"]] = canonical["entity_id"]
                evidence.append(
                    {
                        "duplicate_id": candidate["entity_id"],
                        "canonical_id": canonical["entity_id"],
                        "category_id": candidate.get("category_id"),
                        "containment": float(containment),
                        "jaccard": float(jaccard),
                    }
                )
                break
        if duplicate is None:
            kept.append(candidate)

    kept_ids = {entity["entity_id"] for entity in kept}
    for entity in kept:
        redirected = {}
        for relation in entity.get("relations", []):
            object_id = duplicate_of.get(relation.get("object_id"), relation.get("object_id"))
            if object_id == entity["entity_id"] or object_id not in kept_ids:
                continue
            updated = {**relation, "object_id": object_id}
            key = (updated.get("predicate"), object_id)
            previous = redirected.get(key)
            metric = updated.get("distance", updated.get("center_axis_distance", float("inf")))
            previous_metric = (
                previous.get("distance", previous.get("center_axis_distance", float("inf")))
                if previous
                else float("inf")
            )
            if metric < previous_metric:
                redirected[key] = updated
        entity["relations"] = list(redirected.values())
    result = {
        **document,
        "source_entity_index": str(source.resolve()),
        "deduplication": {
            "minimum_containment": minimum_containment,
            "minimum_jaccard": minimum_jaccard,
            "input_entity_count": len(entities),
            "output_entity_count": len(kept),
            "suppressed_count": len(duplicate_of),
            "evidence": evidence,
        },
        "searchable_entity_count": len(kept),
        "entities": sorted(kept, key=lambda entity: entity["entity_id"]),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
