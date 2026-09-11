from __future__ import annotations

from pathlib import Path
import json


def _normalize(values: list[str] | tuple[str, ...]) -> set[str]:
    return {str(value).strip().casefold() for value in values if str(value).strip()}


def query_entity_index(
    index_path: str | Path,
    *,
    terms: list[str],
    attributes: list[str] | None = None,
    predicate: str | None = None,
    reference_terms: list[str] | None = None,
) -> dict[str, object]:
    """Run deterministic keyword/attribute/relation lookup over an entity index."""
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    requested_terms = _normalize(terms)
    requested_attributes = _normalize(attributes or [])
    entities = index.get("entities", [])

    def matches(entity: dict[str, object], wanted: set[str]) -> bool:
        labels = _normalize(entity.get("labels", []))
        return bool(labels & wanted)

    candidates = [
        entity
        for entity in entities
        if matches(entity, requested_terms)
        and requested_attributes.issubset(_normalize(entity.get("attributes", [])))
    ]

    if predicate or reference_terms:
        if not predicate or not reference_terms:
            raise ValueError("predicate and reference_terms must be supplied together")
        reference_ids = {
            str(entity["entity_id"])
            for entity in entities
            if matches(entity, _normalize(reference_terms))
        }
        normalized_predicate = predicate.strip().casefold()
        candidates = [
            entity
            for entity in candidates
            if any(
                str(relation.get("predicate", "")).strip().casefold() == normalized_predicate
                and str(relation.get("object_id", "")) in reference_ids
                for relation in entity.get("relations", [])
            )
        ]

    return {
        "world_id": index.get("world_id"),
        "query": {
            "terms": sorted(requested_terms),
            "attributes": sorted(requested_attributes),
            "predicate": predicate,
            "reference_terms": reference_terms or [],
        },
        "match_count": len(candidates),
        "matches": candidates,
        "fallback_required": not candidates,
    }
