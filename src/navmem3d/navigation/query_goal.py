from __future__ import annotations

from pathlib import Path
import json

from navmem3d.semantic.query import query_entity_index


def query_navigation_goals(
    entity_index_path: str | Path,
    goal_index_path: str | Path,
    *,
    terms: list[str],
    attributes: list[str] | None = None,
    predicate: str | None = None,
    reference_terms: list[str] | None = None,
    max_goals_per_entity: int = 3,
) -> dict[str, object]:
    """Resolve a deterministic entity query into observation-pose candidates."""
    query = query_entity_index(
        entity_index_path,
        terms=terms,
        attributes=attributes,
        predicate=predicate,
        reference_terms=reference_terms,
    )
    goal_document = json.loads(Path(goal_index_path).read_text(encoding="utf-8"))
    goals_by_entity = {
        entity["entity_id"]: entity.get("goal_candidates", [])
        for entity in goal_document.get("entities", [])
    }
    targets = []
    for entity in query["matches"]:
        candidates = goals_by_entity.get(entity["entity_id"], [])[:max_goals_per_entity]
        targets.append(
            {
                "entity_id": entity["entity_id"],
                "labels": entity.get("labels", []),
                "membership_uri": entity.get("membership_uri"),
                "goal_candidates": candidates,
            }
        )
    return {
        "query": query["query"],
        "match_count": query["match_count"],
        "targets": targets,
        "fallback_required": query["fallback_required"],
        "requires_map_registration": goal_document.get("requires_map_registration", True),
        "reachability_filter_applied": goal_document.get(
            "reachability_filter_applied", False
        ),
    }
