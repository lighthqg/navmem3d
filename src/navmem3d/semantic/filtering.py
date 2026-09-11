from __future__ import annotations

from pathlib import Path
import json


def build_searchable_entity_index(
    geometry_index_path: str | Path,
    output_path: str | Path,
    *,
    structural_axis_ratio: float = 0.45,
    structural_axis_count: int = 2,
    structural_max_axis_ratio: float = 0.85,
    single_view_max_axis_ratio: float = 0.30,
) -> dict[str, object]:
    """Separate likely objects from structural or ambiguous SAM regions.

    All decisions are recorded rather than deleting proposals.  Multi-view
    support admits a proposal unless its geometry is scene-scale; compact
    single-view proposals remain searchable with provisional confidence.
    """
    source = Path(geometry_index_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    low, high = document["scene_robust_bounds"]
    span = [max(float(upper) - float(lower), 1e-9) for lower, upper in zip(low, high)]
    searchable = []
    excluded = []
    decisions = []

    for entity in document.get("entities", []):
        extent = entity["derived_geometry"]["extent"]
        ratios = [float(value) / axis_span for value, axis_span in zip(extent, span)]
        broad_axes = sum(value >= structural_axis_ratio for value in ratios)
        if broad_axes >= structural_axis_count or max(ratios) >= structural_max_axis_ratio:
            status = "structural_region"
            reason = "scene_scale_extent"
        elif int(entity.get("view_count", 1)) >= 2:
            status = "object_candidate"
            reason = "multi_view_support"
        elif max(ratios) <= single_view_max_axis_ratio:
            status = "object_candidate"
            reason = "compact_single_view"
        else:
            status = "ambiguous_region"
            reason = "single_view_noncompact"
        annotated = {
            **entity,
            "candidate_status": status,
            "candidate_reason": reason,
            "extent_scene_ratio": ratios,
        }
        decisions.append(
            {
                "entity_id": entity["entity_id"],
                "status": status,
                "reason": reason,
                "extent_scene_ratio": ratios,
            }
        )
        (searchable if status == "object_candidate" else excluded).append(annotated)

    searchable_ids = {entity["entity_id"] for entity in searchable}
    for entity in searchable:
        entity["relations"] = [
            relation
            for relation in entity.get("relations", [])
            if relation.get("object_id") in searchable_ids
        ]
    result = {
        "schema_version": "0.2",
        "world_id": document.get("world_id"),
        "world_role": "B_internal_twin",
        "evaluation_access": "system",
        "source_geometry_index": str(source.resolve()),
        "scene_robust_bounds": document["scene_robust_bounds"],
        "thresholds": {
            "structural_axis_ratio": structural_axis_ratio,
            "structural_axis_count": structural_axis_count,
            "structural_max_axis_ratio": structural_max_axis_ratio,
            "single_view_max_axis_ratio": single_view_max_axis_ratio,
        },
        "searchable_entity_count": len(searchable),
        "excluded_entity_count": len(excluded),
        "structural_region_count": sum(
            item["status"] == "structural_region" for item in decisions
        ),
        "ambiguous_region_count": sum(
            item["status"] == "ambiguous_region" for item in decisions
        ),
        "entities": searchable,
        "excluded_entities": excluded,
        "decisions": decisions,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
