from __future__ import annotations

from pathlib import Path
import json
import math


def build_observation_goal_index(
    entity_index_path: str | Path,
    virtual_views_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Bind entities to camera poses that actually observed their source masks."""
    entity_source = Path(entity_index_path)
    view_source = Path(virtual_views_path)
    entity_document = json.loads(entity_source.read_text(encoding="utf-8"))
    view_document = json.loads(view_source.read_text(encoding="utf-8"))
    views = {index: view for index, view in enumerate(view_document["views"])}
    entities = []
    for entity in entity_document.get("entities", []):
        center = [float(value) for value in entity["derived_geometry"]["center"]]
        candidates = []
        used_views = set()
        for observation in entity.get("observations", []):
            view_index = int(observation["view_index"])
            if view_index in used_views or view_index not in views:
                continue
            used_views.add(view_index)
            view = views[view_index]
            position = [float(value) for value in view["camera_position"]]
            direction = [target - origin for target, origin in zip(center, position)]
            distance = math.sqrt(sum(value * value for value in direction))
            forward = [value / max(distance, 1e-9) for value in direction]
            candidates.append(
                {
                    "candidate_id": f"{entity['entity_id']}_view_{view_index:03d}",
                    "source_view_id": view.get("view_id", f"view_{view_index:03d}"),
                    "position": position,
                    "look_at": center,
                    "forward": forward,
                    "up": [0.0, -1.0, 0.0],
                    "distance_to_entity_center": distance,
                    "visibility_evidence": "source_mask_observation",
                    "reachability_status": "unvalidated",
                }
            )
        candidates.sort(key=lambda item: item["distance_to_entity_center"])
        entities.append(
            {
                "entity_id": entity["entity_id"],
                "labels": entity.get("labels", []),
                "membership_uri": entity["membership_uri"],
                "goal_candidates": candidates,
            }
        )
    result = {
        "schema_version": "0.2",
        "world_id": entity_document.get("world_id"),
        "world_role": "B_internal_twin",
        "evaluation_access": "system",
        "source_entity_index": str(entity_source.resolve()),
        "source_virtual_views": str(view_source.resolve()),
        "coordinate_frame": view_document.get("coordinate_frame", "source_native_y_down"),
        "source_renderer": view_document.get("renderer"),
        "requires_map_registration": True,
        "reachability_filter_applied": False,
        "entities": entities,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
