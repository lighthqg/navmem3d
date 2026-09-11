from __future__ import annotations

from pathlib import Path
import json


def fuse_proposal_indices(
    proposal_index_paths: list[str | Path],
    output_dir: str | Path,
    *,
    minimum_containment: float = 0.20,
    minimum_jaccard: float = 0.08,
) -> dict[str, object]:
    """Fuse cross-view proposals that share stable Gaussian membership."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("proposal fusion requires NumPy") from exc
    if len(proposal_index_paths) < 2:
        raise ValueError("at least two proposal indices are required")
    proposals = []
    memberships = []
    for view_index, input_path in enumerate(proposal_index_paths):
        path = Path(input_path)
        document = json.loads(path.read_text(encoding="utf-8"))
        for proposal in document.get("proposals", []):
            membership = np.load(path.parent / proposal["membership_uri"], allow_pickle=False)
            proposals.append(
                {
                    "view_index": view_index,
                    "proposal_index": str(path.resolve()),
                    **proposal,
                }
            )
            memberships.append(membership)
    parents = list(range(len(proposals)))
    cluster_views = [{int(proposal["view_index"])} for proposal in proposals]

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: int, right: int) -> bool:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        # Prevent transitive bridges from merging two competing masks from the
        # same view into one 3D entity.
        if cluster_views[left_root] & cluster_views[right_root]:
            return False
        parents[right_root] = left_root
        cluster_views[left_root] |= cluster_views[right_root]
        return True

    candidate_matches = []
    for left in range(len(proposals)):
        for right in range(left + 1, len(proposals)):
            if proposals[left]["view_index"] == proposals[right]["view_index"]:
                continue
            left_category = proposals[left].get("category_id")
            right_category = proposals[right].get("category_id")
            if left_category and right_category and left_category != right_category:
                continue
            intersection = np.intersect1d(
                memberships[left], memberships[right], assume_unique=True
            ).size
            if intersection == 0:
                continue
            containment = intersection / min(memberships[left].size, memberships[right].size)
            jaccard = intersection / (
                memberships[left].size + memberships[right].size - intersection
            )
            if containment >= minimum_containment or jaccard >= minimum_jaccard:
                candidate_matches.append(
                    {
                        "left": left,
                        "right": right,
                        "intersection": int(intersection),
                        "containment": float(containment),
                        "jaccard": float(jaccard),
                    }
                )
    candidate_matches.sort(
        key=lambda match: (match["containment"], match["jaccard"], match["intersection"]),
        reverse=True,
    )
    matches = []
    rejected_view_conflicts = 0
    for match in candidate_matches:
        if union(match["left"], match["right"]):
            matches.append(match)
        else:
            rejected_view_conflicts += 1

    groups: dict[int, list[int]] = {}
    for index in range(len(proposals)):
        groups.setdefault(find(index), []).append(index)
    destination = Path(output_dir)
    membership_dir = destination / "memberships"
    membership_dir.mkdir(parents=True, exist_ok=True)
    entities = []
    for entity_number, group in enumerate(
        sorted(groups.values(), key=lambda values: (-len(values), values[0]))
    ):
        entity_id = f"entity_{entity_number:04d}"
        membership = np.unique(np.concatenate([memberships[index] for index in group])).astype(
            np.uint32
        )
        membership_path = membership_dir / f"{entity_id}.npy"
        np.save(membership_path, membership)
        entities.append(
            {
                "entity_id": entity_id,
                "membership_uri": str(membership_path.relative_to(destination)),
                "gaussian_count": int(membership.size),
                "view_count": len({proposals[index]["view_index"] for index in group}),
                "observations": [
                    {
                        "view_index": proposals[index]["view_index"],
                        "proposal_id": proposals[index]["proposal_id"],
                        "proposal_index": proposals[index]["proposal_index"],
                    }
                    for index in group
                ],
                "category_id": next(
                    (
                        proposals[index].get("category_id")
                        for index in group
                        if proposals[index].get("category_id")
                    ),
                    None,
                ),
                "labels": next(
                    (proposals[index].get("labels", []) for index in group if proposals[index].get("labels")),
                    [],
                ),
            }
        )
    result = {
        "schema_version": "0.2",
        "world_role": "B_internal_twin",
        "evaluation_access": "system",
        "input_view_count": len(proposal_index_paths),
        "input_proposal_count": len(proposals),
        "cross_view_match_count": len(matches),
        "candidate_cross_view_match_count": len(candidate_matches),
        "rejected_view_conflicts": rejected_view_conflicts,
        "entity_count": len(entities),
        "multi_view_entity_count": sum(entity["view_count"] > 1 for entity in entities),
        "thresholds": {
            "minimum_containment": minimum_containment,
            "minimum_jaccard": minimum_jaccard,
        },
        "entities": entities,
        "matches": matches,
    }
    (destination / "fused_entities.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
