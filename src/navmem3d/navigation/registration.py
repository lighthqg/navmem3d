from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def estimate_bounds_registration(
    b_splat_path: str | Path,
    a_scene_dir: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, object]:
    """Estimate a coarse B(RDF)→A(InteriorGS right/back/up) transform.

    This is an initialization only. Marble's converted PLY uses right/down/forward;
    InteriorGS uses right/back/up.
    """
    b_path = Path(b_splat_path)
    if b_path.suffix.lower() == ".splat":
        dtype = np.dtype([("xyz", "<f4", (3,)), ("scale", "<f4", (3,)), ("rgba", "u1", (4,)), ("quat", "u1", (4,))])
        b_xyz = np.asarray(np.memmap(b_path, dtype=dtype, mode="r")["xyz"], dtype=float)
    elif b_path.suffix.lower() == ".ply":
        header = b_path.read_bytes()[:4096]
        if b"element chunk " in header:
            from navmem3d.world.supersplat import load_supersplat_compressed_ply
            b_xyz = np.asarray(load_supersplat_compressed_ply(b_path).means, dtype=float)
        else:
            from plyfile import PlyData
            vertex = PlyData.read(str(b_path)).elements[0]
            names = {prop.name for prop in vertex.properties}
            missing = {"x", "y", "z"} - names
            if missing:
                raise ValueError(f"PLY asset is missing position properties: {sorted(missing)}")
            b_xyz = np.stack([vertex[name] for name in ("x", "y", "z")], axis=1).astype(float)
    else:
        raise ValueError(f"unsupported B geometry asset: {b_path.suffix}")
    b_xyz = b_xyz[np.isfinite(b_xyz).all(axis=1)]
    if b_xyz.size == 0:
        raise ValueError("B geometry asset has no finite Gaussian positions")
    # RDF -> A: (x_right, y_down, z_forward) -> (x_right, y_back, z_up)
    b_a = np.stack([b_xyz[:, 0], b_xyz[:, 2], -b_xyz[:, 1]], axis=1)
    b_low, b_high = np.quantile(b_a, [0.05, 0.95], axis=0)
    scene = Path(a_scene_dir)
    metadata = json.loads((scene / "occupancy.json").read_text())
    a_low = np.asarray(metadata["lower"], dtype=float)
    a_high = np.asarray(metadata["upper"], dtype=float)
    a_low[2], a_high[2] = 0.0, float(a_high[2])
    b_span = np.maximum(b_high - b_low, 1e-6)
    a_span = np.maximum(a_high - a_low, 1e-6)
    scale = float(np.median(a_span / b_span))
    b_center = (b_low + b_high) / 2.0
    a_center = (a_low + a_high) / 2.0
    translation = a_center - scale * b_center
    # B's converted PLY axes are (right, down, forward); A is (right, back, up).
    # Keep this axis rotation in the transform itself so downstream goal poses
    # are registered in the same frame used for bound estimation.
    axis_map = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=float
    )
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = scale * axis_map
    matrix[:3, 3] = translation
    result = {
        "schema_version": "0.1",
        "method": "robust_bounds_axis_mapped_initializer",
        "source_b": str(Path(b_splat_path).resolve()),
        "source_a": str(scene.resolve()),
        "b_coordinate_frame": "RDF",
        "a_coordinate_frame": "interiorgs_xyz_right_back_up",
        "transform_b_to_a": matrix.reshape(-1).tolist(),
        "scale": scale,
        "b_bounds_after_axis_map": [b_low.tolist(), b_high.tolist()],
        "a_bounds": [a_low.tolist(), a_high.tolist()],
        "status": "coarse_initializer_requires_visual_or_landmark_refinement",
    }
    destination = Path(output_path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def register_world_b_to_a(
    goal_index_path: str | Path,
    *,
    transform_b_to_a: list[float] | None = None,
    output_path: str | Path,
    registration_status: str | None = None,
) -> dict[str, object]:
    """Apply a supplied rigid/affine B→A transform to observation goals.

    Marble and InteriorGS do not share a metric frame automatically. The
    first experiment therefore keeps registration explicit: identity is only
    a debug placeholder, while a calibrated matrix must be supplied for
    navigation claims.
    """
    source = Path(goal_index_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    matrix = np.asarray(transform_b_to_a or np.eye(4).reshape(-1).tolist(), dtype=float)
    if matrix.size != 16:
        raise ValueError("transform_b_to_a must contain 16 values")
    matrix = matrix.reshape(4, 4)
    inferred_status = "identity_debug" if np.allclose(matrix, np.eye(4)) else "calibrated"
    status = registration_status or inferred_status
    entities = []
    for entity in document.get("entities", []):
        goals = []
        for goal in entity.get("goal_candidates", []):
            p = np.r_[goal["position"], 1.0]
            q = np.r_[goal["look_at"], 1.0]
            updated = {**goal, "position_a": (matrix @ p)[:3].tolist(), "look_at_a": (matrix @ q)[:3].tolist()}
            updated["registration_status"] = status
            goals.append(updated)
        entities.append({**entity, "goal_candidates": goals})
    result = {**document, "coordinate_frame": "world_b_to_world_a", "requires_map_registration": status in {"identity_debug", "coarse_initializer_requires_visual_or_landmark_refinement"}, "registration": {"transform_b_to_a": matrix.reshape(-1).tolist(), "status": status}, "entities": entities}
    destination = Path(output_path); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
