from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from navmem3d.schemas import SchemaError, WorldManifest


REQUIRED_FILES = (
    "3dgs_compressed.ply",
    "labels.json",
    "occupancy.png",
    "occupancy.json",
    "structure.json",
)


@dataclass(frozen=True)
class InteriorGSScene:
    root: Path
    gaussian: Path
    labels: Path
    occupancy_image: Path
    occupancy_metadata: Path
    structure: Path


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON in {path}: {exc}") from exc


def _collection_size(value: Any, *keys: str) -> int | None:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for key in keys:
            item = value.get(key)
            if isinstance(item, list):
                return len(item)
    return None


def inspect_interiorgs_scene(scene_dir: str | Path) -> dict[str, Any]:
    root = Path(scene_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise SchemaError(f"InteriorGS scene is missing required files: {', '.join(missing)}")

    scene = InteriorGSScene(
        root=root,
        gaussian=root / "3dgs_compressed.ply",
        labels=root / "labels.json",
        occupancy_image=root / "occupancy.png",
        occupancy_metadata=root / "occupancy.json",
        structure=root / "structure.json",
    )
    labels = _load_json(scene.labels)
    occupancy = _load_json(scene.occupancy_metadata)
    structure = _load_json(scene.structure)

    return {
        "scene_id": root.name,
        "scene_dir": str(root),
        "gaussian_path": str(scene.gaussian),
        "gaussian_size_bytes": scene.gaussian.stat().st_size,
        "coordinate_frame": "interiorgs_xyz_right_back_up",
        "metric_scale": 1.0,
        "label_count": _collection_size(labels, "objects", "instances", "labels", "ins"),
        "room_count": _collection_size(structure, "rooms"),
        "structure_instance_count": _collection_size(structure, "ins", "instances"),
        "occupancy_metadata_keys": sorted(occupancy) if isinstance(occupancy, dict) else [],
        "files": {name: str(root / name) for name in REQUIRED_FILES},
    }


def register_interiorgs_scene(
    scene_dir: str | Path, output_manifest: str | Path
) -> tuple[WorldManifest, dict[str, Any]]:
    summary = inspect_interiorgs_scene(scene_dir)
    root = Path(scene_dir).resolve()
    destination = Path(output_manifest).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    gaussian = root / "3dgs_compressed.ply"
    digest = hashlib.sha256()
    with gaussian.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    def relative(path: Path) -> str:
        return Path(os.path.relpath(path, destination.parent)).as_posix()

    manifest = WorldManifest(
        schema_version="0.1",
        world_id=f"interiorgs_{root.name}",
        builder="interiorgs_reference",
        asset_uri=relative(gaussian),
        asset_format="ply",
        coordinate_frame="interiorgs_xyz_right_back_up",
        metric_scale=1.0,
        asset_sha256=digest.hexdigest(),
        metadata={
            "role": "simulation_ground_truth_world_a",
            "occupancy_image_uri": relative(root / "occupancy.png"),
            "occupancy_metadata_uri": relative(root / "occupancy.json"),
            "labels_uri": relative(root / "labels.json"),
            "structure_uri": relative(root / "structure.json"),
            "ground_truth_access": "evaluator_only",
        },
    )
    destination.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest, summary
