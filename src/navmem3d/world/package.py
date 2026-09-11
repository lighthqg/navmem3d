from __future__ import annotations

from pathlib import Path
import hashlib

from navmem3d.schemas import WorldManifest


def inspect_world(manifest_path: str | Path) -> dict[str, object]:
    path = Path(manifest_path)
    manifest = WorldManifest.load(path)
    asset = (path.parent / manifest.asset_uri).resolve()
    collider = None
    if manifest.collider_uri:
        collider = (path.parent / manifest.collider_uri).resolve()
    summary = {
        "world_id": manifest.world_id,
        "builder": manifest.builder,
        "asset_format": manifest.asset_format,
        "coordinate_frame": manifest.coordinate_frame,
        "metric_scale": manifest.metric_scale,
        "asset_path": str(asset),
        "asset_exists": asset.exists(),
        "collider_path": None if collider is None else str(collider),
        "collider_exists": None if collider is None else collider.exists(),
    }
    if asset.exists() and manifest.asset_sha256:
        digest = hashlib.sha256()
        with asset.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        summary["asset_sha256_matches"] = digest.hexdigest() == manifest.asset_sha256
    if asset.exists() and manifest.asset_format == "splat":
        from navmem3d.world.gaussians import inspect_gaussian_asset

        gaussian = inspect_gaussian_asset(asset)
        summary["gaussian_count"] = gaussian.gaussian_count
        summary["robust_position_bounds"] = [gaussian.position_p05, gaussian.position_p95]
    return summary
