from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math


@dataclass(frozen=True)
class VirtualViewConfig:
    asset: Path
    output_dir: Path
    view_count: int = 8
    width: int = 640
    height: int = 480
    horizontal_fov_deg: float = 90.0
    orbit_radius_ratio: float = 0.18
    point_radius_px: int = 1


def _dependencies():
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "virtual view rendering requires NumPy and Pillow; install navmem3d in the habitat environment"
        ) from exc
    return np, Image


def render_virtual_views(config: VirtualViewConfig) -> dict[str, object]:
    """Render deterministic point previews and exact center-pixel Gaussian IDs.

    This is a geometry/ID-linkage baseline, not the final anisotropic 3DGS
    rasterizer. Every non-background output pixel refers to the nearest
    Gaussian center through the matching uint32 ID map.
    """
    np, Image = _dependencies()
    if config.asset.suffix.lower() != ".splat":
        raise ValueError("the baseline renderer currently supports .splat assets")
    if config.view_count <= 0 or config.width <= 0 or config.height <= 0:
        raise ValueError("view count and image dimensions must be positive")
    if config.point_radius_px < 0 or config.point_radius_px > 4:
        raise ValueError("point_radius_px must be between 0 and 4")

    dtype = np.dtype(
        [
            ("xyz", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("rgba", "u1", (4,)),
            ("quat", "u1", (4,)),
        ]
    )
    gaussians = np.memmap(config.asset, dtype=dtype, mode="r")
    xyz = gaussians["xyz"]
    rgba = gaussians["rgba"]
    low = np.quantile(xyz, 0.05, axis=0)
    high = np.quantile(xyz, 0.95, axis=0)
    center = np.median(xyz, axis=0)
    horizontal_span = max(float(high[0] - low[0]), float(high[2] - low[2]))
    radius = max(horizontal_span * config.orbit_radius_ratio, 0.25)
    focal = 0.5 * config.width / math.tan(math.radians(config.horizontal_fov_deg) / 2.0)
    up = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    views = []
    for view_index in range(config.view_count):
        angle = 2.0 * math.pi * view_index / config.view_count
        eye = center.astype(np.float64).copy()
        eye[0] += radius * math.cos(angle)
        eye[2] += radius * math.sin(angle)
        forward = center.astype(np.float64) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        camera_up = np.cross(right, forward)

        delta = xyz.astype(np.float64) - eye
        depth = delta @ forward
        visible = depth > 0.05
        indices = np.flatnonzero(visible)
        projected = delta[visible]
        z = depth[visible]
        u = np.rint(focal * (projected @ right) / z + config.width / 2.0).astype(np.int32)
        v = np.rint(config.height / 2.0 - focal * (projected @ camera_up) / z).astype(np.int32)
        radius_px = config.point_radius_px
        inside = (
            (u >= -radius_px)
            & (u < config.width + radius_px)
            & (v >= -radius_px)
            & (v < config.height + radius_px)
        )
        u, v, z, indices = u[inside], v[inside], z[inside], indices[inside]
        image = np.zeros((config.height, config.width, 3), dtype=np.uint8)
        image[:] = (245, 245, 245)
        id_map = np.full((config.height, config.width), np.iinfo(np.uint32).max, dtype=np.uint32)
        flat_image = image.reshape(-1, 3)
        flat_ids = id_map.reshape(-1)
        depth_map = np.full(config.height * config.width, np.inf, dtype=np.float32)
        offsets = range(-radius_px, radius_px + 1)
        for dy in offsets:
            for dx in offsets:
                uu, vv = u + dx, v + dy
                valid = (uu >= 0) & (uu < config.width) & (vv >= 0) & (vv < config.height)
                pixels = vv[valid].astype(np.int64) * config.width + uu[valid]
                np.minimum.at(depth_map, pixels, z[valid].astype(np.float32))
        for dy in offsets:
            for dx in offsets:
                uu, vv = u + dx, v + dy
                valid = (uu >= 0) & (uu < config.width) & (vv >= 0) & (vv < config.height)
                pixels = vv[valid].astype(np.int64) * config.width + uu[valid]
                local_z = z[valid].astype(np.float32)
                nearest = local_z <= depth_map[pixels] + 1e-6
                chosen_pixel = pixels[nearest]
                chosen_ids = indices[valid][nearest]
                flat_image[chosen_pixel] = rgba[chosen_ids, :3]
                flat_ids[chosen_pixel] = chosen_ids.astype(np.uint32)

        stem = f"view_{view_index:03d}"
        rgb_path = config.output_dir / f"{stem}.png"
        ids_path = config.output_dir / f"{stem}_gaussian_ids.npy"
        Image.fromarray(image).save(rgb_path)
        np.save(ids_path, id_map)
        views.append(
            {
                "view_id": stem,
                "rgb_uri": rgb_path.name,
                "gaussian_id_uri": ids_path.name,
                "camera_position": eye.tolist(),
                "look_at": center.tolist(),
                "covered_pixels": int(np.isfinite(depth_map).sum()),
            }
        )

    result = {
        "renderer": "center-zbuffer-baseline",
        "asset_uri": str(config.asset.resolve()),
        "gaussian_count": int(len(gaussians)),
        "width": config.width,
        "height": config.height,
        "horizontal_fov_deg": config.horizontal_fov_deg,
        "point_radius_px": config.point_radius_px,
        "robust_bounds_p05_p95": [low.tolist(), high.tolist()],
        "views": views,
    }
    (config.output_dir / "virtual_views.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
