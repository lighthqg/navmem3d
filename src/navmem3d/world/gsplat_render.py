from __future__ import annotations

from pathlib import Path
import json
import math
import time


def render_gsplat_views(
    asset_path: str | Path,
    output_dir: str | Path,
    *,
    view_count: int = 4,
    width: int = 640,
    height: int = 480,
    horizontal_fov_deg: float = 90.0,
    orbit_radius_ratio: float = 0.18,
    device: str = "cuda",
    top_contributors: int = 0,
    trajectory_path: str | Path | None = None,
) -> dict[str, object]:
    """Render compact splats and optionally retain exact per-pixel contributors.

    ``top_contributors`` requires gsplat >= 1.6.  IDs are mapped back from the
    packed projection rows to the source asset's Gaussian indices.
    """
    try:
        import numpy as np
        import torch
        from PIL import Image
        import gsplat
        from gsplat.rendering import rasterization
    except ImportError as exc:
        raise RuntimeError("gsplat, PyTorch, NumPy and Pillow are required") from exc
    asset = Path(asset_path)
    if asset.suffix.lower() not in {".splat", ".ply"}:
        raise ValueError("gsplat adapter supports .splat and SuperSplat compressed .ply assets")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if top_contributors < 0:
        raise ValueError("top_contributors must be non-negative")
    top_contributor_fn = getattr(gsplat, "rasterize_top_contributing_gaussian_ids", None)
    if top_contributors and top_contributor_fn is None:
        raise RuntimeError("top-contributor export requires gsplat >= 1.6")

    if asset.suffix.lower() == ".ply":
        header = asset.read_bytes()[:4096]
        if b"element chunk " in header:
            from navmem3d.world.supersplat import load_supersplat_compressed_ply

            decoded = load_supersplat_compressed_ply(asset)
            xyz_np = decoded.means
            scales_np = np.exp(decoded.log_scales)
            colors_np = decoded.colors
            opacities_np = decoded.opacities
            quats_np = decoded.quaternions
            z_up = True
        else:
            # Standard INRIA 3DGS PLY, including Niantic's SPZ-to-PLY output.
            from plyfile import PlyData

            vertex = PlyData.read(str(asset)).elements[0]
            names = {prop.name for prop in vertex.properties}
            required = {
                "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
            }
            missing = required - names
            if missing:
                raise ValueError(f"standard 3DGS PLY is missing properties: {sorted(missing)}")
            xyz_np = np.stack([vertex[name] for name in ("x", "y", "z")], axis=1)
            log_scales = np.stack(
                [vertex[name] for name in ("scale_0", "scale_1", "scale_2")], axis=1
            )
            scales_np = np.exp(log_scales)
            sh0_np = np.stack(
                [vertex[name] for name in ("f_dc_0", "f_dc_1", "f_dc_2")], axis=1
            )
            colors_np = np.clip(0.5 + 0.28209479177387814 * sh0_np, 0.0, 1.0)
            raw_opacities = np.asarray(vertex["opacity"])
            opacities_np = 1.0 / (1.0 + np.exp(-raw_opacities))
            quats_np = np.stack(
                [vertex[name] for name in ("rot_0", "rot_1", "rot_2", "rot_3")], axis=1
            )
            # Converted SPZ PLY preserves the renderer's RDF axes: Y is down.
            z_up = False
    else:
        dtype = np.dtype(
            [
                ("xyz", "<f4", (3,)),
                ("scale", "<f4", (3,)),
                ("rgba", "u1", (4,)),
                ("quat", "u1", (4,)),
            ]
        )
        raw = np.memmap(asset, dtype=dtype, mode="r")
        xyz_np = np.asarray(raw["xyz"])
        scales_np = np.array(raw["scale"], copy=True)
        colors_np = np.array(raw["rgba"][:, :3], copy=True).astype(np.float32) / 255.0
        opacities_np = np.array(raw["rgba"][:, 3], copy=True).astype(np.float32) / 255.0
        quats_np = np.array(raw["quat"], copy=True).astype(np.float32)
        quats_np = (quats_np - 128.0) / 128.0
        z_up = False
    low = np.quantile(xyz_np, 0.05, axis=0)
    high = np.quantile(xyz_np, 0.95, axis=0)
    center = np.median(xyz_np, axis=0).astype(np.float64)
    horizontal_axes = (0, 1) if z_up else (0, 2)
    span = max(float(high[axis] - low[axis]) for axis in horizontal_axes)
    radius = max(span * orbit_radius_ratio, 0.25)
    focal = 0.5 * width / math.tan(math.radians(horizontal_fov_deg) / 2.0)

    means = torch.from_numpy(np.array(xyz_np, copy=True)).to(device)
    scales = torch.from_numpy(np.array(scales_np, copy=True)).to(device)
    colors = torch.from_numpy(np.array(colors_np, copy=True)).to(device).float()
    opacities = torch.from_numpy(np.array(opacities_np, copy=True)).to(device).float()
    quats = torch.from_numpy(np.array(quats_np, copy=True)).to(device).float()
    quats = quats / quats.norm(dim=1, keepdim=True).clamp_min(1e-8)
    intrinsics = torch.tensor(
        [[[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]],
        device=device,
        dtype=torch.float32,
    )
    background = torch.tensor([1.0, 1.0, 1.0], device=device)
    gsplat_version = tuple(int(part) for part in gsplat.__version__.split(".")[:2])
    raster_background = background[None] if gsplat_version >= (1, 6) else background
    up = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if z_up
        else np.array([0.0, -1.0, 0.0], dtype=np.float64)
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records = []

    if trajectory_path is not None:
        trajectory = json.loads(Path(trajectory_path).read_text())
        camera_specs = trajectory["frames"]
    else:
        trajectory = None
        camera_specs = []
        for index in range(view_count):
            angle = 2.0 * math.pi * index / view_count
            eye = center.copy()
            eye[0] += radius * math.cos(angle)
            eye[horizontal_axes[1]] += radius * math.sin(angle)
            camera_specs.append(
                {
                    "frame_id": f"view_{index:03d}",
                    "camera_position": eye.tolist(),
                    "look_at": center.tolist(),
                    "up": up.tolist(),
                }
            )

    for index, camera_spec in enumerate(camera_specs):
        eye = np.asarray(camera_spec["camera_position"], dtype=np.float64)
        target = np.asarray(camera_spec["look_at"], dtype=np.float64)
        camera_world_up = np.asarray(camera_spec.get("up", up), dtype=np.float64)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, camera_world_up)
        right /= np.linalg.norm(right)
        camera_up = np.cross(right, forward)
        rotation = np.stack([right, -camera_up, forward], axis=0)
        view = np.eye(4, dtype=np.float32)
        view[:3, :3] = rotation
        view[:3, 3] = -rotation @ eye
        view_tensor = torch.from_numpy(view).to(device)[None]

        started = time.perf_counter()
        with torch.inference_mode():
            rendered, alpha, metadata = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                view_tensor,
                intrinsics,
                width,
                height,
                near_plane=0.05,
                far_plane=1000.0,
                packed=True,
                backgrounds=raster_background,
                render_mode="RGB",
                rasterize_mode="classic",
            )
            if top_contributors:
                packed_ids, top_weights = top_contributor_fn(
                    metadata["means2d"],
                    metadata["conics"],
                    metadata["opacities"],
                    metadata["isect_offsets"],
                    metadata["flatten_ids"],
                    width,
                    height,
                    metadata["tile_size"],
                    top_contributors,
                )
                projection_ids = metadata["gaussian_ids"]
                if projection_ids is None:
                    source_ids = packed_ids
                else:
                    valid = packed_ids >= 0
                    source_ids = torch.full_like(packed_ids, -1)
                    source_ids[valid] = projection_ids[packed_ids[valid].long()].to(
                        source_ids.dtype
                    )
            torch.cuda.synchronize() if device.startswith("cuda") else None
        elapsed = time.perf_counter() - started
        rgb = (rendered[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        alpha_image = (alpha[0, ..., 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        stem = str(camera_spec.get("frame_id", f"view_{index:03d}"))
        Image.fromarray(rgb).save(destination / f"{stem}.png")
        Image.fromarray(alpha_image).save(destination / f"{stem}_alpha.png")
        contributor_id_uri = None
        contributor_weight_uri = None
        if top_contributors:
            invalid_id = np.iinfo(np.uint32).max
            source_ids_np = source_ids[0].cpu().numpy().astype(np.int64)
            source_ids_np = np.where(source_ids_np < 0, invalid_id, source_ids_np).astype(
                np.uint32
            )
            top_weights_np = top_weights[0].cpu().numpy().astype(np.float16)
            contributor_id_uri = f"{stem}_top_ids.npy"
            contributor_weight_uri = f"{stem}_top_weights.npy"
            np.save(destination / contributor_id_uri, source_ids_np)
            np.save(destination / contributor_weight_uri, top_weights_np)
        records.append(
            {
                "view_id": stem,
                "rgb_uri": f"{stem}.png",
                "alpha_uri": f"{stem}_alpha.png",
                "top_contributor_ids_uri": contributor_id_uri,
                "top_contributor_weights_uri": contributor_weight_uri,
                "camera_position": eye.tolist(),
                "look_at": target.tolist(),
                "timestamp": camera_spec.get("timestamp"),
                "elapsed_s": elapsed,
                "rendered_gaussians": int(metadata["gaussian_ids"].numel())
                if "gaussian_ids" in metadata
                else None,
            }
        )
    result = {
        "renderer": f"gsplat-{gsplat.__version__}",
        "asset_uri": str(asset.resolve()),
        "gaussian_count": int(len(xyz_np)),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "image_size": [width, height],
        "horizontal_fov_deg": horizontal_fov_deg,
        "top_contributors": top_contributors,
        "trajectory_uri": str(Path(trajectory_path).resolve())
        if trajectory_path is not None
        else None,
        "world_role": trajectory.get("world_role") if trajectory else None,
        "purpose": trajectory.get("purpose") if trajectory else "virtual_search_views",
        "views": records,
    }
    (destination / "virtual_views.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
