from __future__ import annotations

from pathlib import Path
import json


def _umeyama_similarity(source, target):
    """Estimate a proper Sim(3), mapping source Nx3 positions to target Nx3."""
    import numpy as np

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both be Nx3")
    if len(source) < 3:
        raise ValueError("at least three camera correspondences are required")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered, target_centered = source - source_mean, target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    variance = float((source_centered * source_centered).sum() / len(source))
    scale = float((singular * np.diag(correction)).sum() / max(variance, 1e-12))
    translation = target_mean - scale * rotation @ source_mean
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    residuals = np.linalg.norm((scale * (rotation @ source.T)).T + translation - target, axis=1)
    return matrix, residuals


def estimate_visual_camera_registration(
    virtual_views_path: str | Path,
    patrol_path: str | Path,
    patrol_render_dir: str | Path,
    output_path: str | Path,
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cuda",
    stride: int = 5,
    minimum_similarity: float = 0.60,
) -> dict[str, object]:
    """Match B virtual views to A patrol RGB, then fit a provisional camera Sim(3).

    The output is a visual initializer, not a navigation-quality registration.
    It intentionally uses only A patrol RGB and poses, never A labels or geometry.
    """
    try:
        import numpy as np
        import open_clip
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("visual registration requires OpenCLIP, PyTorch, NumPy and Pillow") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if stride <= 0:
        raise ValueError("stride must be positive")

    view_document = json.loads(Path(virtual_views_path).read_text(encoding="utf-8"))
    patrol_document = json.loads(Path(patrol_path).read_text(encoding="utf-8"))
    virtual_root = Path(virtual_views_path).parent
    patrol_root = Path(patrol_render_dir)
    b_views = list(view_document.get("views", []))
    a_frames = list(patrol_document.get("frames", []))[::stride]
    image_pairs = []
    for frame in a_frames:
        image_path = patrol_root / f"{frame['frame_id']}.png"
        if image_path.exists():
            image_pairs.append((frame, image_path))
    if len(b_views) < 3 or len(image_pairs) < 3:
        raise ValueError("need at least three B views and three rendered A patrol frames")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()

    def encode(paths):
        chunks = []
        with torch.inference_mode():
            for offset in range(0, len(paths), 32):
                batch = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in paths[offset:offset + 32]]).to(device)
                feature = model.encode_image(batch)
                chunks.append((feature / feature.norm(dim=-1, keepdim=True)).cpu())
        return torch.cat(chunks).numpy()

    b_paths = [virtual_root / str(view["rgb_uri"]) for view in b_views]
    if not all(path.exists() for path in b_paths):
        raise FileNotFoundError("one or more virtual-view RGB files are missing")
    b_features = encode(b_paths)
    a_features = encode([path for _, path in image_pairs])
    similarity = b_features @ a_features.T

    # Greedy one-to-one assignment keeps a repeated restaurant scene from
    # assigning every B view to the same visually generic patrol frame.
    ranked = [
        (float(similarity[b_index, a_index]), b_index, a_index)
        for b_index in range(similarity.shape[0])
        for a_index in range(similarity.shape[1])
    ]
    ranked.sort(reverse=True)
    chosen_b, chosen_a, matches = set(), set(), []
    for score, b_index, a_index in ranked:
        if score < minimum_similarity or b_index in chosen_b or a_index in chosen_a:
            continue
        chosen_b.add(b_index)
        chosen_a.add(a_index)
        matches.append({
            "b_view_id": b_views[b_index].get("view_id", f"view_{b_index:03d}"),
            "b_index": b_index,
            "a_frame_id": image_pairs[a_index][0]["frame_id"],
            "a_index_in_sample": a_index,
            "similarity": score,
            "position_b": b_views[b_index]["camera_position"],
            "position_a": image_pairs[a_index][0]["camera_position"],
        })
        if len(matches) == min(len(b_views), len(image_pairs)):
            break
    if len(matches) < 3:
        raise RuntimeError(
            f"only {len(matches)} visual camera matches passed similarity {minimum_similarity}; cannot fit Sim(3)"
        )
    source = np.asarray([item["position_b"] for item in matches], dtype=float)
    target = np.asarray([item["position_a"] for item in matches], dtype=float)
    matrix, residuals = _umeyama_similarity(source, target)
    for item, residual in zip(matches, residuals):
        item["camera_position_residual_m"] = float(residual)
    result = {
        "schema_version": "0.1",
        "method": "clip_global_image_retrieval_plus_camera_sim3",
        "status": "visual_initializer_requires_landmark_or_execution_validation",
        "virtual_views": str(Path(virtual_views_path).resolve()),
        "patrol": str(Path(patrol_path).resolve()),
        "patrol_render_dir": str(patrol_root.resolve()),
        "model": {"name": model_name, "pretrained": pretrained, "stride": stride},
        "minimum_similarity": minimum_similarity,
        "match_count": len(matches),
        "mean_similarity": float(np.mean([item["similarity"] for item in matches])),
        "median_camera_residual_m": float(np.median(residuals)),
        "max_camera_residual_m": float(np.max(residuals)),
        "transform_b_to_a": matrix.reshape(-1).tolist(),
        "matches": matches,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
