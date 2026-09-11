from __future__ import annotations

from pathlib import Path
import json
import time


def generate_automatic_masks(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    checkpoint: str | Path,
    model_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
    points_per_side: int = 24,
    min_area_fraction: float = 0.001,
    max_area_fraction: float = 0.8,
    max_masks: int = 128,
    device: str = "cuda",
) -> dict[str, object]:
    """Run SAM 2 automatic mask generation and save a portable mask manifest."""
    try:
        import numpy as np
        import torch
        from PIL import Image
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise RuntimeError("SAM 2, PyTorch, NumPy and Pillow are required") from exc

    source = Path(image_path)
    destination = Path(output_dir)
    masks_dir = destination / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    image = np.array(Image.open(source).convert("RGB"), copy=True)
    image_area = image.shape[0] * image.shape[1]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    started = time.perf_counter()
    model = build_sam2(model_config, str(checkpoint), device=device)
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=points_per_side,
        points_per_batch=64,
        pred_iou_thresh=0.78,
        stability_score_thresh=0.90,
        crop_n_layers=0,
        min_mask_region_area=0,
        output_mode="binary_mask",
    )
    with torch.inference_mode():
        if device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                raw_masks = generator.generate(image)
        else:
            raw_masks = generator.generate(image)
    filtered = [
        mask
        for mask in raw_masks
        if min_area_fraction <= mask["area"] / image_area <= max_area_fraction
    ]
    filtered.sort(
        key=lambda mask: (float(mask["predicted_iou"]), float(mask["stability_score"])),
        reverse=True,
    )
    filtered = filtered[:max_masks]
    records = []
    for index, mask in enumerate(filtered):
        mask_id = f"mask_{index:04d}"
        mask_path = masks_dir / f"{mask_id}.png"
        Image.fromarray(mask["segmentation"].astype(np.uint8) * 255).save(mask_path)
        records.append(
            {
                "mask_id": mask_id,
                "mask_uri": str(mask_path.relative_to(destination)),
                "area_px": int(mask["area"]),
                "area_fraction": float(mask["area"] / image_area),
                "bbox_xywh": [float(value) for value in mask["bbox"]],
                "predicted_iou": float(mask["predicted_iou"]),
                "stability_score": float(mask["stability_score"]),
            }
        )
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": "0.1",
        "generator": "sam2.1_hiera_tiny",
        "image_uri": str(source.resolve()),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "points_per_side": points_per_side,
        "raw_mask_count": len(raw_masks),
        "retained_mask_count": len(records),
        "elapsed_s": elapsed,
        "masks": records,
    }
    (destination / "masks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
