from __future__ import annotations

from pathlib import Path
import json
import time


def _box_iou(left, right) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def generate_guided_masks(
    image_paths: list[str | Path],
    output_dir: str | Path,
    *,
    vocabulary_path: str | Path,
    sam_checkpoint: str | Path,
    detector_model: str = "IDEA-Research/grounding-dino-tiny",
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    box_nms_iou: float = 0.60,
    device: str = "cuda",
) -> dict[str, object]:
    """Detect fixed-vocabulary objects, then refine every box with SAM 2."""
    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise RuntimeError("guided masks require Transformers, SAM 2, PyTorch and Pillow") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    vocabulary = json.loads(Path(vocabulary_path).read_text(encoding="utf-8"))
    structural = {"wall", "floor", "ceiling", "unknown_object"}
    def category_id(item: dict[str, object]) -> str:
        return str(item.get("id", item.get("canonical", "object")))
    categories = [item for item in vocabulary["categories"] if category_id(item) not in structural]
    detector_labels = []
    category_by_label = {}
    for category in categories:
        label = next(
            (alias for alias in category["aliases"] if all(ord(char) < 128 for char in alias)),
            category_id(category),
        )
        detector_labels.append(label)
        category_by_label[label] = category
    processor = AutoProcessor.from_pretrained(detector_model)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_model).to(device)
    detector.eval()
    sam_model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml", str(sam_checkpoint), device=device
    )
    predictor = SAM2ImagePredictor(sam_model)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    view_records = []

    for image_index, image_path in enumerate(image_paths):
        started = time.perf_counter()
        source = Path(image_path)
        pil_image = Image.open(source).convert("RGB")
        image = np.array(pil_image, copy=True)
        inputs = processor(images=pil_image, text=[detector_labels], return_tensors="pt").to(
            device
        )
        with torch.inference_mode():
            outputs = detector(**inputs)
        detected = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]
        candidates = []
        for score, phrase, box in zip(
            detected["scores"].detach().cpu().tolist(),
            detected["text_labels"],
            detected["boxes"].detach().cpu().tolist(),
        ):
            phrase_lower = phrase.lower().strip()
            label = next(
                (value for value in detector_labels if value in phrase_lower), None
            )
            if label is None:
                continue
            candidates.append(
                {"category": category_by_label[label], "score": float(score), "box": box}
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        kept = []
        for candidate in candidates:
            if any(
                category_id(item["category"]) == category_id(candidate["category"])
                and _box_iou(item["box"], candidate["box"]) > box_nms_iou
                for item in kept
            ):
                continue
            kept.append(candidate)

        view_dir = destination / f"view_{image_index:03d}"
        masks_dir = view_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        predictor.set_image(image)
        records = []
        for mask_index, candidate in enumerate(kept):
            with torch.inference_mode():
                if device.startswith("cuda"):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        masks, sam_scores, _ = predictor.predict(
                            box=np.asarray(candidate["box"], dtype=np.float32),
                            multimask_output=False,
                        )
                else:
                    masks, sam_scores, _ = predictor.predict(
                        box=np.asarray(candidate["box"], dtype=np.float32),
                        multimask_output=False,
                    )
            mask = masks[0].astype(bool)
            area = int(mask.sum())
            if area < 32 or area > image.shape[0] * image.shape[1] * 0.8:
                continue
            ys, xs = np.nonzero(mask)
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            mask_id = f"mask_{mask_index:04d}"
            mask_path = masks_dir / f"{mask_id}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            records.append(
                {
                    "mask_id": mask_id,
                    "mask_uri": str(mask_path.relative_to(view_dir)),
                    "category_id": category_id(candidate["category"]),
                    "labels": candidate["category"]["aliases"],
                    "detector_score": candidate["score"],
                    "predicted_iou": float(sam_scores[0]),
                    "stability_score": candidate["score"],
                    "area_px": area,
                    "area_fraction": area / (image.shape[0] * image.shape[1]),
                    "bbox_xywh": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                    "detector_box_xyxy": candidate["box"],
                }
            )
        elapsed = time.perf_counter() - started
        manifest = {
            "schema_version": "0.1",
            "generator": "grounding_dino_tiny+sam2.1_hiera_tiny",
            "image_uri": str(source.resolve()),
            "image_size": [image.shape[1], image.shape[0]],
            "device": device,
            "detector_model": detector_model,
            "raw_detection_count": len(candidates),
            "retained_mask_count": len(records),
            "elapsed_s": elapsed,
            "masks": records,
        }
        (view_dir / "masks.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        view_records.append(
            {
                "view_index": image_index,
                "image_uri": str(source.resolve()),
                "mask_manifest_uri": str((view_dir / "masks.json").resolve()),
                "mask_count": len(records),
                "elapsed_s": elapsed,
            }
        )
    result = {
        "schema_version": "0.1",
        "generator": "grounding_dino+sam2",
        "vocabulary_uri": str(Path(vocabulary_path).resolve()),
        "views": view_records,
        "total_mask_count": sum(item["mask_count"] for item in view_records),
    }
    (destination / "guided_masks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
