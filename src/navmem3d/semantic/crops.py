from __future__ import annotations

from pathlib import Path
import json


def export_representative_crops(
    entity_index_path: str | Path,
    output_dir: str | Path,
    *,
    padding_ratio: float = 0.15,
    gallery_columns: int = 6,
) -> dict[str, object]:
    """Export one mask-aware source crop per searchable 3D entity."""
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("representative crop export requires NumPy and Pillow") from exc
    source = Path(entity_index_path)
    document = json.loads(source.read_text(encoding="utf-8"))
    destination = Path(output_dir)
    crops_dir = destination / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    proposal_cache: dict[Path, dict[str, object]] = {}
    mask_manifest_cache: dict[Path, dict[str, object]] = {}
    image_cache: dict[Path, object] = {}
    records = []

    for entity in document.get("entities", []):
        choices = []
        for observation in entity.get("observations", []):
            proposal_path = Path(observation["proposal_index"])
            proposal_document = proposal_cache.setdefault(
                proposal_path, json.loads(proposal_path.read_text(encoding="utf-8"))
            )
            proposal = next(
                item
                for item in proposal_document["proposals"]
                if item["proposal_id"] == observation["proposal_id"]
            )
            choices.append(
                (
                    float(proposal["predicted_iou"]) + float(proposal["stability_score"]),
                    proposal_path,
                    proposal_document,
                    proposal,
                    observation,
                )
            )
        if not choices:
            continue
        _, proposal_path, proposal_document, proposal, observation = max(
            choices, key=lambda item: item[0]
        )
        mask_manifest_path = Path(proposal_document["source_mask_manifest"])
        mask_manifest = mask_manifest_cache.setdefault(
            mask_manifest_path,
            json.loads(mask_manifest_path.read_text(encoding="utf-8")),
        )
        image_path = Path(mask_manifest["image_uri"])
        if image_path not in image_cache:
            image_cache[image_path] = np.array(Image.open(image_path).convert("RGB"), copy=True)
        image = image_cache[image_path]
        mask_path = Path(proposal["source_mask_uri"])
        mask = np.array(Image.open(mask_path).convert("L"), copy=False) > 0
        x, y, width, height = proposal["bbox_xywh"]
        padding = padding_ratio * max(width, height)
        x0 = max(0, int(x - padding))
        y0 = max(0, int(y - padding))
        x1 = min(image.shape[1], int(x + width + padding + 1))
        y1 = min(image.shape[0], int(y + height + padding + 1))
        crop = image[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1]
        crop[~crop_mask] = (crop[~crop_mask].astype(np.float32) * 0.30 + 178.0).clip(
            0, 255
        ).astype(np.uint8)
        crop_path = crops_dir / f"{entity['entity_id']}.png"
        Image.fromarray(crop).save(crop_path)
        records.append(
            {
                "entity_id": entity["entity_id"],
                "crop_uri": str(crop_path.relative_to(destination)),
                "source_image_uri": str(image_path),
                "source_mask_uri": str(mask_path),
                "source_view_index": observation["view_index"],
                "source_proposal_id": proposal["proposal_id"],
                "predicted_iou": proposal["predicted_iou"],
                "stability_score": proposal["stability_score"],
                "category_id": entity.get("category_id"),
            }
        )

    thumb_size = (180, 140)
    rows = max(1, (len(records) + gallery_columns - 1) // gallery_columns)
    gallery = Image.new("RGB", (gallery_columns * 180, rows * 165), "white")
    draw = ImageDraw.Draw(gallery)
    for index, record in enumerate(records):
        tile = Image.open(destination / record["crop_uri"]).convert("RGB")
        tile.thumbnail(thumb_size)
        column, row = index % gallery_columns, index // gallery_columns
        left = column * 180 + (180 - tile.width) // 2
        top = row * 165 + (140 - tile.height) // 2
        gallery.paste(tile, (left, top))
        caption = record["entity_id"]
        if record.get("category_id"):
            caption += f" · {record['category_id']}"
        draw.text((column * 180 + 4, row * 165 + 144), caption, fill="black")
    gallery.save(destination / "gallery.jpg", quality=90)
    result = {
        "schema_version": "0.1",
        "source_entity_index": str(source.resolve()),
        "crop_count": len(records),
        "gallery_uri": "gallery.jpg",
        "entities": records,
    }
    (destination / "representative_crops.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
