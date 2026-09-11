from __future__ import annotations

from pathlib import Path
import json


def label_entity_crops(
    entity_index_path: str | Path,
    crop_manifest_path: str | Path,
    vocabulary_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cuda",
    batch_size: int = 32,
    top_k: int = 5,
) -> dict[str, object]:
    """Assign fixed-vocabulary labels to representative entity crops with CLIP."""
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("CLIP labeling requires open_clip_torch, PyTorch and Pillow") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    entity_source = Path(entity_index_path)
    crop_source = Path(crop_manifest_path)
    vocab_source = Path(vocabulary_path)
    entity_document = json.loads(entity_source.read_text(encoding="utf-8"))
    crop_document = json.loads(crop_source.read_text(encoding="utf-8"))
    vocabulary = json.loads(vocab_source.read_text(encoding="utf-8"))
    categories = vocabulary["categories"]
    templates = vocabulary.get("prompt_templates", ["a photo of a {}"])
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    category_prompts = []
    prompt_category_indices = []
    for category_index, category in enumerate(categories):
        english_aliases = [alias for alias in category.get("aliases", []) if all(ord(char) < 128 for char in alias)]
        names = english_aliases or [category.get("canonical", category.get("id", "object"))]
        for alias in names:
            for template in templates:
                category_prompts.append(template.format(alias))
                prompt_category_indices.append(category_index)
    with torch.inference_mode():
        text_features = model.encode_text(tokenizer(category_prompts).to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        category_features = []
        for category_index in range(len(categories)):
            indices = [
                index
                for index, value in enumerate(prompt_category_indices)
                if value == category_index
            ]
            feature = text_features[indices].mean(dim=0)
            category_features.append(feature / feature.norm())
        category_features = torch.stack(category_features)

    crop_by_id = {record["entity_id"]: record for record in crop_document["entities"]}
    entities = entity_document.get("entities", [])
    for offset in range(0, len(entities), batch_size):
        batch = entities[offset : offset + batch_size]
        images = []
        for entity in batch:
            record = crop_by_id[entity["entity_id"]]
            images.append(preprocess(Image.open(crop_source.parent / record["crop_uri"])))
        image_tensor = torch.stack(images).to(device)
        with torch.inference_mode():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ category_features.T
            values, indices = similarities.topk(min(top_k, len(categories)), dim=1)
        for entity, scores, category_ids in zip(batch, values.cpu(), indices.cpu()):
            ranking = [
                {
                    "category_id": categories[int(category_id)].get("canonical", categories[int(category_id)].get("id")),
                    "score": float(score),
                }
                for score, category_id in zip(scores, category_ids)
            ]
            best = categories[int(category_ids[0])]
            canonical = best.get("canonical", best.get("id"))
            entity["labels"] = [canonical, *best.get("aliases", [])]
            entity.setdefault("attributes", [])
            entity["semantic_prediction"] = {
                "category_id": best.get("canonical", best.get("id")),
                "score": float(scores[0]),
                "margin_to_second": float(scores[0] - scores[1])
                if len(scores) > 1
                else None,
                "ranking": ranking,
                "requires_review": bool(len(scores) > 1 and scores[0] - scores[1] < 0.015),
            }
    result = {
        **entity_document,
        "semantic_compiler": {
            "method": "fixed_vocabulary_clip",
            "model_name": model_name,
            "pretrained": pretrained,
            "vocabulary_uri": str(vocab_source.resolve()),
            "crop_manifest_uri": str(crop_source.resolve()),
        },
        "entities": entities,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
