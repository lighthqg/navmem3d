#!/usr/bin/env bash
# Rebuild the World-B semantic index from an already exported Marble SPZ/PLY.
# This script never reads World-A labels.json or structure.json.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
run_dir="artifacts/interiorgs_0001_839920/world_b"
asset="${run_dir}/marble_1_1/world_b_lowres.ply"
views="${run_dir}/virtual_views_gsplat"
semantic="${run_dir}/semantic_guided"
vocab="configs/restaurant_vocab.zh_en.json"
checkpoint="models/sam2/sam2.1_hiera_tiny.pt"

scripts/run_gpu.sh -m navmem3d.cli render-gsplat-views \
  --asset "${asset}" --output "${views}" --views 8 --width 640 --height 480 --top-contributors 4

view_args=()
for i in 000 001 002 003 004 005 006 007; do
  view_args+=(--image "${views}/view_${i}.png")
done
scripts/run_gpu.sh -m navmem3d.cli generate-guided-masks "${view_args[@]}" \
  --output "${semantic}" --vocabulary "${vocab}" --sam-checkpoint "${checkpoint}" \
  --box-threshold 0.20 --text-threshold 0.15

proposal_args=()
for i in 000 001 002 003 004 005 006 007; do
  PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli lift-mask-set \
    --masks "${semantic}/view_${i}/masks.json" \
    --gaussian-ids "${views}/view_${i}_top_ids.npy" \
    --gaussian-weights "${views}/view_${i}_top_weights.npy" \
    --output "${semantic}/view_${i}_proposals" \
    --min-gaussians 32 --min-weight 0.02 --min-pixel-hits 2
  proposal_args+=(--proposal-index "${semantic}/view_${i}_proposals/proposal_index.json")
done
PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli fuse-proposals "${proposal_args[@]}" --output "${semantic}/entities"
PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli derive-geometry \
  --entities "${semantic}/entities/fused_entities.json" --asset "${asset}" \
  --output "${semantic}/geometry_index.json" --near-threshold-ratio 0.03
PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli filter-entities \
  --geometry-index "${semantic}/geometry_index.json" --output "${semantic}/searchable_entities.json"
PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli deduplicate-entities \
  --entity-index "${semantic}/searchable_entities.json" --output "${semantic}/entity_index.json"
PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli export-entity-crops \
  --entity-index "${semantic}/entity_index.json" --output "${semantic}/entity_crops"
PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli build-observation-goals \
  --entity-index "${semantic}/entity_index.json" --virtual-views "${views}/virtual_views.json" \
  --output "${semantic}/goal_index.json"
PYTHONPATH=src .envs/semantic/bin/python - <<'PY2'
import json
from pathlib import Path
root=Path("artifacts/interiorgs_0001_839920/world_b/semantic_guided")
for name in ("entity_index.json", "goal_index.json"):
    path=root/name; data=json.loads(path.read_text())
    data["world_id"]="marble_bbbc5728"
    data["world_role"]="B_internal_twin"
    data["evaluation_access"]="system"
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
PY2
