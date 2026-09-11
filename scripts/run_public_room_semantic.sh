#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
python_bin="${project_dir}/.envs/semantic/bin/python"
asset="data/worlds/public_room/room-7k.splat"
views="data/worlds/public_room/gsplat_topk_views"

scripts/run_gpu.sh -m navmem3d.cli render-gsplat-views \
  --asset "${asset}" --output "${views}" --views 4 \
  --width 640 --height 480 --top-contributors 4

scripts/run_gpu.sh -m navmem3d.cli generate-guided-masks \
  --image "${views}/view_000.png" --image "${views}/view_001.png" \
  --image "${views}/view_002.png" --image "${views}/view_003.png" \
  --output data/worlds/public_room/guided_masks \
  --vocabulary configs/public_room_vocab.json \
  --sam-checkpoint models/sam2/sam2.1_hiera_tiny.pt

for view_id in 000 001 002 003; do
  PYTHONPATH=src "${python_bin}" -m navmem3d.cli lift-mask-set \
    --masks "data/worlds/public_room/guided_masks/view_${view_id}/masks.json" \
    --gaussian-ids "${views}/view_${view_id}_top_ids.npy" \
    --gaussian-weights "${views}/view_${view_id}_top_weights.npy" \
    --output "data/worlds/public_room/guided_proposals_view_${view_id}" \
    --min-gaussians 32 --min-weight 0.02 --min-pixel-hits 2
done

PYTHONPATH=src "${python_bin}" -m navmem3d.cli fuse-proposals \
  --proposal-index data/worlds/public_room/guided_proposals_view_000/proposal_index.json \
  --proposal-index data/worlds/public_room/guided_proposals_view_001/proposal_index.json \
  --proposal-index data/worlds/public_room/guided_proposals_view_002/proposal_index.json \
  --proposal-index data/worlds/public_room/guided_proposals_view_003/proposal_index.json \
  --output data/worlds/public_room/guided_fused_entities

PYTHONPATH=src "${python_bin}" -m navmem3d.cli derive-geometry \
  --entities data/worlds/public_room/guided_fused_entities/fused_entities.json \
  --asset "${asset}" \
  --output data/worlds/public_room/guided_fused_entities/geometry_index.json \
  --near-threshold-ratio 0.03

PYTHONPATH=src "${python_bin}" -m navmem3d.cli filter-entities \
  --geometry-index data/worlds/public_room/guided_fused_entities/geometry_index.json \
  --output data/worlds/public_room/guided_fused_entities/searchable_entities.json

PYTHONPATH=src "${python_bin}" -m navmem3d.cli deduplicate-entities \
  --entity-index data/worlds/public_room/guided_fused_entities/searchable_entities.json \
  --output data/worlds/public_room/guided_fused_entities/semantic_entity_index.json

PYTHONPATH=src "${python_bin}" -m navmem3d.cli export-entity-crops \
  --entity-index data/worlds/public_room/guided_fused_entities/semantic_entity_index.json \
  --output data/worlds/public_room/guided_entity_crops_final

PYTHONPATH=src python3 -m navmem3d.cli build-observation-goals \
  --entity-index data/worlds/public_room/guided_fused_entities/semantic_entity_index.json \
  --virtual-views data/worlds/public_room/gsplat_topk_views/virtual_views.json \
  --output data/worlds/public_room/guided_fused_entities/observation_goals.json

PYTHONPATH=src python3 -m navmem3d.cli query-index \
  --index data/worlds/public_room/guided_fused_entities/semantic_entity_index.json \
  --term 沙发 --predicate near --reference-term 桌子 --compact
