#!/usr/bin/env bash
# Reproduce the source-only single-room closed loop from existing A 3DGS,
# manual patrol, and an exported Marble B PLY.  It never reads native
# occupancy.png, occupancy.json, labels.json, structure.json, a navmesh, or a collider.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
scene_id="${1:-0007_840137}"
run_dir="artifacts/interiorgs_${scene_id}"
a_asset="data/interiorgs/${scene_id}/3dgs_compressed.ply"
patrol="${run_dir}/patrol/patrol.json"
b_asset="${run_dir}/world_b/marble_1_1/world_b.ply"
m_dir="${run_dir}/m_closed_loop"
b_dir="${run_dir}/world_b/semantic_closed_loop"
vocab="configs/nightclub_vocab.zh_en.json"
checkpoint="models/sam2/sam2.1_hiera_tiny.pt"

for file in "${a_asset}" "${patrol}" "${b_asset}" "${vocab}" "${checkpoint}"; do
  [[ -f "${file}" ]] || { echo "missing required input: ${file}" >&2; exit 2; }
done

# M: source-only Gaussian surface density + actually traversed patrol.
PYTHONPATH=src .envs/semantic/bin/python scripts/build_gaussian_kde_occupancy.py \
  --asset "${a_asset}" --patrol "${patrol}" --output-dir "${m_dir}"
PYTHONPATH=src .envs/semantic/bin/python scripts/build_patrol_topology_from_poses.py \
  --patrol "${patrol}" --occupancy "${m_dir}/observed_occupancy.png" \
  --metadata "${m_dir}/occupancy_metadata.json" --output "${m_dir}/patrol_topology.json" \
  --turn-epsilon-m .8 --max-edge-m 2.5

# B: virtual views, fixed-vocabulary Grounding DINO + SAM2, then 3D entity fusion.
scripts/run_gpu.sh -m navmem3d.cli render-gsplat-views \
  --asset "${b_asset}" --output "${b_dir}/virtual_views" --views 8 --width 640 --height 480 --top-contributors 4
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" scripts/run_gpu.sh -m navmem3d.cli generate-guided-masks \
  --image "${b_dir}/virtual_views/view_000.png" --image "${b_dir}/virtual_views/view_001.png" \
  --image "${b_dir}/virtual_views/view_002.png" --image "${b_dir}/virtual_views/view_003.png" \
  --image "${b_dir}/virtual_views/view_004.png" --image "${b_dir}/virtual_views/view_005.png" \
  --image "${b_dir}/virtual_views/view_006.png" --image "${b_dir}/virtual_views/view_007.png" \
  --output "${b_dir}/guided_masks" --vocabulary "${vocab}" --sam-checkpoint "${checkpoint}" \
  --box-threshold .20 --text-threshold .15

proposal_args=()
for i in 000 001 002 003 004 005 006 007; do
  PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli lift-mask-set \
    --masks "${b_dir}/guided_masks/view_${i}/masks.json" \
    --gaussian-ids "${b_dir}/virtual_views/view_${i}_top_ids.npy" \
    --gaussian-weights "${b_dir}/virtual_views/view_${i}_top_weights.npy" \
    --output "${b_dir}/proposals/view_${i}" --min-gaussians 32 --min-weight .02 --min-pixel-hits 2
  proposal_args+=(--proposal-index "${b_dir}/proposals/view_${i}/proposal_index.json")
done
PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli fuse-proposals "${proposal_args[@]}" --output "${b_dir}/fused"
PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli derive-geometry \
  --entities "${b_dir}/fused/fused_entities.json" --asset "${b_asset}" --output "${b_dir}/geometry.json" --near-threshold-ratio .03
PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli filter-entities \
  --geometry-index "${b_dir}/geometry.json" --output "${b_dir}/searchable.json"
PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli deduplicate-entities \
  --entity-index "${b_dir}/searchable.json" --output "${b_dir}/entities.json"
PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli export-entity-crops \
  --entity-index "${b_dir}/entities.json" --output "${b_dir}/crops"

# B→M: visual candidate binding, deliberately without a global B→A metric transform.
scripts/run_gpu.sh scripts/bind_entities_via_crop_retrieval.py \
  --entities "${b_dir}/entities.json" --crops "${b_dir}/crops/representative_crops.json" \
  --patrol "${patrol}" --patrol-renders "${run_dir}/patrol/renders" \
  --topology "${m_dir}/patrol_topology.json" --output "${b_dir}/entities_to_m.json" --stride 2 --top-k 3

# Query→route demonstration.  The executor must still confirm the target in current RGB.
PYTHONPATH=src .envs/semantic/bin/python scripts/plan_topology_route.py \
  --topology "${m_dir}/patrol_topology.json" --entities "${b_dir}/entities_to_m.json" \
  --term sofa --start-node topo_000 --output "${m_dir}/route_to_sofa.json"

echo "closed loop complete: ${m_dir} + ${b_dir}"
