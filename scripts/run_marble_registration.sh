#!/usr/bin/env bash
# Produce explicit B→A initializers and safety-aware A-grid approach candidates.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"
run_dir="artifacts/interiorgs_0001_839920"
semantic="${run_dir}/world_b/semantic_guided"
asset="${run_dir}/world_b/marble_1_1/world_b_lowres.ply"
scene="data/interiorgs/0001_839920"

PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli estimate-b-to-a \
  --b-splat "${asset}" --a-scene "${scene}" --output "${semantic}/b_to_a_coarse.json"
PYTHONPATH=src .envs/semantic/bin/python -c \
  "import json; print(' '.join(map(str,json.load(open('${semantic}/b_to_a_coarse.json'))['transform_b_to_a'])))" \
  > /tmp/navmem3d_b_to_a_transform.txt
PYTHONPATH=src .envs/semantic/bin/python -m navmem3d.cli register-goals-b-to-a \
  --goal-index "${semantic}/goal_index.json" --transform $(cat /tmp/navmem3d_b_to_a_transform.txt) \
  --registration-status coarse_initializer_requires_visual_or_landmark_refinement \
  --output "${semantic}/goal_index_b_to_a_coarse.json"
PYTHONPATH=src .envs/semantic/bin/python scripts/validate_navigation_targets.py \
  --goals "${semantic}/goal_index_b_to_a_coarse.json" --scene "${scene}" \
  --output "${semantic}/navigation_validation.json" --max-projection-error-m 1.0
