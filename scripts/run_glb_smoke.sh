#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export XDG_CACHE_HOME="/tmp/navmem3d-cache"
export MAMBA_ROOT_PREFIX="/tmp/navmem3d-mamba-root"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH="${project_dir}/src"

"${project_dir}/tools/micromamba/micromamba" run -p "${project_dir}/.envs/habitat" \
  python -m navmem3d.cli capture-glb \
  --scene "${project_dir}/data/habitat_test_scenes/van-gogh-room.glb" \
  --output "${project_dir}/data/captures/van_gogh_smoke" \
  --sequence-id van_gogh_smoke \
  --frames 12 \
  --width 320 \
  --height 240

python3 -m navmem3d.cli inspect-sequence \
  "${project_dir}/data/captures/van_gogh_smoke/capture_sequence.json"

python3 -m navmem3d.cli select-frames \
  "${project_dir}/data/captures/van_gogh_smoke/capture_sequence.json" \
  --count 8 \
  --strategy pose-coverage \
  --copy-rgb-to "${project_dir}/data/captures/van_gogh_smoke/marble_8_images" \
  --output "${project_dir}/data/captures/van_gogh_smoke/selected_8.json"

python3 -m navmem3d.cli make-video \
  --sequence "${project_dir}/data/captures/van_gogh_smoke/capture_sequence.json" \
  --output "${project_dir}/data/captures/van_gogh_smoke/patrol.mp4" \
  --fps 10 \
  --max-duration 30 \
  --max-size-mb 100
