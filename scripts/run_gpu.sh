#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cuda_prefix="${project_dir}/.envs/cuda-12.8"
cuda_home="${cuda_prefix}/targets/x86_64-linux"

export CUDA_HOME="${cuda_home}"
export PATH="${project_dir}/.envs/semantic/bin:${cuda_prefix}/bin:${cuda_prefix}/nvvm/bin:${PATH}"
export PYTHONPATH="${project_dir}/third_party/gsplat:${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-2}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/navmem3d-cache}"
# NavMem3D only needs the classic 3DGS rasterizer and contributor queries.
# Disabling unrelated training/rendering families greatly reduces JIT time.
export BUILD_3DGS="${BUILD_3DGS:-1}"
export BUILD_2DGS="${BUILD_2DGS:-0}"
export BUILD_3DGUT="${BUILD_3DGUT:-0}"
export BUILD_ADAM="${BUILD_ADAM:-0}"
export BUILD_RELOC="${BUILD_RELOC:-0}"
export BUILD_LOSSES="${BUILD_LOSSES:-0}"
export BUILD_CAMERA_WRAPPERS="${BUILD_CAMERA_WRAPPERS:-0}"

exec "${project_dir}/.envs/semantic/bin/python" "$@"
