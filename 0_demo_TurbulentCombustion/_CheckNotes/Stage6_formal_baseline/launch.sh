#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

frozen_ref="stage1-5-optimized-reference-v1"
frozen_sha="169d7c545b9f980aed0fbaff0252e6d4114f3566"
package="_CheckNotes/Stage6_formal_baseline"
run_name="${1:-}"
gpu_id="${GPU_ID:-1}"

case "$run_name" in
  F0)
    config="$package/F0_frozen_current.yaml"
    ;;
  F1)
    config="$package/F1_more_supervision.yaml"
    ;;
  *)
    echo "Usage: GPU_ID=<physical-gpu> bash $package/launch.sh F0|F1" >&2
    exit 2
    ;;
esac

actual_frozen_sha="$(git rev-list -n 1 "$frozen_ref")"
if [[ "$actual_frozen_sha" != "$frozen_sha" ]]; then
  echo "Frozen reference mismatch: expected $frozen_sha, got $actual_frozen_sha" >&2
  exit 3
fi

if ! git diff --quiet "$frozen_ref" -- src Save_config/config_pointcloud_ffm.yaml; then
  echo "Refusing to launch: model/training source differs from $frozen_ref." >&2
  echo "Run the formal baselines before editing Stage 6 architecture code." >&2
  exit 4
fi

conda run --no-capture-output -n phycoflow_env \
  python "$package/validate_configs.py"

mkdir -p "$package/logs" "$package/runs" "$package/gpu_state"
timestamp="$(date +%Y%m%d_%H%M%S)"
nvidia-smi -i "$gpu_id" \
  --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "$package/gpu_state/${run_name}_${timestamp}_initial.csv"

export CUDA_VISIBLE_DEVICES="$gpu_id"
export KEOPS_CACHE_FOLDER="/tmp/keops_stage6_formal_gpu${gpu_id}"

conda run --no-capture-output -n phycoflow_env \
  python src/train_pointcloud_ffm.py --config "$config" \
  2>&1 | tee "$package/logs/${run_name}_${timestamp}.log"

