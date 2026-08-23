#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

package="_CheckNotes/Stage6_compact_query"
gpu_id="${GPU_ID:-2}"

GPU_ID="$gpu_id" bash "$package/benchmark.sh"
conda run --no-capture-output -n phycoflow_env \
  python "$package/summarize_cost.py"
GPU_ID="$gpu_id" bash "$package/launch.sh" full
GPU_ID="$gpu_id" bash "$package/launch.sh" lr
GPU_ID="$gpu_id" bash "$package/evaluate_screen.sh" full
GPU_ID="$gpu_id" bash "$package/evaluate_screen.sh" lr
conda run --no-capture-output -n phycoflow_env \
  python "$package/select_candidate.py"
conda run --no-capture-output -n fig \
  python figures/scripts/plot_stage6_compact_query.py
