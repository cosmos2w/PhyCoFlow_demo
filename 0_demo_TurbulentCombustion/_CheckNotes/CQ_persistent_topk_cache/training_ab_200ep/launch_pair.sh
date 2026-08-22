#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
git_root="$(git -C "$repo_root" rev-parse --show-toplevel)"
cd "$repo_root"

package="_CheckNotes/CQ_persistent_topk_cache/training_ab_200ep"
old_gpu="${OLD_GPU:-0}"
new_gpu="${NEW_GPU:-1}"
old_commit="01d284767af9cbbf6b2e185b2ea52c50545ca607"
new_commit="3f3eefbe5ddeb2d530318bf7686d03b61c051ff4"
old_tree="/tmp/phycoflow_cq_no_topk_01d2847"
new_tree="/tmp/phycoflow_cq_topk_3f3eefb"
project_rel="0_demo_TurbulentCombustion"

if [[ "$old_gpu" == "$new_gpu" ]]; then
  echo "OLD_GPU and NEW_GPU must be different physical GPUs." >&2
  exit 2
fi

conda run --no-capture-output -n phycoflow_env python "$package/validate_configs.py"

ensure_worktree() {
  local path="$1"
  local commit="$2"
  if [[ -e "$path/.git" ]]; then
    actual="$(git -C "$path" rev-parse HEAD)"
    if [[ "$actual" != "$commit" ]]; then
      echo "Existing pinned worktree $path is at $actual, expected $commit." >&2
      exit 3
    fi
  elif [[ -e "$path" ]]; then
    echo "Refusing non-worktree path: $path" >&2
    exit 3
  else
    git -C "$git_root" worktree add --detach "$path" "$commit"
  fi
}

ensure_worktree "$old_tree" "$old_commit"
ensure_worktree "$new_tree" "$new_commit"
old_project="$old_tree/$project_rel"
new_project="$new_tree/$project_rel"

shared_runs="$repo_root/$package/runs"
mkdir -p "$shared_runs"
for pinned_project in "$old_project" "$new_project"; do
  pinned_package="$pinned_project/$package"
  pinned_runs="$pinned_package/runs_shared"
  mkdir -p "$pinned_package"
  if [[ -L "$pinned_runs" ]]; then
    [[ "$(readlink -f "$pinned_runs")" == "$(readlink -f "$shared_runs")" ]] || {
      echo "Pinned run link targets the wrong directory: $pinned_runs" >&2
      exit 3
    }
  elif [[ -e "$pinned_runs" ]]; then
    echo "Refusing non-link pinned run path: $pinned_runs" >&2
    exit 3
  else
    ln -s "$shared_runs" "$pinned_runs"
  fi
done

for gpu in "$old_gpu" "$new_gpu"; do
  free_mb="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
  util_pct="$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
  if [[ "${FORCE_BUSY_GPU:-0}" != "1" && ( "$free_mb" -lt 40000 || "$util_pct" -gt 10 ) ]]; then
    echo "Refusing GPU $gpu: ${free_mb} MiB free, ${util_pct}% utilization." >&2
    exit 4
  fi
done

mkdir -p "$package/logs" "$package/gpu_state" "$package/runs"
timestamp="$(date +%Y%m%d_%H%M%S)"
old_log="$package/logs/CQ_LR_no_persistent_${timestamp}.log"
new_log="$package/logs/CQ_LR_persistent_topk_${timestamp}.log"

nvidia-smi -i "$old_gpu" --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader \
  > "$package/gpu_state/no_persistent_${timestamp}_initial.csv"
nvidia-smi -i "$new_gpu" --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader \
  > "$package/gpu_state/persistent_topk_${timestamp}_initial.csv"

printf 'stage=training\ntimestamp=%s\nold_commit=%s\nnew_commit=%s\n' \
  "$timestamp" "$old_commit" "$new_commit" > "$package/pipeline_status.env"

echo "Launching pre-cache CQ-LR $old_commit on physical GPU $old_gpu"
CUDA_VISIBLE_DEVICES="$old_gpu" KEOPS_CACHE_FOLDER="/tmp/keops_cq_topk_ab_train_old_gpu${old_gpu}" \
  conda run --no-capture-output -n phycoflow_env \
  python "$old_project/src/train_pointcloud_ffm.py" \
  --config "$repo_root/$package/CQ_LR_no_persistent_200ep.yaml" \
  > "$old_log" 2>&1 &
old_pid=$!

echo "Launching persistent-cache CQ-LR $new_commit on physical GPU $new_gpu"
CUDA_VISIBLE_DEVICES="$new_gpu" KEOPS_CACHE_FOLDER="/tmp/keops_cq_topk_ab_train_new_gpu${new_gpu}" \
  conda run --no-capture-output -n phycoflow_env \
  python "$new_project/src/train_pointcloud_ffm.py" \
  --config "$repo_root/$package/CQ_LR_persistent_topk_200ep.yaml" \
  > "$new_log" 2>&1 &
new_pid=$!

printf 'timestamp=%s\nold_gpu=%s\nnew_gpu=%s\nold_pid=%s\nnew_pid=%s\nold_log=%s\nnew_log=%s\nold_commit=%s\nnew_commit=%s\n' \
  "$timestamp" "$old_gpu" "$new_gpu" "$old_pid" "$new_pid" "$old_log" "$new_log" \
  "$old_commit" "$new_commit" > "$package/active_pair.env"

set +e
wait "$old_pid"
old_status=$?
wait "$new_pid"
new_status=$?
set -e
if [[ "$old_status" -ne 0 || "$new_status" -ne 0 ]]; then
  printf 'stage=failed\nold_status=%s\nnew_status=%s\n' "$old_status" "$new_status" \
    > "$package/pipeline_status.env"
  echo "Training failure: no-persistent=$old_status, persistent=$new_status" >&2
  exit 5
fi

old_run="$(ls -dt "$package/runs/CQ_LR_no_persistent_DemoN9520_"* | head -1)"
new_run="$(ls -dt "$package/runs/CQ_LR_persistent_topk_DemoN9521_"* | head -1)"

OLD_GPU="$old_gpu" NEW_GPU="$new_gpu" CURRENT_PROJECT="$new_project" \
  bash "$package/evaluate_pair.sh" "$old_run" "$new_run"

printf 'stage=analysis\n' > "$package/pipeline_status.env"
conda run --no-capture-output -n phycoflow_env \
  python "$package/analyze_pair.py" --baseline-run "$old_run" --persistent-run "$new_run" \
  > "$package/logs/analysis.log" 2>&1

conda run --no-capture-output -n fig \
  python figures/scripts/plot_cq_persistent_training_ab.py \
  --comparison "$package/comparison.json" \
  --output-dir figures/generated/cq_persistent_training_ab \
  > "$package/logs/figure.log" 2>&1

printf 'stage=complete\nold_run=%s\nnew_run=%s\nresults=%s\n' \
  "$old_run" "$new_run" "$package/RESULTS.md" > "$package/pipeline_status.env"
echo "Complete: $package/RESULTS.md"
