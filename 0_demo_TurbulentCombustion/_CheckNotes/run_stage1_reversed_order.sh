#!/usr/bin/env bash
set -u

ROOT_DIR="/home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion"
ARTIFACT_DIR="${ROOT_DIR}/_CheckNotes/Stage1_order_runtime"
mkdir -p "${ARTIFACT_DIR}"
cd "${ROOT_DIR}"

# The updated goal permits a direct shared-GPU check with tuned batch size.
# Capture the co-tenant state so these timings are not mistaken for clean
# absolute performance measurements.
date --iso-8601=seconds > "${ARTIFACT_DIR}/started_at.txt"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits --id=0 > "${ARTIFACT_DIR}/gpu0_initial_state.csv"

run_one() {
  local name="$1"
  local config="$2"
  date --iso-8601=seconds > "${ARTIFACT_DIR}/${name}_started_at.txt"
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader,nounits --id=0 --loop=2 \
    > "${ARTIFACT_DIR}/${name}_gpu_samples.csv" \
    2> "${ARTIFACT_DIR}/${name}_gpu_monitor.stderr" &
  local monitor_pid=$!
  MPLCONFIGDIR="/tmp/stage1-${name}-mpl" \
  KEOPS_CACHE_FOLDER="/tmp/stage1-${name}-keops" \
  /usr/bin/time -v -o "${ARTIFACT_DIR}/${name}_time.txt" \
    conda run --no-capture-output -n phycoflow_env \
    python src/train_pointcloud_ffm.py --config "${config}" \
    > "${ARTIFACT_DIR}/${name}.log" 2>&1
  local status=$?
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  date --iso-8601=seconds > "${ARTIFACT_DIR}/${name}_finished_at.txt"
  printf '%s\n' "${status}" > "${ARTIFACT_DIR}/${name}_exit.txt"
  return "${status}"
}

# Deliberately reverse Round 1 order: optimized first, legacy second.
run_one optimized _CheckNotes/config_stage1_order_optimized_12.yaml
OPTIMIZED_STATUS=$?
run_one legacy _CheckNotes/config_stage1_order_legacy_12.yaml
LEGACY_STATUS=$?
{
  printf 'optimized_exit=%s\n' "${OPTIMIZED_STATUS}"
  printf 'legacy_exit=%s\n' "${LEGACY_STATUS}"
} > "${ARTIFACT_DIR}/exit_status.txt"
if [[ "${OPTIMIZED_STATUS}" -ne 0 || "${LEGACY_STATUS}" -ne 0 ]]; then
  exit 1
fi
