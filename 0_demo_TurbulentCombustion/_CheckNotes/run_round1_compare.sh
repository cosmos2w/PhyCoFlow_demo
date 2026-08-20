#!/usr/bin/env bash
set -u

ROOT_DIR="/home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion"
ARTIFACT_DIR="${ROOT_DIR}/_CheckNotes/Round1_runtime"
mkdir -p "${ARTIFACT_DIR}"

cd "${ROOT_DIR}"

GPU1_UUID="GPU-3ceda40c-fd5c-4b88-6c47-b3301711571e"
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
  | grep -Fq "${GPU1_UUID}"; do
  date --iso-8601=seconds > "${ARTIFACT_DIR}/waiting_for_gpus.txt"
  sleep 30
done

run_one() {
  local name="$1"
  local config="$2"

  date --iso-8601=seconds > "${ARTIFACT_DIR}/${name}_started_at.txt"
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
    --format=csv,noheader,nounits \
    --id=1 \
    --loop=2 > "${ARTIFACT_DIR}/${name}_gpu_samples.csv" \
    2> "${ARTIFACT_DIR}/${name}_gpu_monitor.stderr" &
  local monitor_pid=$!

  MPLCONFIGDIR="/tmp/round1-${name}-mpl" \
  KEOPS_CACHE_FOLDER="/tmp/round1-${name}-keops" \
  /usr/bin/time -v -o "${ARTIFACT_DIR}/${name}_time.txt" \
    conda run --no-capture-output -n phycoflow_env \
    python src/train_pointcloud_ffm.py \
    --config "${config}" \
    > "${ARTIFACT_DIR}/${name}.log" 2>&1
  local status=$?

  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  date --iso-8601=seconds > "${ARTIFACT_DIR}/${name}_finished_at.txt"
  printf '%s\n' "${status}" > "${ARTIFACT_DIR}/${name}_exit.txt"
  return "${status}"
}

date --iso-8601=seconds > "${ARTIFACT_DIR}/started_at.txt"
run_one legacy _CheckNotes/config_round1_legacy_100.yaml
LEGACY_STATUS=$?
run_one optimized _CheckNotes/config_round1_optimized_100.yaml
OPTIMIZED_STATUS=$?
date --iso-8601=seconds > "${ARTIFACT_DIR}/finished_at.txt"
{
  printf 'legacy_exit=%s\n' "${LEGACY_STATUS}"
  printf 'optimized_exit=%s\n' "${OPTIMIZED_STATUS}"
} > "${ARTIFACT_DIR}/exit_status.txt"

if [[ "${LEGACY_STATUS}" -ne 0 || "${OPTIMIZED_STATUS}" -ne 0 ]]; then
  exit 1
fi
