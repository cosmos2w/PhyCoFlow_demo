"""Storage/GPU preflight checks for long cache jobs."""
from __future__ import annotations
import shutil
import subprocess
import fcntl
from pathlib import Path

from .config import RESULTS_DIR


def assert_cache_storage(cfg: dict, *, allow_local: bool = False) -> dict:
    required = bool(cfg["cache"].get("require_results_symlink", True))
    if required and not RESULTS_DIR.is_symlink() and not allow_local:
        raise RuntimeError(
            f"cache_storage_not_symlinked: {RESULTS_DIR}. Create a symlink to the cache disk "
            "or pass --allow-local-cache explicitly."
        )
    target = RESULTS_DIR.resolve()
    usage = shutil.disk_usage(target)
    free_gb = usage.free / 1024**3
    minimum = float(cfg["cache"].get("minimum_free_space_gb", 50))
    if free_gb < minimum:
        raise RuntimeError(f"insufficient_cache_space: {free_gb:.1f} GiB free < {minimum:.1f} GiB required at {target}")
    print(f"[STORAGE] results={RESULTS_DIR} -> {target} | free={free_gb:.1f} GiB | symlink={RESULTS_DIR.is_symlink()}")
    return {"target": str(target), "free_gb": free_gb, "is_symlink": RESULTS_DIR.is_symlink()}


def device_preflight(device: str, cfg: dict) -> dict:
    if not str(device).startswith("cuda:"):
        print(f"[DEVICE] {device}")
        return {"device": device}
    index = int(str(device).split(":", 1)[1])
    query = subprocess.run([
        "nvidia-smi", f"--id={index}",
        "--query-gpu=name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True).stdout.strip().split(",")
    name, total, free, utilization = query[0].strip(), float(query[1]), float(query[2]), float(query[3])
    warn_at = float(cfg["cache"].get("warn_gpu_utilization_percent", 20))
    suffix = " | WARNING: shared/busy GPU" if utilization > warn_at else ""
    print(f"[DEVICE] cuda:{index} {name} | free={free/1024:.1f}/{total/1024:.1f} GiB | util={utilization:.0f}%{suffix}")
    return {"device": device, "name": name, "free_gb": free / 1024, "total_gb": total / 1024, "utilization": utilization}


def acquire_cache_lock(run_id: str):
    lock_path=RESULTS_DIR/"ReconstructionCache"/".locks"/f"{run_id}.lock"
    lock_path.parent.mkdir(parents=True,exist_ok=True); handle=lock_path.open("w")
    try: fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close(); raise RuntimeError(f"cache_run_already_active: {run_id} ({lock_path})") from exc
    handle.write(str(Path('/proc/self').resolve().name)); handle.flush()
    return handle
