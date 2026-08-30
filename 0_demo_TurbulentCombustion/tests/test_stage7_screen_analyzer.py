from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE7 = ROOT / "research" / "stages" / "stage7"
if str(STAGE7) not in sys.path:
    sys.path.insert(0, str(STAGE7))

from analyze_stage7_screen import main


def test_joint_decision_prefers_faster_candidate_inside_one_percent_quality_tie(
    tmp_path: Path, monkeypatch,
):
    definitions = {
        "F0": (0.50, "F0", 100.0, 10.0),
        "CQ-LR-128": (0.60, "Frozen-CQ-LR-128", 70.0, 7.0),
        "S7-A": (0.540, "Stage7-Cond128", 80.0, 8.0),
        "S7-B": (0.535, "Stage7-All256", 85.0, 8.5),
    }
    entries = []
    formal = []
    persistent = []
    summaries = []
    for label, (rf_loss, benchmark_label, step_ms, nfe4_s) in definitions.items():
        fixed_path = tmp_path / f"{label}_fixed.json"
        fixed_path.write_text(json.dumps({
            "summary": {
                f"{label}/epoch_0200.pt": {
                    "epoch": 200,
                    "stored_val_loss": rf_loss + 0.01,
                    "mean_rf_loss": rf_loss,
                    "std_rf_loss": 0.02,
                    "evaluations": 192,
                }
            }
        }))
        recon_path = tmp_path / f"{label}_recon.json"
        recon_path.write_text(json.dumps({
            "metrics": {"CO": rf_loss, "T": rf_loss * 0.5, "obs_count_SenConsis_total": 256}
        }))
        run_dir = tmp_path / f"{label}_run"
        run_dir.mkdir()
        with (run_dir / "loss_history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "train_seconds"])
            writer.writeheader()
            writer.writerow({
                "epoch": 200, "train_loss": rf_loss,
                "val_loss": rf_loss + 0.01, "train_seconds": step_ms,
            })
        entries.extend([
            "--entry", label, str(fixed_path), str(recon_path), str(run_dir), benchmark_label,
        ])
        formal.append({
            "label": benchmark_label, "status": "ok", "full_step_ms": step_ms,
            "peak_allocated_mb": 1000.0 if label == "F0" else 800.0,
        })
        persistent.append({
            "label": benchmark_label, "status": "ok", "steady_nfe4_s": nfe4_s,
        })
        summaries.append({"label": benchmark_label, "total_parameters": 1})

    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps({
        "formal_training_step": formal,
        "persistent_inference": persistent,
        "model_summaries": summaries,
    }))
    output = tmp_path / "decision.json"
    monkeypatch.setattr(sys, "argv", [
        "analyze_stage7_screen.py", *entries,
        "--benchmark", str(benchmark_path),
        "--output", str(output),
    ])
    main()
    result = json.loads(output.read_text())
    assert result["decision"]["continue_label"] == "S7-A"
    assert result["models"]["S7-A"]["train_speedup_vs_f0"] == 1.25
    assert result["models"]["S7-A"]["persistent_nfe4_speedup_vs_f0"] == 1.25
    assert output.with_suffix(".md").is_file()
