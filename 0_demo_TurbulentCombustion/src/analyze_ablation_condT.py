#!/usr/bin/env python
"""CPU-only, cache-only Cond_T ablation reconstruction and physical-fidelity audit.

Uses the archived paper's metric implementations and exact joint-histogram
edges. Accepts a single checkpoint policy directory containing A0..A5 caches.
No checkpoint, training, inference, or plotting is performed here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np

DEMO = Path(__file__).resolve().parents[1]
ARCHIVE = DEMO / "Save_TrainedModel/_TrainedModels"
SCRIPTS = ARCHIVE / "_Scripts"
RESULTS = ARCHIVE / "_Process_Results"
sys.path.insert(0, str(SCRIPTS))
from common.cache import load_cache
from common.pdf_utils import PAIR_FIELDS, histogram
from common.spectral import compare_channel_spectra_batch
from common.statistics import jsd_base2, relative_l2

FIELDS = ("CH4", "CO", "T", "U1", "p")
PAIRS = ("T-U1", "CH4-U1", "p-U1")
METHODS = tuple(f"A{i}" for i in range(6))
# Per-state metric values are reusable across evaluations only when this
# semantic version and the archived metric-source hashes agree.  Keep this
# separate from the whole-file analyzer hash: adding CLI/provenance code must
# not invalidate otherwise identical per-state measurements.
METRIC_IMPLEMENTATION_VERSION = "condT-per-state-v1"
EDGES_SOURCES = {
    "T-U1": ("JointPDF_JSD_bin_edges_paper_full_20260711.csv", "T-U1", False),
    "CH4-U1": ("CH4U1CouplingJSD_bin_edges_paper_full_20260711.csv", "CH4-U1", False),
    "p-U1": ("FlowConsistencyJSD_bin_edges_paper_full_20260711.csv", "U1-p", True),
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _update_digest(digest, name, value):
    """Add an NPZ member to a deterministic, compression-independent digest."""
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    if name == "metadata_json":
        metadata = json.loads(str(np.asarray(value).item()))
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        digest.update(payload)
        return
    array = np.asarray(value)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))


def cache_content_sha256(path):
    """Hash cache semantics rather than the compressed ZIP container bytes.

    NPZ compression and member ordering are allowed to change when a cache is
    copied or regenerated.  All arrays and canonicalized metadata still have
    to match before an old metric row can be reused.
    """
    digest = hashlib.sha256()
    with np.load(Path(path), allow_pickle=False) as data:
        for name in sorted(data.files):
            _update_digest(digest, name, data[name])
    return digest.hexdigest()


def load_cache_metadata(path):
    """Read only metadata from a cache without materializing all arrays."""
    with np.load(Path(path), allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"].item()))


def _metrics_dir(path):
    """Accept either a policy directory or its ``metrics`` subdirectory."""
    path = Path(path).resolve()
    if (path / "per_state_metrics.csv").is_file():
        return path
    candidate = path / "metrics"
    if (candidate / "per_state_metrics.csv").is_file():
        return candidate
    raise FileNotFoundError(f"Expected per_state_metrics.csv in {path} or {candidate}")


def _row_identity(row):
    """Return the stable fields that identify one generated test state."""
    return (str(row.get("method", "")), int(row["snapshot"]), int(row.get("time_index", row["snapshot"])),
            str(row.get("checkpoint", "")), str(row.get("generation_seed", "")),
            str(row.get("checkpoint_sha256", "")))


def _same_reuse_identity(current_meta, old_audit):
    """Check metadata fields that must agree before comparing payload hashes."""
    current_method = str(current_meta.get("variant", current_meta.get("method", "")))
    current = (current_method, int(current_meta["snapshot"]), int(current_meta.get("time_index", current_meta["snapshot"])),
               str(current_meta.get("checkpoint_name", "")), str(current_meta.get("generation_seed", "")),
               str(current_meta.get("checkpoint_sha256", "")))
    old = _row_identity(old_audit)
    return current == old


def _source_hashes_match(previous_metadata, edge_provenance):
    """Validate the semantic inputs needed to reuse previous per-state rows."""
    if not previous_metadata:
        return False, "missing analysis_metadata.json"
    previous_version = previous_metadata.get("metric_implementation_version")
    if previous_version not in (None, METRIC_IMPLEMENTATION_VERSION):
        return False, f"metric implementation version differs ({previous_version!r})"
    expected_sources = {
        str(SCRIPTS / "common/statistics.py"): sha256(SCRIPTS / "common/statistics.py"),
        str(SCRIPTS / "common/spectral.py"): sha256(SCRIPTS / "common/spectral.py"),
        str(SCRIPTS / "common/pdf_utils.py"): sha256(SCRIPTS / "common/pdf_utils.py"),
    }
    if previous_metadata.get("metric_sources") != expected_sources:
        return False, "metric-source hashes differ"
    previous_edges = previous_metadata.get("histogram_edges", {})
    for pair, provenance in edge_provenance.items():
        old = previous_edges.get(pair, {})
        if old.get("sha256") != provenance.get("sha256"):
            return False, f"histogram-edge hash differs for {pair}"
    return True, "source and edge hashes match"


def _prepare_reuse(previous_dir, current_paths, edge_provenance):
    """Build a cache-checked reuse map from a previous policy evaluation.

    The returned map contains only exact semantic matches.  A mismatch is
    deliberately treated as a fresh computation candidate, so a new A0
    checkpoint or a newly generated snapshot cannot silently inherit stale
    values.  ``reuse_log`` is exported to make that decision auditable.
    """
    previous = _metrics_dir(previous_dir)
    metadata_path = previous / "analysis_metadata.json"
    previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
    ok, reason = _source_hashes_match(previous_metadata, edge_provenance)
    if not ok:
        raise RuntimeError(f"Cannot reuse {previous}: {reason}")
    old_rows = read_csv(previous / "per_state_metrics.csv")
    old_audit = read_csv(previous / "cache_audit.csv")
    rows_by_key = defaultdict(list)
    for row in old_rows:
        row["value"] = float(row["value"])
        rows_by_key[(str(row["method"]), int(row["snapshot"]))].append(row)
    audit_by_key = {}
    for row in old_audit:
        key = (str(row["method"]), int(row["snapshot"]))
        if key in audit_by_key:
            raise RuntimeError(f"Duplicate source cache audit identity: {key}")
        audit_by_key[key] = row

    reused = {}
    reuse_log = []
    for path in current_paths:
        key = (path.parent.name, int(re.fullmatch(r"RecCache_s(\d+)\.npz", path.name).group(1)))
        old_audit_row = audit_by_key.get(key)
        old_metric_rows = rows_by_key.get(key, [])
        record = {"method": key[0], "snapshot": key[1], "current_cache": str(path), "reused": False}
        if old_audit_row is None or not old_metric_rows:
            record["reason"] = "no source rows for identity"
            reuse_log.append(record)
            continue
        old_path = Path(old_audit_row.get("cache_path", ""))
        if not old_path.is_file():
            record["reason"] = "source cache missing"
            reuse_log.append(record)
            continue
        try:
            current_meta = load_cache_metadata(path)
            identity_ok = _same_reuse_identity(current_meta, old_audit_row)
            current_digest = cache_content_sha256(path)
            # The evaluator may expose an old cache through a symlink.  In
            # that case the resolved payload is already the same file, so a
            # second full read would only add avoidable I/O.  Future audit
            # files also carry this digest and can skip the source read even
            # when the cache was copied rather than linked.
            source_digest = str(old_audit_row.get("cache_content_sha256", ""))
            old_digest = (current_digest if path.resolve() == old_path.resolve()
                          else source_digest if re.fullmatch(r"[0-9a-f]{64}", source_digest)
                          else cache_content_sha256(old_path))
        except Exception as exc:
            record["reason"] = f"cache read failed: {exc}"
            reuse_log.append(record)
            continue
        if not identity_ok:
            record["reason"] = "cache metadata identity differs"
            reuse_log.append(record)
            continue
        if current_digest != old_digest:
            record["reason"] = "cache payload digest differs"
            record["current_cache_sha256"] = current_digest
            record["source_cache_sha256"] = old_digest
            reuse_log.append(record)
            continue
        current_rows = []
        for row in old_metric_rows:
            copied = dict(row)
            copied["cache_path"] = str(path)
            current_rows.append(copied)
        audit_row = dict(old_audit_row)
        audit_row["cache_path"] = str(path)
        audit_row["cache_content_sha256"] = current_digest
        reused[key] = (current_rows, audit_row)
        record.update({"reused": True, "reason": reason, "cache_content_sha256": current_digest,
                       "source_cache": str(old_path)})
        reuse_log.append(record)
    old_spectra = read_csv(previous / "snapshot0_spectra.csv") if (previous / "snapshot0_spectra.csv").is_file() else []
    spectra_by_key = defaultdict(list)
    for row in old_spectra:
        spectra_by_key[(str(row.get("method", "")), int(row.get("snapshot", 0)))].append(row)
    return reused, spectra_by_key, reuse_log, previous


def load_paper_edges():
    """Preserve float32 CUDA edges for CH4-U1 and float64 legacy edges otherwise."""
    edges, provenance = {}, {}
    for pair, (filename, source_pair, transpose) in EDGES_SOURCES.items():
        path = RESULTS / "JointPDF_JSD" / filename
        rows = [r for r in read_csv(path) if r["pair"] == source_pair
                and r.get("condition", "Cond_T") == "Cond_T"]
        axis_edges = []
        for axis in ("x", "y"):
            selected = [r for r in rows if r["axis"] == axis]
            if not selected:
                raise ValueError(f"Missing histogram edges: {pair}/{axis}")
            if "edge" in selected[0]:
                selected.sort(key=lambda r: int(r["edge_index"]))
                values = [float(r["edge"]) for r in selected]
            else:
                selected.sort(key=lambda r: int(r["bin"]))
                values = [float(r["left"]) for r in selected] + [float(selected[-1]["right"])]
            values = np.asarray(values, dtype=np.float64)
            if len(values) != 65 or not np.all(np.diff(values) > 0):
                raise ValueError(f"Invalid archived 64-bin edges: {pair}/{axis}")
            axis_edges.append(values)
        edges[pair] = tuple(reversed(axis_edges)) if transpose else tuple(axis_edges)
        provenance[pair] = {"path": str(path), "sha256": sha256(path), "source_pair": source_pair,
                            "transposed": transpose, "extent": [float(v) for e in edges[pair] for v in e[[0, -1]]]}
    return edges, provenance


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x*x) * np.sum(y*y))
    return float(np.sum(x*y) / denom) if denom > 0 else float("nan")


def process_cache(job):
    path, edges = job
    arrays, meta = load_cache(Path(path))
    truth = arrays["truth_phys"]
    prediction = arrays["recon_phys"]
    if truth.shape != prediction.shape or truth.shape != (40300, 5):
        raise ValueError(f"Unexpected arrays in {path}: {truth.shape}, {prediction.shape}")
    for key in ("truth_phys", "recon_phys", "truth_norm", "recon_norm", "coords_phys"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Nonfinite {key} in {path}; refusing partial-point metrics")
    method = str(meta.get("variant", meta.get("method", Path(path).parent.name)))
    snapshot = int(meta["snapshot"])
    identity = {"method": method, "snapshot": snapshot, "time_index": int(meta.get("time_index", snapshot)),
                "checkpoint": meta.get("checkpoint_name", Path(path).parents[1].name), "cache_path": str(path)}
    metrics = {}
    phys_l2, norm_l2 = [], []
    sensor_indices = np.asarray(arrays["obs_indices"]).ravel().astype(int)
    sensor_fields = np.asarray(arrays["obs_field_ids"]).ravel().astype(int)
    if len(sensor_indices) != 256 or set(sensor_fields.tolist()) != {2}:
        raise ValueError(f"Expected 256 T sensors in {path}")
    for channel, field in enumerate(FIELDS):
        t, p = truth[:, channel], prediction[:, channel]
        phys_l2.append(relative_l2(t, p))
        norm_l2.append(relative_l2(arrays["truth_norm"][:, channel], arrays["recon_norm"][:, channel]))
        metrics[("physical_relative_l2", field)] = phys_l2[-1]
        metrics[("normalized_relative_l2", field)] = norm_l2[-1]
        metrics[("pointwise_correlation", field)] = correlation(t, p)
        # A dimensionless offset-insensitive companion, especially useful for p.
        fluctuation_norm = np.sqrt(np.sum((t.astype(float)-np.mean(t, dtype=float))**2))
        metrics[("truth_fluctuation_normalized_l2", field)] = float(
            np.sqrt(np.sum((p.astype(float)-t)**2))/(fluctuation_norm+1e-12))
    unobserved = [0, 1, 3, 4]
    metrics[("physical_relative_l2", "Unobserved_mean")] = float(np.mean(np.asarray(phys_l2)[unobserved]))
    metrics[("normalized_relative_l2", "Unobserved_mean")] = float(np.mean(np.asarray(norm_l2)[unobserved]))
    metrics[("physical_relative_l2", "All_fields_mean")] = float(np.mean(phys_l2))
    metrics[("normalized_relative_l2", "All_fields_mean")] = float(np.mean(norm_l2))
    keep = np.ones(truth.shape[0], dtype=bool)
    keep[sensor_indices] = False
    metrics[("physical_relative_l2_excluding_sensors", "T")] = relative_l2(truth[:, 2], prediction[:, 2], keep)
    metrics[("normalized_relative_l2_excluding_sensors", "T")] = relative_l2(arrays["truth_norm"][:, 2], arrays["recon_norm"][:, 2], keep)
    metrics[("sensor_max_abs_normalized_error", "T")] = float(np.max(np.abs(
        arrays["recon_norm"][sensor_indices, 2]-arrays["truth_norm"][sensor_indices, 2])))

    spectra = compare_channel_spectra_batch(truth, prediction, arrays["coords_phys"],
        num_x=403, num_y=100, coordinate_mode="auto", spacing_tolerance=.02,
        remove_mean=True, window="none", use_isotropic_cutoff=True,
        min_shell_count=4, relative_epsilon=1e-12, device="cpu")
    spectral_rows = []
    for field, result in zip(FIELDS, spectra):
        metrics[("spectral_lsd_db", field)] = result["lsd_db"]
        metrics[("spectral_total_energy_ratio", field)] = result["total_energy_ratio"]
        for index, band in enumerate(("low", "mid", "high")):
            metrics[(f"spectral_{band}_energy_ratio", field)] = float(result["band_energy_ratio"][index])
        if snapshot == 0:
            for k, true, pred in zip(result["truth"]["wavenumber"], result["truth"]["spectral_energy"], result["reconstruction"]["spectral_energy"]):
                spectral_rows.append({**identity, "field": field, "wavenumber": float(k),
                                      "truth_energy": float(true), "reconstruction_energy": float(pred)})

    for pair in PAIRS:
        left, right = PAIR_FIELDS[pair]
        t = truth[:, [left, right]]
        p = prediction[:, [left, right]]
        ex, ey = edges[pair]
        ht, hp = histogram(t[:, 0], t[:, 1], (ex, ey)), histogram(p[:, 0], p[:, 1], (ex, ey))
        metrics[("joint_pdf_jsd_base2", pair)] = jsd_base2(ht, hp, pseudocount=1e-12)
        # Supplemental metric only: retain the exact interior paper bins but
        # collect all out-of-range points in axial under/overflow bins.
        # This avoids treating the paper's conditional-on-retention JSD as
        # an assessment of the entire predicted distribution.
        extended = (np.r_[-np.inf, ex, np.inf], np.r_[-np.inf, ey, np.inf])
        ht_full = histogram(t[:, 0], t[:, 1], extended)
        hp_full = histogram(p[:, 0], p[:, 1], extended)
        metrics[("joint_pdf_jsd_with_overflow_base2", pair)] = jsd_base2(ht_full, hp_full, pseudocount=1e-12)
        for label, values in (("truth", t), ("reconstruction", p)):
            retained = ((values[:, 0] >= ex[0]) & (values[:, 0] <= ex[-1]) &
                        (values[:, 1] >= ey[0]) & (values[:, 1] <= ey[-1]))
            metrics[(f"joint_pdf_{label}_retained_fraction", pair)] = float(np.mean(retained))
        metrics[("coupling_truth_correlation", pair)] = correlation(t[:, 0], t[:, 1])
        metrics[("coupling_reconstruction_correlation", pair)] = correlation(p[:, 0], p[:, 1])
        metrics[("coupling_correlation_abs_error", pair)] = abs(metrics[("coupling_truth_correlation", pair)] - metrics[("coupling_reconstruction_correlation", pair)])

    for field, channel in (("CH4", 0), ("CO", 1), ("T", 2), ("p", 4)):
        for label, values in (("truth", truth[:, channel]), ("reconstruction", prediction[:, channel])):
            invalid = values < 0 if channel in (0, 1) else values <= 0
            metrics[(f"{label}_nonphysical_fraction", field)] = float(np.mean(invalid))
            metrics[(f"{label}_minimum", field)] = float(np.min(values))
        if channel in (0, 1):
            negative = np.maximum(-prediction[:, channel].astype(np.float64), 0.)
            metrics[("reconstruction_negative_mean_magnitude", field)] = float(negative.mean())
            metrics[("reconstruction_negative_l2_over_truth_l2", field)] = float(
                np.linalg.norm(negative) / (np.linalg.norm(truth[:, channel].astype(np.float64))+1e-12))
            metrics[("reconstruction_fraction_below_minus_1e4", field)] = float(np.mean(prediction[:, channel] < -1e-4))
    meta_audit = {**identity, "grid_mode": spectra[0]["grid"]["coordinate_mode_used"],
                 "truth_sha256": hashlib.sha256(np.ascontiguousarray(truth).tobytes()).hexdigest(),
                 "sensor_sha256": hashlib.sha256(np.stack([sensor_indices, sensor_fields]).tobytes()).hexdigest(),
                 "generation_seed": meta.get("generation_seed", ""), "checkpoint_sha256": meta.get("checkpoint_sha256", "")}
    rows = [{**identity, "metric": metric, "target": target, "value": float(value)}
            for (metric, target), value in metrics.items()]
    return rows, spectral_rows, meta_audit


def block_bootstrap_means(values, *, block, n_boot, seed):
    """Circular moving-block bootstrap of temporally sorted snapshot values."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if not n or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite nonempty values")
    block = min(int(block), n)
    n_blocks = (n + block - 1) // block
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    # Vectorized block sums avoid n_boot * n * metric-sized arrays.
    doubled = np.concatenate([values, values[:block]])
    prefix = np.concatenate([[0.], np.cumsum(doubled)])
    full_count = n // block
    sums = np.sum(prefix[starts[:, :full_count] + block]-prefix[starts[:, :full_count]], axis=1)
    remainder = n % block
    if remainder:
        last = starts[:, full_count]
        sums += prefix[last + remainder]-prefix[last]
    return sums / n


def describe(values, n_boot, seed):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) != len(values):
        raise ValueError("A metric contains nonfinite values; cannot silently summarize")
    stats = {"n": len(values), "mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values)>1 else 0.,
             "median": float(np.median(values)), "q25": float(np.quantile(values,.25)),
             "q75": float(np.quantile(values,.75)), "p95": float(np.quantile(values,.95)),
             "min": float(values.min()), "max": float(values.max())}
    for block, label in ((1, "iid"), (20, "block20")):
        means = block_bootstrap_means(values, block=block, n_boot=n_boot, seed=seed)
        stats[f"{label}_ci95_low"], stats[f"{label}_ci95_high"] = map(float, np.quantile(means,[.025,.975]))
    return stats


def summarize_rows(rows, n_boot=2000, seed=20260906):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["metric"], row["target"])].append(row)
    summaries, paired = [], []
    for (method, metric, target), items in sorted(groups.items()):
        items.sort(key=lambda row: (int(row["time_index"]), int(row["snapshot"])))
        summaries.append({"method": method, "metric": metric, "target": target,
                          **describe([r["value"] for r in items], n_boot, seed)})
        if method == "A0":
            continue
        for reference in (["A0", "A2"] if method == "A5" else ["A0"]):
            baseline = {int(row["snapshot"]): row for row in groups.get((reference, metric, target), [])}
            paired_items = [row for row in items if int(row["snapshot"]) in baseline]
            if not paired_items:
                continue
            baseline_values = np.asarray([baseline[int(r["snapshot"])]["value"] for r in paired_items], float)
            other = np.asarray([r["value"] for r in paired_items], float)
            delta = other-baseline_values
            entry = {"method": method, "baseline": reference, "metric": metric, "target": target,
                     "n_paired": len(delta), "method_mean": float(other.mean()), "baseline_mean": float(baseline_values.mean()),
                     "mean_difference": float(delta.mean()),
                     "relative_mean_change_percent": float(100*delta.mean()/baseline_values.mean()) if baseline_values.mean()!=0 else "",
                     "fraction_method_less_than_baseline": float(np.mean(delta<0)),
                     "fraction_method_equal_baseline": float(np.mean(delta==0))}
            for block in (5,20,50):
                means = block_bootstrap_means(delta, block=block, n_boot=n_boot, seed=seed)
                low, high = map(float,np.quantile(means,[.025,.975]))
                entry[f"block{block}_ci95_low"], entry[f"block{block}_ci95_high"] = low, high
                entry[f"block{block}_excludes_zero"] = low>0 or high<0
            paired.append(entry)
    return summaries, paired


def paper_crosscheck(rows):
    """Compare the current A0 values to archived paper metrics.

    This is a historical implementation cross-check, not an assertion that
    the current A0 checkpoint is byte-identical to the archived checkpoint.
    """
    current = {(int(r["snapshot"]),r["metric"],r["target"]):r["value"] for r in rows if r["method"]=="A0"}
    comparison = []
    files = [
        (RESULTS / "FieldL2/FieldL2_per_snapshot_paper_full_20260711.csv", "physical_relative_l2", "physical_rel_l2", "field", "snapshot"),
        (RESULTS / "Spectral/SpectralLSD/SpectralLSD_per_snapshot_paper_full_20260711.csv", "spectral_lsd_db", "lsd_db", "field_name", "snapshot_index"),
        (RESULTS / "JointPDF_JSD/JointPDF_JSD_per_snapshot_paper_full_20260711.csv", "joint_pdf_jsd_base2", "jsd_base2", "pair", "snapshot"),
        (RESULTS / "JointPDF_JSD/CH4U1CouplingJSD_per_snapshot_paper_full_20260711.csv", "joint_pdf_jsd_base2", "jsd_base2", "pair", "snapshot"),
        (RESULTS / "JointPDF_JSD/FlowConsistencyJSD_per_snapshot_paper_full_20260711.csv", "joint_pdf_jsd_base2", "jsd_base2", "pair", "snapshot"),
    ]
    for path, metric, value_key, target_key, snapshot_key in files:
        for old in read_csv(path):
            if old.get("method",old.get("model_label"))!="DMF-Gen" or old.get("condition")!="Cond_T" or old.get("status")!="ok":
                continue
            target = old[target_key]
            if target=="U1-p": target="p-U1"
            key=(int(old[snapshot_key]),metric,target)
            if key in current:
                reference=float(old[value_key]); measured=float(current[key])
                comparison.append({"comparison":"current_A0_vs_archived_paper_DMF_Gen_Cond_T",
                    "snapshot":key[0],"metric":metric,"target":target,
                    "archived_value":reference,"new_A0_value":measured,"absolute_difference":abs(measured-reference),"source":str(path)})
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir",type=Path,required=True, help="One checkpoint directory containing A0..A5")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--bootstrap-resamples",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=20260906)
    parser.add_argument("--expected-snapshots",type=int,default=1000)
    parser.add_argument("--allow-partial",action="store_true", help="Explicitly labeled smoke/subset analysis")
    parser.add_argument("--reuse-per-state",action="store_true", help="Rebuild summaries from existing per-state CSV in this output directory")
    parser.add_argument("--reuse-from",type=Path,
                        help="Previous policy evaluation (or metrics directory) from which exact cache-checked per-state rows may be reused")
    args=parser.parse_args()
    if args.reuse_per_state and args.reuse_from:
        parser.error("--reuse-per-state and --reuse-from are mutually exclusive")
    root=args.evaluation_dir.resolve(); output=(args.output_dir or root/"metrics").resolve()
    output.mkdir(parents=True,exist_ok=True)
    edges,edge_provenance=load_paper_edges()
    if args.reuse_per_state:
        rows=read_csv(output/"per_state_metrics.csv")
        for row in rows: row["value"]=float(row["value"])
        audit=read_csv(output/"cache_audit.csv")
        spectra=read_csv(output/"snapshot0_spectra.csv") if (output/"snapshot0_spectra.csv").is_file() else []
        reuse_log=[]
    else:
        paths=sorted(p for p in root.glob("A[0-5]/RecCache_s*.npz")
                     if re.fullmatch(r"RecCache_s\d+\.npz", p.name))
        if not paths: raise FileNotFoundError(f"No A0..A5 caches in {root}")
        counts={method:sum(p.parent.name==method for p in paths) for method in METHODS}
        if not args.allow_partial and any(n!=args.expected_snapshots for n in counts.values()):
            raise RuntimeError(f"Incomplete cache coverage: {counts}; use --allow-partial for explicit smoke only")
        rows=[]; spectra=[]; audit=[]; reuse_log=[]
        reused={}; reusable_spectra={}
        reuse_source=None
        if args.reuse_from:
            reused, reusable_spectra, reuse_log, reuse_source = _prepare_reuse(args.reuse_from, paths, edge_provenance)
            print(f"[reuse] accepted {len(reused)}/{len(paths)} cache identities from {reuse_source}", flush=True)
        fresh_paths=[]
        for path in paths:
            key=(path.parent.name, int(re.fullmatch(r"RecCache_s(\d+)\.npz", path.name).group(1)))
            if key in reused:
                old_rows, old_audit = reused[key]
                rows.extend(old_rows)
                audit.append(old_audit)
                copied_spectra = reusable_spectra.get(key, [])
                if copied_spectra:
                    spectra.extend([{**r, "cache_path": str(path)} for r in copied_spectra])
            else:
                fresh_paths.append(path)
        jobs=((str(path),edges) for path in fresh_paths)
        pool=ProcessPoolExecutor(max_workers=args.workers) if args.workers>1 else None
        iterator=pool.map(process_cache,jobs,chunksize=2) if pool else map(process_cache,jobs)
        try:
            for index,(new_rows,new_spectra,new_audit) in enumerate(iterator,1):
                rows.extend(new_rows); spectra.extend(new_spectra); audit.append(new_audit)
                if index%100==0 or index==len(fresh_paths): print(f"[metrics] {index}/{len(fresh_paths)} fresh caches",flush=True)
        finally:
            if pool: pool.shutdown()
        # Cache content hashes make future cross-directory reuse auditable.
        # Reused rows were hashed in _prepare_reuse; fresh rows are hashed once
        # after metric extraction to avoid hashing the same NPZ in workers.
        fresh_keys={(path.parent.name, int(re.fullmatch(r"RecCache_s(\d+)\.npz", path.name).group(1))): path
                    for path in fresh_paths}
        for item in audit:
            key=(str(item["method"]), int(item["snapshot"]))
            if not item.get("cache_content_sha256"):
                item["cache_content_sha256"] = cache_content_sha256(fresh_keys[key])
        # A source evaluation may omit snapshot-0 spectral rows (for example,
        # an older partial run).  Recompute only those missing rows.
        spectra_keys={(str(row.get("method", "")), int(row.get("snapshot", 0))) for row in spectra}
        missing_s0=[path for path in fresh_paths
                    if path.parent.name in METHODS
                    and int(re.fullmatch(r"RecCache_s(\d+)\.npz", path.name).group(1))==0
                    and (path.parent.name,0) not in spectra_keys]
        if missing_s0:
            for path in missing_s0:
                new_rows,new_spectra,new_audit=process_cache((str(path),edges))
                spectra.extend(new_spectra)
                # Metrics/audit already exist; only the spectral example is
                # needed here.  Its audit is intentionally not appended.
        write_csv(output/"per_state_metrics.csv",rows)
        write_csv(output/"snapshot0_spectra.csv",spectra)
        write_csv(output/"cache_audit.csv",audit)
        if args.reuse_from:
            write_csv(output/"reuse_log.csv",reuse_log)
    # Apply identical coverage requirements to fresh and reused metrics.
    keys=[(r["method"],int(r["snapshot"])) for r in audit]
    if len(keys)!=len(set(keys)):
        raise RuntimeError("Duplicate method/snapshot cache identities")
    expected_methods=set(METHODS)
    if not args.allow_partial:
        expected={(m,s) for m in expected_methods for s in range(args.expected_snapshots)}
        if set(keys)!=expected:
            raise RuntimeError(f"Cache identity coverage differs from six methods x snapshots 0..{args.expected_snapshots-1}")
    row_keys=[(r["method"],int(r["snapshot"]),r["metric"],r["target"]) for r in rows]
    if len(row_keys)!=len(set(row_keys)):
        raise RuntimeError("Duplicate per-state metric identities")
    if {(r[0],r[1]) for r in row_keys}!=set(keys):
        raise RuntimeError("Metric/cache identity mismatch")
    metric_sets=defaultdict(set)
    for method,snapshot,metric,target in row_keys:
        metric_sets[(method,snapshot)].add((metric,target))
    if len({frozenset(values) for values in metric_sets.values()})!=1:
        raise RuntimeError("Per-state metrics have inconsistent metric coverage")
    # Truth and sensor equality are required before paired claims.
    by_snapshot=defaultdict(list)
    for item in audit: by_snapshot[int(item["snapshot"])].append(item)
    mismatches=[snapshot for snapshot,items in by_snapshot.items()
                if len({r["truth_sha256"] for r in items})!=1 or len({r["sensor_sha256"] for r in items})!=1
                or len({str(r["time_index"]) for r in items})!=1]
    if mismatches: raise RuntimeError(f"Paired truth/sensor/time mismatch: {mismatches[:10]}")
    print("[metrics] summarizing with circular block bootstrap",flush=True)
    summaries,paired=summarize_rows(rows,args.bootstrap_resamples,args.seed)
    write_csv(output/"summary_metrics.csv",summaries)
    write_csv(output/"paired_differences.csv",paired)
    comparisons=paper_crosscheck(rows)
    # Keep the historical filename for callers of the 2026-09-06 workflow,
    # but expose an unambiguous canonical name and an explicit comparison
    # label in every row.
    write_csv(output/"archived_A0_crosscheck.csv",comparisons)
    write_csv(output/"A0_paper_crosscheck.csv",comparisons)
    metadata={"evaluation_dir":str(root),"analysis_script":str(Path(__file__).resolve()),
        "analysis_script_sha256":sha256(__file__),"partial":args.allow_partial,
        "metric_implementation_version":METRIC_IMPLEMENTATION_VERSION,
        "cache_count":len(audit),"expected_snapshots_per_method":args.expected_snapshots,
        "snapshots_by_method":{m:len([r for r in audit if r["method"]==m]) for m in sorted({r["method"] for r in audit})},
        "finite_input_required":True,"truth_sensor_temporal_identity_matches":True,
        "spectral_execution":"CPU; canonical common.spectral.compare_channel_spectra_batch",
        "spectral_coordinate_modes":sorted({r["grid_mode"] for r in audit}),
        "paper_joint_pairs":list(PAIRS),"histogram_edges":edge_provenance,
        "joint_pdf_tail_policy":"Outside archived truth 0.5–99.5% edges discarded; remaining histogram normalized. Retained fractions explicitly exported.",
        "joint_pdf_overflow_supplement":"Exact interior 64-bin paper edges extended by -inf/+inf on each axis (66x66), preserving every finite point. Same probability normalization and 1e-12 pseudocount. Not the original panel-d metric.",
        "physical_constraints":"CH4,CO <0; T,p <=0; evaluated on unclipped physical arrays",
        "bootstrap":{"resamples":args.bootstrap_resamples,"seed":args.seed,
            "summary":"IID and circular moving blocks length20", "paired":"difference per common snapshot; circular moving blocks 5,20,50",
            "temporal_order":"time_index then snapshot", "unit":"one test snapshot, never individual spatial pixels",
            "limitations":"One training seed/checkpoint per architecture; temporal block lengths are sensitivity choices, not estimated correlation lengths. CIs are descriptive and not multiplicity-adjusted."},
        "metric_sources":{str(p):sha256(p) for p in (SCRIPTS/"common/statistics.py",SCRIPTS/"common/spectral.py",SCRIPTS/"common/pdf_utils.py")},
        "archived_crosscheck":{"file":"archived_A0_crosscheck.csv",
            "comparison":"current A0 checkpoint versus archived paper DMF-Gen/Cond_T CSV values; descriptive historical check, not checkpoint identity"},
        "paper_crosscheck_max_abs_difference":max((r["absolute_difference"] for r in comparisons),default=None),
        "reuse":{"source":str(reuse_source) if not args.reuse_per_state and args.reuse_from else None,
            "accepted_cache_identities":sum(bool(r.get("reused")) for r in reuse_log),
            "candidate_cache_identities":len(reuse_log),
            "content_hash":"canonical NPZ metadata and array payload SHA256; compression-independent"}}
    (output/"analysis_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(f"[OK] {output}",flush=True)


if __name__=="__main__":
    main()
