"""Source-level regression tests for the Figure 5 V7 ablation package.

These tests deliberately read the saved evaluation products and the compact
tables emitted by the source collector.  They do not run inference or rebuild
any metric.  The timestamp is fixed by the V7 evidence contract so a test
failure identifies a stale or incomplete source package rather than silently
switching to another evaluation.
"""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
EVALUATION_ROOT = (
    REPO_ROOT
    / "0_demo_TurbulentCombustion"
    / "Save_TrainedModel"
    / "ablation_condT"
    / "evaluation_20260910"
)
DERIVED_ROOT = PACKAGE_ROOT / "results" / "derived" / "20260910_1540"
REPORT_HASH = "84d5c9fc708988f51f95d4668ef84786822367570e19bd20f71c466e632a1138"
METHODS = {"A0", "A1", "A2", "A3", "A4", "A5"}
POLICIES = {"last", "best"}
PRIMARY_COLUMNS = ["method", "metric", "target", "mean", "block20_ci95_low", "block20_ci95_high"]
HIGHBAND_COLUMNS = ["policy", "method", "field", "metric", "mean", "block20_ci95_low", "block20_ci95_high"]

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.collect_figure5_v7_ablation_sources import validate_state_table
from utils.figure5_v7_ablation_data import DISPLAY_INFO, METHODS as UTILITY_METHODS, STOCHASTIC_ORDER


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Figure5V7SourceTest(unittest.TestCase):
    """Guard the fixed-source and compact-table contracts."""

    def _raw_summary(self, policy: str) -> pd.DataFrame:
        return pd.read_csv(EVALUATION_ROOT / policy / "metrics" / "summary_metrics.csv")

    def _raw_highband(self) -> pd.DataFrame:
        return pd.read_csv(EVALUATION_ROOT / "high_frequency" / "summary_high_frequency.csv")

    def _derived(self, name: str) -> pd.DataFrame:
        path = DERIVED_ROOT / name
        self.assertTrue(path.is_file(), f"collector output is missing: {path}")
        return pd.read_csv(path, low_memory=False)

    def test_collector_manifest_qa_and_provenance_contract_pass(self) -> None:
        manifest = _json(DERIVED_ROOT / "source_manifest.json")
        qa = _json(DERIVED_ROOT / "source_qa.json")
        self.assertEqual(manifest["schema_version"], "figure5-v7-sources-1")
        self.assertEqual(manifest["status"], "pass")
        self.assertEqual(qa["status"], "pass", qa.get("errors"))
        self.assertEqual(qa.get("errors"), [])
        self.assertEqual(manifest["report_sha256_expected"], REPORT_HASH)
        self.assertGreaterEqual(len(manifest["sources"]), 1)
        protected = [item for item in manifest["sources"].values() if item.get("protected")]
        self.assertTrue(protected)
        self.assertTrue(all(item.get("sha256") == item.get("sha256_after") for item in protected))

        required_tables = {
            "ablation_name_map",
            "panel_reference_map",
            "ablation_provenance",
            "ablation_primary_summary",
            "ablation_primary_paired",
            "ablation_highband_summary",
            "ablation_highband_paired",
            "ablation_spectral_summary",
            "ablation_coupling_audit",
            "ablation_admissibility",
            "deterministic_control_summary",
            "checkpoint_sensitivity",
            "figure5_v7_display_source",
        }
        self.assertTrue(required_tables.issubset(manifest["derived_tables"]))
        self.assertTrue(all((DERIVED_ROOT / f"{name}.csv").is_file() for name in required_tables))

        provenance_columns = {
            "evidence_package",
            "dataset",
            "task",
            "condition",
            "cohort_id",
            "policy",
            "checkpoint_policy",
            "checkpoint_epoch",
            "checkpoint_sha256",
            "source_path",
            "source_sha256",
            "sensor_plan_sha256",
            "state_count",
            "measurement_count",
            "output_point_count",
            "truth_sensor_time_identity",
            "metric_definition",
        }
        for name in ("ablation_primary_summary.csv", "ablation_highband_summary.csv"):
            self.assertTrue(provenance_columns.issubset(self._derived(name).columns), name)

    def test_saved_reports_and_frozen_a0_lineage_are_exact(self) -> None:
        reports = [
            EVALUATION_ROOT / "Ablation_CondT_Evaluation_20260910.md",
            EVALUATION_ROOT.parent / "Ablation_CondT_Evaluation_20260906.md",
        ]
        for report in reports:
            self.assertEqual(_sha256(report), REPORT_HASH, report)

        snapshot = _json(EVALUATION_ROOT / "checkpoint_inputs" / "A0" / "snapshot_manifest.json")
        self.assertTrue(snapshot["source_run_directory"].endswith("A0_baseline_DemoN600_20260906_233723"))
        expected = {
            "last.pt": (7520, 533920, "71a9e1004010558f7137f4140eb976d51d367d39e9bc3473334912257cbaa541"),
            "best.pt": (7095, 503745, "634b839b1be4abdb976fe848524bb81708254df65b1e8cd29b3111a9f63ec654"),
        }
        for name, (epoch, step, digest) in expected.items():
            record = snapshot["checkpoints"][name]
            self.assertEqual(record["epoch"], epoch)
            self.assertEqual(record["global_step"], step)
            self.assertEqual(record["sha256"], digest)
            self.assertTrue(Path(record["frozen_path"]).is_file())

    def test_raw_primary_summaries_have_exact_state_counts_and_unique_keys(self) -> None:
        for policy in sorted(POLICIES):
            table = self._raw_summary(policy)
            self.assertEqual(set(table.method), METHODS)
            self.assertTrue((table.n == 1000).all())
            self.assertFalse(table.duplicated(["method", "metric", "target"]).any())
            numeric = ["n", "mean", "block20_ci95_low", "block20_ci95_high"]
            self.assertTrue(np.isfinite(table[numeric].to_numpy(dtype=float)).all())
            self.assertTrue((table["block20_ci95_low"] <= table["mean"]).all())
            self.assertTrue((table["mean"] <= table["block20_ci95_high"]).all())

            derived = self._derived("ablation_primary_summary.csv")
            self.assertTrue(set(PRIMARY_COLUMNS).issubset(derived.columns))
            selected = derived.loc[derived.policy.eq(policy)]
            self.assertEqual(set(selected.method), METHODS)
            self.assertEqual(len(selected), len(table.loc[table.metric.isin({
                "physical_relative_l2",
                "normalized_relative_l2",
                "truth_fluctuation_normalized_l2",
                "physical_relative_l2_excluding_sensors",
                "normalized_relative_l2_excluding_sensors",
                "pointwise_correlation",
            })]))
            self.assertFalse(selected.duplicated(["policy", "method", "metric", "target"]).any())
            self.assertTrue(np.isfinite(selected[["n", "mean", "block20_ci95_low", "block20_ci95_high"]].to_numpy(dtype=float)).all())

    def test_raw_highband_summary_has_exact_state_counts_and_complete_methods(self) -> None:
        raw = self._raw_highband()
        self.assertEqual(set(raw.policy), POLICIES)
        self.assertEqual(set(raw.method), METHODS)
        self.assertEqual(set(raw.field), {"CH4", "CO", "T", "U1", "p"})
        self.assertTrue((raw.n == 1000).all())
        self.assertFalse(raw.duplicated(["policy", "method", "field", "metric"]).any())
        numeric = ["n", "mean", "block20_ci95_low", "block20_ci95_high"]
        self.assertTrue(np.isfinite(raw[numeric].to_numpy(dtype=float)).all())

        derived = self._derived("ablation_highband_summary.csv")
        self.assertTrue(set(HIGHBAND_COLUMNS).issubset(derived.columns))
        self.assertFalse(derived.duplicated(["policy", "method", "field", "metric"]).any())
        self.assertEqual(set(derived.policy), POLICIES)
        self.assertEqual(set(derived.method), METHODS)
        self.assertTrue((derived.n == 1000).all())

    def test_raw_cache_identities_align_across_methods_and_checkpoint_policies(self) -> None:
        frames = []
        identity_columns = ["truth_sha256", "sensor_sha256", "time_index", "generation_seed"]
        for policy in sorted(POLICIES):
            table = pd.read_csv(EVALUATION_ROOT / policy / "metrics" / "cache_audit.csv")
            table["policy"] = policy
            self.assertEqual(len(table), 6000)
            self.assertEqual(set(table.method), METHODS)
            self.assertFalse(table.duplicated(["method", "snapshot"]).any())
            self.assertTrue(table.groupby("method").snapshot.nunique().eq(1000).all())
            self.assertTrue(table.groupby("snapshot")[identity_columns].nunique().eq(1).all().all())
            frames.append(table)

        both = pd.concat(frames, ignore_index=True)
        self.assertEqual(len(both), 12000)
        self.assertTrue(both.groupby("snapshot")[identity_columns].nunique().eq(1).all().all())
        self.assertEqual(both.groupby(["policy", "method"]).size().min(), 1000)

        identity = both.set_index(["policy", "method", "snapshot"])[identity_columns].sort_index()
        for method in sorted(METHODS):
            last = identity.loc[("last", method)]
            best = identity.loc[("best", method)]
            pd.testing.assert_frame_equal(last, best, check_dtype=False)

        compact_identity = self._derived("evaluation_state_identity.csv")
        self.assertEqual(len(compact_identity), 1000)
        self.assertEqual(compact_identity.snapshot.nunique(), 1000)
        self.assertEqual(compact_identity.time_index.nunique(), 1000)
        self.assertTrue(compact_identity[["truth_sha256", "sensor_sha256"]].notna().all().all())

    def test_state_validator_rejects_duplicates_incomplete_groups_and_nonfinite_values(self) -> None:
        valid = pd.DataFrame(
            {
                "method": "A0",
                "metric": "physical_relative_l2",
                "target": "Unobserved_mean",
                "snapshot": np.arange(1000),
                "time_index": np.arange(1000) + 5,
                "value": np.linspace(0.1, 0.2, 1000),
            }
        )
        self.assertIsNone(validate_state_table(valid))

        duplicate = pd.concat([valid, valid.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_state_table(duplicate)

        with self.assertRaisesRegex(ValueError, "1000"):
            validate_state_table(valid.iloc[:-1].copy())

        nonfinite = valid.copy()
        nonfinite.loc[0, "value"] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            validate_state_table(nonfinite)

    def test_primary_regression_anchors_keep_report_precision(self) -> None:
        expected = {
            "last": {
                "A0": ("0.106321", "0.104495", "0.108229"),
                "A1": ("0.078705", "0.077359", "0.080195"),
                "A2": ("0.126974", "0.124868", "0.129248"),
                "A3": ("0.144703", "0.142757", "0.146664"),
                "A4": ("0.104294", "0.102412", "0.106302"),
                "A5": ("0.354951", "0.351422", "0.358696"),
            },
            "best": {
                "A0": ("0.107893", "0.106044", "0.109840"),
                "A1": ("0.078561", "0.077237", "0.079971"),
                "A2": ("0.129288", "0.127076", "0.131656"),
                "A3": ("0.146284", "0.144486", "0.148103"),
                "A4": ("0.104967", "0.103069", "0.106949"),
                "A5": ("0.356403", "0.352764", "0.360126"),
            },
        }
        for policy, methods in expected.items():
            raw = self._raw_summary(policy)
            row = raw.loc[
                raw.method.isin(METHODS)
                & raw.metric.eq("physical_relative_l2")
                & raw.target.eq("Unobserved_mean")
            ].set_index("method")
            self.assertEqual(set(row.index), METHODS)
            for method, anchors in methods.items():
                observed = tuple(
                    f"{float(row.loc[method, column]):.6f}" for column in ("mean", "block20_ci95_low", "block20_ci95_high")
                )
                self.assertEqual(observed, anchors, f"{policy}/{method}")

            derived = self._derived("ablation_primary_summary.csv")
            drow = derived.loc[
                derived.policy.eq(policy)
                & derived.method.isin(METHODS)
                & derived.metric.eq("physical_relative_l2")
                & derived.target.eq("Unobserved_mean")
            ].set_index("method")
            for method, anchors in methods.items():
                observed = tuple(
                    f"{float(drow.loc[method, column]):.6f}"
                    for column in ("mean", "block20_ci95_low", "block20_ci95_high")
                )
                self.assertEqual(observed, anchors, f"derived {policy}/{method}")

    def test_paired_regression_anchors_keep_ci_precision(self) -> None:
        expected_primary = {
            "last": ("-0.002027", "-0.002634", "-0.001416"),
            "best": ("-0.002926", "-0.003527", "-0.002308"),
        }
        for policy, anchors in expected_primary.items():
            raw = pd.read_csv(EVALUATION_ROOT / policy / "metrics" / "paired_differences.csv")
            row = raw.loc[
                raw.method.eq("A4")
                & raw.baseline.eq("A0")
                & raw.metric.eq("physical_relative_l2")
                & raw.target.eq("Unobserved_mean")
            ]
            self.assertEqual(len(row), 1)
            observed = tuple(
                f"{float(row.iloc[0][column]):.6f}"
                for column in ("mean_difference", "block20_ci95_low", "block20_ci95_high")
            )
            self.assertEqual(observed, anchors, policy)

            derived = self._derived("ablation_primary_paired.csv")
            drow = derived.loc[
                derived.policy.eq(policy)
                & derived.method.eq("A4")
                & derived.baseline.eq("A0")
                & derived.metric.eq("physical_relative_l2")
                & derived.target.eq("Unobserved_mean")
            ]
            self.assertEqual(len(drow), 1)
            observed = tuple(
                f"{float(drow.iloc[0][column]):.6f}"
                for column in ("mean_difference", "block20_ci95_low", "block20_ci95_high")
            )
            self.assertEqual(observed, anchors, f"derived {policy}")

        expected_highband = {
            "last": ("0.951877", "0.909735", "1.001314"),
            "best": ("0.983120", "0.942137", "1.030775"),
        }
        raw = pd.read_csv(EVALUATION_ROOT / "high_frequency" / "paired_high_frequency.csv")
        derived = self._derived("ablation_highband_paired.csv")
        for policy, anchors in expected_highband.items():
            for table, label in ((raw, "raw"), (derived, "derived")):
                row = table.loc[
                    table.policy.eq(policy)
                    & table.method.eq("A4")
                    & table.baseline.eq("A0")
                    & table.field.eq("U1")
                    & table.metric.eq("highband_error_relative_l2")
                ]
                self.assertEqual(len(row), 1, f"{label}/{policy}")
                observed = tuple(
                    f"{float(row.iloc[0][column]):.6f}"
                    for column in ("method_mean_minus_baseline", "block20_ci95_low", "block20_ci95_high")
                )
                self.assertEqual(observed, anchors, f"{label}/{policy}")

    def test_iid_direction_and_canonical_mode_sum_hann_separation_are_retained(self) -> None:
        raw = self._raw_highband()
        for policy in sorted(POLICIES):
            for field in ["CH4", "CO", "T", "U1", "p"]:
                rows = raw.loc[
                    raw.policy.eq(policy)
                    & raw.field.eq(field)
                    & raw.metric.eq("highband_error_relative_l2")
                ].set_index("method")
                self.assertGreater(float(rows.loc["A4", "mean"]), float(rows.loc["A0", "mean"]), f"{policy}/{field}")

        metadata = _json(EVALUATION_ROOT / "high_frequency" / "analysis_metadata.json")
        self.assertTrue(metadata["validation"]["canonical_mode_sum_shell_mean_match"])
        self.assertTrue(metadata["validation"]["canonical_mode_sum_total_energy_match"])
        self.assertIn("separately from mode sums", metadata["grid"]["canonical_shell_weighting"])
        self.assertEqual(metadata["grid"]["high_shell_count"], 68)
        self.assertEqual(metadata["grid"]["high_mode_count"], 17704)

        metrics = set(raw.metric)
        self.assertIn("canonical_shellmean_high_energy_ratio", metrics)
        self.assertIn("reconstruction_to_truth_high_energy_ratio", metrics)
        self.assertIn("hann_reconstruction_to_truth_high_energy_ratio", metrics)
        self.assertIn("hann_highband_error_relative_l2", metrics)
        self.assertTrue(any(metric.startswith("hann_") for metric in metrics))
        self.assertTrue(any(not metric.startswith("hann_") for metric in metrics))
        selected = raw.loc[
            raw.policy.eq("last")
            & raw.method.eq("A0")
            & raw.field.eq("U1")
            & raw.metric.isin(
                {
                    "canonical_shellmean_high_energy_ratio",
                    "reconstruction_to_truth_high_energy_ratio",
                    "hann_reconstruction_to_truth_high_energy_ratio",
                }
            )
        ].set_index("metric")
        self.assertNotEqual(
            float(selected.loc["canonical_shellmean_high_energy_ratio", "mean"]),
            float(selected.loc["reconstruction_to_truth_high_energy_ratio", "mean"]),
        )
        self.assertNotEqual(
            float(selected.loc["reconstruction_to_truth_high_energy_ratio", "mean"]),
            float(selected.loc["hann_reconstruction_to_truth_high_energy_ratio", "mean"]),
        )

    def test_admissibility_events_and_deterministic_control_remain_separate(self) -> None:
        expected_events = {
            ("last", "A4"): [(291, 2769, -1648.564697), (753, 7604, -1397.456299), (935, 9436, -966.496826)],
            ("last", "A5"): [(129, 1258, -4.269897)],
            ("best", "A4"): [(291, 2769, -1764.966797), (753, 7604, -1429.140625), (935, 9436, -970.839600)],
        }
        for policy in sorted(POLICIES):
            state = pd.read_csv(
                EVALUATION_ROOT / policy / "metrics" / "per_state_metrics.csv",
                usecols=["method", "snapshot", "time_index", "metric", "target", "value"],
            )
            temperature = state.loc[
                state.metric.eq("reconstruction_minimum")
                & state.target.eq("T")
                & state.value.le(0)
            ]
            for method in sorted(METHODS):
                observed = list(
                    temperature.loc[temperature.method.eq(method), ["snapshot", "time_index", "value"]]
                    .itertuples(index=False, name=None)
                )
                observed = [(int(snapshot), int(time_index), round(float(value), 6)) for snapshot, time_index, value in observed]
                self.assertEqual(observed, expected_events.get((policy, method), []), f"{policy}/{method}")

            pressure = state.loc[
                state.metric.eq("reconstruction_minimum")
                & state.target.eq("p")
                & state.value.le(0)
            ]
            self.assertTrue(pressure.empty)

        names = self._derived("ablation_name_map.csv")
        self.assertEqual(set(names.internal_run), METHODS)
        self.assertEqual(set(names.loc[names.main_column, "internal_run"]), set(STOCHASTIC_ORDER))
        self.assertFalse(bool(names.loc[names.internal_run.eq("A1"), "main_column"].iloc[0]))
        self.assertEqual(set(UTILITY_METHODS), METHODS)
        self.assertEqual(DISPLAY_INFO["A1"]["label"], "Deterministic regression")

        deterministic = self._derived("deterministic_control_summary.csv")
        self.assertEqual(set(deterministic.method), {"A0", "A1"})
        self.assertEqual(set(deterministic.policy), POLICIES)

    def test_display_source_keeps_package_b_and_package_a_provenance_distinct(self) -> None:
        display = self._derived("figure5_v7_display_source.csv")
        self.assertEqual(set(display.evidence_package.dropna()), {"Package A", "Package B"})
        self.assertTrue(display.source_sha256.notna().all())
        self.assertTrue(display.source_path.notna().all())

        package_b = display.loc[display.evidence_package.eq("Package B")]
        self.assertTrue(package_b.checkpoint_sha256.astype(str).str.startswith("857a505ff96c").any())
        benchmark = pd.read_csv(DERIVED_ROOT / "benchmark_main_f.csv")
        dmf = benchmark.loc[benchmark.method.eq("DMF-Gen")].iloc[0]
        self.assertEqual(f"{float(dmf.mean_unobserved_relative_l2):.3f}", "0.117")
        self.assertTrue((package_b.method != "A0").all())

        package_a = display.loc[display.evidence_package.eq("Package A")]
        self.assertTrue(package_a.internal_key.astype(str).str.startswith("ablation_").any())
        self.assertTrue(package_a.cohort_id.astype(str).str.contains("ablation_condT_1000").any())


if __name__ == "__main__":
    unittest.main()
