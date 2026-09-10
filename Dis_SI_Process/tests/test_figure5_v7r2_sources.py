"""Regression checks for the additive Figure 5 V7R2 source package.

These checks consume the generated local release tables and manifests.  They do
not load a model, launch inference, or repeat the 7,000-cache derivation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
DERIVED_ROOT = PACKAGE_ROOT / "results" / "derived" / "20260910_1707"
METHODS = {"A0", "A1", "A2", "A3", "A4", "A5", "Senseiver"}
MAIN_METHODS = {"A0", "A2", "A3", "A5", "A4", "Senseiver"}
FIELDS = {"CH4", "CO", "T", "U1", "p"}
REPORT_HASH = "84d5c9fc708988f51f95d4668ef84786822367570e19bd20f71c466e632a1138"
SENSEIVER_HASH = "b055ccc8ad86ab2d4d44a527620f4c0db06009c26de26d8d52db44065fc48503"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Figure5V7R2SourceTest(unittest.TestCase):
    def table(self, name: str) -> pd.DataFrame:
        path = DERIVED_ROOT / f"{name}.csv"
        self.assertTrue(path.is_file(), f"missing V7R2 source table: {path}")
        return pd.read_csv(path, low_memory=False)

    def test_source_gate_and_lineage_pass(self) -> None:
        manifest = read_json(DERIVED_ROOT / "source_manifest.json")
        qa = read_json(DERIVED_ROOT / "source_qa.json")
        self.assertEqual(manifest["schema_version"], "figure5-v7r2-source-1")
        self.assertEqual(manifest["status"], "pass")
        self.assertEqual(qa["status"], "pass", qa.get("errors"))
        self.assertEqual(qa.get("errors"), [])
        self.assertEqual(manifest["evaluation_report_sha256"], REPORT_HASH)
        self.assertEqual(manifest["identity_proof"]["array_exact_mismatch_counts"], {
            "truth_phys": 0, "coords_phys": 0, "obs_indices": 0,
            "obs_field_ids": 0, "obs_values_norm": 0,
        })
        self.assertEqual(set(manifest["inherited_panel_sources"]), {"a", "b", "c", "f"})
        self.assertTrue(all(item["evidence_package"] == "PackageB" for values in manifest["inherited_panel_sources"].values() for item in values))
        self.assertTrue((DERIVED_ROOT / "checkpoint_provenance.csv").is_file())

    def test_checkpoint_policy_and_identity(self) -> None:
        checkpoints = self.table("checkpoint_provenance")
        self.assertEqual(set(checkpoints.method), METHODS)
        self.assertEqual(set(checkpoints.policy), {"last"})
        senseiver = checkpoints.loc[checkpoints.method.eq("Senseiver")].iloc[0]
        self.assertEqual(int(senseiver.epoch), 5000)
        self.assertEqual(senseiver.sha256, SENSEIVER_HASH)
        self.assertTrue(all(checkpoints.display_label.astype(str).str.len() > 0))

    def test_reconstruction_states_and_summaries(self) -> None:
        states = self.table("reconstruction_states")
        self.assertEqual(set(states.method), METHODS)
        self.assertEqual(set(states.field), FIELDS | {"Unobserved_mean", "All_fields_mean"})
        self.assertEqual(states.groupby(["method", "field"]).size().min(), 1000)
        self.assertEqual(states.groupby(["method", "field"]).size().max(), 1000)
        self.assertFalse(states.duplicated(["method", "field", "snapshot"]).any())
        self.assertTrue(np.isfinite(states.value.to_numpy(float)).all())

        summary = self.table("reconstruction_summary")
        self.assertEqual(len(summary), len(METHODS) * 7)
        self.assertTrue((summary.n == 1000).all())
        self.assertFalse(summary.duplicated(["method", "field"]).any())
        self.assertTrue(np.isfinite(summary[["mean", "median", "q25", "q75", "block20_ci95_low", "block20_ci95_high"]].to_numpy(float)).all())
        accepted = summary[summary.method.isin(METHODS - {"Senseiver"})]
        self.assertTrue(accepted.ci_method.astype(str).str.startswith("accepted ").all())
        senseiver = summary[(summary.method == "Senseiver") & (summary.field == "Unobserved_mean")].iloc[0]
        self.assertAlmostEqual(float(senseiver["mean"]), 0.1429897546212702, places=5)

    def test_highband_population_and_accepted_parity(self) -> None:
        states = self.table("highband_states")
        self.assertEqual(set(states.method), METHODS)
        self.assertEqual(set(states.field), FIELDS)
        self.assertEqual(states.groupby(["method", "field"]).size().min(), 1000)
        self.assertEqual(states.groupby(["method", "field"]).size().max(), 1000)
        self.assertFalse(states.duplicated(["method", "field", "snapshot"]).any())
        self.assertTrue(np.isfinite(states[["value", "canonical_power_ratio", "canonical_lsd_db"]].to_numpy(float)).all())

        summary = self.table("highband_summary")
        self.assertEqual(len(summary), len(METHODS) * len(FIELDS))
        self.assertTrue((summary.n == 1000).all())
        for column in ("canonical_power_mean", "canonical_lsd_mean"):
            self.assertTrue(np.isfinite(summary[column].to_numpy(float)).all(), column)

    def test_all_field_absolute_spectra_and_diagnostics(self) -> None:
        spectra = self.table("spectra_population")
        self.assertEqual(set(spectra.method), METHODS | {"Truth"})
        self.assertEqual(set(spectra.field), FIELDS)
        self.assertEqual(spectra.groupby(["method", "field"]).size().min(), 198)
        self.assertEqual(spectra.groupby(["method", "field"]).size().max(), 198)
        self.assertTrue((spectra.groupby(["method", "field"])["n"].first() == 1000).all())
        self.assertTrue((spectra["median"] > 0).all())
        self.assertTrue(spectra.groupby(["method", "field"]).shell_index.nunique().eq(198).all())

        diagnostics = self.table("spectral_diagnostics")
        self.assertEqual(len(diagnostics), len(METHODS) * len(FIELDS))
        self.assertFalse(diagnostics.duplicated(["method", "field"]).any())
        self.assertTrue((diagnostics.shell_count == 198).all())
        self.assertTrue((diagnostics.high_shell_count == 68).all())
        self.assertTrue((diagnostics.high_mode_count == 17704).all())
        self.assertTrue((diagnostics.estimator_version == "condT-high-frequency-v2").all())


if __name__ == "__main__":
    unittest.main()
