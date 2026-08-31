from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "0_demo_TurbulentCombustion/tools/benchmark_validation_v4_scale_stress.py"
SPEC = importlib.util.spec_from_file_location("benchmark_validation_v4_scale_stress", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
V4 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V4
SPEC.loader.exec_module(V4)


class TestScaleStressV4(unittest.TestCase):
    def test_dummy_sequence_is_deterministic_sensor_prefixed_and_prefix_consistent(self) -> None:
        spec = V4.DummyQuerySpec(
            seed=17,
            dimension=3,
            low=(0.0, 0.0, 0.0),
            high=(1.0, 1.0, 0.0),
        )
        sensors = np.asarray(
            [[0.1, 0.2, 0.0], [0.7, 0.8, 0.0], [0.3, 0.4, 0.0]],
            dtype=np.float32,
        )
        small = V4.generate_dummy_query_coordinates(32, spec, sensor_coords=sensors)
        large = V4.generate_dummy_query_coordinates(96, spec, sensor_coords=sensors)

        self.assertEqual(small.shape, (32, 3))
        self.assertEqual(large.shape, (96, 3))
        np.testing.assert_array_equal(small[: len(sensors)], sensors)
        np.testing.assert_array_equal(large[:32], small)
        np.testing.assert_array_equal(
            small,
            V4.generate_dummy_query_coordinates(32, spec, sensor_coords=sensors),
        )
        self.assertTrue(np.all(small >= np.asarray(spec.low, dtype=np.float32)))
        self.assertTrue(np.all(small <= np.asarray(spec.high, dtype=np.float32)))
        self.assertEqual(V4.hash_coordinates(small), V4.hash_coordinates(small.copy()))
        self.assertEqual(
            V4.query_hash(small, spec, sensor_coords_sha256=V4.hash_coordinates(sensors)),
            V4.query_hash(small, spec, sensor_coords_sha256=V4.hash_coordinates(sensors)),
        )

    def test_query_counts_are_declared_before_adaptive_doubling(self) -> None:
        self.assertEqual(
            V4.predeclared_query_counts(),
            (100_000, 250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000),
        )
        self.assertEqual(
            V4.predeclared_query_counts(16_000_000),
            (100_000, 250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000),
        )
        with self.assertRaises(ValueError):
            V4.predeclared_query_counts(3_999_999)

    def test_actual_architecture_audit_has_only_four_scaling_eligible_methods(self) -> None:
        cfg = V4.load_config()
        rows = V4.audit_native_query_support(cfg["methods"])
        self.assertEqual([row["method"] for row in rows], list(V4.METHODS))
        eligible = {row["method"] for row in rows if row["query_scaling_eligible"]}
        self.assertEqual(eligible, {"DMF-Gen", "FFM-Perceiver", "MLP-RBF", "Senseiver"})
        for row in rows:
            self.assertTrue(row["decision_basis"])
            self.assertNotIn("loader limitation", row["decision_basis"].lower())
            self.assertTrue(row["evidence_source"])
            self.assertGreater(int(row["evidence_line"]), 0)
        geofno = next(row for row in rows if row["method"] == "Geo-FNO")
        self.assertFalse(geofno["native_query_supported"])
        self.assertIn("variant=fno", geofno["decision_basis"])

    def test_strict_qa_requires_shared_hashes_and_boundary_rows(self) -> None:
        cfg = V4.load_config()
        support_rows = V4.audit_native_query_support(cfg["methods"])
        eligible = [row["method"] for row in support_rows if row["query_scaling_eligible"]]
        counts = (100_000, 250_000)
        query_rows = []
        summaries = []
        geometry = []
        repeats = []
        boundaries = []
        for method in eligible:
            for count in counts:
                query_hash = f"q-{count}"
                query_rows.append({"method": method, "N": count, "query_sha256": query_hash})
                summaries.append(
                    {
                        "method": method,
                        "N": count,
                        "status": "ok",
                        "median_latency_ms": 1.0,
                        "latency_iqr_ms": 0.1,
                        "repeats": 30,
                        "throughput_only": True,
                        "accuracy_claim": False,
                        "query_sha256": query_hash,
                    }
                )
                geometry.append({"method": method, "N": count, "geometry_prepare_ms": 0.0})
                repeats.append({"method": method, "N": count})
            boundaries.append(
                {
                    "method": method,
                    "largest_success_N": 250_000,
                    "first_failure_N": "",
                }
            )
        qa = V4.strict_scale_qa(
            support_rows=support_rows,
            summary_rows=summaries,
            repeat_rows=repeats,
            geometry_rows=geometry,
            boundary_rows=boundaries,
            query_rows=query_rows,
            candidate_counts=V4.predeclared_query_counts(),
            query_hashes_by_count={100_000: "q-100000", 250_000: "q-250000"},
            expected_eligible=eligible,
        )
        self.assertEqual(qa["status"], "pass")
        self.assertTrue(qa["shared_query_hash_per_count"])
        self.assertTrue(qa["largest_success_first_failure_recorded"])


if __name__ == "__main__":
    unittest.main()
