"""CPU-only regression tests for paired ablation evaluation utilities."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SRC / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load('analyze_ablation_condT')
evaluation = load('evaluate_ablation_condT')


class TestAblationEvaluation(unittest.TestCase):
    def test_cache_content_digest_ignores_npz_container_details(self):
        metadata = {"method": "A1", "snapshot": 0, "generation_seed": 7}
        arrays = {
            "truth_phys": np.arange(12, dtype=np.float32).reshape(3, 4),
            "recon_phys": np.arange(12, dtype=np.float32).reshape(3, 4) + .25,
            "obs_indices": np.array([1, 4], dtype=np.int64),
            "metadata_json": np.array(json.dumps(metadata, indent=2)),
        }
        with tempfile.TemporaryDirectory() as directory:
            compressed = Path(directory) / "compressed.npz"
            plain = Path(directory) / "plain.npz"
            np.savez_compressed(compressed, **arrays)
            # Deliberately change member order and JSON whitespace.  Reordering
            # or recompressing a cache must not invalidate reusable metrics.
            np.savez(plain, metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
                     obs_indices=arrays["obs_indices"], recon_phys=arrays["recon_phys"],
                     truth_phys=arrays["truth_phys"])
            self.assertEqual(analysis.cache_content_sha256(compressed), analysis.cache_content_sha256(plain))
            changed = dict(arrays)
            changed["recon_phys"] = changed["recon_phys"].copy()
            changed["recon_phys"][0, 0] += 1
            altered = Path(directory) / "altered.npz"
            np.savez(altered, **changed)
            self.assertNotEqual(analysis.cache_content_sha256(compressed), analysis.cache_content_sha256(altered))

    def test_bootstrap_matches_explicit_circular_blocks(self):
        values = np.arange(13, dtype=float) ** 2
        for block in [1, 5, 13, 20]:
            actual = analysis.block_bootstrap_means(values, block=block, n_boot=31, seed=7)
            size = min(block, len(values))
            starts = np.random.default_rng(7).integers(0, len(values), size=(31, (len(values)+size-1)//size))
            indices = ((starts[:, :, None] + np.arange(size)) % len(values)).reshape(31, -1)[:, :len(values)]
            np.testing.assert_allclose(actual, values[indices].mean(axis=1), rtol=0, atol=1e-12)

    def test_paired_differences_use_snapshot_identity(self):
        rows = []
        for method, offset, order in [('A0', 0., [2, 0, 1]), ('A2', .2, [0, 2, 1]), ('A5', .5, [1, 0, 2])]:
            for snapshot in order:
                rows.append(dict(method=method, snapshot=snapshot, time_index=snapshot*10,
                                 metric='error', target='field', value=snapshot+1+offset))
        _, paired = analysis.summarize_rows(rows, n_boot=50)
        pairs = {(r['method'], r['baseline']): r for r in paired}
        self.assertEqual(set(pairs), {('A2', 'A0'), ('A5', 'A0'), ('A5', 'A2')})
        self.assertAlmostEqual(pairs[('A5', 'A2')]['mean_difference'], .3)
        self.assertEqual(pairs[('A5', 'A2')]['n_paired'], 3)
        self.assertAlmostEqual(pairs[('A2', 'A0')]['block20_ci95_low'], .2)

    def test_metric_identity_and_scale(self):
        truth = np.array([1., 2., 3.])
        self.assertEqual(analysis.relative_l2(truth, truth), 0.)
        self.assertAlmostEqual(analysis.relative_l2(truth, truth*1.1), .1)
        edges, _ = analysis.load_paper_edges()
        for x, y in edges.values():
            self.assertEqual((len(x), len(y)), (65, 65))
            hist = analysis.histogram(np.array([x[20], x[40]]), np.array([y[20], y[40]]), (x, y))
            self.assertEqual(analysis.jsd_base2(hist, hist), 0.)

    def test_paper_plan_and_seed(self):
        plan = evaluation.read_plan(evaluation.PLAN)
        self.assertEqual(set(plan), set(range(1000)))
        self.assertTrue(all(len(rows) == 256 for rows in plan.values()))
        self.assertEqual(evaluation.stable_seed(20260711, 'generation', 'DMF-Gen', 'Cond_T', 0), 812309383)
        self.assertEqual(set(evaluation.discover_runs()), {f'A{i}' for i in range(6)})

    def test_overflow_metric_detects_discarded_tail_mass(self):
        edges = np.linspace(0., 1., 65)
        truth = np.array([.5, .5])
        pred = np.array([.5, -1.])
        paper_truth = analysis.histogram(truth, truth, (edges, edges))
        paper_pred = analysis.histogram(pred, pred, (edges, edges))
        self.assertEqual(analysis.jsd_base2(paper_truth, paper_pred), 0.)
        extended = np.r_[-np.inf, edges, np.inf]
        full_truth = analysis.histogram(truth, truth, (extended, extended))
        full_pred = analysis.histogram(pred, pred, (extended, extended))
        self.assertGreater(analysis.jsd_base2(full_truth, full_pred), .3)
        self.assertAlmostEqual(full_pred.sum(), 1.)


if __name__ == '__main__':
    unittest.main()
