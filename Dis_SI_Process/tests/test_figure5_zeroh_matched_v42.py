from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from Dis_SI_Process.utils.figure5_v41_panels import _shared_log_error_limits, draw_accuracy_cost
from Dis_SI_Process.utils.figure5_zeroh_matched_v42_data import load_zeroh_matched_v42


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((REPO_ROOT / "Dis_SI_Process" / "configs" / "zeroh_matched_v42.yaml").read_text())


def _data():
    return load_zeroh_matched_v42(CONFIG, REPO_ROOT)


def test_matched_method_coverage_and_formal_sources():
    data = _data()
    assert list(data["uq_crps"]["method"]) == ["DMF-Gen", "FFM-Perceiver"]
    assert list(data["cost_native"]["method"]) == ["DMF-Gen", "FFM-Perceiver", "MLP-RBF", "Senseiver"]
    assert set(data["uq_spread"]["method"]).isdisjoint({"MLP-RBF", "Senseiver"})
    assert data["uq"]["qa"]["status"] == "pass"
    assert data["cost"]["qa"]["status"] == "pass"
    assert data["cost"]["qa"]["no_cond_t_cost_reuse"] is True


def test_uq_counts_and_bootstrap_reconstruction():
    data = _data()
    assert len(data["uq_crps_samples"]) == 2 * 200
    assert set(data["uq_crps_samples"]["sample_kind"]) == {"paired_unique_case_time_state"}
    assert data["uq"]["manifest"]["draws_per_state"] == 64
    assert len(data["uq_spearman_bootstrap"]) == 2 * 2000
    rebuilt = data["uq_spearman_bootstrap"].groupby("method")["spearman_rho"].quantile([0.025, 0.975]).unstack()
    formal = data["uq_spread"].set_index("method")
    assert np.allclose(rebuilt[0.025], formal["spearman_ci_low"], rtol=0, atol=5e-13)
    assert np.allclose(rebuilt[0.975], formal["spearman_ci_high"], rtol=0, atol=5e-13)


def test_cost_domain_and_accuracy_cohort():
    data = _data()
    assert set(data["cost_native"]["N"].astype(int)) == {16384}
    assert set(data["cost_native"]["sensor_count"].astype(int)) == {256}
    assert set(data["cost_native"]["error_n"].astype(int)) == {300}
    assert set(data["training_cost"]["batch_size"].astype(int)) == {512}
    assert set(data["training_cost"]["resolution_weights"]) == {"L=0.5;M=0.5;H=0"}


def test_cost_renderers_are_loglog():
    data = _data()
    limits = _shared_log_error_limits(data["cost_native"], data["training_cost"])
    for panel, table in (("c", data["cost_native"]), ("d", data["training_cost"])):
        fig, ax = plt.subplots()
        draw_accuracy_cost(ax, table, CONFIG, title="test", xlabel="test", panel_label=panel, ylabel="test", ylim=limits)
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close(fig)
