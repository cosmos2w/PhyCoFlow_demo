from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from Dis_SI_Process.utils.figure5_v41_panels import _shared_log_error_limits, draw_accuracy_cost
from Dis_SI_Process.utils.figure5_zeroh_matched_v42_data import load_superres_matched


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((REPO_ROOT / "Dis_SI_Process" / "configs" / "mixed_hml_matched_v43.yaml").read_text())


def _data():
    return load_superres_matched(CONFIG, REPO_ROOT)


def test_mixed_hml_formal_coverage_and_domain():
    data = _data()
    assert list(data["uq_crps"]["method"]) == ["DMF-Gen", "FFM-Perceiver"]
    assert list(data["cost_native"]["method"]) == ["DMF-Gen", "FFM-Perceiver", "MLP-RBF", "Senseiver"]
    assert len(data["uq_crps_samples"]) == 400
    assert data["uq"]["manifest"]["draws_per_state"] == 64
    assert set(data["cost_native"]["N"].astype(int)) == {16384}
    assert set(data["cost_native"]["error_n"].astype(int)) == {300}


def test_mixed_hml_training_metric_is_canonical_three_resolution_mix():
    data = _data()
    assert set(data["training_cost"]["resolution_weights"]) == {"L=0.333333333333;M=0.333333333333;H=0.333333333333"}
    for resolution in "LMH":
        assert f"{resolution}_median_ms" in data["training_cost"]
        assert f"{resolution}_stability_fraction" in data["training_cost"]


def test_mixed_hml_cost_panels_are_loglog():
    data = _data()
    limits = _shared_log_error_limits(data["cost_native"], data["training_cost"])
    for panel, table in (("c", data["cost_native"]), ("d", data["training_cost"])):
        fig, ax = plt.subplots()
        draw_accuracy_cost(ax, table, CONFIG, title="test", xlabel="test", panel_label=panel, ylabel="test", ylim=limits)
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close(fig)
