from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from Dis_SI_Process.utils.figure5_v41_data import load_figure5_v41_data
from Dis_SI_Process.utils.figure5_v41_panels import _shared_log_error_limits, draw_accuracy_cost
from Dis_SI_Process.utils.figure5_v41_style import apply_style


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((REPO_ROOT / "Dis_SI_Process" / "configs" / "figure5_v41.yaml").read_text())


def _data():
    return load_figure5_v41_data(CONFIG, REPO_ROOT)[0]


def test_uq_distribution_units_and_counts():
    data = _data()
    assert len(data["uq_crps_samples"]) == 5 * 200
    assert set(data["uq_crps_samples"]["sample_kind"]) == {"paired_held_out_state"}
    assert len(data["uq_spearman_bootstrap"]) == 5 * 2000
    assert set(data["uq_spearman_bootstrap"]["sample_kind"]) == {"temporal_moving_block_bootstrap"}
    rebuilt = data["uq_spearman_bootstrap"].groupby("method")["spearman_rho"].quantile([0.025, 0.975]).unstack()
    formal = data["uq_spread"].set_index("method")
    assert np.allclose(rebuilt[0.025], formal["spearman_ci_low"], rtol=0, atol=5e-13)
    assert np.allclose(rebuilt[0.975], formal["spearman_ci_high"], rtol=0, atol=5e-13)


def test_zeroh_source_is_complete_and_audited():
    data = _data()
    assert data["zeroh_errors"] == []
    assert len(data["zeroh"]) == 4 * 300
    assert set(data["zeroh"]["recipe"]) == {"4_ZeroH_Balanced"}
    assert data["zeroh_metadata"]["audit"]["passed"] is True


def test_main_cost_renderer_is_loglog():
    data = _data()
    apply_style(CONFIG["style"]["font_family"])
    fig, ax = plt.subplots()
    limits = _shared_log_error_limits(data["cost_native"], data["training_cost"])
    draw_accuracy_cost(
        ax,
        data["cost_native"],
        CONFIG,
        title="test",
        xlabel="test",
        panel_label="c",
        ylabel="test",
        ylim=limits,
    )
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_layout_contract_has_tight_independent_gutters_and_memory_only_e():
    layout = CONFIG["figure"]["layout"]
    assert layout["ab_wspace"] < 0.20
    assert layout["cd_wspace"] < 0.20
    assert CONFIG["figure"]["panel_map"]["e"] == "scalability_memory_only"
