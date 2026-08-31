from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from Dis_SI_Process.utils.figure5_v41_panels import _shared_log_error_limits, draw_accuracy_cost
from Dis_SI_Process.utils.figure5_v41_style import apply_style
from Dis_SI_Process.utils.figure5_v42_data import load_figure5_v42_data


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((REPO_ROOT / "Dis_SI_Process" / "configs" / "figure5_v42.yaml").read_text())


def _data():
    return load_figure5_v42_data(CONFIG, REPO_ROOT)[0]


def test_panel_d_restores_training_update_time_and_preserves_v4_coordinates():
    data = _data()
    assert data["training_metric"] == "training_update_time_ms"
    original = pd.read_csv(data["run_metadata"]["training"]["directory"] / "training_cost_summary.csv")
    original = original.loc[original["status"].astype(str).str.lower().eq("ok")].set_index("method")
    revised = data["training_cost"].set_index("method")
    assert np.array_equal(
        original[["cost_value", "cost_low", "cost_high"]].to_numpy(float),
        revised.loc[original.index, ["cost_value", "cost_low", "cost_high"]].to_numpy(float),
    )
    assert revised.loc["DMF-Gen", "cost_value"] == 527.5089871138334


def test_geofno_is_clean_formal_two_gpu_wall_timing_and_latent_is_unavailable():
    data = _data()
    geo = data["geofno_timing"]
    assert geo["manifest"]["protocol"]["promoted_metric"] == "synchronized wall ms/global optimizer update"
    assert geo["qa"]["gpu_clean_before"] is True
    assert geo["qa"]["gpu_clean_after"] is True
    assert int(geo["summary"].iloc[0]["measured_updates"]) == 100
    table = data["training_cost"].set_index("method")
    assert int(table.loc["Geo-FNO", "device_count"]) == 2
    assert table.loc["Latent FM", "status"] == "unavailable"


def test_panel_d_renderer_is_loglog():
    data = _data()
    apply_style(CONFIG["style"]["font_family"])
    fig, ax = plt.subplots()
    draw_accuracy_cost(ax, data["training_cost"], CONFIG, title="test", xlabel="test", panel_label="d", ylabel="test", ylim=_shared_log_error_limits(data["cost_native"], data["training_cost"]))
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    plt.close(fig)
