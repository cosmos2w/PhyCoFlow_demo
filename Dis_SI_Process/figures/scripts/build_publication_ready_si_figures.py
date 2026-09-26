"""Render the four publication-ready SI figures from frozen paper evaluation artifacts.

Run: conda run -n fig python Dis_SI_Process/figures/scripts/build_publication_ready_si_figures.py
No training, checkpoint loading, or prediction inference is performed here.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SI = ROOT / "Dis_SI_Process" / "SI_Source" / "figures_final"
OUT = ROOT / "Dis_SI_Process" / "figures" / "generated" / "publication_ready_si_20260926"
MIXED = ROOT / "1_SubTask_SuperResolution" / "Save_TrainedModel" / "_TrainedModels"
COMB = ROOT / "0_demo_TurbulentCombustion" / "Save_TrainedModel" / "_TrainedModels"
MR = MIXED / "_Process_Results"
CR = COMB / "_Process_Results"
COLORS = {"DMF-Gen": "#C94053", "FFM-Perceiver": "#4C86A6",
          "Senseiver": "#8D9BAD", "MLP-RBF": "#4C9E91"}
FIELD_LABELS = [r"$Y_{\rm CH_4}$", r"$Y_{\rm CO}$", "$T$", "$U_1$", "$p$"]
FIELD_KEYS = ["CH4", "CO", "T", "U1", "p"]
CONDITIONS = ["Cond_T", "Cond_TU1", "Cond_COTU1P"]
COND_LABELS = ["T", "$T+U_1$", "CO+$T+U_1+p$"]

for font_path in (Path.home() / ".local" / "share" / "fonts" / "Arial").glob("*.TTF"):
    font_manager.fontManager.addfont(str(font_path))

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.0,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "axes.titlesize": 7.4, "axes.labelsize": 7.0,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.55, "lines.linewidth": 1.05,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SI.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(SI / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def panel_label(ax, label: str) -> None:
    ax.text(-0.13, 1.12, label, transform=ax.transAxes, fontsize=8,
            fontweight="bold", ha="left", va="bottom")


def render_mixed_resolution() -> None:
    """300 paired case-time identities, 256 nested sensors, saved best checkpoints."""
    a = pd.read_csv(MR / "QuestionA_DataBenefit" / "QuestionA_per_snapshot_2026-08-02_12-50.csv")
    b = pd.read_csv(MR / "QuestionB_ZeroH" / "QuestionB_per_snapshot_2026-08-02_12-50.csv")
    data = pd.concat([a, b[b.recipe != "1_H_only"]], ignore_index=True)
    assert len(data) == 6000 and data.status.eq("ok").all()
    assert data.sensor_count.eq(256).all() and data.checkpoint_kind.eq("best").all()
    assert data.groupby(["model", "recipe"]).size().eq(300).all()
    wave = pd.read_csv(MR / "MultiscaleWavelet" / "MultiscaleWavelet_per_snapshot_20260802_1250.csv")
    wave = wave[(wave.recipe == "5_ZeroH_MRich") & wave.model_label.isin(["DMF-Gen", "Senseiver"])]
    assert wave.groupby(["model_label", "scale_group"]).size().eq(300).all()
    recipes = ["1_H_only", "2_H_limited", "3_Mixed_HML", "4_ZeroH_Balanced", "5_ZeroH_MRich"]
    labels = ["H only", "H limited", "Mixed HML", "Zero H balanced", "Zero H M rich"]
    methods = list(COLORS)
    fig = plt.figure(figsize=(183 / 25.4, 113 / 25.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=.62, wspace=.35)
    ax = fig.add_subplot(gs[:, 0])
    for j, method in enumerate(methods):
        samples = [data[(data.recipe == recipe) & (data.model_label == method)].physical_rel_l2.to_numpy() for recipe in recipes]
        assert all(len(x) == 300 for x in samples)
        pos = np.arange(5) + (j - 1.5) * .17
        bp = ax.boxplot(samples, positions=pos, widths=.145, patch_artist=True,
                        showfliers=False, medianprops={"color": "#263238", "linewidth": .65})
        for box in bp["boxes"]:
            box.set(facecolor=COLORS[method], edgecolor=COLORS[method], alpha=.72, linewidth=.45)
        for key in ("whiskers", "caps"):
            for artist in bp[key]: artist.set(color=COLORS[method], linewidth=.5)
        ax.plot([], [], color=COLORS[method], linewidth=4, label=method)
    ax.set_xticks(range(5), labels, rotation=25, ha="right")
    ax.set_xlim(-.55, 4.55)
    ax.set_ylabel("Physical relative $L_2$")
    ax.grid(axis="y", color="#e4e7ea", linewidth=.45)
    ax.legend(ncol=2, loc="upper left", frameon=False, fontsize=6)
    panel_label(ax, "a")
    for i, (metric, title, ylabel) in enumerate([
        ("pattern_correlation", "Wavelet pattern correlation", "Cosine correlation"),
        ("variance_fraction_bias_pp", "Squared-energy allocation bias", "Predicted − reference (pp)"),
    ]):
        ax = fig.add_subplot(gs[i, 1])
        for j, method in enumerate(["DMF-Gen", "Senseiver"]):
            samples = [wave[(wave.model_label == method) & (wave.scale_group == scale)][metric].to_numpy()
                       for scale in ["large", "intermediate", "fine"]]
            pos = np.arange(3) + (j - .5) * .24
            bp = ax.boxplot(samples, positions=pos, widths=.21, patch_artist=True, showfliers=False,
                            medianprops={"color": "#263238", "linewidth": .7})
            for box in bp["boxes"]:
                box.set(facecolor=COLORS[method], edgecolor=COLORS[method], alpha=.75, linewidth=.5)
            for key in ("whiskers", "caps"):
                for artist in bp[key]: artist.set(color=COLORS[method], linewidth=.55)
        ax.set_xticks(range(3), ["Large", "Intermediate", "Fine"])
        ax.set_xlim(-.55, 2.55)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        if i: ax.axhline(0, color="#777", linewidth=.6, linestyle="--")
        ax.grid(axis="y", color="#e4e7ea", linewidth=.4)
        panel_label(ax, "bc"[i])
    fig.subplots_adjust(left=.085, right=.99, bottom=.17, top=.94)
    save(fig, "si01_mixed_resolution_distributions_final")


def combustion_cache(condition: str, state: int = 0) -> dict[str, np.ndarray]:
    manifest = rows(CR / "ReconstructionCache" / "ReconstructionCache_manifest_paper_full_20260711.csv")
    matches = [r for r in manifest if r["method"] == "DMF-Gen" and r["condition"] == condition
               and int(r["snapshot"]) == state and r["status"] == "ok"]
    assert len(matches) == 1
    with np.load(matches[0]["cache_path"]) as archive:
        return {key: archive[key].copy() for key in archive.files if key != "metadata_json"}


def grid_order(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert coords.shape == (40300, 3)
    idx = np.lexsort((coords[:, 0], coords[:, 1]))
    xy = coords[idx]
    x, y = xy[:, 0].reshape(100, 403), xy[:, 1].reshape(100, 403)
    assert np.all(np.diff(x, axis=1) > 0) and np.all(np.diff(y, axis=0) > 0)
    return idx, x, y


def render_combustion_fields() -> None:
    caches = [combustion_cache(cond) for cond in CONDITIONS]
    limits = json.loads((COMB / "_Process_Figures" / "_Contours" /
                         "ContourColorLimits_paper_full_20260711.json").read_text())
    truth = caches[0]["truth_phys"]
    assert all(np.array_equal(truth, cache["truth_phys"]) for cache in caches[1:])
    idx, x, y = grid_order(caches[0]["coords_phys"])
    fig, axes = plt.subplots(5, 7, figsize=(183 / 25.4, 121 / 25.4),
                             gridspec_kw={"wspace": .055, "hspace": .2})
    for field, key in enumerate(FIELD_KEYS):
        vmin, vmax = limits["physical_limits"][key]
        emax = limits["robust_error_limits"][key][1]
        cmap = "coolwarm" if key == "U1" else ("inferno" if key == "T" else "viridis")
        vals = [truth[:, field]]
        for cache in caches:
            vals.extend([cache["recon_phys"][:, field],
                         np.abs(cache["recon_phys"][:, field] - truth[:, field])])
        for col, values in enumerate(vals):
            ax = axes[field, col]
            norm = Normalize(0, emax) if col > 0 and col % 2 == 0 else Normalize(vmin, vmax)
            ax.pcolormesh(x, y, values[idx].reshape(100, 403), shading="nearest",
                          cmap="magma" if col > 0 and col % 2 == 0 else cmap,
                          norm=norm, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if field == 0:
                ax.set_title(["Reference", "Cond-$T$", "Abs. error", "Cond-$(T,U_1)$", "Abs. error",
                              "Four-channel", "Abs. error"][col], fontsize=6.7, pad=3)
            if col == 0:
                ax.set_ylabel(FIELD_LABELS[field], rotation=0, labelpad=13, va="center", fontsize=7)
    for col, label in zip([0, 1, 3, 5], "abcd"):
        axes[0, col].text(-.10, 1.34, label, transform=axes[0, col].transAxes,
                          fontsize=8, fontweight="bold", ha="left", va="bottom")
    fig.subplots_adjust(left=.07, right=.995, top=.92, bottom=.04)
    save(fig, "si02_complete_combustion_fields_final")


def render_spectral_diagnostics() -> None:
    sys.path.insert(0, str(COMB / "_Scripts"))
    from common.spectral import compare_channel_spectra_batch
    caches = [combustion_cache(cond) for cond in CONDITIONS]
    coords = caches[0]["coords_phys"]
    fig, axes = plt.subplots(5, 2, figsize=(183 / 25.4, 160 / 25.4),
                             gridspec_kw={"width_ratios": [1.35, 1], "hspace": .38, "wspace": .3})
    cond_colors = ["#E63946", "#457B9D", "#2A9D8F"]
    lsd = pd.read_csv(CR / "Spectral" / "SpectralLSD" /
                      "SpectralLSD_per_snapshot_paper_full_20260711.csv")
    lsd = lsd[(lsd.model_label == "DMF-Gen") & lsd.status.eq("ok")]
    for field, key in enumerate(FIELD_KEYS):
        ax = axes[field, 0]
        for ci, cache in enumerate(caches):
            spec = compare_channel_spectra_batch(cache["truth_phys"][:, [field]],
                  cache["recon_phys"][:, [field]], coords[:, :2], num_x=403, num_y=100,
                  coordinate_mode="auto", remove_mean=True, window="none",
                  min_shell_count=4, use_isotropic_cutoff=True, relative_epsilon=1e-12)[0]
            truth = spec["truth"]
            pred = spec["reconstruction"]
            if ci == 0:
                ax.plot(truth["wavenumber"], truth["normalized_spectral_energy"],
                        color="#20272b", label="Reference", linewidth=1.1)
            ax.plot(pred["wavenumber"], pred["normalized_spectral_energy"],
                    color=cond_colors[ci], label=COND_LABELS[ci], alpha=.92)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylabel(FIELD_LABELS[field], rotation=0, labelpad=15, va="center")
        ax.grid(color="#e9ecee", linewidth=.35)
        if field == 0: ax.set_title("Held-out state 0: normalized radial spectra", loc="left")
        if field == 4: ax.set_xlabel("Topological wavenumber (rad cell$^{-1}$)")
        bx = axes[field, 1]
        vals = [lsd[(lsd.condition == cond) & (lsd.field_name == key)].lsd_db.to_numpy()
                for cond in CONDITIONS]
        assert all(len(v) == 1000 for v in vals), (key, list(map(len, vals)))
        bp = bx.boxplot(vals, widths=.55, patch_artist=True, showfliers=False,
                        medianprops={"color": "#273137", "linewidth": .7})
        for j, box in enumerate(bp["boxes"]):
            box.set(facecolor=cond_colors[j], edgecolor=cond_colors[j], alpha=.78, linewidth=.5)
        for part in ("whiskers", "caps"):
            for artist in bp[part]: artist.set(color="#64727a", linewidth=.55)
        bx.set_xticks([1, 2, 3], ["T", "$T+U_1$", "4 fields"])
        bx.set_ylabel("LSD (dB)")
        bx.grid(axis="y", color="#e9ecee", linewidth=.35)
        if field == 0: bx.set_title("1,000 held-out evaluation states", loc="left")
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right", fontsize=5.4)
    panel_label(axes[0, 0], "a"); panel_label(axes[0, 1], "b")
    fig.subplots_adjust(left=.115, right=.99, top=.94, bottom=.07)
    save(fig, "si03_complete_spectral_diagnostics_final")


def render_conditional_ensembles() -> None:
    visual = rows(CR / "ValidationV2" / "Uncertainty" / "u2_formal_20260830_v1" / "visual_cases.csv")
    assert [int(r["state"]) for r in visual] == [0, 502, 999]
    maps = []
    for record in visual:
        with np.load(record["path"]) as z:
            maps.append({k: z[k].copy() for k in z.files})
    cache = combustion_cache("Cond_T", 0)
    scales = []
    offsets = []
    for field in range(5):
        norm = cache["truth_norm"][:, field].astype(float)
        physical = cache["truth_phys"][:, field].astype(float)
        scale = np.sum((norm - norm.mean()) * (physical - physical.mean())) / np.sum((norm - norm.mean()) ** 2)
        offset = physical.mean() - scale * norm.mean()
        assert np.max(np.abs(physical - (offset + scale * norm))) < .01
        scales.append(scale); offsets.append(offset)
    fig, axes = plt.subplots(6, 4, figsize=(183 / 25.4, 148 / 25.4),
                             gridspec_kw={"hspace": .21, "wspace": .05})
    shared_spread_max = {field: max(float(np.quantile(scales[field] * payload["ensemble_std_norm"][:, field], .99))
                                    for payload in maps) for field in (0, 3)}
    colors = json.loads((COMB / "_Process_Figures" / "_Contours" /
                         "ContourColorLimits_paper_full_20260711.json").read_text())
    for state_row, (record, payload) in enumerate(zip(visual, maps)):
        assert int(payload["state"]) == int(record["state"]) and len(payload["generation_seeds"]) == 64
        idx, x, y = grid_order(payload["coords"])
        for k, field in enumerate([0, 3]):
            row = 2 * state_row + k
            true = offsets[field] + scales[field] * payload["truth_norm"][:, field]
            mean = offsets[field] + scales[field] * payload["ensemble_mean_norm"][:, field]
            spread = scales[field] * payload["ensemble_std_norm"][:, field]
            key = FIELD_KEYS[field]
            vmin, vmax = colors["physical_limits"][key]
            emax = colors["robust_error_limits"][key][1]
            vals = [true, mean, np.abs(mean - true), spread]
            norms = [Normalize(vmin, vmax), Normalize(vmin, vmax), Normalize(0, emax),
                     Normalize(0, max(shared_spread_max[field], 1e-12))]
            for col, (values, norm) in enumerate(zip(vals, norms)):
                ax = axes[row, col]
                ax.pcolormesh(x, y, values[idx].reshape(100, 403), shading="nearest",
                              norm=norm, cmap="magma" if col > 1 else ("coolwarm" if field == 3 else "viridis"), rasterized=True)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_visible(False)
                if row == 0:
                    ax.set_title(["Reference", "Ensemble mean", "Absolute error", "Ensemble SD"][col], pad=3)
                if col == 0:
                    ax.set_ylabel(f"State {record['state']}\n{FIELD_LABELS[field]}", rotation=0,
                                  labelpad=18, va="center", fontsize=6.3)
    for col, label in enumerate("abcd"):
        axes[0, col].text(-.08, 1.22, label, transform=axes[0, col].transAxes,
                          fontsize=8, fontweight="bold", ha="left", va="bottom")
    fig.text(.995, .985, "256 temperature measurements fixed across 64 draws within each state",
             ha="right", va="top", fontsize=6.5, color="#263238")
    fig.subplots_adjust(left=.12, right=.995, top=.92, bottom=.045)
    save(fig, "si04_spatial_conditional_ensembles_final")


if __name__ == "__main__":
    render_mixed_resolution(); print("Supplementary Figure 1 complete", flush=True)
    render_combustion_fields(); print("Supplementary Figure 2 complete", flush=True)
    render_spectral_diagnostics(); print("Supplementary Figure 3 complete", flush=True)
    render_conditional_ensembles(); print("Supplementary Figure 4 complete", flush=True)
