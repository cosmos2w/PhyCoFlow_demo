"""Training monitoring keeps progress artifacts current and restart-safe."""

import json

from phycoflow_reconstruction.training.monitoring import TrainingMonitor


def test_monitor_loads_history_and_updates_loss_figure(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "history.jsonl").write_text(
        json.dumps({"step": 1, "total": 2.0}) + "\n",
        encoding="utf-8",
    )
    monitor = TrainingMonitor(
        tmp_path,
        start_step=1,
        final_step=2,
        configured_steps=2,
        steps_per_epoch=2,
        description="test:model",
        enabled=False,
        plot_every_steps=1,
    )
    monitor.record({"step": 2, "total": 1.0}, lr=1.0e-4)
    assert monitor._epoch_coordinates([1, 2, 3, 4]) == [0.5, 1.0, 1.5, 2.0]
    monitor.close()

    assert (tmp_path / "loss_history.png").stat().st_size > 0


def test_monitor_updates_detailed_coherence_figure_with_all_families(tmp_path):
    (tmp_path / "metrics").mkdir()
    monitor = TrainingMonitor(
        tmp_path,
        start_step=0,
        final_step=1,
        configured_steps=1,
        steps_per_epoch=1,
        description="post:test",
        enabled=False,
        plot_every_steps=1,
    )
    row = {
        "step": 1,
        "data_loss": 0.5,
        "coherence_loss": 6.0,
        "global_distribution.total": 1.0,
        "global_distribution.self.marginal_w2": 0.2,
        "global_distribution.mutual.pairwise_swd": 0.3,
        "global_distribution.cross.joint_topk_swd": 0.5,
        "cross_spectrum.total": 2.0,
        "cross_spectrum.same_frequency.magnitude_squared": 0.8,
        "cross_spectrum.cross_frequency.band_energy_coupling": 1.2,
        "topology.total": 3.0,
        "topology.self.betti_curves": 1.4,
        "topology.mutual.fibered_betti_curves": 1.6,
        "data_grad_norm": 0.4,
        "coherence_grad_norm": 4.0,
        "combined_grad_norm": 2.0,
        "gradient_cosine": -0.25,
        "gradient_conflict": True,
        "config_fallback_used": False,
    }
    monitor.record(row)

    assert (tmp_path / "coherence_history.png").stat().st_size > 0
    assert set(monitor._values).issuperset(
        {
            "global_distribution.total",
            "cross_spectrum.total",
            "topology.total",
            "gradient_cosine",
            "gradient_conflict",
        }
    )
    monitor.close()


def test_monitor_resets_progress_at_each_epoch(tmp_path):
    (tmp_path / "metrics").mkdir()
    monitor = TrainingMonitor(
        tmp_path,
        start_step=0,
        final_step=6,
        configured_steps=6,
        steps_per_epoch=3,
        description="test:model",
        enabled=True,
        plot_every_steps=100,
    )

    monitor.record({"step": 1, "total": 2.0})
    assert monitor.active_epoch == 1
    assert monitor.progress.total == 3
    assert monitor.progress.n == 1

    monitor.record({"step": 2, "total": 1.5})
    monitor.record({"step": 3, "total": 1.0})
    monitor.record({"step": 4, "total": 0.8})
    assert monitor.active_epoch == 2
    assert monitor.progress.total == 3
    assert monitor.progress.n == 1
    monitor.close()
