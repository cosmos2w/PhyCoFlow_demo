"""Unit checks for quantitative training-preview annotations."""

import warnings

import numpy as np

from phycoflow_reconstruction.training.preview import (
    _absolute_error_title,
    _plot_preview,
    _relative_l2_error,
)


def test_relative_l2_error_uses_field_reference_norm():
    truth = np.asarray([3.0, 4.0])
    estimate = np.asarray([0.0, 0.0])

    value = _relative_l2_error(estimate, truth)

    assert value == 1.0
    assert _absolute_error_title(value) == "Absolute error\nRelative $L_2$ = 1.000e+00"


def test_relative_l2_error_marks_zero_reference_as_unavailable():
    value = _relative_l2_error(np.ones(4), np.zeros(4))

    assert value is None
    assert _absolute_error_title(value) == "Absolute error\nRelative $L_2$ = N/A"


def test_subsampled_pointcloud_preview_draws_sensor_legend_without_warning(tmp_path):
    query_coords = np.asarray(
        [[0.0, 0.0], [0.3, 0.7], [0.8, 0.2], [1.0, 1.0]], dtype=np.float32
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        outputs = _plot_preview(
            tmp_path / "pointcloud_preview",
            prediction=np.asarray([[0.0], [0.4], [0.7], [1.0]], dtype=np.float32),
            target=np.asarray([[0.1], [0.3], [0.8], [0.9]], dtype=np.float32),
            query_coords=query_coords,
            obs_coords=np.asarray([[0.2, 0.4], [0.7, 0.8]], dtype=np.float32),
            obs_values=np.asarray([0.2, 0.8], dtype=np.float32),
            obs_fields=np.asarray([0, 0], dtype=np.int64),
            obs_valid=np.asarray([True, True]),
            field_names=("T",),
            logical_shape=(2, 3),
            epoch=1.0,
        )

    assert all(path.stat().st_size > 0 for path in outputs)
