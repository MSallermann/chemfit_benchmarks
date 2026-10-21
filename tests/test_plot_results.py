from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch


PLOT_RESULTS = Path(__file__).resolve().parents[1] / "plot_results.py"
SPEC = importlib.util.spec_from_file_location("plot_results", PLOT_RESULTS)
assert SPEC is not None and SPEC.loader is not None
plot_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_results)


def test_linear_ideal_uses_every_observed_x_coordinate(tmp_path: Path) -> None:
    results = [
        {
            "configuration": {
                "name": "scaling",
                "backend": "thread",
                "task_kind": "sleep",
                "task_duration": 0.01,
                "components": 1,
                "graph_depth": 1,
                "evaluations": 1,
                "batch_size": 1,
                "workers": workers,
            },
            "metrics": {"speedup": float(workers)},
        }
        for workers in (1, 2, 4, 8)
    ]
    figure = MagicMock()
    axes = MagicMock()

    with patch("matplotlib.pyplot.subplots", return_value=(figure, axes)):
        plot_results._plot(
            results,
            tmp_path,
            filename="speedup.png",
            x_field="workers",
            metric="speedup",
            ylabel="Speedup",
            ideal="linear",
        )

    ideal_call = next(
        call for call in axes.plot.call_args_list if call.kwargs.get("label") == "ideal"
    )
    assert ideal_call.args[:2] == ([1.0, 2.0, 4.0, 8.0], [1.0, 2.0, 4.0, 8.0])


def test_weak_scaling_points_share_one_series(tmp_path: Path) -> None:
    results = [
        {
            "configuration": {
                "name": "weak-scaling",
                "backend": "thread",
                "task_kind": "sleep",
                "task_duration": 0.1,
                "components": 32,
                "graph_depth": 1,
                "evaluations": workers,
                "batch_size": workers,
                "workers": workers,
                "scaling": "weak",
            },
            "metrics": {"weak_scaling_efficiency": 1.0},
        }
        for workers in (1, 2, 4, 8)
    ]
    figure = MagicMock()
    axes = MagicMock()

    with patch("matplotlib.pyplot.subplots", return_value=(figure, axes)):
        plot_results._plot(
            results,
            tmp_path,
            filename="weak.png",
            x_field="workers",
            metric="weak_scaling_efficiency",
            ylabel="Weak-scaling efficiency",
            ideal="unity",
        )

    series_call = next(call for call in axes.plot.call_args_list if call.kwargs)
    assert series_call.args[:2] == ([1.0, 2.0, 4.0, 8.0], [1.0, 1.0, 1.0, 1.0])
    assert "evals/worker=1" in series_call.kwargs["label"]
    axes.set_xlabel.assert_called_once_with("Worker threads")


def test_scaling_metrics_use_smallest_measured_worker_count() -> None:
    results = [
        {
            "configuration": {
                "name": "scaling",
                "backend": "mpi",
                "task_kind": "busy",
                "task_duration": 0.1,
                "components": 1024,
                "graph_depth": 1,
                "evaluations": 32,
                "batch_size": 32,
                "workers": workers,
                "scaling": "strong",
            },
            "metrics": {"median_runtime_seconds": runtime},
        }
        for workers, runtime in ((144, 20.0), (288, 10.0), (576, 5.0))
    ]

    plot_results._derive_scaling_metrics(results)

    assert [result["metrics"]["speedup"] for result in results] == [1.0, 2.0, 4.0]
    assert [result["metrics"]["parallel_efficiency"] for result in results] == [
        1.0,
        1.0,
        1.0,
    ]


def test_mpi_batching_settings_identify_plot_series() -> None:
    base = {
        "name": "scaling",
        "backend": "mpi",
        "task_kind": "busy",
        "task_duration": 0.1,
        "components": 1024,
        "graph_depth": 1,
        "evaluations": 32,
        "batch_size": 32,
        "workers": 1008,
        "scaling": "strong",
    }
    legacy = {"configuration": dict(base), "metrics": {}}
    defaults = {
        "configuration": {
            **base,
            "mpi_message_task_cap": 8,
            "mpi_target_messages_per_worker": 8,
        },
        "metrics": {},
    }
    tuned = {
        "configuration": {
            **base,
            "mpi_message_task_cap": 16,
            "mpi_target_messages_per_worker": 2,
        },
        "metrics": {},
    }

    assert plot_results._scaling_series_key(legacy) == plot_results._scaling_series_key(
        defaults
    )
    assert plot_results._scaling_series_key(tuned) != plot_results._scaling_series_key(
        defaults
    )
    label = plot_results._series_label(tuned, "workers")
    assert "message task cap=16" in label
    assert "target messages/worker=2" in label
