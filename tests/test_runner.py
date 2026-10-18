from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


RUNNER = Path(__file__).resolve().parents[1] / "benchmark" / "chemfit_runner.py"
ROOT = RUNNER.parents[1]
SPEC = importlib.util.spec_from_file_location("chemfit_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def make_config(**changes):
    values = {
        "name": "test",
        "backend": "serial",
        "task_kind": "sleep",
        "task_duration": 0.0,
        "components": 3,
        "evaluations": 4,
        "batch_size": 4,
        "workers": 1,
        "repeats": 2,
        "warmup_evaluations": 1,
    }
    values.update(changes)
    return runner.BenchmarkConfig(**values)


@pytest.mark.parametrize("backend", ["direct", "serial"])
@pytest.mark.parametrize("task_kind", ["sleep", "busy"])
def test_evaluate_many_result_validation(backend: str, task_kind: str) -> None:
    config = make_config(backend=backend, task_kind=task_kind)
    objective = runner.make_objective(config)
    scheduler = (
        runner.chemfit.DirectScheduler()
        if backend == "direct"
        else runner.chemfit.SerialGraphScheduler()
    )
    inputs = runner._inputs(4, 0)
    with scheduler.prepare(objective) as schedule:
        requests = [
            runner.chemfit.EvaluationRequest(value, runner.chemfit.EvaluateContext())
            for value in inputs
        ]
        results = list(schedule.evaluate_many(requests))
    assert sorted(result.index for result in results) == list(range(4))
    for result in results:
        assert not isinstance(result.value, Exception)
        expected = 3 * (inputs[result.index]["x"] + 1.0)
        assert math.isclose(float(result.value), expected)


def test_thread_run_has_repeats_and_single_worker_reference() -> None:
    config = make_config(backend="thread", workers=2)
    target, baseline = runner.run(config)
    assert target is not None
    assert baseline is not None
    assert len(target["samples_seconds"]) == 2
    assert len(baseline["samples_seconds"]) == 2
    assert target["evaluations_per_second"] > 0
    assert "first_evaluation_seconds" in target
    assert "measured_startup_overhead_seconds" in target


@pytest.mark.parametrize("backend", ["direct", "serial", "thread"])
def test_deep_layered_graph_has_expected_value(backend: str) -> None:
    workers = 2 if backend == "thread" else 1
    config = make_config(
        backend=backend,
        workers=workers,
        components=2,
        graph_depth=3,
    )
    objective = runner.make_objective(config)
    if backend == "direct":
        scheduler = runner.chemfit.DirectScheduler()
        executor = None
    elif backend == "serial":
        scheduler = runner.chemfit.SerialGraphScheduler()
        executor = None
    else:
        executor = runner.ThreadPoolExecutor(max_workers=workers)
        scheduler = runner.chemfit.ExecutorGraphScheduler(executor=executor)
    try:
        with scheduler.prepare(objective) as schedule:
            # 2 -> 2*(2+1) -> 2*(6+1) -> 2*(14+1)
            assert schedule({"x": 2.0}) == 30.0
    finally:
        if executor is not None:
            executor.shutdown()


def test_toml_sweep_expansion(tmp_path: Path) -> None:
    path = tmp_path / "suite.toml"
    path.write_text(
        """
schema_version = 1
result_directory = "measurements"

[defaults]
backend = "thread"
task_kind = "sleep"
task_duration = 0.0
components = 2
evaluations = 4
batch_size = 4
repeats = 2
warmup_evaluations = 0

[[benchmark]]
name = "workers"
workers = [1, 2, 4]
""",
        encoding="utf-8",
    )
    suite = runner.load_config(path)
    assert suite.result_directory == Path("measurements")
    assert [config.workers for config in suite.benchmarks] == [1, 2, 4]
    assert {config.graph_depth for config in suite.benchmarks} == {1}


@pytest.mark.parametrize(
    "filename",
    [
        "smoke.toml",
        "local.toml",
        "mpi.toml",
        "deep_graph.toml",
        "deep_graph_mpi.toml",
    ],
)
def test_bundled_configuration_is_valid(filename: str) -> None:
    suite = runner.load_config(ROOT / "config" / filename)
    assert suite.benchmarks
    backends = {config.backend for config in suite.benchmarks}
    assert "mpi" not in backends or backends == {"mpi"}


def test_json_result_contains_raw_samples_and_is_exclusively_created(
    tmp_path: Path,
) -> None:
    config = make_config()
    measurements = {
        "first_evaluation_seconds": 0.01,
        "warmup_seconds": 0.02,
        "samples_seconds": [1.0, 1.1],
        "median_seconds": 1.05,
        "median_absolute_deviation_seconds": 0.05,
        "minimum_seconds": 1.0,
        "maximum_seconds": 1.1,
        "evaluations_per_second": 4 / 1.05,
        "objective_terms_per_second": 12 / 1.05,
        "scheduler_construction_seconds": 0.001,
        "schedule_preparation_seconds": 0.002,
        "measured_startup_overhead_seconds": 0.003,
    }
    result = runner.build_result(config, measurements, measurements)
    first = runner.write_result(result, tmp_path)
    second = runner.write_result(result, tmp_path)
    assert first != second
    stored = json.loads(first.read_text(encoding="utf-8"))
    assert stored["timings"]["samples_seconds"] == [1.0, 1.1]
    assert stored["configuration"]["components"] == 3
    assert stored["configuration"]["graph_depth"] == 1
    assert stored["software"]["chemfit_git"]["commit"]
