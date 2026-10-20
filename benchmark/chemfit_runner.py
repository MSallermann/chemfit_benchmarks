#!/usr/bin/env python3
"""Standalone ChemFit scheduling benchmarks.

The workload intentionally uses only ChemFit's public objective, composition,
scheduler, and prepared-schedule APIs. A prepared schedule remains alive for
the first-use check, warm-up, and every timed repetition.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib

import chemfit


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    backend: str
    task_kind: str
    task_duration: float
    components: int
    evaluations: int
    batch_size: int
    workers: int
    repeats: int
    warmup_evaluations: int
    graph_depth: int = 1
    scaling: str = "strong"

    def validate(self) -> None:
        if not self.name:
            raise ValueError("benchmark name cannot be empty")
        if self.backend not in {"direct", "serial", "thread", "mpi"}:
            raise ValueError(f"unknown backend {self.backend!r}")
        if self.task_kind not in {"sleep", "busy"}:
            raise ValueError("task kind must be 'sleep' or 'busy'")
        if self.scaling not in {"strong", "weak"}:
            raise ValueError("scaling must be 'strong' or 'weak'")
        if self.task_duration < 0:
            raise ValueError("task duration cannot be negative")
        for field_name in ("components", "evaluations", "batch_size", "workers", "repeats"):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be positive")
        if self.warmup_evaluations < 0:
            raise ValueError("warmup_evaluations cannot be negative")
        if self.graph_depth < 1:
            raise ValueError("graph_depth must be positive")
        if self.scaling == "weak" and (
            self.evaluations % self.workers or self.batch_size % self.workers
        ):
            raise ValueError("weak-scaling evaluations and batch size must be divisible by workers")
        if self.backend in {"direct", "serial"} and self.workers != 1:
            raise ValueError(f"{self.backend} has exactly one worker")


@dataclass(frozen=True)
class SuiteConfig:
    benchmarks: list[BenchmarkConfig]
    result_directory: Path


def _busy_wait(seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


@chemfit.objective()
def synthetic_task(
    value: dict[str, float] | float,
    *,
    duration: float,
    task_kind: str,
) -> float:
    """One deterministic objective term with configurable execution cost."""
    if task_kind == "sleep":
        time.sleep(duration)
    else:
        _busy_wait(duration)
    input_value = value["x"] if isinstance(value, dict) else value
    return input_value + 1.0


def make_objective(config: BenchmarkConfig):
    """Build layered fan-out/fan-in work with public composition APIs."""
    layers = [
        chemfit.combine(
            *[
                synthetic_task.bind(
                    duration=config.task_duration,
                    task_kind=config.task_kind,
                )
                for _ in range(config.components)
            ]
        )
        for _ in range(config.graph_depth)
    ]
    objective = layers[0]
    for layer in layers[1:]:
        objective = chemfit.pipeline(objective, layer)
    return objective


def _inputs(count: int, sequence: int) -> list[dict[str, float]]:
    return [{"x": 1.0 + sequence / 1000.0 + index / max(count, 1)} for index in range(count)]


def evaluate_many_checked(
    schedule: Any,
    config: BenchmarkConfig,
    *,
    count: int,
    sequence: int,
) -> float:
    """Time checked batches through ``PreparedSchedule.evaluate_many()``."""
    inputs = _inputs(count, sequence)
    requests = [chemfit.EvaluationRequest(value, chemfit.EvaluateContext()) for value in inputs]
    values: list[Any] = [None] * count
    completed: set[int] = set()

    started = time.perf_counter()
    for offset in range(0, count, config.batch_size):
        batch = requests[offset : offset + config.batch_size]
        batch_indices: set[int] = set()
        for result in schedule.evaluate_many(batch):
            if result.index in batch_indices or not 0 <= result.index < len(batch):
                raise AssertionError(f"invalid or duplicate result index {result.index}")
            batch_indices.add(result.index)
            global_index = offset + result.index
            completed.add(global_index)
            values[global_index] = result.value
        if batch_indices != set(range(len(batch))):
            raise AssertionError("evaluate_many() did not yield every batch result")
    elapsed = time.perf_counter() - started

    if completed != set(range(count)):
        raise AssertionError("evaluate_many() completion count mismatch")
    for input_, value in zip(inputs, values, strict=True):
        if isinstance(value, Exception):
            raise AssertionError("an individual evaluation failed") from value
        expected = input_["x"]
        for _ in range(config.graph_depth):
            expected = config.components * (expected + 1.0)
        if not math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"incorrect result: {value!r} != {expected!r}")
    return elapsed


def _measure_coordinator(schedule: Any, config: BenchmarkConfig) -> dict[str, Any]:
    # This includes useful work and any lazy worker initialization, so it is
    # intentionally not reported as pure worker-startup time.
    first_evaluation_seconds = evaluate_many_checked(schedule, config, count=1, sequence=-2)
    warmup_started = time.perf_counter()
    if config.warmup_evaluations:
        evaluate_many_checked(
            schedule,
            config,
            count=config.warmup_evaluations,
            sequence=-1,
        )
    warmup_seconds = time.perf_counter() - warmup_started
    samples = [
        evaluate_many_checked(
            schedule,
            config,
            count=config.evaluations,
            sequence=repeat,
        )
        for repeat in range(config.repeats)
    ]
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return {
        "first_evaluation_seconds": first_evaluation_seconds,
        "warmup_seconds": warmup_seconds,
        "samples_seconds": samples,
        "median_seconds": median,
        "median_absolute_deviation_seconds": mad,
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "evaluations_per_second": config.evaluations / median,
        "objective_terms_per_second": (
            config.evaluations * config.components * config.graph_depth / median
        ),
    }


def _run_local(config: BenchmarkConfig) -> dict[str, Any]:
    objective = make_objective(config)
    executor: ThreadPoolExecutor | None = None
    construction_started = time.perf_counter()
    if config.backend == "direct":
        scheduler = chemfit.DirectScheduler()
    elif config.backend == "serial":
        scheduler = chemfit.SerialGraphScheduler()
    else:
        executor = ThreadPoolExecutor(max_workers=config.workers)
        scheduler = chemfit.ExecutorGraphScheduler(executor=executor)
    scheduler_construction_seconds = time.perf_counter() - construction_started

    preparation_started = time.perf_counter()
    schedule = scheduler.prepare(objective)
    schedule_preparation_seconds = time.perf_counter() - preparation_started
    try:
        with schedule:
            result = _measure_coordinator(schedule, config)
    finally:
        if executor is not None:
            executor.shutdown()
    result["scheduler_construction_seconds"] = scheduler_construction_seconds
    result["schedule_preparation_seconds"] = schedule_preparation_seconds
    result["measured_startup_overhead_seconds"] = (
        scheduler_construction_seconds + schedule_preparation_seconds
    )
    return result


def _run_mpi_communicator(config: BenchmarkConfig, comm: Any) -> dict[str, Any] | None:
    from chemfit.mpi_graph_schedule import MPIGraphScheduler

    objective = make_objective(config)
    construction_started = time.perf_counter()
    scheduler = MPIGraphScheduler(comm=comm, mpi_debug_log=False)
    scheduler_construction_seconds = time.perf_counter() - construction_started
    preparation_started = time.perf_counter()
    schedule = scheduler.prepare(objective)
    schedule_preparation_seconds = time.perf_counter() - preparation_started

    result = None
    with schedule:
        if comm.Get_rank() == 0:
            result = _measure_coordinator(schedule, config)
            result["scheduler_construction_seconds"] = scheduler_construction_seconds
            result["schedule_preparation_seconds"] = schedule_preparation_seconds
            result["measured_startup_overhead_seconds"] = (
                scheduler_construction_seconds + schedule_preparation_seconds
            )
        else:
            schedule.worker_loop()
    comm.Barrier()
    return result


def run(config: BenchmarkConfig) -> dict[str, Any] | None:
    """Run one configured benchmark point."""
    config.validate()
    if config.backend != "mpi":
        return _run_local(config)

    from mpi4py import MPI

    world = MPI.COMM_WORLD
    required_ranks = config.workers + 1
    if world.Get_size() < required_ranks:
        raise ValueError(
            f"MPI backend requested {config.workers} workers ({required_ranks} ranks), "
            f"but the allocation has {world.Get_size()} ranks"
        )

    # This permits a worker-count sweep in one allocation. Excess ranks wait
    # at the world barrier between configurations.
    active_comm = world.Split(
        0 if world.Get_rank() < required_ranks else MPI.UNDEFINED,
        world.Get_rank(),
    )
    target = None
    if active_comm != MPI.COMM_NULL:
        target = _run_mpi_communicator(config, active_comm)
        active_comm.Free()
    world.Barrier()
    return target


def _git_info_for_module(module_file: str) -> dict[str, Any]:
    path = Path(module_file).resolve().parent
    try:
        root = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", root, "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def metadata() -> dict[str, Any]:
    try:
        chemfit_version = importlib.metadata.version("chemfit")
    except importlib.metadata.PackageNotFoundError:
        chemfit_version = "unknown"
    slurm = {
        key: os.environ[key]
        for key in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_CLUSTER_NAME",
            "SLURM_NNODES",
            "SLURM_NTASKS",
            "SLURM_CPUS_PER_TASK",
            "SLURM_JOB_PARTITION",
            "SLURM_NODELIST",
        )
        if key in os.environ
    }
    return {
        "software": {
            "chemfit_version": chemfit_version,
            "chemfit_git": _git_info_for_module(chemfit.__file__),
            "python_version": platform.python_version(),
        },
        "machine": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "slurm": slurm,
        },
    }


_CONFIG_FIELDS = {field.name for field in fields(BenchmarkConfig)}


def _expand_benchmark(defaults: dict[str, Any], table: dict[str, Any]) -> list[BenchmarkConfig]:
    unknown = set(table) - _CONFIG_FIELDS
    if unknown:
        raise ValueError(f"unknown benchmark keys: {sorted(unknown)}")
    values = {**defaults, **table}
    missing = _CONFIG_FIELDS - set(values)
    if missing:
        raise ValueError(f"missing benchmark keys: {sorted(missing)}")

    names: list[str] = []
    axes: list[list[Any]] = []
    for field in fields(BenchmarkConfig):
        value = values[field.name]
        axis = value if isinstance(value, list) else [value]
        if not axis:
            raise ValueError(f"sweep axis {field.name!r} cannot be empty")
        names.append(field.name)
        axes.append(axis)

    expanded: list[BenchmarkConfig] = []
    for combination in itertools.product(*axes):
        config_values = dict(zip(names, combination, strict=True))
        if config_values["scaling"] == "weak":
            # Weak-scaling TOML values describe one worker's share. Store the
            # expanded totals in BenchmarkConfig and therefore in result JSON.
            config_values["evaluations"] *= config_values["workers"]
            config_values["batch_size"] *= config_values["workers"]
        config = BenchmarkConfig(**config_values)
        config.validate()
        expanded.append(config)
    return expanded


def load_config(path: Path) -> SuiteConfig:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    allowed_top_level = {"schema_version", "result_directory", "defaults", "benchmark"}
    unknown = set(document) - allowed_top_level
    if unknown:
        raise ValueError(f"unknown top-level TOML keys: {sorted(unknown)}")
    if document.get("schema_version", 1) != 1:
        raise ValueError("only configuration schema_version = 1 is supported")
    defaults = {
        "graph_depth": 1,
        "scaling": "strong",
        **document.get("defaults", {}),
    }
    unknown_defaults = set(defaults) - _CONFIG_FIELDS
    if unknown_defaults:
        raise ValueError(f"unknown default keys: {sorted(unknown_defaults)}")
    tables = document.get("benchmark", [])
    if not tables:
        raise ValueError("configuration must contain at least one [[benchmark]] table")
    benchmarks = [config for table in tables for config in _expand_benchmark(defaults, table)]
    return SuiteConfig(
        benchmarks=benchmarks,
        result_directory=Path(document.get("result_directory", "results")),
    )


def build_result(
    config: BenchmarkConfig,
    target: dict[str, Any],
) -> dict[str, Any]:
    metrics = {
        "median_runtime_seconds": target["median_seconds"],
        "throughput_evaluations_per_second": target["evaluations_per_second"],
        "throughput_objective_terms_per_second": target["objective_terms_per_second"],
    }

    return {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(config),
        "worker_count": config.workers,
        "timings": target,
        "metrics": metrics,
        **metadata(),
    }


def _safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "-" for character in value)
    return cleaned.strip("-") or "benchmark"


def write_result(result: dict[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    config = result["configuration"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = (
        f"{timestamp}_{_safe_name(config['name'])}_{config['backend']}_"
        f"w{config['workers']}_{result['run_id'][:10]}"
    )
    path = directory / f"{stem}.json"
    # Exclusive creation prevents replacement even if names collide on a
    # shared filesystem.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def _parse_args() -> tuple[list[BenchmarkConfig], Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="TOML suite configuration")
    parser.add_argument("--results-dir", type=Path, help="override result directory")
    parser.add_argument("--name", default="command-line")
    parser.add_argument("--backend", choices=("direct", "serial", "thread", "mpi"))
    parser.add_argument("--task-kind", choices=("sleep", "busy"))
    parser.add_argument("--task-duration", type=float)
    parser.add_argument("--components", type=int)
    parser.add_argument("--evaluations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-evaluations", type=int, default=2)
    parser.add_argument("--graph-depth", type=int)
    parser.add_argument("--scaling", choices=("strong", "weak"))
    args = parser.parse_args()

    direct_names = (
        "backend",
        "task_kind",
        "task_duration",
        "components",
        "evaluations",
        "batch_size",
        "workers",
        "graph_depth",
        "scaling",
    )
    supplied_directly = any(getattr(args, name) is not None for name in direct_names)
    if args.config:
        if supplied_directly:
            parser.error("benchmark arguments cannot be combined with --config")
        suite = load_config(args.config)
        return suite.benchmarks, args.results_dir or suite.result_directory

    required_direct_names = tuple(
        name for name in direct_names if name not in {"graph_depth", "scaling"}
    )
    missing = [name for name in required_direct_names if getattr(args, name) is None]
    if missing:
        parser.error("without --config these arguments are required: " + ", ".join(missing))
    config_values = {
        "name": args.name,
        "backend": args.backend,
        "task_kind": args.task_kind,
        "task_duration": args.task_duration,
        "components": args.components,
        "evaluations": args.evaluations,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "repeats": args.repeats,
        "warmup_evaluations": args.warmup_evaluations,
        "graph_depth": 1 if args.graph_depth is None else args.graph_depth,
        "scaling": "strong" if args.scaling is None else args.scaling,
    }
    if config_values["scaling"] == "weak":
        config_values["evaluations"] *= config_values["workers"]
        config_values["batch_size"] *= config_values["workers"]
    config = BenchmarkConfig(
        **config_values,
    )
    config.validate()
    return [config], args.results_dir or Path("results")


def main() -> None:
    configs, result_directory = _parse_args()
    backends = {config.backend for config in configs}
    if "mpi" in backends and backends != {"mpi"}:
        raise ValueError("MPI and local backends must use separate configuration files")

    for position, config in enumerate(configs, start=1):
        target = run(config)
        if target is None:
            continue
        result = build_result(config, target)
        path = write_result(result, result_directory)
        print(
            f"[{position}/{len(configs)}] {config.name}: {config.backend}, "
            f"{config.workers} worker(s), median "
            f"{target['median_seconds']:.6g} s -> {path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
