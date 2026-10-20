"""
Measure synthetic end-to-end throughput of ChemFit scheduler backends.

Unlike ``coordinator_throughput.py``, which replaces backend execution with an
in-memory completion queue, this benchmark sends atomic tasks through the real
direct, serial-graph, thread-pool, process-pool, or MPI schedule.  The atomic
task can be a no-op or perform a controlled sleep/busy wait, and its round-trip
payload size is configurable.  Scheduler construction, worker startup, and
shutdown are outside timed samples.

The reported rate includes the work performed by the selected scheduling path:
component invocation in every case, plus graph coordination, backend submission
and completion notification, and context-state transport where applicable.  It
is therefore an end-to-end scheduler/backend ceiling rather than a pure
transport measurement.  The direct scheduler bypasses the graph and serves as
a component-lifecycle baseline; one direct evaluation is counted as one
completion for comparison with the graph schedules.

Examples::

    python backend_throughput.py --quick
    python backend_throughput.py \
        --backends direct,serial,threadpool,processpool \
        --workers 1,4,12 \
        --batch-sizes 1,12,768 \
        --payload-sizes 0,1024 \
        --work-times 0,0.001,0.01 \
        --work-modes sleep,busy \
        --output backend-throughput.json
    mpiexec -n 13 python backend_throughput.py \
        --backends mpi --batch-sizes 12,768
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import sys
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend_throughput_workload import (
    SyntheticInput,
    SyntheticOutput,
    WorkMode,
    synthetic_task,
)
from chemfit.evaluate_context import EvaluateContext
from chemfit.executor_graph_schedule import ExecutorGraphScheduler
from chemfit.graph_schedule_base import SerialGraphScheduler
from chemfit.scheduling import DirectScheduler, EvaluationRequest

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_BACKENDS = "direct,serial,threadpool,processpool"
DEFAULT_WORKERS = "1,4"
DEFAULT_BATCH_SIZES = "1,64,1024"
DEFAULT_PAYLOAD_SIZES = "0"
DEFAULT_WORK_TIMES = "0"
DEFAULT_WORK_MODES = "sleep,busy"

QUICK_WORKERS = (1, 2)
QUICK_BATCH_SIZES = (1, 64)
QUICK_PAYLOAD_SIZES = (0,)
QUICK_WORK_TIMES = (0.0,)


class Backend(str, Enum):
    """Scheduler backend exercised by one benchmark case."""

    direct = "direct"
    serial = "serial"
    threadpool = "threadpool"
    processpool = "processpool"
    mpi = "mpi"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One scheduler/backend parameter combination."""

    backend: Backend
    workers: int
    batch_size: int
    payload_bytes: int
    work_seconds: float
    work_mode: WorkMode


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Measurements and summary statistics for one backend case."""

    backend: str
    workers: int
    batch_size: int
    payload_bytes: int
    work_seconds: float
    work_mode: str
    iterations_per_sample: int
    completions_per_sample: int
    sample_seconds: list[float]
    completions_per_second: list[float]
    median_completions_per_second: float
    median_nanoseconds_per_completion: float
    median_absolute_deviation: float


def make_scheduler(case: BenchmarkCase, resources: ExitStack) -> Any:
    """Construct the selected scheduler while retaining owned resources."""
    if case.backend == Backend.direct:
        return DirectScheduler()

    if case.backend == Backend.serial:
        return SerialGraphScheduler()

    if case.backend == Backend.threadpool:
        from concurrent.futures import ThreadPoolExecutor

        executor = resources.enter_context(ThreadPoolExecutor(max_workers=case.workers))
        return ExecutorGraphScheduler(executor=executor)

    if case.backend == Backend.processpool:
        try:
            from loky import ProcessPoolExecutor
        except ImportError as exception:
            if hasattr(exception, "add_note"):
                exception.add_note(
                    "Install ChemFit's test dependencies to benchmark loky: "
                    "`pip install chemfit[test]`."
                )
            raise

        executor = resources.enter_context(
            ProcessPoolExecutor(max_workers=case.workers)
        )
        return ExecutorGraphScheduler(executor=executor)

    from chemfit.mpi_graph_schedule import MPIGraphScheduler

    return MPIGraphScheduler(mpi_debug_log=False)


def make_requests(case: BenchmarkCase) -> tuple[EvaluationRequest[SyntheticInput], ...]:
    """Create reusable caller-owned contexts outside the measured region."""
    payload = bytes(case.payload_bytes)
    return tuple(
        EvaluationRequest(
            input=SyntheticInput(
                task_id=index,
                payload=payload,
                work_seconds=case.work_seconds,
                work_mode=case.work_mode,
            ),
            ctx=EvaluateContext(),
        )
        for index in range(case.batch_size)
    )


def validate_case(
    schedule: Any,
    requests: Sequence[EvaluationRequest[SyntheticInput]],
) -> None:
    """Check result indices and round-trip payloads before timing."""
    results = sorted(schedule.evaluate_many(requests), key=lambda result: result.index)
    if len(results) != len(requests):
        msg = f"Expected {len(requests)} results, got {len(results)}."
        raise AssertionError(msg)

    for index, (request, result) in enumerate(zip(requests, results, strict=True)):
        expected = SyntheticOutput(
            task_id=request.input.task_id,
            payload=request.input.payload,
        )
        if result.index != index or result.value != expected:
            msg = (
                f"Incorrect result at index {index}: "
                f"got ({result.index}, {result.value!r}), expected {expected!r}."
            )
            raise AssertionError(msg)


def consume_iterations(
    schedule: Any,
    requests: Sequence[EvaluationRequest[SyntheticInput]],
    iterations: int,
) -> float:
    """Run and consume complete batches, returning measured wall time."""
    start = time.perf_counter()
    for _ in range(iterations):
        deque(schedule.evaluate_many(requests), maxlen=0)
    return time.perf_counter() - start


def calibrated_iterations(
    schedule: Any,
    requests: Sequence[EvaluationRequest[SyntheticInput]],
    target_seconds: float,
    max_iterations: int,
) -> int:
    """Choose a repeat count that makes each sample long enough to be stable."""
    gc.collect()
    elapsed = consume_iterations(schedule, requests, iterations=1)
    if elapsed <= 0.0:
        return max_iterations

    estimate = math.ceil(target_seconds / elapsed)
    return min(max(1, estimate), max_iterations)


def run_case(
    case: BenchmarkCase,
    *,
    samples: int,
    warmups: int,
    target_seconds: float,
    max_iterations: int,
    mpi_rank: int,
) -> BenchmarkResult | None:
    """Validate, calibrate, and measure one backend-throughput case."""
    with ExitStack() as resources:
        scheduler = make_scheduler(case, resources)
        schedule = resources.enter_context(scheduler.prepare(synthetic_task))

        if case.backend == Backend.mpi and mpi_rank != 0:
            schedule.worker_loop()
            return None

        requests = make_requests(case)
        validate_case(schedule, requests)
        iterations = calibrated_iterations(
            schedule,
            requests,
            target_seconds,
            max_iterations,
        )

        for _ in range(warmups):
            consume_iterations(schedule, requests, iterations)

        sample_seconds = []
        for _ in range(samples):
            gc.collect()
            sample_seconds.append(consume_iterations(schedule, requests, iterations))

    completions_per_sample = case.batch_size * iterations
    completion_rates = [completions_per_sample / elapsed for elapsed in sample_seconds]
    median_rate = statistics.median(completion_rates)
    rate_mad = statistics.median(abs(rate - median_rate) for rate in completion_rates)

    return BenchmarkResult(
        backend=case.backend.value,
        workers=case.workers,
        batch_size=case.batch_size,
        payload_bytes=case.payload_bytes,
        work_seconds=case.work_seconds,
        work_mode=case.work_mode.value,
        iterations_per_sample=iterations,
        completions_per_sample=completions_per_sample,
        sample_seconds=sample_seconds,
        completions_per_second=completion_rates,
        median_completions_per_second=median_rate,
        median_nanoseconds_per_completion=1e9 / median_rate,
        median_absolute_deviation=rate_mad,
    )


def parse_backends(value: str) -> tuple[Backend, ...]:
    """Parse a comma-separated backend selection."""
    try:
        backends = tuple(Backend(item.strip().lower()) for item in value.split(","))
    except ValueError as exception:
        choices = ", ".join(backend.value for backend in Backend)
        msg = f"backends must be selected from: {choices}"
        raise argparse.ArgumentTypeError(msg) from exception

    if not backends:
        msg = "at least one backend is required"
        raise argparse.ArgumentTypeError(msg)
    return backends


def parse_work_modes(value: str) -> tuple[WorkMode, ...]:
    """Parse synthetic work modes, excluding the implicit no-op mode."""
    try:
        modes = tuple(WorkMode(item.strip().lower()) for item in value.split(","))
    except ValueError as exception:
        msg = "work modes must be selected from: sleep, busy"
        raise argparse.ArgumentTypeError(msg) from exception

    if not modes or WorkMode.noop in modes:
        msg = "work modes must contain one or both of: sleep, busy"
        raise argparse.ArgumentTypeError(msg)
    return modes


def parse_positive_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated sequence of positive integers."""
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exception:
        msg = f"expected comma-separated integers, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exception

    if not values or any(item < 1 for item in values):
        msg = "all values must be positive integers"
        raise argparse.ArgumentTypeError(msg)
    return values


def parse_nonnegative_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated sequence of nonnegative integers."""
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exception:
        msg = f"expected comma-separated integers, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exception

    if not values or any(item < 0 for item in values):
        msg = "all values must be nonnegative integers"
        raise argparse.ArgumentTypeError(msg)
    return values


def parse_nonnegative_floats(value: str) -> tuple[float, ...]:
    """Parse a comma-separated sequence of finite nonnegative floats."""
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exception:
        msg = f"expected comma-separated numbers, got {value!r}"
        raise argparse.ArgumentTypeError(msg) from exception

    if not values or any(item < 0.0 or not math.isfinite(item) for item in values):
        msg = "all values must be finite nonnegative numbers"
        raise argparse.ArgumentTypeError(msg)
    return values


def make_cases(
    backends: Sequence[Backend],
    workers: Sequence[int],
    batch_sizes: Sequence[int],
    payload_sizes: Sequence[int],
    work_times: Sequence[float],
    work_modes: Sequence[WorkMode],
    mpi_workers: int,
) -> list[BenchmarkCase]:
    """Expand CLI axes without duplicating serial, MPI, or no-op cases."""
    cases = []
    for backend in backends:
        if backend in {Backend.direct, Backend.serial}:
            backend_workers = (1,)
        elif backend == Backend.mpi:
            backend_workers = (mpi_workers,)
        else:
            backend_workers = workers

        for worker_count in backend_workers:
            for batch_size in batch_sizes:
                for payload_size in payload_sizes:
                    for work_seconds in work_times:
                        modes = (WorkMode.noop,) if work_seconds == 0.0 else work_modes
                        cases.extend(
                            BenchmarkCase(
                                backend=backend,
                                workers=worker_count,
                                batch_size=batch_size,
                                payload_bytes=payload_size,
                                work_seconds=work_seconds,
                                work_mode=mode,
                            )
                            for mode in modes
                        )
    return cases


def result_document(
    results: Sequence[BenchmarkResult],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a self-describing JSON-serializable result document."""
    return {
        "benchmark": "chemfit synthetic scheduler/backend throughput",
        "description": (
            "End-to-end synthetic atomic throughput through real scheduling "
            "paths; the direct scheduler is the non-graph baseline."
        ),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "settings": {
            "samples": args.samples,
            "warmups": args.warmups,
            "target_seconds": args.target_seconds,
            "max_iterations": args.max_iterations,
            "garbage_collection": "enabled; full collection before each sample",
            "worker_startup": "excluded from timed samples",
            "worker_shutdown": "excluded from timed samples",
        },
        "results": [asdict(result) for result in results],
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        type=parse_backends,
        default=parse_backends(DEFAULT_BACKENDS),
        help=("comma-separated backends: direct, serial, threadpool, processpool, mpi"),
    )
    parser.add_argument(
        "--workers",
        type=parse_positive_ints,
        default=parse_positive_ints(DEFAULT_WORKERS),
        help="executor worker counts; ignored by direct, serial, and MPI",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_positive_ints,
        default=parse_positive_ints(DEFAULT_BATCH_SIZES),
        help="comma-separated numbers of concurrent atomic evaluations",
    )
    parser.add_argument(
        "--payload-sizes",
        type=parse_nonnegative_ints,
        default=parse_nonnegative_ints(DEFAULT_PAYLOAD_SIZES),
        help="comma-separated round-trip payload sizes in bytes",
    )
    parser.add_argument(
        "--work-times",
        type=parse_nonnegative_floats,
        default=parse_nonnegative_floats(DEFAULT_WORK_TIMES),
        help="comma-separated synthetic task durations in seconds",
    )
    parser.add_argument(
        "--work-modes",
        type=parse_work_modes,
        default=parse_work_modes(DEFAULT_WORK_MODES),
        help="work modes used for positive durations: sleep, busy",
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--target-seconds", type=float, default=0.5)
    parser.add_argument("--max-iterations", type=int, default=1_000_000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a small smoke benchmark by overriding the numeric axes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the complete JSON result document",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the human-readable progress table",
    )
    return parser


def mpi_world_info(backends: Sequence[Backend]) -> tuple[int, int]:
    """Return MPI rank and size only when the MPI backend is requested."""
    if Backend.mpi not in backends:
        return 0, 1

    from mpi4py import MPI

    return MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    mpi_size: int,
) -> None:
    """Reject invalid or ambiguous measurement settings."""
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if args.target_seconds <= 0.0:
        parser.error("--target-seconds must be positive")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")
    if Backend.mpi in args.backends and mpi_size < 2:
        parser.error("the MPI backend requires at least two MPI ranks")
    if Backend.mpi in args.backends and len(args.backends) != 1:
        parser.error("run the MPI backend separately from local backends")


def print_header() -> None:
    """Print the compact human-readable result header."""
    print(
        f"{'backend':<12} {'workers':>7} {'batch':>7} {'bytes':>9} "
        f"{'work':>9} {'mode':>6} {'iterations':>11} "
        f"{'completions/s':>16} {'ns/completion':>15}"
    )


def print_result(result: BenchmarkResult) -> None:
    """Print one compact human-readable result row."""
    print(
        f"{result.backend:<12} {result.workers:>7} "
        f"{result.batch_size:>7} {result.payload_bytes:>9} "
        f"{result.work_seconds:>9.3g} {result.work_mode:>6} "
        f"{result.iterations_per_sample:>11} "
        f"{result.median_completions_per_second:>16,.0f} "
        f"{result.median_nanoseconds_per_completion:>15,.1f}"
    )


def main() -> None:
    """Run the configured backend-throughput matrix."""
    parser = build_parser()
    args = parser.parse_args()
    mpi_rank, mpi_size = mpi_world_info(args.backends)
    validate_args(parser, args, mpi_size)

    if args.quick:
        args.workers = QUICK_WORKERS
        args.batch_sizes = QUICK_BATCH_SIZES
        args.payload_sizes = QUICK_PAYLOAD_SIZES
        args.work_times = QUICK_WORK_TIMES
        args.samples = min(args.samples, 3)
        args.warmups = min(args.warmups, 1)
        args.target_seconds = min(args.target_seconds, 0.05)

    cases = make_cases(
        args.backends,
        args.workers,
        args.batch_sizes,
        args.payload_sizes,
        args.work_times,
        args.work_modes,
        mpi_workers=mpi_size - 1,
    )

    if mpi_rank == 0 and not args.quiet:
        print_header()

    results = []
    for case in cases:
        result = run_case(
            case,
            samples=args.samples,
            warmups=args.warmups,
            target_seconds=args.target_seconds,
            max_iterations=args.max_iterations,
            mpi_rank=mpi_rank,
        )
        if result is not None:
            results.append(result)
            if not args.quiet:
                print_result(result)

    if mpi_rank == 0 and args.output is not None:
        with args.output.open("w", encoding="utf-8") as output_file:
            json.dump(result_document(results, args), output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
