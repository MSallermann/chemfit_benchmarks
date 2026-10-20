"""
Measure the graph coordinator's synthetic completion throughput.

This benchmark deliberately removes worker execution and backend transport from
the measurement.  Its in-memory backend converts every submitted ``NodeTask``
directly into a trivial ``NodeCompletion`` while the production
``GraphScheduleBase.evaluate_many`` driver performs all graph bookkeeping,
dependency release, structural lifecycle work, and result emission.

The reported rate is therefore a coordinator ceiling, not an estimate of MPI,
process-pool, or thread-pool throughput.  Compare it with an end-to-end backend
benchmark to determine how much of a real workload is coordinator-bound.

Examples::

    python coordinator_throughput.py
    python coordinator_throughput.py --quick
    python coordinator_throughput.py \
        --shapes atomic,fanout,pipeline \
        --batch-sizes 1,256,4096 \
        --nodes-per-evaluation 1,8,64 \
        --drain-sizes 1,8,all \
        --output coordinator-throughput.json
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chemfit.abstract_objective_function import (
    Component,
    PipelineComponent,
    TupleJoinedComponent,
    component,
)
from chemfit.callgraph import CallGraph, objective_to_call_graph
from chemfit.evaluate_context import EvaluateContext
from chemfit.graph_schedule_base import (
    BatchState,
    GraphScheduleBase,
    NodeCompletion,
    NodeTask,
)
from chemfit.scheduling import EvaluationRequest

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_SHAPES = "atomic,fanout,pipeline"
DEFAULT_BATCH_SIZES = "1,16,256,4096"
DEFAULT_NODES_PER_EVALUATION = "1,8,64"
DEFAULT_DRAIN_SIZES = "1,8,64,all"

QUICK_BATCH_SIZES = (1, 64)
QUICK_NODES_PER_EVALUATION = (1, 8)
QUICK_DRAIN_SIZES: tuple[int | None, ...] = (1, None)


class GraphShape(str, Enum):
    """Synthetic graph topology exercised by one benchmark case."""

    atomic = "atomic"
    fanout = "fanout"
    pipeline = "pipeline"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One coordinator-throughput parameter combination."""

    shape: GraphShape
    batch_size: int
    nodes_per_evaluation: int
    drain_size: int | None

    @property
    def completions_per_iteration(self) -> int:
        """Return the number of atomic completions in one input batch."""
        return self.batch_size * self.nodes_per_evaluation


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Measurements and summary statistics for one benchmark case."""

    shape: str
    batch_size: int
    nodes_per_evaluation: int
    drain_size: int | str
    iterations_per_sample: int
    evaluations_per_sample: int
    completions_per_sample: int
    sample_seconds: list[float]
    completions_per_second: list[float]
    median_completions_per_second: float
    median_nanoseconds_per_completion: float
    median_evaluations_per_second: float
    median_absolute_deviation: float


class ImmediateCompletionSchedule(GraphScheduleBase[float, Any]):
    """Graph schedule whose backend completes submitted tasks immediately."""

    def __init__(self, graph: CallGraph, drain_size: int | None) -> None:
        """Bind a graph to an in-memory completion queue policy."""
        super().__init__(graph)
        self.drain_size = drain_size
        self.submitted_completions = 0

    def submit_tasks(
        self,
        tasks: Sequence[NodeTask],
        batch_state: BatchState,
    ) -> None:
        """Convert submitted tasks into already-available completions."""
        if not hasattr(batch_state, "completions"):
            batch_state.completions = deque()

        batch_state.completions.extend(
            NodeCompletion(
                run_id=task.run_id,
                node_id=task.node_id,
                # Preserving the input makes both pipelines and fan-outs
                # produce deterministic, easily validated results.
                outcome=task.input,
                ctx_result_state=None,
            )
            for task in tasks
        )
        self.submitted_completions += len(tasks)

    def wait_for_completions(
        self,
        batch_state: BatchState,
    ) -> Sequence[NodeCompletion]:
        """Return one configured burst for graph drivers with batched drains."""
        completions: deque[NodeCompletion] = batch_state.completions
        if not completions:
            msg = "Synthetic backend has no completion available."
            raise RuntimeError(msg)

        count = len(completions)
        if self.drain_size is not None:
            count = min(count, self.drain_size)

        return [completions.popleft() for _ in range(count)]

    def next_completion(
        self,
        batch_state: BatchState,
    ) -> NodeCompletion:
        """Return one completion for graph drivers using the original API."""
        completions: deque[NodeCompletion] = batch_state.completions
        if not completions:
            msg = "Synthetic backend has no completion available."
            raise RuntimeError(msg)

        return completions.popleft()


@component()
def _passthrough(value: float) -> float:
    """Define an atomic leaf; the synthetic backend never calls its body."""
    return value


def make_objective(
    shape: GraphShape,
    nodes_per_evaluation: int,
) -> Component[float, Any]:
    """Construct one graph containing the requested number of atomic leaves."""
    if nodes_per_evaluation < 1:
        msg = "nodes_per_evaluation must be positive"
        raise ValueError(msg)

    if shape == GraphShape.atomic:
        if nodes_per_evaluation != 1:
            msg = "the atomic shape always has exactly one atomic node"
            raise ValueError(msg)
        return _passthrough

    leaves = [_passthrough for _ in range(nodes_per_evaluation)]
    if shape == GraphShape.fanout:
        return TupleJoinedComponent(tuple(leaves))

    if nodes_per_evaluation == 1:
        return leaves[0]

    pipeline = PipelineComponent(leaves[0], leaves[1])
    for leaf in leaves[2:]:
        pipeline = pipeline.then(leaf)
    return pipeline


def make_requests(batch_size: int) -> tuple[EvaluationRequest[float], ...]:
    """Create caller-owned contexts outside the measured region."""
    return tuple(
        EvaluationRequest(input=float(index), ctx=EvaluateContext())
        for index in range(batch_size)
    )


def expected_value(case: BenchmarkCase, input_value: float) -> Any:
    """Return the synthetic graph's expected result for one request."""
    if case.shape == GraphShape.fanout:
        return (input_value,) * case.nodes_per_evaluation
    return input_value


def validate_case(
    schedule: ImmediateCompletionSchedule,
    requests: Sequence[EvaluationRequest[float]],
    case: BenchmarkCase,
) -> None:
    """Check result indices, values, and completion count before timing."""
    submitted_before = schedule.submitted_completions
    results = list(schedule.evaluate_many(requests))
    submitted = schedule.submitted_completions - submitted_before

    if submitted != case.completions_per_iteration:
        msg = (
            f"Expected {case.completions_per_iteration} completions, "
            f"but the synthetic backend received {submitted}."
        )
        raise AssertionError(msg)
    if len(results) != case.batch_size:
        msg = f"Expected {case.batch_size} results, got {len(results)}."
        raise AssertionError(msg)

    results_by_index = sorted(results, key=lambda result: result.index)
    for index, result in enumerate(results_by_index):
        expected = expected_value(case, float(index))
        if result.index != index or result.value != expected:
            msg = (
                f"Incorrect result at index {index}: "
                f"got ({result.index}, {result.value!r}), expected {expected!r}."
            )
            raise AssertionError(msg)


def consume_iterations(
    schedule: ImmediateCompletionSchedule,
    requests: Sequence[EvaluationRequest[float]],
    case: BenchmarkCase,
    iterations: int,
) -> float:
    """Run and consume complete batches, returning measured wall time."""
    submitted_before = schedule.submitted_completions
    start = time.perf_counter()
    for _ in range(iterations):
        # deque consumes the result iterator without retaining a result list or
        # adding a Python-level loop body to every emitted evaluation.
        deque(schedule.evaluate_many(requests), maxlen=0)
    elapsed = time.perf_counter() - start

    expected = case.completions_per_iteration * iterations
    submitted = schedule.submitted_completions - submitted_before
    if submitted != expected:
        msg = f"Expected {expected} timed completions, got {submitted}."
        raise AssertionError(msg)
    return elapsed


def calibrated_iterations(
    schedule: ImmediateCompletionSchedule,
    requests: Sequence[EvaluationRequest[float]],
    case: BenchmarkCase,
    target_seconds: float,
    max_iterations: int,
) -> int:
    """Choose a repeat count that makes each sample long enough to be stable."""
    gc.collect()
    elapsed = consume_iterations(schedule, requests, case, iterations=1)
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
) -> BenchmarkResult:
    """Validate, calibrate, and measure one coordinator-throughput case."""
    objective = make_objective(case.shape, case.nodes_per_evaluation)
    schedule = ImmediateCompletionSchedule(
        objective_to_call_graph(objective),
        drain_size=case.drain_size,
    )
    requests = make_requests(case.batch_size)

    with schedule:
        validate_case(schedule, requests, case)
        iterations = calibrated_iterations(
            schedule,
            requests,
            case,
            target_seconds,
            max_iterations,
        )

        for _ in range(warmups):
            consume_iterations(schedule, requests, case, iterations)

        sample_seconds = []
        for _ in range(samples):
            gc.collect()
            sample_seconds.append(
                consume_iterations(schedule, requests, case, iterations)
            )

    completions_per_sample = case.completions_per_iteration * iterations
    evaluations_per_sample = case.batch_size * iterations
    completion_rates = [completions_per_sample / elapsed for elapsed in sample_seconds]
    evaluation_rates = [evaluations_per_sample / elapsed for elapsed in sample_seconds]
    median_rate = statistics.median(completion_rates)
    median_evaluation_rate = statistics.median(evaluation_rates)
    rate_mad = statistics.median(abs(rate - median_rate) for rate in completion_rates)

    return BenchmarkResult(
        shape=case.shape.value,
        batch_size=case.batch_size,
        nodes_per_evaluation=case.nodes_per_evaluation,
        drain_size="all" if case.drain_size is None else case.drain_size,
        iterations_per_sample=iterations,
        evaluations_per_sample=evaluations_per_sample,
        completions_per_sample=completions_per_sample,
        sample_seconds=sample_seconds,
        completions_per_second=completion_rates,
        median_completions_per_second=median_rate,
        median_nanoseconds_per_completion=1e9 / median_rate,
        median_evaluations_per_second=median_evaluation_rate,
        median_absolute_deviation=rate_mad,
    )


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


def parse_shapes(value: str) -> tuple[GraphShape, ...]:
    """Parse a comma-separated sequence of graph shapes."""
    try:
        shapes = tuple(GraphShape(item.strip().lower()) for item in value.split(","))
    except ValueError as exception:
        choices = ", ".join(shape.value for shape in GraphShape)
        msg = f"shapes must be selected from: {choices}"
        raise argparse.ArgumentTypeError(msg) from exception

    if not shapes:
        msg = "at least one shape is required"
        raise argparse.ArgumentTypeError(msg)
    return shapes


def parse_drain_sizes(value: str) -> tuple[int | None, ...]:
    """Parse positive drain sizes, using ``all`` for an unbounded drain."""
    drain_sizes: list[int | None] = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if item == "all":
            drain_sizes.append(None)
            continue
        try:
            drain_size = int(item)
        except ValueError as exception:
            msg = f"drain sizes must be positive integers or 'all', got {item!r}"
            raise argparse.ArgumentTypeError(msg) from exception
        if drain_size < 1:
            msg = "drain sizes must be positive"
            raise argparse.ArgumentTypeError(msg)
        drain_sizes.append(drain_size)

    if not drain_sizes:
        msg = "at least one drain size is required"
        raise argparse.ArgumentTypeError(msg)
    return tuple(drain_sizes)


def make_cases(
    shapes: Sequence[GraphShape],
    batch_sizes: Sequence[int],
    nodes_per_evaluation: Sequence[int],
    drain_sizes: Sequence[int | None],
) -> list[BenchmarkCase]:
    """Expand CLI axes into benchmark cases without duplicate atomic cases."""
    return [
        BenchmarkCase(
            shape=shape,
            batch_size=batch_size,
            nodes_per_evaluation=node_count,
            drain_size=drain_size,
        )
        for shape in shapes
        for batch_size in batch_sizes
        for node_count in ((1,) if shape == GraphShape.atomic else nodes_per_evaluation)
        for drain_size in drain_sizes
    ]


def result_document(
    results: Sequence[BenchmarkResult],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build a self-describing JSON-serializable result document."""
    return {
        "benchmark": "chemfit coordinator synthetic completion throughput",
        "description": (
            "In-memory completion ceiling through GraphScheduleBase.evaluate_many; "
            "excludes worker execution and backend transport."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
        },
        "results": [asdict(result) for result in results],
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=parse_shapes(DEFAULT_SHAPES),
        help="comma-separated graph shapes: atomic, fanout, pipeline",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_positive_ints,
        default=parse_positive_ints(DEFAULT_BATCH_SIZES),
        help="comma-separated evaluation batch sizes",
    )
    parser.add_argument(
        "--nodes-per-evaluation",
        type=parse_positive_ints,
        default=parse_positive_ints(DEFAULT_NODES_PER_EVALUATION),
        help="comma-separated atomic-node counts for fanout and pipeline shapes",
    )
    parser.add_argument(
        "--drain-sizes",
        type=parse_drain_sizes,
        default=parse_drain_sizes(DEFAULT_DRAIN_SIZES),
        help="comma-separated completion burst sizes; use 'all' for unbounded",
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--target-seconds", type=float, default=0.5)
    parser.add_argument("--max-iterations", type=int, default=1_000_000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a small smoke benchmark, overriding the three numeric axes",
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


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject invalid scalar measurement settings."""
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if args.target_seconds <= 0.0:
        parser.error("--target-seconds must be positive")
    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")


def print_header() -> None:
    """Print the compact human-readable result header."""
    print(
        f"{'shape':<9} {'batch':>7} {'nodes':>7} {'drain':>7} "
        f"{'iterations':>11} {'completions/s':>16} {'ns/completion':>15}"
    )


def print_result(result: BenchmarkResult) -> None:
    """Print one compact human-readable result row."""
    print(
        f"{result.shape:<9} {result.batch_size:>7} "
        f"{result.nodes_per_evaluation:>7} {result.drain_size!s:>7} "
        f"{result.iterations_per_sample:>11} "
        f"{result.median_completions_per_second:>16,.0f} "
        f"{result.median_nanoseconds_per_completion:>15,.1f}"
    )


def main() -> None:
    """Run the configured coordinator-throughput matrix."""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    if args.quick:
        args.batch_sizes = QUICK_BATCH_SIZES
        args.nodes_per_evaluation = QUICK_NODES_PER_EVALUATION
        args.drain_sizes = QUICK_DRAIN_SIZES
        args.samples = min(args.samples, 3)
        args.warmups = min(args.warmups, 1)
        args.target_seconds = min(args.target_seconds, 0.05)

    cases = make_cases(
        args.shapes,
        args.batch_sizes,
        args.nodes_per_evaluation,
        args.drain_sizes,
    )

    if not args.quiet:
        print_header()

    results = []
    for case in cases:
        result = run_case(
            case,
            samples=args.samples,
            warmups=args.warmups,
            target_seconds=args.target_seconds,
            max_iterations=args.max_iterations,
        )
        results.append(result)
        if not args.quiet:
            print_result(result)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output_file:
            json.dump(result_document(results, args), output_file, indent=2)
            output_file.write("\n")
        if not args.quiet:
            print(f"\nWrote JSON results to {args.output}")


if __name__ == "__main__":
    main()
