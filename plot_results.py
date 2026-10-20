#!/usr/bin/env python3
"""Create scientific figures from standalone ChemFit benchmark JSON files."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_results(directory: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("schema_version") != 1:
            raise ValueError(f"unsupported result schema in {path}")
        if "configuration" not in result or "metrics" not in result:
            raise ValueError(f"not a ChemFit benchmark result: {path}")
        result["_path"] = str(path)
        results.append(result)
    return results


def _series_label(result: dict[str, Any], x_field: str) -> str:
    config = result["configuration"]
    commit = result.get("software", {}).get("chemfit_git", {}).get("commit")
    scaling = config.get("scaling", "strong")
    parts = [
        config["name"],
        config["backend"],
        config["task_kind"],
        f"terms={config['components']}",
        f"depth={config.get('graph_depth', 1)}",
    ]
    if scaling == "weak" and x_field == "workers":
        workers = config["workers"]
        parts.extend(
            (
                "weak scaling",
                f"evals/worker={config['evaluations'] // workers}",
                f"batch/worker={config['batch_size'] // workers}",
            )
        )
    else:
        parts.extend(
            (
                f"evals={config['evaluations']}",
                f"batch={config['batch_size']}",
            )
        )
    if x_field != "workers":
        parts.append(f"workers={config['workers']}")
    if x_field != "task_duration":
        parts.append(f"duration={config['task_duration']:g}s")
    if commit:
        parts.append(f"commit={commit[:8]}")
    return ", ".join(parts)


def _scaling_series_key(result: dict[str, Any]) -> str:
    config = dict(result["configuration"])
    workers = config.pop("workers")
    if config.get("scaling", "strong") == "weak":
        config["evaluations"] //= workers
        config["batch_size"] //= workers
    commit = result.get("software", {}).get("chemfit_git", {}).get("commit")
    return json.dumps((config, commit), sort_keys=True)


def _derive_scaling_metrics(results: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[_scaling_series_key(result)].append(result)

    for series in grouped.values():
        metric = (
            "weak_scaling_efficiency"
            if series[0]["configuration"].get("scaling", "strong") == "weak"
            else "speedup"
        )
        if all(metric in result["metrics"] for result in series):
            continue
        reference_workers = min(result["configuration"]["workers"] for result in series)
        reference_runtime = statistics.median(
            result["metrics"]["median_runtime_seconds"]
            for result in series
            if result["configuration"]["workers"] == reference_workers
        )
        for result in series:
            runtime_ratio = reference_runtime / result["metrics"]["median_runtime_seconds"]
            if metric == "weak_scaling_efficiency":
                result["metrics"][metric] = runtime_ratio
            else:
                worker_ratio = result["configuration"]["workers"] / reference_workers
                result["metrics"]["speedup"] = runtime_ratio
                result["metrics"]["parallel_efficiency"] = runtime_ratio / worker_ratio


def _plot(
    results: list[dict[str, Any]],
    output: Path,
    *,
    filename: str,
    x_field: str,
    metric: str,
    ylabel: str,
    ideal: str | None = None,
    runtime_errors: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    metric_results = [result for result in results if metric in result["metrics"]]
    if not metric_results:
        return

    grouped: dict[str, dict[float, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for result in metric_results:
        config = result["configuration"]
        grouped[_series_label(result, x_field)][float(config[x_field])].append(result)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for label, points in sorted(grouped.items()):
        xs = sorted(points)
        ys = [statistics.median(item["metrics"][metric] for item in points[x]) for x in xs]
        if runtime_errors:
            errors = [
                statistics.median(
                    item["timings"]["median_absolute_deviation_seconds"] for item in points[x]
                )
                for x in xs
            ]
            ax.errorbar(xs, ys, yerr=errors, marker="o", capsize=3, label=label)
        else:
            ax.plot(xs, ys, marker="o", label=label)

    all_xs = [float(result["configuration"][x_field]) for result in metric_results]
    if ideal == "linear" and all_xs:
        # On a logarithmic x-axis, joining only the two endpoints would draw a
        # straight line in display coordinates rather than the function y=x.
        # Include every observed x coordinate so the reference follows ideal
        # linear speedup at each worker count.
        ideal_xs = sorted(set(all_xs))
        reference_x = min(ideal_xs)
        ax.plot(
            ideal_xs,
            [value / reference_x for value in ideal_xs],
            "k--",
            linewidth=1,
            label="ideal",
        )
    elif ideal == "unity":
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="ideal")

    if x_field == "workers":
        ax.set_xscale("log", base=2)
        backends = {result["configuration"]["backend"] for result in metric_results}
        if backends == {"mpi"}:
            ax.set_xlabel("MPI workers (coordinator rank 0 excluded)")
        elif backends == {"thread"}:
            ax.set_xlabel("Worker threads")
        else:
            ax.set_xlabel("Worker count")
    else:
        if all(value > 0 for value in all_xs):
            ax.set_xscale("log")
        else:
            ax.set_xscale("symlog", linthresh=1e-6)
        ax.set_xlabel("Atomic task duration (s)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if grouped:
        ax.legend(fontsize="xx-small")
    fig.savefig(output / filename, dpi=200)
    fig.savefig(output / filename.replace(".png", ".pdf"))
    plt.close(fig)


def make_plots(results: list[dict[str, Any]], output: Path) -> None:
    if not results:
        raise ValueError("no JSON results found")
    _derive_scaling_metrics(results)
    output.mkdir(parents=True, exist_ok=True)
    _plot(
        results,
        output,
        filename="runtime_vs_workers.png",
        x_field="workers",
        metric="median_runtime_seconds",
        ylabel="Median runtime (s)",
        runtime_errors=True,
    )
    _plot(
        results,
        output,
        filename="speedup_vs_workers.png",
        x_field="workers",
        metric="speedup",
        ylabel="Speedup relative to smallest worker count",
        ideal="linear",
    )
    _plot(
        results,
        output,
        filename="efficiency_vs_workers.png",
        x_field="workers",
        metric="parallel_efficiency",
        ylabel="Parallel efficiency relative to smallest worker count",
        ideal="unity",
    )
    _plot(
        results,
        output,
        filename="weak_scaling_efficiency_vs_workers.png",
        x_field="workers",
        metric="weak_scaling_efficiency",
        ylabel="Weak-scaling efficiency",
        ideal="unity",
    )
    _plot(
        results,
        output,
        filename="throughput_vs_task_duration.png",
        x_field="task_duration",
        metric="throughput_evaluations_per_second",
        ylabel="Complete objective evaluations / s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory containing JSON results")
    parser.add_argument("--output", type=Path, default=Path("plots"))
    args = parser.parse_args()
    results = load_results(args.results)
    make_plots(results, args.output)
    print(f"plotted {len(results)} result(s) in {args.output}")


if __name__ == "__main__":
    main()
