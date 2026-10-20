"""Importable synthetic workload used by ``backend_throughput.py``."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from chemfit.abstract_objective_function import component


class WorkMode(str, Enum):
    """Synthetic work performed by an atomic task."""

    noop = "noop"
    sleep = "sleep"
    busy = "busy"


@dataclass(frozen=True, slots=True)
class SyntheticInput:
    """Pickleable task input with a controlled round-trip payload."""

    task_id: int
    payload: bytes
    work_seconds: float
    work_mode: WorkMode


@dataclass(frozen=True, slots=True)
class SyntheticOutput:
    """Synthetic result returned through the backend and context state."""

    task_id: int
    payload: bytes


@component()
def synthetic_task(input_: SyntheticInput) -> SyntheticOutput:
    """Perform controlled work and return the configured payload."""
    if input_.work_mode == WorkMode.sleep:
        time.sleep(input_.work_seconds)
    elif input_.work_mode == WorkMode.busy:
        end = time.perf_counter() + input_.work_seconds
        while time.perf_counter() < end:
            pass

    return SyntheticOutput(task_id=input_.task_id, payload=input_.payload)
