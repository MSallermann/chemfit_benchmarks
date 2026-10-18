#!/bin/bash
#SBATCH --job-name=chemfit-mpi-benchmark

set -euo pipefail

config=${1:-${CHEMFIT_BENCH_CONFIG:-config/mpi.toml}}
results=${CHEMFIT_RESULTS_DIR:-results}
python_executable=${CHEMFIT_PYTHON:-python}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

# Supply --nodes, --ntasks, and any site policy options to sbatch. The task
# count must be at least the largest configured worker count plus rank 0.
srun "$python_executable" benchmark/chemfit_runner.py \
    --config "$config" \
    --results-dir "$results"
