#!/bin/bash
#SBATCH --job-name=chemfit-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

config=${1:-${CHEMFIT_BENCH_CONFIG:-config/local.toml}}
results=${CHEMFIT_RESULTS_DIR:-results}
python_executable=${CHEMFIT_PYTHON:-python}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

"$python_executable" benchmark/chemfit_runner.py \
    --config "$config" \
    --results-dir "$results"
