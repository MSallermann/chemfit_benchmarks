# ChemFit scheduling benchmarks

This is a small standalone benchmark suite for finding scheduling bottlenecks
before ChemFit 4.0.0. Python runs the benchmarks, TOML describes parameter
sweeps, each configuration produces one JSON file, and Matplotlib produces the
figures. There is no benchmark framework or database.

The workload is deliberately close to ordinary ChemFit code. It creates
objectives with `chemfit.objective()` and `.bind()`, composes independent terms
with `chemfit.combine()`, and calls `PreparedSchedule.evaluate_many()` through
`DirectScheduler`, `SerialGraphScheduler`, `ExecutorGraphScheduler`, or
`MPIGraphScheduler`. The same prepared schedule stays open through first use,
warm-up, and every measured repetition.

The included configurations cover:

- strong scaling with fixed useful work;
- weak scaling with fixed useful work per worker;
- throughput from many concurrent evaluations;
- overhead for inexpensive tasks;
- independent objective terms versus batches of top-level evaluations;
- layered fan-out/fan-in graphs with sequential dependencies.

Sleeping tasks expose scheduler concurrency. CPU-bound tasks are also
available, but the threaded backend's busy loop holds the Python GIL; use the
MPI backend when measuring CPU parallelism.

## Install

Create an environment for the ChemFit commit or release being measured, then
install this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "chemfit[mpi] @ git+https://github.com/MSallermann/chemfit.git@next"
python -m pip install -e ".[all]"
```

The MPI extra is needed only for MPI runs. ChemFit itself is not modified.

## Synthetic throughput microbenchmarks

Two focused microbenchmarks complement the configurable workload runner. Use
them when a full scaling campaign shows a throughput limit and you want to
locate it more precisely:

| Benchmark | What is timed | What is excluded |
| --- | --- | --- |
| [`coordinator_throughput.py`](benchmark/coordinator_throughput.py) | Graph bookkeeping, dependency release, completion draining, and result emission in `GraphScheduleBase.evaluate_many()` | Component execution, executor or MPI submission, serialization, and transport |
| [`backend_throughput.py`](benchmark/backend_throughput.py) | Complete atomic evaluations through a real direct, serial-graph, thread-pool, process-pool, or MPI scheduling path | Scheduler construction, worker startup, and worker shutdown |

Both scripts validate every case before measuring it, calibrate the number of
iterations used in each sample, run warm-up samples, and report the median of
the raw completion rates. They print a compact table by default and optionally
write one self-describing JSON document containing the complete matrix. They
do not use the TOML configurations or emit the per-configuration result schema
used by `chemfit_runner.py` and `plot_results.py`.

### Coordinator throughput

Run a short smoke measurement with:

```bash
python benchmark/coordinator_throughput.py --quick \
  --output coordinator-throughput.json
```

The benchmark uses an in-memory backend that turns every submitted node into
an immediately available completion. The component bodies are never called.
This makes the result a ceiling for the coordinator on the current Python and
machine, not an estimate of thread-pool, process-pool, or MPI performance.

A more targeted sweep looks like:

```bash
python benchmark/coordinator_throughput.py \
  --shapes atomic,fanout,pipeline \
  --batch-sizes 1,256,4096 \
  --nodes-per-evaluation 1,8,64 \
  --drain-sizes 1,8,all \
  --samples 7 --target-seconds 0.5 \
  --output coordinator-throughput.json
```

The axes have the following meanings:

| Option | Meaning |
| --- | --- |
| `--shapes` | `atomic` is one node, `fanout` contains independent nodes, and `pipeline` contains sequentially dependent nodes. |
| `--batch-sizes` | Number of concurrent top-level evaluations passed to one `evaluate_many()` call. |
| `--nodes-per-evaluation` | Atomic nodes in each fan-out or pipeline. It is ignored for `atomic`, which always has one node. |
| `--drain-sizes` | Maximum immediately available completions returned to the graph driver at once. `all` drains the complete available burst. |
| `--samples` | Number of measured samples retained in the JSON output. |
| `--warmups` | Number of untimed, calibrated warm-up samples. |
| `--target-seconds` | Approximate target duration used to calibrate the iterations in one sample. |
| `--max-iterations` | Upper bound on calibrated iterations per sample. |

`fanout` and `pipeline` perform the same number of atomic completions but expose
different dependency structure. Compare them to quantify dependency-release
and frontier-refill costs. Varying `drain-size` shows the cost of processing
completions individually versus in bursts. Large batches keep many graph
instances active at once and show whether that fixed work amortizes.

The primary metric is `median_completions_per_second`; its reciprocal is also
reported as `median_nanoseconds_per_completion`. The JSON additionally records
evaluation throughput, every sample duration and completion rate, the
calibrated iteration count, and the median absolute deviation of completion
rates. One evaluation contains `nodes_per_evaluation` completions, so compare
completion throughput rather than evaluation throughput across different node
counts.

### End-to-end backend throughput

Run a short local smoke measurement without the optional process-pool
dependency:

```bash
python benchmark/backend_throughput.py --quick \
  --backends direct,serial,threadpool \
  --output backend-throughput.json
```

The benchmark sends an atomic component through the selected real scheduling
path. The component can do no work, sleep, or busy-wait, and can return a byte
payload of a controlled size. A useful local sweep is:

```bash
python benchmark/backend_throughput.py \
  --backends direct,serial,threadpool,processpool \
  --workers 1,4,12 \
  --batch-sizes 1,12,768 \
  --payload-sizes 0,1024 \
  --work-times 0,0.001,0.01 \
  --work-modes sleep,busy \
  --output backend-throughput.json
```

The process-pool backend uses `loky`. Install ChemFit's `test` extra before
selecting it; for the Git installation above, use `chemfit[mpi,test]` instead
of `chemfit[mpi]`. The remaining axes are:

| Option | Meaning |
| --- | --- |
| `--backends` | Any local combination of `direct`, `serial`, `threadpool`, and `processpool`, or `mpi` by itself. |
| `--workers` | Worker counts for thread and process pools. Direct and serial always use one worker; MPI infers workers from the launched ranks. |
| `--batch-sizes` | Number of concurrent atomic evaluations in each `evaluate_many()` call. |
| `--payload-sizes` | Bytes carried in both the synthetic input and returned output. |
| `--work-times` | Synthetic component duration in seconds. Zero selects one deduplicated `noop` case. |
| `--work-modes` | For positive durations, `sleep` releases the GIL and `busy` runs a Python busy loop. |

The measurement includes component invocation and the scheduling path used by
the selected backend. For graph schedulers it also includes graph coordination
and backend submission/completion handling; process-pool and MPI cases include
the applicable serialization and transport. `direct` bypasses the graph and
is a component-lifecycle baseline. As in the coordinator benchmark, the
primary metric is median completions per second, with raw durations, rates,
and rate MAD retained in JSON.

Run MPI separately from all local backends. Rank 0 is the coordinator and all
other ranks are workers, so this example records eight workers:

```bash
mpiexec -n 9 python benchmark/backend_throughput.py \
  --backends mpi \
  --batch-sizes 8,256 \
  --payload-sizes 0,1024 \
  --work-times 0,0.001 \
  --output backend-throughput-mpi.json
```

Use no-op cases to expose fixed per-completion cost, payload sweeps to expose
transport sensitivity, and positive work times to find where useful work
amortizes scheduling overhead. Sleeping work is appropriate for thread
concurrency; the Python busy loop holds the GIL, so do not interpret threaded
busy-mode results as CPU parallel scaling.

To relate the two microbenchmarks, compare the backend result with the
coordinator's `atomic` shape at the same batch size. The coordinator number is
the graph driver's in-memory ceiling. The gap to `serial`, `threadpool`,
`processpool`, or `mpi` includes component and backend costs. Measurements are
most useful when Python version, ChemFit commit, host, affinity, and MPI stack
are held constant; the JSON records Python and platform information but not a
ChemFit Git commit or cluster allocation metadata.

## Run one benchmark locally

```bash
python benchmark/chemfit_runner.py \
  --name local-example \
  --backend thread --workers 4 \
  --task-kind sleep --task-duration 0.01 \
  --components 8 --graph-depth 1 \
  --evaluations 32 --batch-size 32 \
  --repeats 5 --warmup-evaluations 2 \
  --results-dir results
```

`direct`, `serial`, and `thread` are the local backend names. `thread` uses an
`ExecutorGraphScheduler` with a `ThreadPoolExecutor`. `direct` and `serial`
require one worker.

## Run a configured sweep

The small smoke suite is safe for checking an installation:

```bash
python benchmark/chemfit_runner.py --config config/smoke.toml
```

The local suite contains all general study shapes:

```bash
python benchmark/chemfit_runner.py \
  --config config/local.toml \
  --results-dir results/local-commit-a
```

A TOML file has shared defaults and one or more benchmark tables. Every
benchmark field can be either a single value or a list, whether it appears in
`[defaults]` or a `[[benchmark]]` table. This includes `name`, `backend`,
`task_kind`, `task_duration`, `components`, `graph_depth`, `evaluations`,
`batch_size`, `workers`, `repeats`, and `warmup_evaluations`. A list creates a
Cartesian sweep; a single value stays fixed. `scaling` may be `strong` or
`weak` and defaults to `strong`. The top-level `schema_version`
and `result_directory` fields must be single values. Command-line arguments
are also single-valued, so use a TOML file for sweeps:

```toml
schema_version = 1
result_directory = "results"

[defaults]
backend = "thread"
task_kind = "sleep"
task_duration = 0.01
components = 32
graph_depth = 1
evaluations = 32
batch_size = 32
workers = 1
repeats = 5
warmup_evaluations = 2

[[benchmark]]
name = "strong-scaling"
workers = [1, 2, 4, 8]
```

The configuration fields have deliberately narrow meanings:

| Field | Meaning |
| --- | --- |
| `name` | Human-readable series name stored in JSON and plot labels. |
| `backend` | `direct`, `serial`, `thread`, or `mpi`. |
| `workers` | Thread workers, or MPI worker ranks excluding coordinator rank 0. Must be 1 for `direct` and `serial`. |
| `components` | Number of independent objective terms in every graph layer. This is the layer's fan-out width. |
| `graph_depth` | Number of sequential combined-objective layers. A value of 1 is the original flat workload. |
| `evaluations` | Number of complete top-level objective evaluations in each timed repetition. |
| `batch_size` | Maximum requests passed to one `evaluate_many()` call. Smaller values split a repetition into consecutive batches. |
| `task_duration` | Target duration in seconds for every atomic objective term. |
| `task_kind` | `sleep` for concurrency studies or `busy` for a Python CPU loop. |
| `warmup_evaluations` | Untimed evaluations after first use and before samples are collected. |
| `repeats` | Number of raw timing samples retained in the JSON result. |
| `scaling` | `strong` keeps the configured workload fixed. `weak` treats `evaluations` and `batch_size` as per-worker values and scales both by `workers`. |

Lists are expanded independently. For example, four worker counts and three
depths produce twelve configurations. There is no pairing or zip behavior.
Keep MPI and local backends in separate files because MPI launches every
Python process under the same communicator.

## Benchmark case catalog

The supplied TOML files contain named cases rather than opaque benchmark IDs.
Each case answers a different scheduling question.

### Smoke and scheduler baselines

[config/smoke.toml](config/smoke.toml) is the installation check. It runs:

- the same small flat objective with `DirectScheduler` and
  `SerialGraphScheduler`;
- `ExecutorGraphScheduler` with 1, 2, and 4 workers;
- one depth-three executor graph with 2 workers.

The suite has six configurations with 1 ms sleeping terms and three samples.
Use it before submitting a larger campaign. Direct scheduling is a semantic
and framework-overhead baseline; it does not expose graph-level parallelism.

### Strong scaling

`strong-scaling` in [config/local.toml](config/local.toml) and
`mpi-strong-scaling` in [config/mpi.toml](config/mpi.toml) keep useful work
fixed while worker count changes. Every sample contains 32 evaluations of a
32-term, depth-one objective with 10 ms sleeping terms. The local case uses
1, 2, 4, and 8 threads; MPI uses 1, 3, and 7 worker ranks.

Use runtime, speedup, and parallel efficiency to answer whether adding workers
reduces time for the same workload. The workload contains 1,024 atomic term
executions per sample, so it offers concurrency both within objectives and
across evaluations.

### Weak scaling

`weak-scaling` in [config/local.toml](config/local.toml) and
`mpi-weak-scaling` in [config/mpi.toml](config/mpi.toml) hold useful work per
worker constant. They configure one evaluation per worker, with 32 independent
terms in each evaluation, so every worker receives 32 atomic terms. The cases
cover 10 ms and 100 ms terms.

Weak mode couples workload to worker count rather than taking a Cartesian
product. In this example:

```toml
[[benchmark]]
name = "weak-scaling"
scaling = "weak"
workers = [1, 2, 4, 8]
evaluations = 1
batch_size = 1
```

the expanded configurations contain 1, 2, 4, and 8 evaluations respectively.
The JSON records these expanded totals and also records the reduced one-worker
reference configuration. Weak-scaling efficiency is calculated as
`one_worker_runtime / target_runtime`; ideal weak scaling therefore remains at
1.0 as workers and total work increase together.

### Throughput across task durations

`throughput` and `mpi-throughput` use one term per objective but submit 128
evaluations together. They sweep 1 ms, 10 ms, and 100 ms terms. This removes
within-objective fan-out and asks how efficiently a scheduler handles many
independent top-level requests.

Compare evaluation throughput across durations. Poor throughput at 1 ms that
converges toward ideal behavior at 100 ms indicates fixed scheduling costs
that are amortized by more expensive work.

### Scheduling overhead for inexpensive tasks

`scheduling-overhead` is in the local suite. It submits 256 one-term
evaluations and sweeps 1 microsecond, 10 microseconds, and 100 microseconds.
Useful work is intentionally tiny, so this case measures Python submission,
completion processing, graph traversal, and executor coordination more than
the synthetic task itself.

Treat the configured duration as a target, not a hard real-time guarantee.
Operating-system timer resolution and Python overhead dominate the shortest
points. This case is useful for relative comparisons between ChemFit commits
on the same machine, not for claiming absolute microsecond task accuracy.

### Independent components versus batched evaluations

These two cases contain the same 32 atomic terms but expose different
parallelism:

| Case | Components | Evaluations | Batch size | Scheduling shape |
| --- | ---: | ---: | ---: | --- |
| `independent-components` | 32 | 1 | 1 | One wide combined objective. |
| `batched-evaluations` | 1 | 32 | 32 | Thirty-two narrow objective instances. |

The MPI suite provides the corresponding `mpi-` cases. Comparing them reveals
whether ChemFit schedules ready nodes inside one graph differently from ready
nodes across many `evaluate_many()` requests.

### Deep layered graphs

`deep-layered-graph` in the local suite uses eight terms per layer, 16
concurrent evaluations, 1 ms terms, and depths 2, 4, and 8. The MPI suite
provides depths 2 and 4. These cases exercise repeated dependency release and
fan-out/fan-in barriers under a steady supply of concurrent graph instances.

The focused [config/deep_graph.toml](config/deep_graph.toml) and
[config/deep_graph_mpi.toml](config/deep_graph_mpi.toml) additionally compare
one graph in flight with 16 graphs in flight, including depth-one reference
points. The next section explains this topology and its interpretation in
detail.

### Which configuration should I run?

| Goal | Configuration |
| --- | --- |
| Verify the environment | `config/smoke.toml` |
| Run every local study | `config/local.toml` |
| Run local depth studies only | `config/deep_graph.toml` |
| Run the main distributed studies | `config/mpi.toml` |
| Run distributed depth studies only | `config/deep_graph_mpi.toml` |

## Deep-graph benchmarks

`graph_depth=1` is the original flat combined objective. A larger value builds
a structural pipeline of combined-objective layers:

```text
input -> [N independent terms -> sum] -> [N independent terms -> sum] -> ...
```

Here `components=N` is the fan-out width at every layer. Each layer must
finish before the next becomes runnable, so the workload exercises dependency
release, repeated frontier refill, and synchronization barriers. Work is
accounted for as:

```text
atomic terms per evaluation = components * graph_depth
atomic terms per timed sample = components * graph_depth * evaluations
nominal useful time = atomic terms per timed sample * task_duration
```

The result is deterministic. Starting with input `x`, every layer applies
`next_value = components * (current_value + 1)`. The runner recomputes this
recurrence and checks every completed evaluation after its timed region.

### Run one deep graph

This example measures a four-layer graph with eight independent terms per
layer, or 32 atomic term executions per complete evaluation:

```bash
python benchmark/chemfit_runner.py \
  --name deep-four-layers \
  --backend thread --workers 8 \
  --task-kind sleep --task-duration 0.001 \
  --components 8 --graph-depth 4 \
  --evaluations 1 --batch-size 1 \
  --warmup-evaluations 2 --repeats 5 \
  --results-dir results/deep-one
```

With one evaluation and a batch size of one, at most `components` atomic terms
are runnable at a time. In this example, worker counts above eight cannot add
within-graph parallelism.

### Run the focused local sweep

[config/deep_graph.toml](config/deep_graph.toml) varies workers over 1, 2, 4,
and 8 and depth over 1, 2, 4, and 8. It contains two complementary series:

- `deep-single-evaluation` keeps one graph in flight and exposes every layer
  barrier;
- `deep-batched-evaluations` keeps 16 graphs in flight, allowing ready terms
  from other evaluations to occupy workers while one graph changes layers.

The two series produce 32 configurations in total. Run all of them with:

```bash
python benchmark/chemfit_runner.py \
  --config config/deep_graph.toml \
  --results-dir results/deep-local
```

For a quick installation check, the smoke suite contains one small
depth-three case:

```bash
python benchmark/chemfit_runner.py \
  --config config/smoke.toml \
  --results-dir results/smoke
```

### Design fair comparisons

Choose parameters according to the question being asked:

- For strong scaling, hold everything except `workers` fixed.
- For weak scaling, set `scaling="weak"`; `evaluations` and `batch_size` then
  specify one worker's share and grow with `workers`.
- To study increasing depth, hold `components` fixed and vary `graph_depth`.
  Total work grows with depth, so compare objective-term throughput as well as
  raw runtime.
- To isolate topology while holding atomic work fixed, create separate tables
  with `(components, graph_depth)` pairs such as `(64, 1)`, `(32, 2)`,
  `(16, 4)`, and `(8, 8)`. Do not put both fields in lists: lists form a
  Cartesian product rather than correlated pairs.
- To expose layer barriers in one graph, use `evaluations=1` and
  `batch_size=1`. To test whether concurrent candidates hide those barriers,
  increase both while keeping them equal.
- Use short sleeping terms to emphasize coordinator and dependency-release
  overhead. Longer terms show whether this overhead is amortized.
- The threaded `busy` workload holds the Python GIL. Use MPI when the goal is
  measuring CPU parallelism.

The deep workload uses only `chemfit.objective()`, `.bind()`,
`chemfit.combine()`, and `chemfit.pipeline()`. It introduces no custom
component subclasses or private scheduler APIs. This is a regular layered
graph, not yet an irregular DAG with shared intermediate nodes.

### Run the focused MPI sweep

[config/deep_graph_mpi.toml](config/deep_graph_mpi.toml) runs equivalent
single-evaluation and 16-evaluation cases at 1, 3, and 7 workers. The largest
point needs eight ranks: seven workers plus coordinator rank 0.

For a two-rank MPI smoke test outside Slurm:

```bash
mpiexec -n 2 python benchmark/chemfit_runner.py \
  --name mpi-deep-smoke \
  --backend mpi --workers 1 \
  --task-kind sleep --task-duration 0.001 \
  --components 4 --graph-depth 3 \
  --evaluations 2 --batch-size 2 \
  --warmup-evaluations 1 --repeats 3 \
  --results-dir results/mpi-deep-smoke
```

The allocation must contain at least `workers + 1` ranks. A TOML worker sweep
can use a larger allocation: each point creates an active subcommunicator and
excess ranks wait for the next configuration.

## Results and comparisons

Every expanded configuration creates a new, uniquely named JSON file. Files
are opened with exclusive creation, so existing measurements are never
replaced. Each record contains:

- the complete configuration and worker count;
- every timing sample, median, median absolute deviation, and throughput;
- the ChemFit version and Git commit, including dirty-tree state;
- Python, host, operating-system, CPU-count, and available Slurm metadata.

Each configuration executes only the requested worker count; the runner does
not perform an additional one-worker reference measurement. Scheduler
construction and schedule preparation are recorded as measured startup
overhead. The separately named
`first_evaluation_seconds` includes useful task time and any lazy worker
initialization; it is deliberately not described as pure worker startup.

Correctness is checked outside each timed region after every batch: all result
indices must be present exactly once, individual failures are rejected, and
the combined numerical result must match the expected sum.

Keep a separate results directory for each campaign if convenient, or put all
runs together: the commit metadata and unique run IDs remain in every file.

Inspect a result without installing another reporting tool:

```bash
python -m json.tool results/deep-local/RESULT_FILE.json
```

For deep graphs, focus on:

- `configuration.graph_depth`, `components`, `evaluations`, and `batch_size`;
- `timings.samples_seconds` and `timings.median_seconds`;
- `timings.objective_terms_per_second`, which includes terms from every layer;
- `timings.first_evaluation_seconds`, which contains one real evaluation and
  possible lazy initialization and therefore is not pure startup overhead.

Falling objective-term throughput or flattening speedup as depth increases is
evidence of dependency-release or scheduling overhead, particularly for short
tasks. If the single-evaluation series degrades but the batched series does
not, other ready graph instances are successfully hiding layer barriers.

## Plot results

```bash
python plot_results.py results/local-commit-a --output plots/local-commit-a
```

For the focused deep-graph sweep, use:

```bash
python plot_results.py results/deep-local --output plots/deep-local
```

The script recursively reads JSON files and writes PNG and PDF versions of:

- runtime versus worker count, with timing MAD error bars;
- speedup relative to the smallest measured worker count, with ideal scaling;
- parallel efficiency relative to the smallest measured worker count;
- weak-scaling efficiency versus worker count, when weak results are present;
- evaluation throughput versus atomic task duration.

Results with the same configuration, ChemFit commit, and x coordinate are
combined by their median for plotting. The original run files remain intact.
Scaling metrics are derived in memory from the smallest worker count in each
series, so plotting adds no benchmark work and does not alter result files.
Graph depth appears in each series label, so a focused deep sweep produces one
worker-scaling curve per depth for both the single and batched cases.

## Running on a Slurm cluster

The example scripts let Slurm allocate resources and let Python execute the
actual benchmark sweep. They contain no account, partition, module name,
virtual-environment path, or cluster-specific filesystem path.

### 1. Prepare the environment

From the repository root on a shared filesystem:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "chemfit[mpi] @ git+https://github.com/MSallermann/chemfit.git@next"
python -m pip install -e ".[all]"
```

Load the site's Python and MPI modules before creating the environment if the
cluster requires them. `mpi4py` must be built against the same MPI
implementation used by Slurm's `srun` on the compute nodes.

Point the submission scripts at the exact interpreter and choose an absolute
result directory. Both variables are exported into the batch job by normal
Slurm environment propagation:

```bash
export CHEMFIT_PYTHON="$PWD/.venv/bin/python"
export CHEMFIT_RESULTS_DIR="$PWD/results/cluster-campaign"
```

If the site does not export the submission environment by default, pass
`--export=ALL` to `sbatch`. If compute nodes need module commands or explicit
environment activation, copy the example script and add those site-specific
commands before the Python or `srun` line.

### 2. Run local schedulers on one node

[slurm/single_node.sh](slurm/single_node.sh) requests one node and one process.
Set `--cpus-per-task` to at least the largest `workers` value in the selected
TOML file:

| Configuration | Maximum workers | Required CPUs per task |
| --- | ---: | ---: |
| `config/smoke.toml` | 4 | 4 |
| `config/local.toml` | 8 | 8 |
| `config/deep_graph.toml` | 8 | 8 |

Run the installation smoke test first:

```bash
sbatch --time=00:10:00 --cpus-per-task=4 \
  slurm/single_node.sh config/smoke.toml
```

Then submit the complete local study or only the focused deep study:

```bash
sbatch --time=01:00:00 --cpus-per-task=8 \
  slurm/single_node.sh config/local.toml

sbatch --time=01:00:00 --cpus-per-task=8 \
  slurm/single_node.sh config/deep_graph.toml
```

Add the site's policy options on the `sbatch` command line, for example
`--account`, `--partition`, `--reservation`, or `--qos`. The supplied scripts
do not guess these values. `OMP_NUM_THREADS` and `MKL_NUM_THREADS` default to
one in the scripts so a synthetic objective term does not create nested thread
pools.

The single-node script must be launched as one Slurm task. Do not use `srun`
with several ranks for a local TOML file: every rank would independently run
the same local sweep.

### 3. Validate MPI with two ranks

Before a full allocation, run one coordinator and one worker. In an interactive
allocation this can be done with:

```bash
salloc --nodes=1 --ntasks=2 --cpus-per-task=1 --time=00:10:00

srun "$CHEMFIT_PYTHON" benchmark/chemfit_runner.py \
  --name mpi-cluster-smoke \
  --backend mpi --workers 1 \
  --task-kind sleep --task-duration 0.001 \
  --components 4 --graph-depth 3 \
  --evaluations 2 --batch-size 2 \
  --warmup-evaluations 1 --repeats 3 \
  --results-dir "$CHEMFIT_RESULTS_DIR/mpi-smoke"
```

This verifies imports, MPI ABI compatibility, rank startup, worker shutdown,
and shared result-path access before consuming multiple nodes.

### 4. Run multi-node MPI suites

MPI needs one coordinator rank in addition to the configured worker ranks:

```text
required Slurm tasks = maximum configured workers + 1
```

Both supplied MPI files have a maximum of seven workers, so they require eight
Slurm tasks. A two-node example is:

```bash
sbatch --nodes=2 --ntasks=8 --ntasks-per-node=4 \
  --cpus-per-task=1 --time=01:00:00 \
  slurm/multi_node_mpi.sh config/mpi.toml
```

Run only the focused deep-graph MPI cases with:

```bash
sbatch --nodes=2 --ntasks=8 --ntasks-per-node=4 \
  --cpus-per-task=1 --time=01:00:00 \
  slurm/multi_node_mpi.sh config/deep_graph_mpi.toml
```

Choose `--nodes` and `--ntasks-per-node` to match the machine. The benchmark
does not require two nodes, but spreading ranks across nodes is necessary when
the goal is to include inter-node communication. Add account and partition
options on the command line as required by the site.

The script launches exactly one `srun` step. For each configuration, global
rank 0 and the requested workers form an active communicator. Extra allocated
ranks wait while smaller worker-count points run. Rank 0 drives
`evaluate_many()`, worker ranks stay in `worker_loop()`, and closing the
prepared schedule releases the workers before the next point.

### 5. Select configuration and output through environment variables

Passing the TOML file as the first script argument is clearest, but both scripts
also accept environment variables:

```bash
export CHEMFIT_BENCH_CONFIG=config/deep_graph_mpi.toml
export CHEMFIT_RESULTS_DIR="$PWD/results/deep-mpi-commit-a"

sbatch --nodes=2 --ntasks=8 --ntasks-per-node=4 \
  --cpus-per-task=1 --time=01:00:00 \
  slurm/multi_node_mpi.sh
```

Command-line `--results-dir` is handled inside the script through
`CHEMFIT_RESULTS_DIR`. Every expanded configuration writes a separate unique
JSON file, so a partially completed job retains all points finished before a
timeout or failure.

### 6. Collect and compare results

Slurm normally starts a job in the submission directory; use `sbatch --chdir`
if the site behaves differently. Standard output goes to Slurm's usual job log
and includes one summary line per completed configuration. JSON files go to
`CHEMFIT_RESULTS_DIR`.

After the job finishes, plot on a login or analysis node:

```bash
source .venv/bin/activate
python plot_results.py \
  results/deep-mpi-commit-a \
  --output plots/deep-mpi-commit-a
```

For ChemFit commit comparisons, use the same TOML file, Slurm resource request,
module stack, and node type. Install each commit in its own environment and
write it to a separate results directory. Every JSON record also stores the
detected ChemFit version, Git commit, dirty-tree state, hostname, and Slurm
allocation metadata.

### Common cluster mistakes

- Fewer than `workers + 1` MPI ranks causes the runner to reject the point.
- Running a local TOML file under a multi-rank `srun` duplicates the entire
  local sweep on every rank.
- Building `mpi4py` with a different MPI implementation from `srun` commonly
  causes import failures, startup failures, or hangs.
- Requesting fewer CPUs than threaded workers oversubscribes the allocation
  and invalidates scaling measurements.
- Mixing results from different node types or resource requests makes commit
  comparisons difficult even though metadata is retained.
- Relative result paths are resolved from the job's working directory; use an
  absolute `CHEMFIT_RESULTS_DIR` on unfamiliar systems.

## Development checks

```bash
pytest -q
ruff check .
```

The test suite covers the direct, serial, and threaded paths, TOML expansion,
correct result ordering and values, raw timing preservation, and non-overwrite
result behavior. MPI and Slurm require an MPI-capable host or cluster.
