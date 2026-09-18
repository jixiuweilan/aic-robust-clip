# FP32 performance package and T4 acceptance

This is an opt-in execution optimization, not a new research method. No speedup
or one-hour ten-epoch run is claimed until measured on the enrolled server.
The [internal submission gate](research/submission-gate.md) applies: this
engineering package and the B03/B04 controls are intermediate/root results,
not leaderboard submission candidates. Only an optimized, evidenced leaf of
the declared experiment tree may advance to official submission.
Keep the current job, source tree, archive mapping, cache and shared HEAD3
untouched. Deploy in a separate code tree/environment or after the job exits.
Do not update an editable installation used by a live job. Downloads remain
user-operated; all runtime commands are offline.

## Compatibility and controls

Existing JSON files need no changes: microbatch, serial loading, configuration
digests and persisted artifact schemas remain compatible. An optional top-level
`performance` object accepts:

| Field | Absent default | T4 candidate |
| --- | --- | --- |
| `cache_batch_size` | training `batch_size` | 64 |
| `eval_batch_size` | training `batch_size` | 64 |
| `head_batch_size` | training `batch_size` | 128 |
| `num_workers` | 0 | 4 |
| `prefetch_factor` | 2 | 2 |
| `pin_memory` | false | true |

All batch sizes are positive integers. Training/head batches must divide the
declared effective batch; an explicit accumulation override must agree.
Workers range from 0 to 16; prefetch factor from 1 to 4. Unknown fields and
boolean/inexact integer substitutes are rejected. Smoke requires all batches
1, zero workers and no pinning. Validation precedes artifact/image access.
Training microbatch remains `batch_size`. HEAD3 independently derives its
accumulation from the head batch. Dev/confirm evaluation and training reliability
scoring use the eval batch; test prediction is unchanged. Cached features and
HEAD3 stay zero-worker regardless of the pixel-loading worker setting.

The loader uses spawn, ordered delivery and persistent workers. Each request
carries its epoch/index; augmentation remains sample-addressed. Checkpoints
track delivered samples only. Partial reset discards outstanding work; restoring
never advances to the worker dispatch cursor. Worker exceptions fail closed,
with no silent fallback or retry. Archive verification stays process-local:
every new worker verifies the archive independently. Do not edit archives or
mapping files during a job. Worker IPC requires sufficient `/dev/shm` and local
process communication permissions. The worker timeout is 300 seconds; if a
cold archive verification exceeds it, report the failure rather than retrying.

FP32, optimizer, effective batch 128, image processing, epoch budgets, validation
frequency and selection rules are unchanged. Batched kernels can differ in
roundoff: newly extracted features need not be byte-identical. Cache indices,
records and shared-head identity formats do not change; artifact SHA-256 still
identifies actual bytes. Reuse existing verified complete caches and HEAD3.
Do not edit a run's configuration and resume it; keep its original source/config
or begin a new output directory. B04/B03 must use the same chosen performance
settings and the same actual shared-head hash.

## Prepare candidates without touching current runs

Use the server's **actual** B03/B04 ten-epoch config, including its machine file,
relocated archive setup, split, weights, cache and HEAD3 paths. The generator
resolves those paths before copying them. It writes fourteen candidate JSONs
(seven execution variants for each control), but launches nothing. This is not
authorization to run fourteen formal experiments.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Retain the verified AIC_ARCHIVE_LOCATIONS used by the current server task.
.conda/aic-robust-clip/bin/python -m aic_robust_clip.performance_configs \
  --config /ABSOLUTE/PATH/TO/ACTUAL-B03-CONFIG.json \
  --output-dir outputs/t4-performance/candidates-v1
```

The destination must not exist. `current` retains the supplied execution
settings; m16/m32 use microbatch 16/32 and w0/w2/w4 use 0/2/4 workers. The
generator preserves source inputs and optimizer settings, uses effective batch
128, and allocates new formal output paths. It rejects non-CE/shortened profiles.
Existing cache/head paths are intentionally retained: don't run cache/head
generation against an existing directory. If the old cache is partial, keep it
and coordinate a separate, new destination before restarting anything.

The module command below works immediately in an existing editable checkout.
To expose the new `aic-benchmark` console entry point, refresh only the local
package after the active job exits (no dependencies or network):

```bash
.conda/aic-robust-clip/bin/python -m pip install \
  --no-index --no-deps --no-build-isolation -e .
```

## One bounded server acceptance task

First run focused/full tests, compileall, pip check and diff checks with CUDA
hidden and offline variables. Missing dependencies are a blocker, not grounds
to skip tests or download packages. The new multiprocessing fixtures use tiny
generated images/vectors only; a two-update restoration test mocks the formal
policy solely to exercise the production prefetch path. It never uses real
data, weights, a full epoch or a GPU. CPU fixture results are not speed evidence.

```bash
CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -p test_performance.py -v
CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -v
.conda/aic-robust-clip/bin/python -m compileall -q src tests
.conda/aic-robust-clip/bin/python -m pip check
git diff --check
```

Then use one operator-allocated T4 consistently, with no concurrent workload on
that GPU. Check permission/allocation, not just apparent idle status. Record
`nvidia-smi`, all logs, exit codes and source/environment identities.

```bash
# GPU 0 is an example; the execution agent must use the allocated GPU index.
CUDA_VISIBLE_DEVICES=0 .conda/aic-robust-clip/bin/python -m aic_robust_clip.benchmark \
  --config outputs/t4-performance/candidates-v1/B03-m16-w0.json \
  --phase train --warmup-steps 2 --measure-steps 10 \
  --output outputs/t4-performance/measurements/B03-m16-w0-train
```

The benchmark accepts B03/B04 plain CE formal CUDA configs only. It enforces
the normal enrollment/local-host guards. `--phase` is cache/train/eval, never
confirm/test. Warmup is 0..5 steps (default 2), measurement 1..20 (default 10).
Cache/eval and effective training batches are capped at 128. A training step
is an effective-batch optimizer update; other steps are batches. Training
must stop before the first epoch boundary. No validation/model selection or
automatic continuation follows it. Train/eval require the existing shared
HEAD3; cache measurement does not. Up to `workers * prefetch_factor` extra
pixel batches may be read ahead, exclusively in the permitted partition;
the GPU/update limits remain exact. Workers are closed on completion/failure.

Benchmark outputs are isolated and never overwritten. Cache measurements do
not write a cache index or shards. Train measurements save one full checkpoint
to time serialization, marked `BENCHMARK`; the normal checkpoint loader rejects
it for training/inference. These outputs are not scores, formal model results,
resumable experiments or initializer artifacts. Keep them out of deliverable
model directories.

Measure `current`, then m16-w0, then m16-w4. Compare m32-w0/m32-w4 if memory
permits. If four workers lose throughput or exhaust shared resources, measure
the matching w2 configuration. OOM is a recorded failure: use a new candidate
and output directory for a smaller batch, never an automatic retry. If cache
or eval batch 64 OOMs, create an explicit new config with that batch 32.
Select identical settings for the pair using both B03 and B04 measurements,
not B04 alone. Prefer at least 15% GPU memory headroom, accounting for other
processes; within 5% throughput choose fewer resources. Keep the serial
configuration if prefetch makes it slower. Do not run all candidates blindly.

## Interpret results and promote

`benchmark.json` records the resolved config, code/source digest, installed
dependencies, GPU, cold-start/warmup time, steady wall throughput, data wait,
CUDA transfer/compute/optimizer regions, checkpoint write time and peak memory.
CUDA regions are measured by events read after the window, not CPU submission
timestamps; they include idle/host-dispatch gaps and are not additive kernel
profiler timings. The first window may still include cold workers; compare
cold and steady costs rather than hiding archive verification.
Event recording itself has overhead, especially at microbatch 1: treat these
as instrumented short-window estimates and confirm with production epoch time.

Cache throughput includes extraction and CPU feature normalization, **not shard
disk writes**. Training includes loss/gradient checks and CPU logging. Reports
exclude configuration preflight from the measured window; external command
wall time and the first complete epoch must be retained for end-to-end totals.

Estimate an epoch as:

`ceil(train_count / 128) * measured_update_seconds +
 ceil(dev_count / eval_batch) * measured_eval_batch_seconds + checkpoint_cost`.

Budget two checkpoint writes when an epoch improves (best and last), plus cold
startup and cleanup. Multiply epoch cost by 3/10 only as an estimate; include
one-time preparation separately. Never infer LoRA speed from cache speed.
Recalibrate against the first complete production epoch, and again when the
paired jobs share disks/CPUs on separate GPUs. Short-window extrapolations do
not establish convergence or one-hour completion.

Promote only after correctness passes and measured improvement; preserve the
baseline artifacts and original configuration. Begin the new paired ten-epoch
runs with the same HEAD3 and approved settings. A three-epoch screening strategy
is a separate experiment decision, not an engineering speedup. If FP32 remains
too slow, report the measured bottleneck before planning FP16 or DDP.
