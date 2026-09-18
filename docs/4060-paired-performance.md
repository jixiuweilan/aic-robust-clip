# Two-host B03/B04 performance handoff

One teammate owns B03 and another owns B04, each on their own RTX 4060 Linux/WSL
host. Do not update, stop, resume or replace either live T4 run. This assignment
validates an execution profile and runs the matched controls, not a leaderboard
submission. The [leaf-only gate](research/submission-gate.md) remains mandatory.

## Shared inputs and isolated execution

Both teammates must use the same delivered code/source digest, dependency
versions, seed 17, frozen manifest/split/class map, official CLIP weights and
the **same actual** completed server HEAD3 (`head.pt` and `head.json`). Record
both head file hashes before running. Do not independently rebuild HEAD3, use
a benchmark checkpoint, or resume a T4 checkpoint under the new profile.
Online B03/B04 do not require regenerating fixed feature caches.

Use a separate checkout and project-local `.conda/aic-robust-clip` on each
machine. Downloads and asset transfers are user-operated; agents must report
missing inputs, not download them. Enroll each machine separately with
`aic-doctor --bind-training`; never copy another host's binding. Retain the
development-host prohibition. See [team setup](team-setup.md) and
[archive relocation](archive-relocation.md) for environment and path contracts.
The executing agent asks its operator for connection details, authorized GPU
and local paths; no secrets belong in Git or the handoff evidence.

Adapt each host's formal input config to its own existing files, binding and
new output directory. Keep frozen records unchanged and use an absolute
`AIC_ARCHIVE_LOCATIONS` JSON path for relocated archives. Generate candidates
in a new directory as described in [the performance guide](t4-performance.md).
Only use the assigned recipe's `m16-w4` candidate, initially:

```json
{
  "batch_size": 16,
  "effective_batch_size": 128,
  "performance": {
    "cache_batch_size": 64,
    "eval_batch_size": 64,
    "head_batch_size": 128,
    "num_workers": 4,
    "eval_num_workers": 0,
    "prefetch_factor": 2,
    "pin_memory": true
  }
}
```

This is an execution-settings fragment, not a complete runnable config.
Preserve the original ten-epoch CE/optimizer/scheduler/augmentation settings;
accumulation is 8. The generator removes a stale explicit accumulation override.
Both recipes use FP32. Do not infer 4060 throughput or memory use from T4 data.
Config/output paths, host bindings and archive mappings are host-local; shared
scientific inputs and execution settings must match across the pair.

## Acceptance, then the assigned control

Keep `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`. Run these CPU-only checks in each checkout first:

```bash
CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -p test_loader_lifecycle.py -v
CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -p test_performance.py -v
CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -v
.conda/aic-robust-clip/bin/python -m compileall -q src tests
.conda/aic-robust-clip/bin/python -m pip check
git diff --check
```

All tests must pass without skips. Use the delivery receipt's counts: a
pre-existing uncommitted workflow test makes the development tree's count one
higher than the delivered commit. Do not silently skip a missing dependency.
Run the existing generated-image `aic-check-model` bounded CUDA startup check
for the assigned recipe, as described in team setup, on each new host.

Next run one training window (2 warmup + 10 measured optimizer updates) and
three separate evaluation windows (2 + 10 batches each) for the assigned
recipe, each in a new benchmark output directory:

```bash
# Replace paths and GPU index with this host's verified values.
CUDA_VISIBLE_DEVICES=0 .conda/aic-robust-clip/bin/python -m aic_robust_clip.benchmark \
  --config /ABSOLUTE/ASSIGNED-m16-w4.json \
  --phase train --warmup-steps 2 --measure-steps 10 \
  --output outputs/4060-acceptance/train-01
CUDA_VISIBLE_DEVICES=0 .conda/aic-robust-clip/bin/python -m aic_robust_clip.benchmark \
  --config /ABSOLUTE/ASSIGNED-m16-w4.json \
  --phase eval --warmup-steps 2 --measure-steps 10 \
  --output outputs/4060-acceptance/eval-01
```

For the next two eval windows use `eval-02` and `eval-03`; stop on any failure.
These are planned acceptance repetitions, never automatic retries. Evaluation
must record zero workers with batch 64; training must record four workers and
microbatch 16. Preserve stdout **and stderr**, JSON artifacts, exit status,
`nvidia-smi`, `pip freeze`, source identity, cold/steady times, data wait, and
allocated/reserved memory. A successful GPU loop followed by cleanup failure
is a failed run. OOM/SIGABRT/hang: retain the failure, stop, and report; do not
silently fall back, alter precision, reduce epochs or relaunch formal training.

For diagnosing the original worker defect, create a separate config copy with
only `eval_num_workers: 4` changed. On an authorized otherwise-idle training
host, run up to three planned eval windows with unique `diag-eval-w4-01/02/03`
outputs, stopping at the first failure. Retain `failure.json` with its stage
and traceback if present. Never edit the accepted production config in place.
This diagnostic is not required to enable the isolated worker-zero eval path,
but the original SIGABRT cannot be declared fixed without multiworker CUDA
retesting. A pass on 4060 does not establish a fix on the original T4 stack;
T4 retesting must use its separate bench checkout/GPU, not the live training tree.

Once **both** hosts pass their worker-zero profile acceptance, compare timing
and memory headroom (target at least 15%), approve the common configuration,
then each teammate starts only their assigned ten-epoch control in its new
formal output directory. Do not resume benchmark artifacts. The module exposes
the existing `aic-train --config /ABSOLUTE/ASSIGNED-m16-w4.json` entry point.
Calibrate the time estimate against the first complete epoch including full
validation and checkpoint writes; short windows do not guarantee one-hour runs.

Return the matched input/head/source hashes, resolved config, machine/env
reports, acceptance logs, per-epoch train/dev metrics and time, checkpoint
metadata/hashes, and completion/failure status. Do not upload data, weights,
bindings or checkpoints to GitHub. Keep the accelerated pair separately named
from the ongoing T4 controls. Neither pair is eligible for organizer scoring
merely by completing training.
