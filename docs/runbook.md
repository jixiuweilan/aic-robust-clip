# Implementation Runbook

The N01–N05 command workflow is implemented. See [the current handoff](handoff.md)
for Miniconda setup, user-operated downloads, exact commands and remaining
validation gates. Formal work belongs on a separately enrolled RTX 4060 or
Tesla T4 machine. See [FP32 performance rollout](t4-performance.md) for opt-in
batching, safe prefetch and bounded remote measurements. A passing fixture/startup check is not a score or a completed
experiment. Official-weight B03 CPU startup passed; target-machine CUDA startup
remains pending.

## Local development machine

Run the dependency-free contract and fixture suite from the repository root:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .conda/aic-robust-clip/bin/python -m compileall -q src tests
PYTHONPATH=src .conda/aic-robust-clip/bin/python -m aic_robust_clip.submission --help
```

The optional finite startup check uses only generated tensors and has a hard
limit of eight samples and two optimizer updates:

```bash
.conda/aic-robust-clip/bin/python -m pip install --no-index --no-deps --no-build-isolation -e .
.conda/aic-robust-clip/bin/aic-startup-check
```

Do not point the startup check at competition archives. Do not turn the smoke
configuration into a formal run, repeat it as training, or use its loss as an
accuracy result. System Python does not have the training stack. Synthetic
CPU regressions were verified in an isolated `/tmp` environment with the
versions in `requirements-test-cpu.txt` (Python 3.14.6). That lock deliberately
does not contain transformers or pretrained weights and is not a formal GPU
environment. This older check did not verify real-model startup; the subsequent
project-Conda B03 official-weight CPU check is recorded below.

Current setup uses Miniconda, not venv. All downloads are delegated to the user.
The earlier `/tmp` environment is historical and is no longer available.
To prepare and verify the current CPU regression environment:

```bash
# The project environment already exists on this development machine.
# On a new checkout only, with the required Conda packages cached:
# conda create --offline --prefix "$PWD/.conda/aic-robust-clip" python=3.11 pip
# USER performs the following dependency download:
.conda/aic-robust-clip/bin/python -m pip install -r requirements/cpu-test.in
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -v
```

The tests use temporary generated PNGs/tensors only. Every trainer invocation
is smoke mode with at most three cumulative optimizer updates. No competition
archives, real weights, full epochs or benchmark scores are involved. Tensor
tests are explicitly skipped when torch is absent; a dependency-free green
suite alone is not tensor-path validation.

Current Conda verification on 2026-09-16 after user dependency installation:
**45 passed, zero skipped**, including new tensor/E2E checks and NumPy RNG
restoration. The synthetic startup stopped after two samples/two updates.
See [the validation record](validation-20260916.md) for exact provenance and
remaining target-GPU checks. This supersedes the earlier 29-pass/16-skip run.
After user weight provisioning, offline hash checks and one real ViT-B/32 B03
CPU startup also passed, stopping at two samples/two updates. The verified
directory is `checkpoints/openai-clip-vit-b32`; the interrupted download is
retained separately. No other real-weight recipe or target CUDA check ran here.

Historical review-fix verification on 2026-09-16: **36/36 tests passed** in the isolated
CPU environment; system Python passed 25 and explicitly skipped 11 tensor
tests. The startup command stopped after two samples and two optimizer updates.
Compilation, CLI help and whitespace checks passed. No real competition images,
pretrained weights, GPU work or formal training were used. NumPy-dependent RNG
restoration remains untested in this CPU environment; Python/Torch RNG resume
was checked with stochastic fixtures.

## Separate data-preparation/training machine

Provision the exact dependency lock and an immutable Hugging Face revision of
`openai/clip-vit-base-patch32` before running. Record the revision, local
weight-directory digest, code revision, configuration digest, seed, and data
artifact hashes in the run output. Keep each competition stage in its own
registered data namespace.

Run a complete decoded training audit and a packaging/integrity-only test audit
with separate output paths:

```bash
aic-audit-archive data/train.zip --stage preliminary --role train \
  --output outputs/preliminary/train-audit.json
aic-audit-archive data/test.zip --stage preliminary --role test --no-decode \
  --output outputs/preliminary/test-audit.json
aic-build-class-map outputs/preliminary/train-audit.json \
  --output outputs/preliminary/class-map.json
aic-make-split outputs/preliminary/train-audit.json \
  --output outputs/preliminary/split.json \
  --report outputs/preliminary/split-report.json
```

The split algorithm is now `grouped-observed-label-stratified-v2`. Regenerate
old splits and dependent caches/checkpoints: the v1 partition objective was
incorrect. With 200 independent samples per class, v2 gives 160/20/20; duplicate
groups can still force deviations, and their integrity takes priority.

The split command rejects partial manifests, decode failures, test records,
mixed stages, and stale manifest digests. Review audit failures and conflicting
duplicate-label groups before any formal run; no tool silently changes labels.
The near-duplicate screen is not implemented as confirmed scene identity, so
do not treat candidate similarity as proof of independence.

## Baseline and research handoff

Use the fixed class map and split digests in every resolved training
configuration. B01 consumes normalized cached CLIP features; B04 consumes the
same official preprocessing through a frozen encoder; B03 adds only visual
Q/V LoRA and uses the same classifier contract. W/P/I and GCE/SCE execute in
the common trainer and are disabled unless explicitly configured. See
`configs/research-methods.json`: merge a recipe's `parameters` into the run
configuration, then call `TrainConfig.from_run(run)`. Method presets are not
complete formal experiment profiles and have no measured run.

### Data loading and research API

For ordinary PyTorch DataLoader use `collate_fn=collate_samples` from
`aic_robust_clip.data.loading`; transforms must decode bytes and return tensors.
For checkpoint/resume use `StatefulBatchLoader` instead. Its default has no workers;
formal configurations can opt into ordered spawn prefetch while checkpoints
still persist only the delivered-sample cursor and deterministic epoch order.
Close streams explicitly or use their context manager. Construct them with
`max_samples=run.max_samples` in smoke mode, where prefetch is forbidden. Local
PyTorch loaders must use `num_workers=0` and the resolved small batch size.

Construct `ManifestDataset.from_split` using the **complete parent manifest**
and complete split records; the constructor validates their digest, coverage,
stage and group isolation before selecting a partition.

The common `train_baseline` API additionally accepts:

- `training_labels`: exact active training sample-ID to index mapping for W/I.
  For a bounded stream, these are the IDs selected in `train_loader.order`, not
  all IDs in the archive. I counts these labels before weighting.
- `scoring_loader`: a separate, unshuffled stateful stream over the same active
  IDs, records and class map, using a deterministic view and purpose `scoring`.
  W scores raw CE and updates its history; dev/confirm/test records are rejected.
  A partial smoke pass does not advance formal warm-up epochs.
- `reference_encoder`: an independently loaded, frozen original encoder for P.
  It must share no parameters with the adapted model. The model must provide
  `forward_with_features` (implemented by `VisualLoRAModel`). Both branches see
  the exact same input tensor; the reference is not needed for prediction.
- `reliability`: optional explicit tiny history fixtures for testing the
  post-warm-up path. Resume restores the saved state instead.

Training consumes at most eight samples in smoke; each auxiliary reference or
scoring pass also has a separate cap of eight, and evaluation has at most two
batches. Actual counts are returned. These are correctness budgets, not epochs.
I modifies training logits only; evaluation/prediction retain raw model logits.

### Checkpoints and exact resume

Supply `checkpoint_dir` and `checkpoint_metadata` to write `last.pt` and, when
dev evaluation is supplied, `best.pt`. The metadata must match the run lineage
and use `configuration_digest=train_config.digest` (not just `run.digest`).
Actual configuration, dependency versions, epoch logs, prior and W history are
stored alongside optimizer, scheduler, RNG and stream state; checkpoint SHA-256
values are returned in the result. Saves use atomic file replacement.

Use `stop_after_updates=1` for a bounded pause at a completed update, then pass
`resume_from=.../last.pt` with the **same** resolved TrainConfig and a fresh
matching stateful stream. Do not reseed or replay consumed samples. Smoke caps
are cumulative across resume; a completed smoke checkpoint cannot receive a
fresh budget, and formal/smoke configurations cannot silently interchange.
Do not pause at the final smoke cap; omit the pause to finish bounded evaluation.
Checkpoint boundaries never contain pending accumulated gradients.

The configured scheduler (`constant`, `step` or `warmup_cosine`) is covered by
the config digest and persisted state. The command workflow resolves CACHE20 /
ONLINE10, AdamW head/LoRA groups, one-epoch LR warm-up, cosine decay, and effective
batch 128 on the experiment machine. Smoke resolves accumulation to one and
never executes the formal epoch budgets. Shared HEAD3 initialization, sharded
memory-mapped caches and CLI orchestration are now implemented; their execution
is always explicit. Final target-machine acceptance remains pending.

The v2 checkpoint selection tuple is dev macro recall, micro Top-1, then earliest
epoch. Old scalar-selection checkpoints cannot resume under the new trainer.
Pipeline checkpoints also bind weight files, preprocessing, initialization and
execution mode. Effective-batch weighting uses total weight mass; preservation
uses actual sample count, including incomplete batches. See `tests/test_workflow.py`.

Before formal fitting, verify that the model reports only the intended
trainable parameters, that smoke mode cannot resume a formal output namespace,
and that all auxiliary passes have finite sample/update caps in local checks.
Formal feature caching, head initialization, epochs, validation selection, and
the research matrix are not run in this checkout.

## Prediction and packaging

Before preparing an official leaderboard submission, satisfy the
[internal leaf-only eligibility gate](research/submission-gate.md). A baseline,
screening winner, or passing CSV validator alone is not eligible.

After recipe selection on the separate machine, load one checkpoint, one class
map, and one preprocessing flow. Predict only from the explicit test manifest;
perform no fitting or adaptation. Write headerless `pred_results.csv`, then
validate it against the exact test filename list and class map:

```bash
aic-validate-submission outputs/pred_results.csv \
  --expected-files data/test_filenames.txt \
  --class-map outputs/preliminary/class-map.json
aic-package-submission outputs/pred_results.csv outputs/submission.zip \
  --expected-files data/test_filenames.txt \
  --class-map outputs/preliminary/class-map.json
```

The ZIP must contain only `pred_results.csv` at its root. Validate the final
ZIP again. A validated local archive is a software artifact, not an official
submission; do not commit data, weights, checkpoints, or generated outputs.
