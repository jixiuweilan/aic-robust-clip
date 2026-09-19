# Implementation Task Plan and Agent Handoff

Current update (2026-09-17): B01 execution artifacts and a separate-machine B03
CUDA startup report have been reviewed; supplementary prediction/cache checks
pass. Historical logs are lost, a rerun is teammate-reported, and the user elected
to proceed with the [B04/B03 matched comparison](next-online-pair.md).
The implementation ledger below is the historical 2026-09-16 checkpoint, not a
claim that the received B01 run never happened or that full T09 is now closed.

Prepared **2026-09-16**. Status: **review defects repaired with bounded synthetic
regression coverage; full task acceptance remains pending**. The N01–N05
engineering implementation and command handoff are available in
[the current handoff](handoff.md); T09 acceptance and research experiments
remain pending. This document defines work for
agents assigned by the user. It does not authorize running formal training on
the current machine.

## Goal and current evidence

Deliver a reproducible pipeline from official archives to audited manifests,
fixed splits, frozen-CLIP baselines, bounded training startup checks, and valid
prediction packages. Then implement the minimum research comparisons as
independent modules. Formal experiments execute on a separate training machine.

Observed archive-directory facts from 2026-09-16:

| Archive | Compressed bytes | Images | Organization |
| --- | ---: | ---: | --- |
| `data/train.zip` | 19,978,676,862 | 103,218 JPGs | 500 numeric class folders; 201–213 images per class |
| `data/test.zip` | 1,012,335,166 | 24,967 JPGs | Flat filenames |

Counts match the archived preliminary-stage specification. This is a stage
inference, not proof of source provenance. ZIP directory inspection found no
duplicate member paths, unsafe paths, encryption, or nonimage files. CRCs,
image decoding, image-level duplicates, and full archive SHA-256 remain
unchecked. No class-name mapping or separate clean validation set was found
inside these two archives. Both archives are ignored by Git.

The implementation now contains versioned contracts, read-only archive audit,
exact-duplicate grouping and deterministic splits, explicit manifest loaders,
the locked-route CLIP loader, baseline loop/checkpoint/inference structure,
and a stricter CSV/ZIP validator. The [29 experiment entries](research/experiment-matrix.csv)
remain planned; task completion must not mark them as experimentally
completed. System Python remains dependency-free; an isolated CPU-only test
environment was used for synthetic image/tensor regressions. The project Conda
environment subsequently passed one bounded official-weight B03 CPU startup;
target-machine CUDA and other real-weight recipe paths remain unverified here.

Implemented handoff state:

- T01–T06: grouped split v2 corrects the ratio objective; stale parent manifests
  are rejected; collation and a resumable zero-worker stream are available.
  The common loop restores optimizer/scheduler/RNG/cursor state, writes last
  and dev-selected checkpoints, and rejects incompatible configurations.
- T07–T08: GCE/SCE and W/P/I now execute inside the common loop. Synthetic
  fixtures check method composition, frozen-reference gradients, bounded
  scoring and Q/V-only updates. B03 also passed two real CLIP Q/V LoRA CPU
  forward/backward/update steps on generated images; this is not a scored run.
- N01–N05 code now includes official revision/hash checks, optimizer groups,
  warm-up/cosine profiles, shared head initialization, sharded caches and CLI
  orchestration. With user-installed dependencies in `.conda/aic-robust-clip`,
  [45 CPU tests pass without skips](validation-20260916.md), including tensor/E2E
  checks; a synthetic startup stopped after two samples/two optimizer updates.
- Remaining acceptance work: verify bounded real-model startup on the separate
  CUDA machine and record its
  compatible installed GPU lock. Do not infer this evidence from implemented
  code or historical synthetic checks. T09 is not closed.

## Authority and execution boundaries

Every implementing agent reads [AGENTS.md](../AGENTS.md),
[competition rules](competition.md), [candidate definitions](research/candidates.md),
and [validation protocol](research/validation.md) before its task.

- Current machine: code development, tests, and explicitly bounded startup
  checks only. No full epochs, reported-score classifier fitting, repeated
  smoke checks used as training, full feature extraction, tuning, or matrix runs.
- Proposed local smoke defaults: batch size 1, at most 8 training images,
  2 optimizer updates, a hard maximum of 3 updates per invocation, and at most
  2 evaluation batches. These are implementation defaults for this plan, not
  new organizer rules. Bound all auxiliary scoring/reference passes too.
- Smoke mode skips the formal three-epoch head initialization and all automatic
  full-dataset preprocessing. It must never continue into normal training.
  It uses a dedicated output namespace and cannot silently resume formal runs.
- Test data is prediction-only. Development checks use tiny programmatically
  generated fixtures or a bounded current-stage training subset, never official
  test images. Fixtures are software tests, not a new training dataset.
- Exercise audit/split tools locally on bounded fixtures. Schedule complete
  decoding, hashing, duplicate analysis and feature extraction in the separate
  data-preparation/training environment; do not bundle them into a local smoke.
- Archives remain immutable in their current locations. Stage isolation can be
  expressed through registered paths and manifests; avoid copying 20 GB merely
  to rename directories. Derived artifacts live in ignored stage directories.
- Agent assignment does not authorize account actions, remote provisioning,
  uploading competition data, official submissions, or formal training here.

## Milestones and dependency graph

| ID | Task | Depends on | Proposed ownership | Priority |
| --- | --- | --- | --- | --- |
| T01 | Shared contracts, environment and bounded execution | None | `contracts.py`, `runtime.py`, dependency/config conventions | P0 |
| T02 | Archive inventory and image audit tools | T01 | `data/audit.py`, audit tests | P0 |
| T03 | Duplicate groups, deterministic splits and loaders | T02 | `data/splits.py`, `data/dataset.py`, split tests | P0 |
| T04 | Official CLIP loading and frozen classifier | T01 | `models/clip.py`, `models/classifier.py`, model tests | P0 |
| T05 | Baseline training, evaluation and checkpoints | T03, T04 | `training/`, `metrics.py`, baseline configurations | P0 |
| T06 | Inference, class-ID validation and packaging | T05 | `inference.py`, existing `submission.py`, inference tests | P0 |
| T07 | Visual LoRA and matched online control | T05 | `models/lora.py`, LoRA/online configurations and tests | P1 |
| T08 | GCE and W/P/I modules with isolated controls | T07 | `training/objectives.py`, `training/reliability.py`, experiment configs | P1 |
| T09 | Integration review and training-machine handoff | T06, T08 | Integration tests and runbook | P1 |

All proposed modules are under `src/aic_robust_clip/`; these names describe
future ownership, not existing APIs. Freeze exact interface names in T01.
T01's owner also integrates shared `pyproject.toml`, CLI registration and
top-level documentation changes. Other agents submit requested shared edits to
that owner to avoid simultaneous rewrites.

**First useful milestone:** T01–T06, a locally startup-validated baseline
pipeline ready for a training machine. **Research-code milestone:** T07–T09,
the first seven experiment recipes implemented and handed off. Neither
milestone includes a trained model or a measured accuracy claim.

If the user assigns several agents, T02 and T04 can proceed independently after
T01. T06 and T07 can proceed independently after T05. Keep integration and
shared-file changes under one owner. No agent needs to implement the entire
29-row research menu for these milestones.

## T01 — shared contracts and execution mode

Deliver versioned schemas and examples for:

- Dataset registration: stage, source archive paths, declared provenance,
  expected counts, completeness status and optional official-name mapping.
- Sample record: stage, role (`train` or `test`), archive identity, normalized
  member path, stable sample ID, byte size, CRC, optional content/pixel hashes,
  decode status and nullable class ID. Test records cannot carry fitted labels.
- Class map: exact official string ID to contiguous training index and inverse;
  preserve leading zeros; prohibit silent remapping between checkpoints.
- Split record: sample ID, group ID, partition, seed, algorithm/schema version
  and parent-manifest digest. Partial audits are explicitly marked partial.
- Run configuration: stage, execution mode, manifest/split/class-map digests,
  official weight identity, parameters, seed, output root and bounded smoke
  limits. Resolve defaults once and save the resolved configuration.
- Checkpoint metadata: model family, class map, stage, configuration, data and
  weight identities, code revision, optimizer/scheduler/RNG state, progress,
  and optional module state. Define which mismatches reject resume/inference.

Use OpenAI's Hugging Face CLIP ViT-B/32 package as the initial implementation
route (`openai/clip-vit-base-patch32`); lock an inspected revision and compatible
dependencies during implementation. This is a project choice within the
competition's permitted routes. No unverified version numbers are prescribed
by this plan. Keep a CPU-capable test setup and document the separate GPU
installation. Do not vendor weights or silently download them during tests.

Acceptance: invalid stages/roles/configurations fail clearly; local execution
defaults to bounded smoke and rejects full-run profiles; parameter limits
apply before loading/scanning entire data; missing weights produce an actionable
error. Contract fixtures must support the downstream tasks without real data.

## T02 — inventory and audit tools

Implement ZIP inspection without mandatory extraction. Full mode should stream
archive hashes and member decoding with bounded memory. Verify member CRCs,
read images with Pillow, and report every failure with a stable sample ID.
Separate strict decoding failures from explicitly enabled truncated-image
recovery; record the policy and affected files. Never silently discard images.

Define byte hashes and decoded-pixel hashes exactly: decoder/version, EXIF
orientation policy, RGB conversion, dimensions, framing and byte order must
be fixed. Read-only analysis must not rewrite archives or labels. Reject path
traversal, duplicate paths, unsupported encrypted members and unsafe extraction
targets. If extraction is offered, make it optional and deterministic.

Deliver sample/class manifests, an audit summary, provenance and artifact
hashes. Test archive inventory is isolated from training analysis and limited
to packaging/integrity needs; do not use its content to filter or select
training examples. Full audit commands and bounded fixture commands must be
distinct and their completion states distinguishable.

Acceptance: fixtures cover valid data, bad CRC/decode, truncation recovery,
duplicate names, unsafe paths and conflicting class IDs. Repeated runs with
unchanged inputs/policy produce identical logical manifests. Partial audits
must not masquerade as complete manifests for formal training.

## T03 — groups, splits and data loading

Build duplicate groups using current-stage training data only. Exact byte/pixel
duplicates are mandatory; near-duplicate handling must record its detector,
threshold and automatic grouping rule, with candidate-only output clearly
distinguished from confirmed grouping. A simple deterministic perceptual-hash
screen is a possible first implementation; its matches are not proof of the
same scene. Report false-positive/chaining risk and avoid claiming complete
duplicate removal. Grouping revisions invalidate dependent splits.

Implement the frozen, grouped, observed-label-stratified 80/10/10 protocol with
seed 17 from [validation](research/validation.md). Prioritize training support
where scarcity prevents three partitions. Keep conflicting-label groups intact;
do not alter labels. Report per-class split counts, deviations and coverage.

Load data from an explicit manifest, not by recursively discovering all images.
Support worker-safe ZIP access or a validated extracted-path backend. Enforce
stage, split and role at loader construction. Train/scoring/history loaders may
not accept dev, confirm or test records. Keep confirm access explicit and
separate from routine per-epoch evaluation.

Acceptance: deterministic output across discovery order, no group crossing,
no sample missing/duplicated across partitions, correct small-class behavior,
rejected stage mixing, rejected test training, and stable sample identities
across workers. The split report calls held-out noisy labels `noisy_proxy`.

## T04 — official CLIP and classifier

Load only the locked OpenAI ViT-B/32 weights and matching preprocessing. Record
the actual revision and weight digest. Implement normalized image features and
the linear classifier specified in [candidates](research/candidates.md).
Freeze the intended parameters and expose trainable-parameter names/counts.

Implement a streaming feature-cache interface keyed by stage, source/split
hashes, weight identity and exact preprocessing. Formal cache generation runs
on the training machine; local tests use small fixtures. Reject stale or
mismatched caches. Do not cache random online augmentations as if they were
the fixed-view baseline. Numeric class labels suffice for this task.

Acceptance: feature shapes/norms are correct; deterministic evaluation repeat
is reproducible within declared tolerance; frozen weights receive no updates;
one classifier optimizer step changes the intended parameters. Real pretrained
weight validation and mocked structural tests are reported separately.

## T05 — baseline loop, metrics and checkpoints

Implement B01 frozen-feature classification and B04 online frozen-encoder
control, plus a common training loop. Use [experiment profiles](research/experiments.md)
as formal-mode specifications. Smoke uses their code paths with strict bounded
data and updates, not their full epoch budgets or three-epoch initialization.

Provide micro Top-1, macro recall with class coverage, fixed head/mid/tail
recall, per-epoch logs and per-image predictions. Selection reads dev only.
Implement explicit no-tuning confirm evaluation. Save last and dev-selected
checkpoints without silently choosing by test/confirm scores.

Checkpoint/resume must include optimizer, scheduler, random states, sampler
progress and lineage. Test split/stage/class-map incompatibility. Preserve
gradient accumulation semantics for incomplete batches. All local startup runs
must stop automatically, including error/retry paths; no hidden epoch scans.

Acceptance: compare a tiny uninterrupted run to a paused/resumed run under
deterministic conditions; hand-computed metrics match; dev cannot enter the
training loader; smoke asserts actual optimizer-update and sample counts.
No local accuracy benchmark or full baseline fit is part of acceptance.

## T06 — prediction and submission

Load one checkpoint and its exact class map/preprocessing. Support bounded
prediction checks locally using fixtures or held-out training images. The
formal prediction path streams the official test manifest only after recipe
selection and performs no adaptation or fitting.

Extend the existing validator to check allowed class IDs and exact filename
coverage from manifests. Reject leading/trailing filename whitespace rather
than silently stripping it; distinguish filenames from paths on both slash
conventions. Preserve current command compatibility where feasible. Generate
headerless `pred_results.csv`, retain four-digit IDs, and package it at ZIP root
without extra files. Validate before and after packaging; fail instead of
silently dropping a failed prediction or overwriting an existing run.

Acceptance: fixtures cover missing/extra/duplicate images, out-of-map IDs,
filename case/whitespace, invalid columns, empty input, malformed ZIP and
nested paths. Mock logits must map back to the expected official IDs. A
locally validated archive is a software artifact, not a competition submission.

## T07 — visual LoRA and the matched control

Implement B03 using visual Q/V-only LoRA, leaving K, original backbone and
text weights frozen. Use T05's same classifier, views and initialization
contract as B04. Implement formal three-epoch head initialization as a separate
training-machine step; local smoke uses a bounded initializer.

Acceptance: zero-initialized LoRA initially matches the base within tolerance;
only allowed LoRA/head parameters update; Q/K/V slice tests catch accidental
K updates; checkpoint reload preserves predictions; any merge/export path
matches unmerged evaluation. One bounded local forward/backward/update check
passes, or a hardware/missing-weight limitation is recorded without claiming
real-model verification. An OOM must not trigger automatic larger/full retries.

## T08 — first research modules

Implement R01 (GCE), F100 (W), F010 (P), and F001 (I), with the equations,
warm-up and data restrictions in [candidates](research/candidates.md). Keep
all modules disabled in the B03 control. Their combinations should be
configurable so later factorial cells do not need another trainer rewrite.

- W: detached per-sample EMA losses, within-observed-class ranks, tie/scarce
  class behavior, next-epoch application, and resumable state. Local tests can
  inject explicit tiny history fixtures to exercise post-warm-up logic;
  never run epochs merely to reach it.
- P: the same view through adapted and frozen original encoders; no teacher
  gradients; no teacher at inference; include its pass in the local budget.
- I: counts from the train partition before weighting; correct train/inference
  logit signs; no post-hoc double correction or test-derived priors.
- GCE: stable probability math, expected limits/gradients, no hidden W/P/I.

Acceptance: all-off matches B03; GCE/weighting/prior fixtures have known expected
values; buffers cannot accept dev/test IDs; save/resume includes module state;
bounded startup checks exercise each changed path without formal experiments.
Mark the seven recipe configurations implementation-ready, not measured.

## T09 — integration and training-machine handoff

Review task interactions and run the full relevant test suite, configuration
loading, CLI help, documentation links, packaging checks and `git diff --check`.
Use a bounded real-model smoke where hardware permits. Verify step/sample caps
and prove no smoke path invokes full audits, feature caching or formal fitting.

Provide one runbook containing environment recreation, required data/weight
locations and digests, full audit/group/split commands, baseline/cache commands,
the first seven experiment configurations, resume commands, metrics locations,
and prediction/package commands. Explicitly distinguish commands for this
machine from commands for the separate training environment.

End with a handoff report containing code revision, resolved dependency lock,
completed tasks, tests, actual startup-check results, unverified hardware paths,
and remaining training-machine inputs. Full-data audit/split artifacts are
required before formal experiments, but need not be fabricated to close a
local code milestone. Do not include competition data, weights or generated
predictions in Git or transfer bundles.

## Deferred work

- Official-name prompt branch: P00–P06 remains conditional on an organizer
  class-name map; it is not a prerequisite for the numeric-label baseline.
- Adapters, SCE/ELR, full factorial execution, extra sensitivities and the
  multi-seed/final refit phases follow measured initial results and compute
  allocation. Their research design remains available without implementing
  every feature now.
- Formal runs require the separate machine's configuration and an execution
  allocation. Official stage provenance, baseline and submission limits must
  be resolved before formal competition use. These gaps do not block fixture
  tests or implementation of the current contracts.

## Copyable task assignment

> Implement task Txx from `docs/implementation-plan.md` in
> `/home/t/projects/new`. Read `AGENTS.md` and the task's linked contracts first.
> Work only on this task and its tests/documentation; preserve other agents'
> changes. Coordinate shared-file changes with the integration owner. Use
> bounded fixtures/startup checks locally; do not run formal training or full
> preprocessing. Deliver the implementation, relevant validation, a clear
> list of unverified paths, and a task-scoped commit. Do not mark a research
> experiment completed based on a passing smoke check.

Task status handoff uses `planned -> in_progress -> implemented -> verified`.
`verified` means that task's stated code checks passed; it never means model
quality was established. If real-model smoke is unavailable, distinguish
fixture verification from the pending real-model check in the report.
