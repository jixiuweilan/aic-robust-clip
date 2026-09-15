# Validation and Decision Protocol

Status: proposed (2026-09-15). This protocol governs all rows in the
[experiment registry](experiment-matrix.csv). It is deliberately explicit
about which labels support a conclusion.

## Gate D: establish what data is actually available

Before model experiments, record current-stage archive hashes, file count,
class-ID mapping, optional official names, decoder failures, dimensions,
observed class counts, and duplicate groups. Preserve original files. Use byte
hashes plus deterministic decoded-pixel hashes for exact duplicates. Record
candidate near duplicates for group construction using only current-stage
training data. Never change labels based on manual review.

Create `train`, `dev`, and `confirm` manifests from official training data.
Target an 80/10/10 split with seed 17, stratified by observed class and grouped
by duplicates. A group must never span partitions. Conflicting-label duplicate
groups remain together; label conflicts are reported, not manually corrected.
Use deterministic group assignment minimizing classwise target-count deviation,
breaking ties by seeded group ID. Store actual counts and deviations.

Where class/group scarcity makes three-way splitting impossible, prioritize
retaining training support and report unavailable dev/confirm classes. Do not
break a group or silently omit such classes from reported coverage. Freeze
the manifests before experiments. A revised grouping requires new split IDs
and rerunning affected baselines.

An organizer-provided clean validation set, if it exists and is permitted for
selection, can instead be divided into frozen dev/confirm subsets while keeping
the official training set for fitting. Record its provenance and actual usage
rules. Never assume such a set exists from the balanced clean **test** description.

## Label-quality levels

| Available labels | Honest interpretation | Permitted use |
| --- | --- | --- |
| Organizer-confirmed clean validation | Clean in-domain validation | Development and locked confirmation under organizer rules |
| Held-out official noisy training labels | Agreement with noisy observed labels | Provisional selection; all metrics explicitly marked `noisy_proxy` |
| Automatic high-confidence subset | Model-selected proxy with selection bias | Supplementary diagnostic only |
| Official hidden test labels | Unavailable to team | Final prediction and permitted organizer evaluation only |

A noisy holdout cannot reliably identify clean-test accuracy under arbitrary
instance-dependent noise. Multiple seeds reduce optimization uncertainty but
do not repair incorrect validation labels. No new manual clean labels are
required or introduced by this plan. If clean validation is unavailable, report
the unresolved limitation and treat method rankings as provisional.

## What each operation may access

| Operation | Train | Dev | Confirm | Official test |
| --- | --- | --- | --- | --- |
| Fit weights, prototypes, priors, reliability, ELR history | Yes | No | No | No |
| Tune hyperparameters / choose checkpoints | Fit on train | Score only | No | No |
| Compare frozen shortlisted methods once | Existing trained models | Existing decisions | Score only | No |
| Refit selected fixed recipe | All eligible current-stage official training partitions | Included only if originally official training | Same restriction | No |
| Final inference after recipe lock | No updates | No updates | No updates | Prediction only |

The final refit rebuilds counts, filtering, buffers, and classifier from scratch
on the eligible full training set. Do not reuse a split-trained reliability
cache without rebuilding it. Never fold a separate organizer validation set
into training unless its rules explicitly permit that use.

## Metrics

- Micro Top-1: all correct predictions / all evaluated images. This matches
  the organizer's metric definition, but local label quality still matters.
- Macro recall: mean of classwise correct / classwise support, over classes
  present in the evaluation split. Always report present classes / total classes.
  This better reflects balanced-class evaluation when the holdout is long-tailed.
- Head/mid/tail macro recall: rank all classes by **training** count, then
  deterministic class ID; divide into three roughly equal groups of classes.
  Freeze these groups for every comparison. They are project diagnostics, not
  an organizer metric. Show support, missing classes, and minimum class recall.
- Noise diagnostics: raw-loss distribution, classwise weight mass, ESS,
  pseudo-label change rates where applicable, and train–dev gap. Without clean
  training labels, do not report noise-detection precision or true noise rate.
- Feature drift: mean `1-cos(u_theta,u0)` on a fixed training-only diagnostic
  subset and view. It supports P analysis; it never decides the winner alone.
- Resources: peak allocated/reserved GPU memory, total training seconds,
  scoring/teacher/OT time, images per second, trainable parameter count, and
  inference latency with warm-up, batch size, device and precision recorded.

## Selection without feedback leakage

Use dev macro recall as the primary local ranking statistic, with micro Top-1
and tail recall reported alongside. If labels are noisy, call this ranking
provisional. For screened runs, evaluate each epoch and select the highest dev
macro recall, then dev micro Top-1, then earliest epoch. Report last-epoch scores
too. Never select a seed because it wins on confirm or the leaderboard.

Use seeds 17, 29, 43 for the shortlisted baseline and candidates. Compare paired
seeds with identical partitions and preprocessing. Save per-image predictions
for a paired bootstrap of the metric difference, resampling duplicate groups
(not independent images) on dev. Use 2,000 replicates and a fixed bootstrap seed
101; report a 95% interval and mean/standard deviation over seeds separately.
With three seeds, characterize uncertainty cautiously. Group bootstrap does
not account for mislabeled ground truth or repeated hyperparameter search.

Predeclared **project heuristic**, not a significance theorem: provisionally
advance a more complex method if mean dev macro recall improves by at least
0.5 percentage points, at least two of three paired seeds improve, and tail
macro recall does not drop by more than 1 point. If the paired interval crosses
zero, mark the finding inconclusive. Keep cheaper tied methods. Any override
must be documented before confirm is examined.

Open confirm only after the shortlist, coefficients, epochs, and selection rule
are frozen. Report every shortlisted result. A confirm failure may invalidate
the proposed gain but must not trigger repeated tuning against that same set.
Use future independent evidence for a new claim. A confirm set containing noisy
labels remains a proxy even if it agrees with dev.

## Synthetic perturbation experiments

An optional stress study may add deterministic label flips **only to training
labels**, without overwriting the original manifest. Call 10%/20% the added
flip rate, not the total true noise rate. Flips can accidentally repair an
already wrong label. Preserve dev/confirm, architecture and seed; record the
corruption seed and transition rule. A within-training subset is allowed;
external CIFAR/Food101N datasets from papers are not part of this AIC plan.

## Artifacts required to interpret a run

Record experiment ID, parent/control ID, stage, split hashes, label-quality
level, class map and name source, official weight hash, code revision,
dependency lock, complete resolved parameters, seeds, per-epoch metrics,
per-image predictions, resource measurements, selected checkpoint hash, and
resume state (optimizer/scheduler/RNG/weights/history). Store sensitive data
and outputs in ignored local directories.

Final packaging records the exact model, preprocessing, class map and any
fixed prior correction. Inference uses one crop and one selected model; seeds
are for evaluation of stability, never an ensemble. The existing CSV validator
does not yet verify stage-specific allowed class IDs; implement that check
against the actual class map before submission. Exact test filename coverage
must also be checked.
