# Project Instructions

## Purpose and authority

This repository targets the 2026 AIC task "Robust Fine-Grained Image
Recognition Fine-Tuning with Noisy Labels."

1. Treat `references/official/task-rules.pdf` and newer organizer notices as
   authoritative.
2. Keep `docs/competition.md` aligned with the latest official rules.
3. Record the source URL and retrieval date whenever an official artifact is
   added or replaced, and refresh `references/SHA256SUMS`.

## Competition guardrails

- Use OpenAI CLIP ViT-B/32 as the only backbone.
- Use only OpenAI's public ViT-B/32 pretrained weights through OpenAI CLIP or
  the corresponding Hugging Face implementation.
- Use only the current stage's organizer-provided training and validation data.
- Never use test images for training, pseudo-labeling, representation learning,
  model selection, or other supervised, self-supervised, or unsupervised work.
- Do not introduce external image datasets or manually supplied labels.
- Keep preliminary, second-round, and semifinal data physically and logically
  isolated. Earlier-stage data cannot be used in a later stage.
- Final evaluation must use one model and one inference flow. Do not ensemble,
  fuse, or vote across independently trained models.
- Any noise filtering or relabeling required by the final method must be fully
  automatic and reproducible from code.
- Do not commit competition data, checkpoints, credentials, or generated
  submissions.

## Engineering expectations

- Prefer configuration-driven experiments with explicit seeds and data-stage
  identifiers.
- Record dependency versions, configuration, seed, source revision, metrics,
  and checkpoint hash for every result used in a report.
- Keep data auditing separate from training and ensure audit outputs cannot
  silently alter labels or splits.
- Validate `pred_results.csv` with the repository validator before packaging.
- Run focused tests, the full available suite, and `git diff --check` before
  considering a change complete.
