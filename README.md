# Robust Fine-Grained Recognition with Noisy Labels

Starter repository for the 2026 AIC competition task **Robust Fine-Grained
Image Recognition Fine-Tuning with Noisy Labels**.

- Official task page: <https://www.aicomp.cn/tracks/tracks-1/3714.html>
- Track notice: <https://www.aicomp.cn/notice/notice-1/3629.html>
- Local official documents: [`references/official/`](references/official/)
- Source review date: **2026-09-15**

The repository contains the competition brief, archived reference material,
a [research design](docs/research-notes.md), explicit manifest/audit/split
contracts, frozen-CLIP baseline code, bounded startup training, inference, and
a submission-format validator. The research design includes a literature
synthesis, candidate methods, validation protocol, and 29 experiment entries.
As of 2026-09-17, a separate-machine B01 delivery supports completion of the
first baseline. Supplementary prediction/cache checks pass; historical test
logs are unavailable, and the teammate's rerun is reported rather than independently
verified. The user elected to proceed with these documented limitations.
The next prepared task is the [matched B04/B03 comparison](docs/next-online-pair.md).

The current setup, commands and acceptance ledger are in
[`docs/handoff.md`](docs/handoff.md); implementation details and historical
checks are in [`docs/runbook.md`](docs/runbook.md). Use the project Miniconda
environment `.conda/aic-robust-clip`. All downloads are user-operated.

For teammates cloning the public repository, start with
[team setup and CUDA startup acceptance](docs/team-setup.md). Data, weights,
environments and generated outputs are not included in Git.

For implementation agents: [task plan and handoff](docs/implementation-plan.md)
defines dependencies, deliverables, acceptance criteria and local execution
limits. The reviewed data/trainer defects have regression coverage, including
bounded synthetic resume and research-loss checks. This is not full T01–T08
acceptance: formal profiles and CLI orchestration are implemented, and
[45 CPU tests passed on September 16](docs/validation-20260916.md), including
tensor/E2E checks; the [paired preparation check](docs/next-online-pair.md)
raised the suite to 46 tests. The [T4/relocation regressions](docs/archive-relocation.md)
add another 15 tests while preserving frozen artifact identities.
One bounded official-weight B03 CPU startup also passed (two updates, generated
images only). A separate-machine B03 CUDA startup report has since been
received and reviewed; this machine has not reproduced CUDA execution.

## Local execution limit

This machine is for code-correctness tests and short, explicitly step-limited
training startup checks only, because its GPU memory is insufficient for
formal training. Run formal baselines, tuning, and research experiments on a
separate machine with sufficient memory. See [project rules](AGENTS.md).

## Non-negotiable competition constraints

| Area | Requirement |
| --- | --- |
| Backbone | OpenAI CLIP ViT-B/32 only |
| Pretrained weights | Only OpenAI's public ViT-B/32 weights, via OpenAI CLIP or the corresponding Hugging Face package |
| Training data | Only the official data for the current stage; data from earlier stages must not be reused |
| Test data | Prediction only; it must not participate in supervised, self-supervised, or unsupervised training |
| External data | Not allowed, including public, private, or manually added labels |
| Final inference | A single model / single inference flow; no ensembles, fusion, or voting |
| Online metric | Top-1 accuracy |
| Reproducibility | Training, automatic noise handling, validation, and inference must be reproducible from submitted code |

If this summary conflicts with a newer organizer announcement, the newer
official announcement wins. See [`docs/competition.md`](docs/competition.md)
for the complete working brief.

## Repository layout

```text
configs/             Experiment configuration files
data/                Local competition data (ignored by Git)
checkpoints/         Local model weights (ignored by Git)
outputs/             Logs, predictions, and submissions (ignored by Git)
docs/                Competition brief and research notes
references/official/ Archived organizer PDFs
references/papers/   Public papers named by the organizer
src/aic_robust_clip/ Project Python package
tests/               Lightweight repository tests
```

## Submission validation

The organizer requires a headerless `pred_results.csv` with exactly two
columns per row: image filename and a zero-padded four-digit class ID. The CSV
must then be the sole file at the root of a ZIP archive.

Validate either the CSV or its final ZIP:

```bash
PYTHONPATH=src .conda/aic-robust-clip/bin/python -m aic_robust_clip.submission outputs/pred_results.csv
PYTHONPATH=src .conda/aic-robust-clip/bin/python -m aic_robust_clip.submission outputs/submission.zip
```

Optionally compare filenames against an official test-image list containing
one filename per line:

```bash
PYTHONPATH=src .conda/aic-robust-clip/bin/python -m aic_robust_clip.submission \
  outputs/submission.zip --expected-files data/test_filenames.txt
```

Run the current test suite:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -v
```

Use `aic-audit-archive` and `aic-make-split` for explicit archive/manifest
preparation. Full-data auditing, feature caching, and formal runs are
separate-machine operations; this machine is limited to fixture tests and the
resolved smoke bounds in `configs/baseline-smoke.json`.

## Data handling

Competition datasets, model weights, and generated outputs are deliberately
excluded from Git. Keep every competition stage isolated under `data/`; never
copy samples from one stage into another. See [`data/README.md`](data/README.md).
