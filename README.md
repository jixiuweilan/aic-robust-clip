# Robust Fine-Grained Recognition with Noisy Labels

Starter repository for the 2026 AIC competition task **Robust Fine-Grained
Image Recognition Fine-Tuning with Noisy Labels**.

- Official task page: <https://www.aicomp.cn/tracks/tracks-1/3714.html>
- Track notice: <https://www.aicomp.cn/notice/notice-1/3629.html>
- Local official documents: [`references/official/`](references/official/)
- Source review date: **2026-09-15**

The repository contains the competition brief, archived reference material,
a [research design](docs/research-notes.md), a project layout, and a
submission-format validator. The research design includes a literature
synthesis, candidate methods, validation protocol, and 29 planned experiment
entries. It does not yet contain a training or inference implementation.

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
PYTHONPATH=src python3 -m aic_robust_clip.submission outputs/pred_results.csv
PYTHONPATH=src python3 -m aic_robust_clip.submission outputs/submission.zip
```

Optionally compare filenames against an official test-image list containing
one filename per line:

```bash
PYTHONPATH=src python3 -m aic_robust_clip.submission \
  outputs/submission.zip --expected-files data/test_filenames.txt
```

Run the current test suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Data handling

Competition datasets, model weights, and generated outputs are deliberately
excluded from Git. Keep every competition stage isolated under `data/`; never
copy samples from one stage into another. See [`data/README.md`](data/README.md).
