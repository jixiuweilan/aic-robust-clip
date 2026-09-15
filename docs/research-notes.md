# Research Plan: Robust CLIP under Noisy Labels

Design review: **2026-09-15**. Status: **research design complete; experiments
not run**. All designs follow [`competition.md`](competition.md).

## Working decision and deliverables

Start with a frozen OpenAI CLIP ViT-B/32 image encoder and a learned classifier.
Use visual LoRA as the main adaptation candidate, then test noise weighting,
feature preservation, and class-prior correction separately and in combination.
If the organizer supplies meaningful class names, add a prompt-learning branch.
A frozen-feature adapter is the low-cost fallback. This ordering is an
engineering hypothesis about diagnostic value; it does not predict the winner.

| Document | Question answered |
| --- | --- |
| [Literature synthesis](research/literature.md) | What does each paper contribute, and what transfers to this task? |
| [Candidate methods](research/candidates.md) | What exactly would we implement and why? |
| [Validation protocol](research/validation.md) | How do we evaluate when training labels are noisy? |
| [Experiment matrix](research/experiments.md) | Which comparisons run first and how do we decide? |
| [Experiment registry](research/experiment-matrix.csv) | Stable experiment IDs, controls, dependencies, seeds, and status |
| [Source register](../references/research-sources.md) | Primary URLs and the evidence actually inspected |

Official class names, current-stage data access, clean validation availability,
hardware, and submission limits remain unknown. The registry is a design
artifact, not an executable trainer configuration. Coefficients are proposed
starting points, and no experimental results or measured speedups are claimed.

## Suggested reading order

1. **CLIP** establishes the required backbone and its image-text representation.
2. **CoOp** adapts CLIP with learned prompt context while freezing pretrained
   parameters, providing a parameter-efficient baseline family.
3. **LoRA** supplies a general low-rank adaptation mechanism for transformer
   layers; applying it to CLIP still requires competition-specific validation.
4. **JoAPR** directly studies prompt learning under label noise and is the most
   task-aligned public paper in the organizer's list.
5. **NLPrompt** adds a recent prompt-specific robust-loss and transport approach.
6. **TrustCLIP** remains a follow-up reading item: its title and venue were
   verified, but its full method was not obtained in this review.

Local copies of five openly downloadable papers and additional primary links are
catalogued in [`../references/README.md`](../references/README.md).

## Experimental questions to answer before optimization

- How much does a frozen CLIP representation retain if the taxonomy contains
  only numeric IDs, and what simple classifier is a defensible baseline?
- How does estimated label reliability vary by class frequency and sample
  difficulty?
- Does automatic weighting improve the available validation metric without
  collapsing minority classes? A noisy held-out set is not clean ground truth.
- Which trainable parameter subset gives the best accuracy-to-memory tradeoff:
  prompt parameters, adapters, LoRA, or a restricted backbone subset?
- How much representation drift from the pretrained CLIP model predicts
  validation degradation?
- Are improvements stable across multiple seeds and across head/mid/tail class
  groups, not just aggregate accuracy?

## Reproducibility checklist for future experiments

- current competition stage and immutable dataset manifest;
- exact OpenAI CLIP ViT-B/32 implementation and weight identifier;
- source revision, environment lock, configuration, and random seed;
- deterministic train/validation split construction using training data only;
- automatic noise-estimation outputs and their hashes;
- aggregate and per-frequency-group validation metrics;
- checkpoint hash and the exact command used to produce predictions;
- proof that final inference uses one model and one inference flow.
