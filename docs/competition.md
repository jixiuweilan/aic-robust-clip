# Competition Brief

This document is a working summary of the organizer's materials for the 2026
AIC task **Robust Fine-Grained Image Recognition Fine-Tuning with Noisy
Labels**. It was checked against the official website and PDFs on 2026-09-15.
The organizer's latest notice always takes precedence over this summary.

## Task

Design a robust fine-tuning method for fine-grained classification of natural
plants and animals. Training labels contain noise; the later online stages also
contain long-tailed class distributions. The organizer evaluates predictions
on balanced, manually verified test sets.

The required backbone is OpenAI CLIP ViT-B/32. Suggested directions include
prompt tuning, adapters, LoRA, robust losses, sample filtering, pseudo-label
refinement, representation constraints, and mitigation of catastrophic
forgetting.

## Stage data

| Stage | Classes | Training images | Test images | Noted characteristics |
| --- | ---: | ---: | ---: | --- |
| Preliminary | 500 | 103,218 | 24,967 | Noisy training labels |
| Second round | 750 | 148,695 | 37,444 | Noisy labels and long tail |
| Semifinal | 500 | 90,197 | 24,912 | Noisy labels and long tail |

All stages are single-label classification. Training images are organized by
class directory. Test labels are hidden. The organizer notes that some image
files may appear truncated to ordinary viewers while remaining readable by
Pillow.

## Hard constraints

1. The backbone must be CLIP ViT-B/32; no other or larger vision foundation
   model may replace it.
2. Only the current stage's official dataset may be used. Earlier-stage data
   cannot be carried into later stages.
3. The test set is prediction-only and cannot participate in training by any
   supervised, self-supervised, or unsupervised method.
4. No external public/private dataset or manually added labels may be used.
5. Pretrained weights are limited to OpenAI's public CLIP ViT-B/32 weights,
   obtained through OpenAI CLIP or the corresponding Hugging Face package.
6. Commercial closed-model APIs and online foundation-model inference cannot
   replace the core recognition flow.
7. The final result must come from one model or one inference flow. Ensembles,
   model fusion, and voting are prohibited.
8. The complete training path, including noise screening, must be reproducible
   from submitted code. Manual cleaning cannot be a required preprocessing
   step.
9. Competition data is limited to this competition and must not be leaked or
   reused elsewhere.

## Evaluation

The online metric is Top-1 accuracy:

```text
accuracy = correct predictions / total test images
```

The preliminary score is used for familiarization and advancement but is not
part of the final online aggregate. The organizer states that the second-round
score contributes 40% and the semifinal score contributes 60%. A submission
below a published baseline may be treated as invalid.

The final combines objective evaluation with an offline defense. The defense
focuses on innovation, technical completeness, reproducibility, and
presentation; later final-round instructions are authoritative.

## Prediction submission

Create a headerless CSV in this form:

```csv
xxxxxxxxxxxx.jpg,0001
xxxxxxxxxxxy.jpg,0123
xxxxxxxxxxxz.jpg,0456
```

Requirements:

- exactly two fields per row: exact test filename and class ID;
- preserve filename case and extension;
- class ID is exactly four decimal digits, left-padded with zeroes;
- name the file `pred_results.csv`;
- place that file in a ZIP archive for upload.

The same prediction format applies to the first three online stages. Later
stages also require reproducible code, run instructions, the complete
train/validation/inference path, environment details, and a runnable model or
container. The semifinal additionally requires a technical PDF. Final-round
deliverables are subject to a later notice.

## Track schedule and participation

The track notice says:

- registration opened on 2026-04-28; each task's registration deadline in the
  registration system is authoritative;
- the second round was planned to start by mid-September 2026;
- the semifinal was planned to start by 2026-10-10;
- the national final was planned for mid-to-late November 2026, with its date
  and location to be announced separately.

Teams may contain one to three students from the same institution, with up to
two advisors. The track notice lists a fee of CNY 500 per team. Confirm all
deadlines, eligibility, fees, and deliverables in the live registration system
before acting on them.

## Contacts

- Task QQ group: `1090224462`
- Task email: `zerens@njust.edu.cn`
- Challenge-track QQ group: `981069628`
- Organizer email: `office@aicomp.cn`

## Official sources

- [Task page](https://www.aicomp.cn/tracks/tracks-1/3714.html)
- [Challenge-track notice](https://www.aicomp.cn/notice/notice-1/3629.html)
- [Competition home and registration entry](https://www.aicomp.cn/)
- Archived copies and hashes: [`../references/README.md`](../references/README.md)
