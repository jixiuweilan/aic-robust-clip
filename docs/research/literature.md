# Literature Synthesis

Review date: **2026-09-15**. Source IDs refer to the
[primary-source register](../../references/research-sources.md). Transfer
judgments below are our assessment, not results on AIC data.

## What each family contributes

| Family / sources | Source mechanism | Project use | Main uncertainty |
| --- | --- | --- | --- |
| CLIP, S01 | Image/text alignment supports transfer and text-similarity classification | Frozen representation; optional semantic baseline | Fine-grained name availability and alignment quality |
| CoOp, S02 | Learn prompt context with pretrained weights frozen | Baseline when official names exist | Frozen weights do not prevent prompt overfitting |
| LoRA, S03 | Low-rank updates to frozen transformer matrices | Main visual adaptation family | Original evidence concerns language models, not noisy AIC classification |
| JoAPR, S04 | Adaptive loss partition, label refurbishment, prompt retraining | Conditional noise-handling comparator | Few-shot RN50 evidence may not transfer to large ViT-B/32 long-tail data |
| TrustCLIP, S05 | Title identifies semantic verification and gradient projection | Follow-up reading | Full method unavailable |
| GCE, S06 | Loss family between CE and MAE-like behavior | Cheap single-loss control | Hard clean minority examples may also be downweighted |
| SCE, S07 | Combine CE and reverse CE | Independent robust-loss control | Clipping and coefficient scales |
| ELR, S08 | Regularize toward historical predictions | Single-network temporal alternative | Early predictions may already be systematically wrong |
| DivideMix, S09 | Mixture partition, two networks, semi-supervised learning | Partitioning and confirmation-bias reference | Higher cost; final ensemble output is prohibited |
| Balanced Softmax, S10 | Training-logit correction using class counts | Long-tail loss ablation | Noisy labels distort observed priors; full BALMS is different |
| Logit adjustment, S11 | Prior correction after CE training or within the loss | Cheap post-hoc control | Double correction or wrong sign |
| CLIP-Adapter, S12 | Residual bottleneck on frozen features | Low-cost visual adapter | Numeric classifier is our adaptation of the text-head recipe |
| NLPrompt, S13 | PromptMAE; text-prototype OT partition with CE/MAE routing | Recent prompt comparator | Uniform training-class marginals may conflict with long tails |
| LP-FT, S14 | Fit classifier before adapting backbone | Common head initialization | OOD evidence does not establish noise robustness |

## JoAPR: the mechanism worth testing

S04 §§4.1–4.4 use confidence-penalized warm-up, a two-component Gaussian
mixture over sample losses, joint adaptive thresholds, and label refurbishment.
JoAPR adds predictive entropy to CE for partitioning; JoAPR* omits that term.
Losses are averaged over five epochs. The clean partition uses an **OR** between
its two criteria. Predictions refine even labels in the clean partition; the
noisy partition uses predicted targets. Retraining includes Mixup and a
prediction-distribution regularizer.

The transferable idea is stable selection without treating the clean/noisy
split as certain. Our simpler weighting proposal does not reproduce JoAPR.
Failure hypotheses include overlapping mixture components, overconfident
feedback, and hard minority examples being assigned to noise. Uniform
prediction regularization also needs examination under long-tail training.
[S04](../../references/papers/joapr-2024.pdf), pp. 28697–28700.

## NLPrompt: the simple control before optimal transport

S13 separates PromptMAE from PromptOT. MAE provides a cheap test of whether a
robust loss is already sufficient. PromptOT uses text prototypes to partition
samples; the subsets receive CE or MAE. Its formulation imposes uniform sample
and class marginals.

Balanced test classes do **not** imply balanced training classes. Equal
assignments on long-tail training may create incorrect pseudo-labels. Compare
uniform marginals with a training-prior variant only after simple prompt
controls; that variant is a project extension. The printed cost takes a log of
similarities, but raw cosine scores can be nonpositive. Check the code's
conversion to a positive transport kernel before porting it.
[S13](../../references/papers/nlprompt-2025.pdf), §§4–6.3.

## Loss choices are competing explanations

For probability `p_y` assigned to the observed label:

- GCE uses `(1 - p_y^q) / q`, `0 < q <= 1`; `q -> 0` gives CE. It tests whether
  reducing gradients from poorly fitted labels helps.
  [S06](https://arxiv.org/html/1805.07836v4).
- SCE combines CE and reverse CE. Finite clipping of zero label probabilities
  is necessary. It tests a different loss shape without explicit selection.
  [S07](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.html).
- Basic ELR adds `lambda * log(1 - dot(p, t))` to CE, where `t` is a detached
  running prediction target per training sample. Checkpoint this state. ELR+
  adds mechanisms beyond this variant; its scores are not basic ELR scores.
  [S08](https://arxiv.org/html/2007.00151v2), §§4.2–4.3.

Do not combine these losses with weighting in the first comparison. If GCE
matches a complicated filter, that supports the simpler mechanism. A wrong
early target is a failure mode for temporal methods, especially for classes
poorly represented from the start.

## Long-tail correction: avoid counting one idea twice

For logits `z_c` and observed training counts `n_c`, Balanced Softmax applies
CE to `z_c + log(n_c)`. Training-time logit-adjusted CE at `tau=1` is equivalent
up to a common constant when `pi_c = n_c / N`. They are not independent modules.
After this training, infer with `z_c`.

Post-hoc adjustment takes an ordinary-CE model and predicts with
`z_c - tau * log(pi_c)`. Compare it against training-time correction; do not
apply both by default. Class-balanced sampling changes the comparison, so the
initial experiments share a natural-frequency sampler.
[S10](https://arxiv.org/html/2007.10740v3), §3 and component analysis;
[S11](https://arxiv.org/html/2007.07314v2), §§4–5.

## Synthesis

1. Is the frozen representation sufficient? Compare classifier, adapter, and
   LoRA with matched preprocessing.
2. Is erroneous supervision causing degradation? Compare CE, robust losses,
   temporal regularization, and weighting.
3. Does adaptation erase useful features? Compare the same LoRA with and
   without a feature constraint.
4. Does imbalance explain the error? Test prior correction and report fixed
   frequency-group recall.

Combine modules after measuring their effects. Novelty is unestablished:
combining known modules is an engineering candidate. A research contribution
would need repeatable evidence for a specific interaction, such as classwise
weighting preserving minority recall under noisy supervision.
