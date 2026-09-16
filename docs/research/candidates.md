# Candidate Methods

Status: **proposed, partially implemented, unmeasured** (2026-09-16). The
numeric-label baseline, visual Q/V-only LoRA structure, and GCE/W/P/I paths
in the common trainer have bounded synthetic regression coverage; this remains a design
specification for measured experiments. Source mechanisms and transfer limits are in
[the literature synthesis](literature.md).

## Candidate map

| Candidate | Inputs needed | Trainable state | Priority |
| --- | --- | --- | --- |
| C0: frozen representation + classifier | Images and official numeric labels | Linear classifier | Required baseline |
| C1: residual feature adapter | Same as C0 | Small adapter and classifier | Low-cost fallback |
| C2: visual LoRA with independent W/P/I modules | Same as C0 | Visual attention LoRA and classifier | Main research candidate |
| C3: LoRA with ELR | Same as C0 and stable sample IDs | Same parameters as C2; prediction-history buffer | Alternative to filtering |
| C4: prompt adaptation | Official meaningful names mapped to numeric IDs | Shared text prompt; frozen CLIP | Conditional comparator |

Class-name availability changes C4, not the viability of C0–C3. Do not treat a
folder named `0007` as meaningful text supervision. Do not obtain species names,
descriptions, captions, or images from external models or datasets.

## Common architecture

Let `f0` be the official OpenAI CLIP ViT-B/32 image encoder and `u0` its
L2-normalized output. C0 uses `z = W_head u0 + b_head`, a randomly initialized
linear classifier fitted only on the current training partition. All visual
comparisons use this same classifier form and official class-index mapping.

C1 uses `u = normalize(u0 + A_up(ReLU(A_down(u0))))`; start with bottleneck
width 64 and zero-initialize `A_up`, so initial features equal C0. This is an
adapter-inspired numeric-label model, not an exact CLIP-Adapter reproduction.

C2 inserts `Delta W = (alpha/r) B A` into the image encoder's query and value
projections in every transformer block. Start with `r=4`, `alpha=4`, and zero
`B`; freeze the original backbone and all text parameters. If the implementation
stores Q/K/V in one matrix, modify only Q/V slices and verify K remains frozen.
The trainable classifier is initialized from the same C0 head for each paired
seed. LoRA still needs activation memory and backward computation through
the image encoder; low parameter count does not imply negligible GPU cost.

All visual adaptation runs share a fixed three-epoch head-fitting initialization
on cached training features. Main training then starts at epoch 1 with fresh
optimizer state. The initialization is part of the reproducible pipeline and
its cost is recorded. It is not the best epoch of a validation-selected C0 run.

## C2 modules: W = weighting, P = preservation, I = prior correction

These are **project-designed mechanisms**, motivated by S03/S04/S08/S10/S14.
They must earn their place in the [factorial experiment](experiments.md).

### W: soft, classwise reliability weighting

Warm up for two main-training epochs with weights 1. At the end of each epoch,
score every training sample on one deterministic view with the current model
in evaluation mode. Scores are raw, unadjusted CE against observed labels.
Keep an EMA of these scores (`beta=0.7`, first observation initializes it).
Do not fit the EMA or ranks on validation/test images.

For each observed class with at least five training samples, let `r_i` be the
ascending midrank of its EMA loss, rescaled to `[0,1]`. Lowest loss has rank 0.
Define next epoch's detached weight:

```text
w_i = w_min + (1 - w_min) * (1 - r_i),    w_min = 0.2
```

Average ranks for ties; if all scores in the class are equal, use weights 1.
Classes with fewer than five samples also use weights 1. Recompute weights
once per epoch. Keep every sample; do not overwrite the organizer's labels.

This rank is a reliability heuristic, **not a calibrated probability that the
label is correct**. It gives similarly sized weight ranges to easy and hard
classes but cannot detect an entirely mislabeled class. Strong weighting can
still suppress useful difficult examples. Compare it to global ranks and to
`w_min=0` as diagnostic ablations, logging per-class weight mass and
`ESS_c = (sum_c w)^2 / sum_c(w^2)`.

### P: preserve pretrained image features

For the same augmented training image, compare the adapted normalized feature
`u_theta` with `u0` from the frozen original OpenAI encoder:

```text
L_preserve = mean_i (1 - dot(u_theta_i, stop_gradient(u0_i)))
```

Use every training sample, independently of W, to avoid conflating the two
modules. Begin with `lambda_preserve=0.1`. This second encoder pass is a
training-only reference with the same permitted pretrained weights; it does
not contribute logits at inference. Release it for final prediction. A fixed
training-view cache can accelerate a separate experiment but is not equivalent
to matching both branches on the same random view.

This constraint may preserve irrelevant features and inhibit necessary
adaptation. A lower feature drift is a diagnostic, not a success metric.

### I: correct the observed training prior

Count labels from the **training partition before weighting**, with Laplace
smoothing: `pi_c = (n_c + 1) / (N + C)`. Freeze this vector for the entire run.
It estimates the observed noisy-label prior, not the true class prior.

```text
L_supervised = sum_i w_i * CE(z_i + tau * log(pi), y_i) / sum_i w_i
L_total = L_supervised + lambda_preserve * L_preserve
```

Use `tau=1` for I on and `tau=0` for I off. W off means `w_i=1`; P off means
`lambda_preserve=0`. At inference, use **raw logits** for an I-trained model.
The smoothing makes this a specified project variant of the Balanced Softmax
form. Freeze counts to isolate weighting effects; if W substantially changes
effective class mass, log the mismatch as a limitation and test effective-mass
priors only in a separately identified extension.

The separate post-hoc control uses an ordinary-CE model and
`z - tau_post * log(pi)`, with `tau_post` selected on development data. Do not
combine that correction with I or add class-balanced sampling to this matrix.

### Loss alternatives

Test GCE (`q=0.7`) and SCE (`alpha=1`, `beta=1`) against the plain C2 CE run,
with W/P/I off. For SCE, use
`alpha*CE + beta*(-sum_c p_c*log(clip(one_hot(y)_c,1e-4,1)))`.
Use stable log-softmax for CE. These coefficients are starting choices, not
paper optima transferred to AIC.

## C3: temporal regularization alternative

Use basic single-network ELR on the same LoRA architecture; W/P/I off.
Maintain `t_i <- beta*t_i + (1-beta)*detach(p_i)`, normalize the buffer to
sum 1 before use, and use `beta=0.7`, `lambda_elr=1` initially. The first
observation initializes `t_i` to the detached prediction. The loss is
`CE + lambda_elr*log(clamp(1-dot(p_i,t_i),min=1e-7))`.
Persist the buffer and stable ID-to-row mapping in resumable checkpoints.

This is an explicitly specified basic-ELR adaptation, not ELR+. Use only
training-sample predictions in the history. Do not average independent models
or average predictions at inference. A dense float32 buffer for all 148,695
second-round examples and 750 classes is about 426 MiB before overhead; the
actual training partition is smaller. CPU storage is possible but transfer
cost must be profiled.

## C4: official-name prompt branch

Use a frozen CLIP image encoder and shared learned text context with 16 tokens.
Keep class names and their numeric-ID mapping fixed. Start with CoOp+CE,
CoOp+GCE, and PromptMAE (`sum_c abs(p_c-one_hot(y)_c)`), each on the same data
and matched budget. The zero-shot reference uses one fixed template,
`a photo of a {official_class_name}.`; it is available only if supplied names
are suitable for the tokenizer and semantically meaningful.

Then compare JoAPR* adapted to ViT-B/32 and an NLPrompt port. Record every
deviation from the paper. Inspect source equations, initialization, and loss
normalization before implementing either. Compare NLPrompt's uniform OT
class marginal with `pi_c` only when long-tail training and names are confirmed.
The latter is an extension, not an NLPrompt reproduction. Keep the full
training population represented in OT; arbitrary small batches may omit most
of 500–750 classes.

Cache frozen image features only when preprocessing is exactly fixed, and
recompute learned text features after updates. Precompute the final text
prototype matrix for inference. Prediction then uses one image encoder and
one prototype matrix, with no mixing of independent classifier predictions.

## Deferred directions and reasons

- DivideMix's two-network recipe: study its mechanism; defer its higher-cost
  adaptation. Training-only teacher use is discussed by the organizer, but
  any eventual implementation must still export a single permitted inference
  model. The final ensemble used by many recipes is not a valid shortcut.
- Test-time adaptation, extra foundation-model teachers, external class
  descriptions, cross-stage data/checkpoint reuse: excluded by the project
  constraints or by the conservative stage-isolation design.
- Hard pseudo-label replacement: defer until a comparator supports its value.
  Misidentified or out-of-class images need not belong to any known class;
  confident relabeling can reinforce a false association.
- Full fine-tuning, higher resolution, and heavy augmentation: defer until
  controlled adaptation and memory profiling justify their cost.

Any final method starts each competition stage from permitted OpenAI weights;
carry the selected algorithm across stages, not earlier-stage learned state.
