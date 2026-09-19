# Experiment Matrix and Execution Order

Status (2026-09-17): **B01 reviewed with documented evidence limitations;
B04/B03 prepared, not run**. The
[CSV registry](experiment-matrix.csv) is the authoritative list of experiment
IDs and dependencies. It is not consumed by a training program. Read
[candidates](candidates.md) for module definitions and
[validation](validation.md) for the selection rule.

Registry status `reviewed_with_limitations` means B01 checkpoint receipts,
prediction arithmetic and cache-index lineage pass, but old execution logs are
unavailable and a teammate-reported rerun has not been independently observed.
The user elected to proceed without further historical-log requests. `prepared`
means code/configuration and a bounded fixture check are ready, not that a
formal experiment passed. Other entries remain `planned`; neither status
closes upstream data/provenance gates. See the [next paired task](../next-online-pair.md)
for the current evidence boundary and release conditions.

## Execution location

Per the [local machine rule](../../AGENTS.md), this machine may run only
code-correctness tests and bounded training startup checks. Every training
profile and scored training run below is intended for a separate machine with
sufficient GPU memory, including frozen-feature classifier fitting. Local
startup checks are not completed matrix runs and do not produce performance
evidence. Code and configurations can be prepared locally.

## 1. Shared profiles

All settings below are starting choices, not measured optima. Use the complete
current training partition, not an externally downloaded benchmark.

| Profile | Training budget | Data / augmentation | Initial optimizer settings |
| --- | --- | --- | --- |
| CACHE20 | 20 epochs on frozen cached features | Official 224px deterministic preprocessing; one feature per image | AdamW; head/adapter LR 1e-3; weight decay 1e-4 |
| ONLINE10 | 10 main epochs after shared 3-epoch head initialization | 224px random resized crop, scale 0.8–1.0, ratio 3/4–4/3; flip probability 0.5; official normalization | AdamW; LoRA LR 1e-4; head/adapter LR 1e-3; weight decay 1e-4 |
| PROMPT20 | 20 epochs | Same image augmentation as ONLINE10; train text prompt only | SGD; LR 0.002; momentum 0.9; weight decay 5e-4 |
| LONG30 | 30 epochs, fresh run for each shortlisted recipe | Same augmentation and optimizer family as that recipe | Keep screened coefficients and LR; stretch the schedule to 30 epochs |

AdamW betas are `(0.9,0.999)`, epsilon `1e-8`; exclude biases and normalization
parameters from weight decay. Use an effective batch of 128, with a smaller
microbatch and accumulation if needed, natural-frequency shuffling, and one
linear LR warm-up epoch followed by cosine decay to zero. W remains disabled
for its first two main epochs. These are separate warm-up mechanisms.
Use float32 initially; a measured mixed-precision port is a separate engineering
change applied consistently to paired controls. Record the final partial batch
and gradient-accumulation normalization. Fix transform implementation/interpolation
versions; B03 and B04 receive the same seeded image stream.

No Mixup, class-balanced sampler, test-time augmentation, EMA parameter model,
or backbone change in the visual core. JoAPR* has its own specified Mixup and
partitioning, which must be recorded as recipe differences. Give prompt
comparators the same 20 epochs, but measure extra partition/OT passes separately.

No GPU-hour estimate is justified yet. On the separate training machine,
before scheduling, measure at least
100 representative steps after warm-up, one scoring pass, and peak memory for
each distinct family. Estimate total cost from epoch steps plus scoring,
reference-encoder and transport passes. Fit the microbatch to the actual
machine; do not assume a named GPU will fit.

## 2. First decision: representation versus adaptation

| IDs | Comparison | Interpretation |
| --- | --- | --- |
| B01 / B02 | Frozen classifier versus frozen-feature adapter | Does a nonlinear feature head help cheaply? |
| B04 / B05 / B03 | Online frozen classifier / online adapter / visual LoRA | Which adaptation helps under the same views and training budget? |
| R01 / R02 / R03 versus B03 | GCE / SCE / basic ELR versus CE | Is noise handling beneficial before adding multiple mechanisms? |
| L01 versus B03 | Post-hoc prior correction | Does simple class-prior adjustment explain a gain? |

B01's cached performance is a starting baseline, not an augmentation-matched
comparison with B03. Use B04 for that claim. B05 makes an adapter–LoRA
comparison fairer. Reuse common head initialization only when its training
partition and seed match. L01 evaluates `tau_post` in `{0,0.5,1}` on dev,
adds no training run, and uses the stored B03 logits.

## 3. Main factorial: W, P, I

Run all cells with the same backbone, initialization, ONLINE10 profile and
seed. The `000` cell is B03 and is reused. Seven new cells complete a 2×2×2
factorial. This reveals interactions that a single add-one-module chain hides.

| ID | W | P | I | Main contrast |
| --- | ---: | ---: | ---: | --- |
| B03 | 0 | 0 | 0 | Plain LoRA+CE |
| F100 | 1 | 0 | 0 | Weighting alone |
| F010 | 0 | 1 | 0 | Feature preservation alone |
| F001 | 0 | 0 | 1 | Prior correction alone |
| F110 | 1 | 1 | 0 | Weighting × preservation |
| F101 | 1 | 0 | 1 | Weighting × prior correction |
| F011 | 0 | 1 | 1 | Preservation × prior correction |
| F111 | 1 | 1 | 1 | Full proposed combination |

For example, at I=0 the W×P interaction is
`score(F110)-score(F100)-score(F010)+score(B03)`, in percentage points.
Repeat at I=1 to examine dependence on prior correction. A single-seed
interaction is exploratory. Repeat the required four cells over all three
seeds before claiming a robust interaction; Q01 alone is not that complete
factorial replication.

A01 replaces classwise ranks with global ranks in F100. A02 changes only
`w_min` to zero. These diagnose whether class conditioning and a positive
weight floor explain the outcome. They do not create ground-truth noise labels.
Reject claims of correct noise detection based solely on their selected samples.

## 4. Conditional semantic branch

The `official_names` gate requires a verified organizer mapping with meaningful
names. A zero-shot reference (P00) has no training run. P01/P02/P03 compare
CoOp+CE, CoOp+GCE and PromptMAE. Proceed to P04 JoAPR* or P05 NLPrompt only
after the baseline works. P06 changes only the OT class marginal from uniform
to the smoothed training prior; run it only if long tails are established.

Inspect source implementation details before P04–P06. Port to ViT-B/32 and
record differences from the papers; importing their dataset defaults is not
part of the experiment. For OT, profile cost at the actual class/image counts.
Do not claim that a 20-epoch local port reproduces a 200-epoch paper result.

## 5. Budget tiers and bounded tuning

1. **Minimum useful screen: seven training runs**, all seed 17: B01, B03, B04,
   R01, F100, F010, F001. D00 is a read-only audit; shared head initialization
   is additional recorded work. This identifies several hypotheses at low cost.
2. **Core expansion:** four remaining factorial cells, B02/B05, R02/R03,
   A01/A02, and L01. Choose additions based on the first screen and measured
   budget; do not silently claim unrun comparisons.
3. **Names available:** P00–P03 first; P04–P06 only if justified.
4. **Confirmation preparation:** Q01 restarts B03 and at most two shortlisted
   recipes for LONG30, each with seeds 17/29/43: at most nine new training runs.
   Select recipes on dev, apply the predeclared validation rule, then Q02
   evaluates the locked models on confirm without updates.
5. **Final refit:** Q03 runs one selected recipe, seed 17, from official weights
   on all eligible current-stage training data. Fix its epoch count before
   refitting to the rounded median dev-selected epoch across Q01's three seeds
   for that recipe (nearest integer, halves rounded up). No confirm or test
   score decides this epoch count. Seed 17 is predetermined, not a lucky seed.

The full registry is an optional menu, not a commitment to run every row.
Q01 is a budget ceiling: if only one candidate survives, use six runs; if none
survives, retain the baseline. Long-run reruns reset optimizer, weights and all
sample state. They do not continue a selected screening checkpoint.

For at most one survivor, allow one-at-a-time sensitivity checks before Q01:
LoRA rank `{2,4,8}`, GCE q `{0.3,0.7,1}`, preservation coefficient
`{0,0.1,1}`, or weight floor `{0.2,0.5,1}`. Test only relevant factors and add
at most four configurations in total, with explicit derived IDs and control
rows. Do not take the Cartesian product. Preserve separate baseline tuning
budget if learning rate is varied. Parameters absent from a recipe are not
tuned. All additional runs must be registered before execution.

## 6. Stop and handoff criteria

- Halt an individual run for nonfinite loss, corrupted class mapping, incorrect
  trainable parameters, leakage, or a reproducibility failure; report it failed.
- If weighting concentrates or eliminates effective support in a class, inspect
  its ESS and recall. Disable it if validation support does not justify the
  added complexity. Low training loss alone is not a reason to promote it.
- Do not call a recipe successful unless it beats its matched control under
  the [validation rule](validation.md), with label-quality limits stated.
- Record unsuccessful and inconclusive experiments. Deliver the selected
  recipe, all tested controls, per-seed metrics, cost, and checkpoint provenance.

Actual execution additionally needs current-stage data, an implemented trainer,
and a defined compute allocation. The design work is complete without assuming
those inputs or starting training.
