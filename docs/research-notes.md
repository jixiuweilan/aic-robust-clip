# Research Starting Points

This is a compact reading map, not an implemented method or a claim about
competition performance. All proposed experiments must remain inside the data
and model restrictions in [`competition.md`](competition.md).

## Suggested reading order

1. **CLIP** establishes the required backbone and its image-text representation.
2. **CoOp** adapts CLIP with learned prompt context while freezing pretrained
   parameters, providing a parameter-efficient baseline family.
3. **LoRA** supplies a general low-rank adaptation mechanism for transformer
   layers; applying it to CLIP still requires competition-specific validation.
4. **JoAPR** directly studies prompt learning under label noise and is the most
   task-aligned public paper in the organizer's list.
5. **TrustCLIP** is explicitly named by the organizer and studies semantic label
   verification plus gradient control, but its full text was not publicly
   downloadable from the sources checked on 2026-09-15.

Local copies of the four openly downloadable papers and links for all five are
catalogued in [`../references/README.md`](../references/README.md).

## Experimental questions to answer before optimization

- How much does a frozen CLIP representation retain under the anonymous class
  taxonomy, and what simple single-model classifier is a defensible baseline?
- How does estimated label reliability vary by class frequency and sample
  difficulty?
- Does an automatic warm-up/filter/reweight schedule improve clean validation
  accuracy without collapsing minority classes?
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
