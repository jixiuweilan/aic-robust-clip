# Experiment Configurations

Configurations belong here. Each configuration must state the competition
stage, manifest/split identities, exact CLIP ViT-B/32 weight source and
revision, trainable parameter subset, seed, optimization settings, execution
mode, and output location. `baseline-smoke.json` is a bounded startup-check
example; it is not a formal experiment profile and must not be used to claim a
score.

`research-methods.json` maps the first seven recipes to model classes and
`RunConfig.parameters` overrides consumed by `TrainConfig.from_run(run)`.
It selects objectives, not an execution mode: inherit the bounded smoke run
configuration for local checks. W/P/I can be combined by merging their distinct
parameter keys; GCE/SCE reject combinations with W/P/I. These are method presets,
not complete formal profiles (data/weight identities, optimizer groups, schedule,
head initialization and training-machine provisioning remain required).

Complete schema-v2 command templates now live in `formal/` for B01, B04, B03,
R01, F100, F010 and F001. They reference stage-specific inputs, shared HEAD3,
CACHE20/ONLINE10 defaults and a separate-machine binding. Paths resolve relative
to each configuration file. These templates are not measured experiments or
authorization to run formal training. `smoke/B03.json` uses separate bounded
fixture artifacts. See [the current handoff](../docs/handoff.md) for prerequisites
and the still-pending real-weight/target-machine acceptance checks. The local
CPU tensor and CLI fixture suite passed on 2026-09-16.
