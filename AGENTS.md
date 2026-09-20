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

## Internal leaderboard submission gate

User rule established on 2026-09-18: only a fully optimized leaf of the declared
experiment tree is eligible for organizer leaderboard submission. Baseline
roots (including B03/B04), engineering profiles, and intermediate screening
results are not eligible. Completing baseline training or validating a CSV
does not establish submission eligibility. Follow
`docs/research/submission-gate.md` and retain the leaf's selection evidence.
This is an internal project rule, not an organizer requirement.

用户于 2026-09-20 明确授权本次一次排行榜占位结果，作为上述内部叶子门槛的
单次例外；不将该结果标记为已完成优化的叶子。模型先按现有 dev 证据固定，
test 仅作单模型推理，仍须通过提交格式和覆盖校验。例外不取消比赛规则，
不自动授权后续占位提交；只生成供用户上传的文件，不代为操作排行榜。

## Local machine execution limit

User rule established on 2026-09-15: this machine has insufficient GPU memory
for formal training. Local execution is limited to code-correctness tests and
bounded checks that training can start successfully.

- Training startup checks must use a small batch and an explicit, finite step
  limit, sufficient to verify data loading, forward/backward passes and an
  optimizer update. Stop when that check is complete.
- Do not run formal training, full training epochs, baseline fitting for
  reported scores, hyperparameter searches, or the research experiment matrix
  on this machine. Moving the work to CPU or reducing the model/batch size
  does not remove this limit.
- Run formal experiments and sustained performance profiling on a separate
  machine with sufficient memory. Preparing their code/configuration locally
  does not authorize executing them locally.
- Report local checks as correctness/startup validation only, not evidence of
  convergence, accuracy, or completed experiments.
- Future training entry points must offer an explicitly bounded startup-check
  mode; it must never automatically continue into formal training.

## Engineering expectations

- 用户于 2026-09-19 明确：本仓库每次授权更新完成并通过相关验证后，agent
  自动提交并推送到 GitHub，无需再次询问。组员直接从 GitHub 查看最新任务说明，
  用户只需发送通知。仅提交本次相关文件，不包含私有回传、数据、权重或检查点；
  不强制推送。推送失败须如实报告，不宣称远端已更新。此规则不授权更新运行中的 T4。

- 用户于 2026-09-20 明确：T4 执行任务指令直接在对话中交付给用户，由用户转交
  执行 agent，不上传 GitHub。代码修复和组员文档仍按上述规则自动提交、推送。

- 用户于 2026-09-19 明确：不设正式实验的绝对耗时门槛。“一小时一次迭代”是
  尽快获得有效反馈的软目标，不是超时停止、淘汰方法或限制训练预算的条件。
  比赛分数优先；有收益证据或明确提升假设的方法，即使更慢，也值得追加预算。
  通常优先缩短迭代时间，但须区分方法固有成本与可避免的工程低效；相对性能告警
  用于排查工程问题，不代替模型效果判断。追加实验须记录依据与下一观察点，
  不自动无限续跑。本机仅允许正确性/有界启动检查的限制不变。

- 面向组员的任务说明、操作步骤和交接说明一律使用简体中文，要求清晰、简洁、完整。
  命令、路径、配置键和必要技术名称保留原样；明确任务目标、前置条件、执行步骤、
  验收标准、失败停止条件和回传材料。此规则优先于通用的技术文档英文约定。

- All downloads are delegated to the user (rule established 2026-09-16).
  Agents must not initiate package, model-weight, or dataset downloads.
  Prepare explicit user-run commands or offline installation instructions;
  report missing assets rather than silently fetching them. Provisioning
  commands are user-operated and must never be invoked by tests or training.
- Use Miniconda with the project-local prefix `.conda/aic-robust-clip`.
  Do not install dependencies into `base` or another project's environment.
  Agents may create the environment from cached packages with `--offline`;
  downloading missing Conda or pip packages remains the user's responsibility.

- Prefer configuration-driven experiments with explicit seeds and data-stage
  identifiers.
- Record dependency versions, configuration, seed, source revision, metrics,
  and checkpoint hash for every result used in a report.
- Keep data auditing separate from training and ensure audit outputs cannot
  silently alter labels or splits.
- Validate `pred_results.csv` with the repository validator before packaging.
- Run focused tests, the full available suite, and `git diff --check` before
  considering a change complete.
