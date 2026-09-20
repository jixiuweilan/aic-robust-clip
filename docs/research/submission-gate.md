# Internal submission eligibility gate

This is a project governance rule. It is stricter than, and must not be
presented as, an organizer rule.

## 本次占位例外（2026-09-20）

用户明确要求生成一次可上传排行榜的占位结果，授权本次跳过“充分优化叶子”这一
内部门槛。结果应记录为占位、未完成优化；不补写已通过多种子或 confirm 的结论。
在读取 test 进行预测前，根据已有 dev 证据固定一个 checkpoint，保留模型、配置、
源码、依赖和测试资产摘要，执行单模型推理并校验 CSV、ZIP 和完整文件名覆盖。
test 仍不得用于训练、拟合或模型选择；生成文件交用户上传，不代为操作排行榜。
本次例外不取消下方默认门槛，也不自动授权后续占位提交。

## Rule

Only a fully optimized leaf of the project's experiment tree is eligible for
an official leaderboard submission. Root and intermediate nodes are not
submission candidates.

- **Root node:** an initial baseline or control family, such as B03 or B04.
  It establishes a reference and is required for comparison, but cannot be
  submitted as the final result.
- **Intermediate node:** an engineering profile, short screening run, partial
  training result, or unresolved method/hyperparameter branch. It may guide
  selection but has no submission eligibility.
- **Leaf node:** a complete, reproducible method choice with its data-stage
  contract, automatic noise-handling settings, hyperparameters, seed policy,
  training budget, source revision, and checkpoint identity fixed. It must have
  completed the required formal training, locked selection/confirmation
  evaluation, and passed the single-model prediction and package validators.

“Fully optimized” means fully resolved within the declared experiment tree; it
does not mean an unbounded search over every possible method. Before a leaf is
declared, the parent alternatives and the screening budget must be recorded.
The leaf must not retain an unresolved choice that is being made after looking
at test images or test predictions.

## Current consequence

B03 and B04 are baseline/control roots. The engineering acceleration branch is
an intermediate node. Neither is currently eligible for leaderboard
submission. The next eligible result must come from a predeclared shortlist,
short-budget comparison, one selected branch, and a full final training run
whose checkpoint is then locked for confirmation and one-flow prediction.

An internally green software test, a cache, a shared HEAD3 initializer, a
benchmark report, a three-epoch screen, or a dev-only winner is evidence for
the next decision, not a submission result.

## Required leaf record

Before packaging `pred_results.csv`, retain one record containing:

1. the root and intermediate ancestors;
2. the final method and all fixed parameters;
3. stage, manifest, split, class-map and official-weight identities;
4. seed, training/evaluation budget and source/dependency identities;
5. final checkpoint hash and locked confirmation-selection record;
6. validator output for the single-model prediction package.

This gate prevents a convenient baseline or an unfinished branch from being
mistaken for the result of full optimization.
