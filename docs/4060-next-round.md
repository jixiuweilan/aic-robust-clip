# 两台 4060：缓存修复验收与 LoRA 学习率短程对照

本轮只部署到两台 4060 的独立 Linux／WSL2 Linux 文件系统目录。**不更新、不重启、不操作正在运行的 T4 代码、环境、配置或产物。** T4 继续 W/P 方向；4060 探索 LoRA 学习率。旧 B03/B04 十轮基线不再重复，已有共同 HEAD3 不重建。

| 负责人 | 本机同期对照 | 本机候选 | 固定条件 |
|---|---|---|---|
| 组员 A | LoRA LR `1e-4` | `3e-5` | B03、普通 CE、seed 17、分类头 LR `1e-3` |
| 组员 B | LoRA LR `1e-4` | `3e-4` | 同上，同一份已核验 HEAD3 |

每人先对照、后候选，均在 **3 个完整轮次后暂停**。配置仍为 ONLINE10 十轮计划，保留原增强、优化器、warmup 和学习率调度。不能把 `epochs` 改为 3。不使用 confirm/test，不提交排行榜；本轮不自动晋级。

## 1. 前置条件与部署

执行 agent 在后续执行时自行向对应组员确认连接地址、工作目录、获准使用的 GPU；本次交付不建立 SSH 连接。两台机器使用相同补丁及其前置源码，核对任务包 `SHA256SUMS`；不能只复制某个修复文件后沿用旧准入。

- 保留全部历史回传和失败报告。`feedback/` 已忽略，不能清理或覆盖。B04 原报告记载的 WSL 缓存测试问题仍是历史失败；只有在该组员原机器上使用新源码通过原测试和新回归，才构成本轮复验证据。
- 使用 `.conda/aic-robust-clip`。依赖、官方 OpenAI CLIP ViT-B/32 权重、当前赛段数据均应已存在；缺失则列清单交用户下载或传输，agent 不下载、不安装缺失包。
- 复用冻结 manifest、split、class_map、缓存和共同 HEAD3。核对 `head.json` 与 `head.pt` 的 SHA-256，两机必须一致；运行前现有 `load_head` 还会校验实际 checkpoint 内容与身份。不得重建缓存／HEAD3 或改写冻结文件来解决路径问题。
- 原配置复制成新的本机 B03 配置，填写已有资产与本机机器绑定的绝对路径；输出指向新目录。保留 seed 17、有效 batch 128、普通 CE、分类头 LR `1e-3`、LoRA LR `1e-4`、完整 ONLINE10。两机配置的非路径字段保持一致。
- ZIP 必须在本地 Linux 文件系统中；WSL2 使用 Linux 卷，不用 `/mnt/c`、远端挂载或共享盘。新 worker 首次完整校验 ZIP 属于冷启动，不因暂时没有输出而重复启动。训练期间 ZIP 和映射均不可变。

在项目根目录设置以下变量，路径和 GPU 编号必须换成已核对的值。每条命令的 stdout、stderr 与退出码都要保留。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export AIC_ARCHIVE_LOCATIONS=/绝对路径/archive-locations.json
AIC_PY="$PWD/.conda/aic-robust-clip/bin/python"
AIC_SOURCE=/绝对路径/本机B03正式配置.json
AIC_MEMBER=A  # 组员 B 填 B
AIC_TASK="$PWD/outputs/4060-lr-round1-v1"  # 必须为新目录
mkdir "$AIC_TASK"
```

已有有效 `machine-training.json` 直接复用，不覆盖。如果尚未登记，只在独立 4060 上执行 `.conda/aic-robust-clip/bin/aic-doctor --bind-training machine-training.json`；开发机不得登记。

## 2. 零失败、零跳过测试与合成 CUDA 启动验收

```bash
"$AIC_PY" -m aic_robust_clip.rtx4060 check \
  --config "$AIC_SOURCE" --output "$AIC_TASK/checks"
```

该入口先验证是已登记的独立 4060，然后依次运行 relocation 专项、完整 suite、compileall、pip check、git diff --check；保存完整日志。随后 FP32、FP16 分别做一次合成 CUDA 启动检查，**每次仅两次更新**，不读比赛图像。检查结果及依赖清单写入 `checks/checks.json`；任一失败或跳过即停止，不生成通过回执。B04 原失败测试 `test_verification_cached_only_for_unchanged_file_and_process` 必须实际执行并通过，不能删除、跳过或仅改变预期值来放行。

每台都执行。B04 所在组员回传新测试日志及原失败报告路径，由负责人确认来自原机器。新准入不能把旧报告改写成通过。测试回执只证明当前软件正确性和启动成功，不证明收敛或分数。

## 3. 只生成配置，不训练、不重建资产

```bash
"$AIC_PY" -m aic_robust_clip.rtx4060 prepare \
  --config "$AIC_SOURCE" --member "$AIC_MEMBER" --output "$AIC_TASK/bundle"
AIC_BUNDLE="$AIC_TASK/bundle/bundle.json"
```

生成器只读取配置和身份元数据，创建 6 组 × 2 个配置。运行目录按工程配置／control、candidate 隔离；原配置及旧产物不改动。`bundle.json` 记录元数据摘要和配置摘要，后续修改配置会被拒绝。

| 配置目录名 | 训练精度 | microbatch | 累积次数 | train/eval workers |
|---|---|---:|---:|---|
| `fp32-m16-w4-e0` | FP32 | 16 | 8 | 4/0 |
| `fp32-m32-w4-e4` | FP32 | 32 | 4 | 4/4 |
| `fp16-m32-w4-e4` | FP16 | 32 | 4 | 4/4 |
| `fp16-m64-w4-e4` | FP16 | 64 | 2 | 4/4 |
| `fp16-m128-w4-e4` | FP16 | 128 | 1 | 4/4 |
| `fp16-m64-w2-e2` | FP16 | 64 | 2 | 2/2 |

全部有效 batch 128；验证始终 FP32、batch 64；prefetch 2、pin_memory=true。只测速 control 配置，避免把学习率收益混入工程选择。

## 4. 每台独立测完六个候选

每个候选一次 train、一次 eval；每次独立进程，2 次预热＋10 次测量。只在独立 4060 执行，不在开发机执行。运行前确认该 GPU 无其他任务。训练短窗会生成专用测速 checkpoint，禁止用于正式训练或恢复。

```bash
for AIC_PROFILE in fp32-m16-w4-e0 fp32-m32-w4-e4 fp16-m32-w4-e4 fp16-m64-w4-e4 fp16-m128-w4-e4 fp16-m64-w2-e2; do
  AIC_CONFIG="$AIC_TASK/bundle/$AIC_PROFILE/control.json"
  mkdir -p "$AIC_TASK/windows/$AIC_PROFILE"
  for AIC_PHASE in train eval; do
    "$AIC_PY" -m aic_robust_clip.benchmark \
      --config "$AIC_CONFIG" --phase "$AIC_PHASE" --warmup-steps 2 --measure-steps 10 \
      --output "$AIC_TASK/windows/$AIC_PROFILE/$AIC_PHASE" \
      > "$AIC_TASK/windows/$AIC_PROFILE/$AIC_PHASE.stdout.log" \
      2> "$AIC_TASK/windows/$AIC_PROFILE/$AIC_PHASE.stderr.log"
    AIC_STATUS=$?
    echo "$AIC_STATUS" > "$AIC_TASK/windows/$AIC_PROFILE/$AIC_PHASE.exitcode"
    if [ "$AIC_STATUS" -ne 0 ]; then
      echo "候选 $AIC_PROFILE 失败，保留证据，停止此候选；不得改参重试。"
      break
    fi
  done
done
"$AIC_PY" -m aic_robust_clip.rtx4060 summarize \
  --bundle "$AIC_BUNDLE" --directory "$AIC_TASK/windows" \
  --checks "$AIC_TASK/checks/checks.json" --output "$AIC_TASK/summary-$AIC_MEMBER.json"
```

若 shell 设置了 `set -e`，失败会直接退出循环；保留现场，再由负责人确认是否继续其他尚未运行的候选，不重跑失败候选。不要重复执行整个循环覆盖 stdout/stderr。同一候选 eval 失败也视为整组失败。

汇总校验数据量、源代码／配置／依赖／机器、FP32 验证、退出清理及至少 15% 分配／预留显存余量。训练测量 1,280 个样本，含预热共 1,536；验证测量 640，含预热共 768；派发上限应分别相同，无额外派发。worker 数量匹配配置，全部 `exitcode=0`、`alive=false`、`forced=false`，无残留 worker，无 `failure.json`。

预计总轮时 = train 样本数／训练吞吐 + dev 样本数／验证吞吐 + 2 × 实测 checkpoint 写入耗时。最后一项保守估计 last 与 best 的保存成本；冷启动另列，不把短窗估计当成实测整轮时间。报告同时保留数据等待、冷启动、峰值显存。现场 `nvidia-smi` 须确认没有其他进程占用，allocator 统计不包含其他进程显存。

## 5. 共同选择与最终三次退出复验

两人回传各自 `summary-A.json`、`summary-B.json` 和原始测速证据。负责人在同版本源码上进行只读选择，输出必须为新文件：

```bash
"$AIC_PY" -m aic_robust_clip.rtx4060 select \
  --summary-a /回传目录/summary-A.json --summary-b /回传目录/summary-B.json \
  --output /新输出目录/selection.json
```

只有两台均通过的配置可选，选择两机预计总轮时之和最小者；相对最优值差距不足 5% 时，先按 train worker 更少、再按 eval worker 更少、最后按 microbatch 更小选择。并列时按预计轮时及固定名称排序以保证可重复。没有共同通过配置则停止正式开跑并反馈。

把同一份 `selection.json` 交两人，各自保存到新的 `$AIC_TASK/selection.json`，读出选中配置，**两机各运行三次新的独立验证窗口**：

```bash
AIC_PROFILE=$("$AIC_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile"])' "$AIC_TASK/selection.json")
AIC_CONFIG="$AIC_TASK/bundle/$AIC_PROFILE/control.json"
mkdir "$AIC_TASK/final-eval"
for AIC_REPEAT in 1 2 3; do
  "$AIC_PY" -m aic_robust_clip.benchmark --config "$AIC_CONFIG" \
    --phase eval --warmup-steps 2 --measure-steps 10 \
    --output "$AIC_TASK/final-eval/eval-$AIC_REPEAT" \
    > "$AIC_TASK/final-eval/eval-$AIC_REPEAT.stdout.log" \
    2> "$AIC_TASK/final-eval/eval-$AIC_REPEAT.stderr.log"
  AIC_STATUS=$?
  echo "$AIC_STATUS" > "$AIC_TASK/final-eval/eval-$AIC_REPEAT.exitcode"
  if [ "$AIC_STATUS" -ne 0 ]; then break; fi
done
"$AIC_PY" -m aic_robust_clip.rtx4060 accept \
  --bundle "$AIC_BUNDLE" --selection "$AIC_TASK/selection.json" \
  --eval-directory "$AIC_TASK/final-eval" --checks "$AIC_TASK/checks/checks.json" \
  --output "$AIC_TASK/acceptance.json"
```

任一次失败停止重复窗口，缺少任意窗口会使 accept 拒绝。三次全部通过才生成本机回执。回执绑定源码摘要（含未提交源码）、依赖、Python/CUDA、机器/GPU、两支配置、checks、原测速和最终窗口文件哈希。代码、配置、身份元数据或证据变化必须重新验收，不能复用旧回执。两机回执均交负责人核验后才正式开跑；其中 B04 原失败机器的复验不可缺。

## 6. 同期对照与学习率候选，完整三轮即停

```bash
"$AIC_PY" -m aic_robust_clip.rtx4060 run \
  --bundle "$AIC_BUNDLE" --receipt "$AIC_TASK/acceptance.json" --role control
# 上一条成功，result.json 显示三轮完整且 paused=true 后再执行：
"$AIC_PY" -m aic_robust_clip.rtx4060 run \
  --bundle "$AIC_BUNDLE" --receipt "$AIC_TASK/acceptance.json" --role candidate
```

入口固定传递 `stop_after_epochs=3`，保留十轮调度；先检查本机对照，再允许候选。每支都从同一 HEAD3 初始化，输出必须是全新目录。不得从旧 B03/B04 或测速 checkpoint 恢复。仅进程中断且故障已经确认后，使用相同命令追加 `--resume`，它只读取本支配置输出目录内的 `last.pt`，仍在第三轮暂停。OOM、非有限 loss、身份错误、读取错误、SIGABRT、异常清理时保留现场并反馈，不改 batch/workers/精度自动续跑。

完成量以冻结 split 实际 train 数计算：每轮 `ceil(train/128)` 更新、完整 train 样本访问。核对 `completed_epochs=3`、3 条 `complete=true`、`paused=true`、`stopped_by_limit=false`，并保留完整 dev 输出。若本赛段 train 为 82,586，则三轮累计 1,938 更新、247,758 次样本访问。不能以进程退出码或 `status` 单字段代替完成量。

## 7. 回传与下一观察点

每台机器交付一个新的私有版本目录／压缩包，保留原始材料，不提交比赛图片、权重、检查点或提交包到 Git：

- 本轮任务包版本／补丁与 SHA-256、Git revision＋源码摘要、全部本地修改 diff、`pip-freeze.txt`、Python/CUDA/GPU/驱动信息、机器指纹及授权 GPU 编号。
- checks 下专项和完整测试日志、compileall／pip check／diff check 日志、两次合成 CUDA 报告。B04 原机器同时注明历史失败报告路径、新复验日期与日志，不改写历史。
- 本机来源配置、完整 bundle 配置、映射 JSON 与摘要、冻结 manifest/split/class_map 摘要、共享 HEAD3 两文件摘要；不得重打标签或混入其他赛段。
- 六个候选所有 train/eval 原始 JSON、stdout、stderr、退出码、失败 JSON；本机 summary、共同 selection、三次最终 eval 窗口、本机 acceptance。两机负责人核对汇总与原始材料对应。
- control/candidate 各自的 `resolved.json`、`result.json`、`epochs.json`、全部 `dev-epoch-*.json`、`dev-predictions.json`、`best.pt`、`last.pt` 及 SHA-256，完整启动／异常／恢复日志。
- 每轮 macro recall、micro top-1、逐类指标、逐图预测、真实 train/dev/save 轮时、数据等待、峰值显存、更新／样本数、最佳轮次与检查点摘要。以**候选减本机同期对照**为主比较，报告三轮趋势；不能把两机绝对分数差直接当成 LR 收益。
- 整个回传包 SHA-256，以及排除清单自身后的文件校验清单。

首轮只筛选，不自动晋级。负责人结合效果、改善趋势、T4 W/P 结果决定下一观察点；若续跑，记录依据和明确预算／观察点，从各自 `last.pt` 开始，不自动无限续跑。比赛分数优先，无绝对耗时淘汰线；“一小时反馈”只是软目标。测速慢先区分方法成本与可避免的工程低效，有收益证据或明确提升假设的慢方法仍可追加预算。
