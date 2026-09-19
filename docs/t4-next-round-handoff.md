# 四张 T4：工程验收与首轮模型改进任务

状态更新（2026-09-19）：N00/N02/N03 已完成十轮回传并通过指标复核，N01 冻结。见 [十轮终窗评审](research/t4-epoch10-results-20260919.md)。下文保留原工程与首窗操作记录，**不作为重新 prepare、重跑基线或自动续训的指令**。

目标：在服务器的四张 T4 上独立运行 N00/B03（CE）、N01/R01（GCE）、
N02/F100（类内软降权）、N03/F010（特征保持），先消除明显工程瓶颈，再各跑至
第 3 个完整训练轮次。**不占用两位组员的 4060，不启动六卡分布式训练，不提交排行榜。**

本任务覆盖准备、测速、性能准入、首个训练窗口及回传，是一个完整任务包。
本文件是本次四 T4 执行入口，取代旧研究计划中工程待实现和仅两台 PC 的排期；
并不自动批准旧计划里的后续实验或晋级。
本机仅做合成数据正确性测试；本文件中的 CUDA 命令全部由工程 agent 在服务器执行。
SSH 地址和项目目录由工程 agent 向用户确认，不索取或回传密码/私钥。

## 1. 前置条件与边界

- 使用本次交付提交和项目 Miniconda 环境 `.conda/aic-robust-clip`。不要安装到 base。
  缺依赖、权重或数据时停止并交给用户下载，不运行 provision/pip install 等下载命令。
- 旧训练已完成的 GPU 才可使用。调度器拒绝已有计算进程的卡，并按 GPU UUID 加锁；
  锁仅协调本仓库启动的任务，不能替代共享服务器的使用约定。
- 保留旧 B03/B04 输出、冻结 manifest/split/class-map、官方权重和 seed 17 HEAD3。
  从旧配置复制路径，不重建已有缓存或 HEAD3，不改写 manifest 中的绝对路径。
- 迁移文件通过 `AIC_ARCHIVE_LOCATIONS` 显式指定，沿用摘要强校验；路径不确定先核对。
- 不访问 confirm/test；不新增标签，不融合模型。首轮仍只是中间节点。
- 建议输出磁盘至少留 30 GiB，测量与模型文件都不推送 GitHub，不自动清理旧证据。

以下所有命令在服务器仓库根目录执行。`SOURCE_CONFIG` 必须指向服务器现有的
**正式 B03/B04、有效 batch 128、ONLINE10 的 CE 配置**，不接受 smoke 配置。
路径变量示例需要替换；不要照抄一个不存在的路径。

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export AIC_ARCHIVE_LOCATIONS=/实际路径/archive-locations.json
PY=.conda/aic-robust-clip/bin/python
SOURCE_CONFIG=/实际路径/原B03正式配置.json
PACK=outputs/t4-next-round-v1
```

## 2. 正确性验收与只生成配置

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest discover -s tests -v
"$PY" -m compileall -q src tests
"$PY" -m pip check
git diff --check
"$PY" -m aic_robust_clip.t4_runner prepare --config "$SOURCE_CONFIG" --output "$PACK"
```

全套应零失败零跳过；本次交付回执给出测试数。以上不下载、不训练，也不登记机器。
已有训练机登记继续有效；缺失或不匹配时由操作人检查现有 `aic-doctor --bind-training`
流程，开发机仍不能登记。准备操作生成 6 组、每组 4 个隔离配置，不创建 run 输出。

生成的配置保持完整 10 轮 LR 计划、有效 batch 128、同一增强、同一 HEAD3；
只改变微批次/加载进程/训练精度及明确登记的 GCE/W/P 方法。
默认训练进程使用 2 个计算线程，加载 worker 每个 1 线程，避免四卡进程争抢全部 CPU。

## 3. 先测速，再释放正式训练

调度器默认物理 GPU 0/1/2/3 分别执行 N00/N01/N02/N03；可用 `--gpus` 更改顺序。
每次 `bench` 四卡并发，各方法测 2 个 warmup + 10 个更新；随后重复 3 次 dev 窗口，
W 另外重复 3 次 train 固定视图评分窗口。窗口严格有限，不跨完整训练轮次。
任何失败返回非零，不自动重试；其他已在运行的独立任务可以完成并保留证据。

依次测以下配置；**每条完成并检查结果后再发下一条，不并行启动两组四卡测速**：

```bash
"$PY" -m aic_robust_clip.t4_runner bench --bundle "$PACK/bundle.json" --profile fp32-m16-e0 --output "$PACK/bench-fp32-m16-e0"
"$PY" -m aic_robust_clip.t4_runner bench --bundle "$PACK/bundle.json" --profile fp32-m16-e4 --output "$PACK/bench-fp32-m16-e4"
"$PY" -m aic_robust_clip.t4_runner bench --bundle "$PACK/bundle.json" --profile fp32-m32-e4 --output "$PACK/bench-fp32-m32-e4"
"$PY" -m aic_robust_clip.t4_runner bench --bundle "$PACK/bundle.json" --profile fp16-m32-e4 --output "$PACK/bench-fp16-m32-e4"
```

- `e0/e4` 是 eval/scoring workers；train workers 为 4。eval batch 64、W scoring batch 128。
  e0 用于量化串行验证的代价，不是默认生产选择。
- m32 相对 m16 的收益在多 worker 下重新测，不能沿用旧串行加载下“收益不足 5%”的判断。
- FP16 若成功且更快，再测 `fp16-m64-e4`。四卡并发若 CPU/IO 等待明显，再测
  `fp32-m32-w2-e2`，不要盲目把 workers 加到 8/16。
- FP16 只用于训练前向，采用 GradScaler；loss、特征归一化、正式 dev 与 W 评分保持
  FP32。遇到非有限 loss/gradient 直接失败，不偷偷跳过 optimizer 更新或切回 FP32。
  FP16 性能通过不代表精度已经无损；其对照 N00 和三候选必须使用同一精度重新比较。
- W 训练窗口使用**明确标记的合成非均匀权重**，只检验降权计算路径与开销；单独
  scoring 窗口测真实读取/前向/CE 成本，不声称已完成两轮 warmup 或拟合可靠性。
  P 窗口包含第二编码器真实前向。测速 checkpoint 标记 BENCHMARK，不能续正式训练。

如 eval/scoring 多 worker 仍 SIGABRT，停止该配置，回传 `failure.json` 与日志。
保留 e0 已通过证据，但不要自行宣布“修复完成”或用明显慢的配置直接训练。
显存不足、进程异常、文件不存在也按失败处理，不临时改参数重试同一目录。

## 4. 选择工程配置并创建准入回执

逐方法比较 `benchmark.json`：吞吐、数据等待时间、冷启动、显存、worker 退出；
同时保存 `nvidia-smi`/CPU/磁盘监控文本。**以四卡并发结果为主，不用单卡短窗乘四。**
至少比较一个不同配置；上面三组 FP32 与一组 FP16 的调查均须有结果或明确失败原因。
不能只因为某组能运行就结束优化。

选择原则：没有异常退出，显存 reserved 留至少 10%，整体预计轮次耗时接近已测最优，
并逐项解释剩余主要瓶颈。若更大 batch/AMP 带来明显收益但未验证，不得标记为工程验收完成。
接受器会拒绝选中配置的四任务总预计轮次耗时比提交的比较组慢超过 10%。
该门槛不等于证明全局最优；每个方法的延迟也必须单独查看。
它比较同一组方法的工程配置，不是按绝对耗时淘汰计算量更大的方法。

示例仅演示语法，**不能预定 m32 FP32 是胜者**。把实测选中的 profile 与目录代入；
通过的其他比较组用多个 `--comparison` 一并提供，失败组在回传说明原因，不伪造成功报告。

```bash
PROFILE=fp32-m32-e4
"$PY" -m aic_robust_clip.t4_runner accept \
  --bundle "$PACK/bundle.json" --profile "$PROFILE" \
  --bench-dir "$PACK/bench-$PROFILE" \
  --comparison "$PACK/bench-fp32-m16-e0" \
  --comparison "$PACK/bench-fp32-m16-e4" \
  --rationale '替换为实测选择理由及其他候选去留，包含FP16结果' \
  --output "$PACK/performance-accepted.json"
```

回执绑定源码提交/源摘要、依赖、主机、四个配置和测速报告 SHA-256；修改任一项后不能
沿用旧准入。输出包含每个方法的一轮和三轮耗时估算；W 加上全 train 评分，P 已包含
在训练吞吐，checkpoint 成本也计入。短窗估算不含冷启动，不承诺所有方法一小时结束。

## 5. 正式首窗：四卡各到第三个完整轮次

```bash
"$PY" -m aic_robust_clip.t4_runner train \
  --bundle "$PACK/bundle.json" --profile "$PROFILE" \
  --acceptance "$PACK/performance-accepted.json" \
  --through 3 --output "$PACK/dispatch-epoch3"
```

训练期间检查每个 run 的 `timing-epoch-0000.json`。它记录训练/数据等待、W 评分、dev、
checkpoint、总耗时和实际样本数。预计每轮 train 82,586、dev 10,378；以冻结 split 为准。
若完整轮次总耗时比短窗估算高 30% 以上，先检查冷启动、存储争抢和 worker 重启；
暂停后续晋级并回传，不在慢配置下默默跑完十轮。不要把 benchmark 的训练吞吐当总耗时。
此 30% 是工程异常排查触发线，不是方法淘汰线，也不是一小时绝对上限。
负责人结合冷启动、稳定轮时、并发状态及模型收益，可说明理由后批准继续；
无需仅为达成一小时目标而减轮次、删除必要评分或放弃有潜力的慢方法。
调度器不会因为某个方法先完成就替它自动开下一轮，也不会自动改变研究配方。

每个 run 完成第 3 轮的训练、W 评分、dev 验证、日志、best/last 保存后退出，
`result.json.status=paused`、`completed_epochs=3`，调度记录为 complete（表示本次派发成功）。
此时三轮结果只是筛选证据，不是模型提交资格。不要使用旧 `--stop-after-updates 1938`
代替完整轮次暂停，也不要把 formal_epochs 改成 3。

后续晋级由代码/研究负责人看结果决定。如下只是获得晋级指令后续跑某两个任务的语法：

```bash
"$PY" -m aic_robust_clip.t4_runner train \
  --bundle "$PACK/bundle.json" --profile "$PROFILE" \
  --acceptance "$PACK/performance-accepted.json" \
  --jobs N00,N02 --gpus 0,2 --resume --through 6 \
  --output "$PACK/dispatch-selected-epoch6"
```

只从各自 last.pt 恢复；best.pt 用于选择，不能拿来续跑。续跑保持 LR/RNG/optimizer/
GradScaler/W 历史和样本游标，完成第 10 轮返回 complete。不同精度、batch 或方法不能
互相恢复。每次 dispatch 使用全新输出目录；已完成任务不能用原目录再次“重跑”。

## 6. 回传与停止条件

回传一个完整任务包，不把任务拆成零散截图：

1. 提交、源摘要、完整测试/compileall/pip check/diff 日志、依赖与 GPU 环境。
2. bundle/configs、所有 bench 目录（含失败项）、监控记录、性能选择回执；比较表列出
   四个方法的 train/dev/scoring 吞吐、预计/实际轮次耗时、显存和最终选择理由。
3. 首窗所有 dispatch 日志、每轮指标/逐图预测/timing、W 的 reliability 统计、resolved、
   result、best/last 及 HEAD3 摘要。模型回传方式另行约定，不能推送公共 GitHub。
4. 调度器自动导出的 `data-lineage.json`（冻结划分/类映射摘要、train 类频数、dev
   duplicate-group 映射），以及 confirm 未用于调参的声明；缺失不能自行重划分或
   访问 confirm 补齐。该 JSON 不包含图像或 confirm 的评估结果。
5. SHA256SUMS 排除自身，保留原文件；失败原因不得改写为通过。

任何 worker 异常、非有限数、身份或样本覆盖错误、意外读到 confirm/test、资产缺失、
磁盘/显存不足均停止受影响任务。不得自动下载、跳过失败样本、降配置续同一 run、
覆盖旧结果、杀掉别人的 GPU 进程。负责人未批准前，不启动额外种子、消融或最终重训。

## 实现说明

- 已实现独立 scoring batch/workers、全轮次暂停、四方法有界测速和四卡任务隔离。
- 训练从每 microbatch 同步检查改为每有效 batch、optimizer 更新前检查；仍拒绝 NaN/Inf。
  dev/W 评分只保留小型脱离计算图的预测/标量，集中回传 CPU，不缓存全数据图像或特征。
- 最佳模型原子引用已保存的 last 快照，避免同轮重复序列化；后续替换 last 不改变 best。
  checkpoint 摘要流式计算，避免额外一次数百 MB 的整文件内存分配。
- FP16 累计梯度后只 unscale 一次，保存恢复 scaler；遵循
  [PyTorch 2.7 AMP 文档](https://docs.pytorch.org/docs/2.7/notes/amp_examples.html)
  （2026-09-19 查阅）。本机 CPU fixture 只能验证控制流，不能替代 T4 的数值和性能验收。
