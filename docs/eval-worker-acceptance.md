# eval worker SIGABRT：修复说明与组员验收

目标：在原 T4 环境复验 B04 的多 worker 验证路径，确认读取、计算、退出均成功。原报告中 B04 的三次同类失败来自 `3f2bbed`；尚无修复后 T4 的复验证据。开发机 CPU 回归通过不能代替此项验收。

## 已处理的路径

- `2824689`：正常关闭时停止派发，收完已派发的数据，再关闭 worker；提供训练与验证各自的 worker 配置。
- 本次补充：测速在启动 worker 前限制派发范围，只读取预热和测量需要的批次；训练仍保留完整 epoch 长度以计算学习率。
- 退出后检查每个 worker 的退出码。非零退出码、强制终止或残留进程均使任务失败；不捕获后当作成功，也不切换到单进程重试。
- 读取失败后禁止在同一 loader 上重新读取。失败 JSON 新增已完成步数、样本数、派发上限和 worker 状态；worker 启用故障堆栈输出，务必保存 stderr。

本机新增的两批 batch64 / 224×224 生成图像测试覆盖实际验证张量尺寸，验证派发不越界、游标不推进、worker 正常退出。另一个确定性回归覆盖“worker 已以 -6 退出但 PyTorch 关闭函数正常返回”的漏报。

原始 SIGABRT 日志没有堆栈或完成步数，无法确定发生在读取还是退出。原生退出竞争是待验证假设；上游也有 [spawn、persistent workers 和大张量触发类似退出问题的报告](https://github.com/Lightning-AI/pytorch-lightning/issues/21703)（查阅于 2026-09-19，环境为 PyTorch 2.9.1，与本项目 2.7.0 不同），不能据此直接认定根因。

## 执行步骤

1. 在获准使用的 T4 GPU 和独立验收代码目录部署本次交付；保留旧训练结果。使用原有 `.conda/aic-robust-clip` 或已配置的独立验收环境，不安装依赖、不下载。先跑生命周期专项、性能专项和完整测试，要求零失败零跳过，并通过 compileall、pip check、git diff --check。
2. 核对官方权重、冻结划分、原 HEAD3、机器绑定和 `AIC_ARCHIVE_LOCATIONS`。保持离线变量、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，同一 GPU 不并发其他任务。新 worker 校验压缩包属于冷启动，不重复启动进程。
3. 从既有 B04-m16-w4 配置复制两份到新目录：分别明确设置 `performance.eval_num_workers=2` 和 `4`，均保持 eval batch 64、pin_memory=true、prefetch=2、FP32；其他字段一致。仅改 `num_workers` 不会覆盖已显式设置的 `eval_num_workers=0`。
4. 每份配置执行三次预先计划的验证短窗，每次新输出目录、2 次预热和 10 次测量。任一次失败立即停止，不继续后续重复。两组均通过后，B03 用相同设置做一次验证回归和一次有界训练回归。

```bash
# 在执行机器根目录，替换已核对的路径和 GPU 编号。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export AIC_ARCHIVE_LOCATIONS=/绝对路径/archive-locations.json
CUDA_VISIBLE_DEVICES=2 .conda/aic-robust-clip/bin/python -m aic_robust_clip.benchmark \
  --config /绝对路径/B04-eval-w2.json \
  --phase eval --warmup-steps 2 --measure-steps 10 \
  --output outputs/eval-worker-acceptance/B04-w2-01
```

接着分别使用 `B04-w2-02/03`、`B04-w4-01/02/03` 新目录；w4 运行要同时切换配置文件。B03 的训练短窗使用 `--phase train`，同样为 2+10 次更新，不运行完整训练或 confirm/test。

## 通过标准与回传

- 进程退出码为 0，stderr 无 SIGABRT；`benchmark.json` 存在，`failure.json` 不存在。
- 验证为 10 次测量、640 个测量样本，含预热共读取 768 个样本；`loader_after_cleanup.delivered_samples=768`、`dispatch_stop=768`。
- `prefetch_extra_samples_upper_bound=0`；`loader_after_cleanup.worker_exits` 数量为配置的 2 或 4，全部 `exitcode=0`、`alive=false`、`forced=false`。
- 回传源码提交及 source digest、依赖清单、配置、GPU 信息、所有 JSON、stdout/stderr 和每条命令退出码。失败时保留已有完整证据和 `failure.json`，不要改配置自动重试。

旧两组正式训练已完成，无需为本验收重建 HEAD3 或重跑 10 轮。正式方案暂保留 eval workers=0，待此项通过后再决定是否启用多 worker。回传中的 `SHA256SUMS` 不应包含它自身；生成新清单时排除自身，原始证据包保持不动。
