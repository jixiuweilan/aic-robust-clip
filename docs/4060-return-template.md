# 4060 本轮回传表

复制到新的私有任务目录后填写；未知项填“待核验”，不要预填通过。每次回传保留旧版。

## 基本信息与前置核验

- 组员姓名 / A 或 B：
- 日期、时区、机器名称、GPU 型号/总显存/编号、授权使用范围：
- 项目绝对路径 / 本轮任务绝对路径 / 来源配置路径：
- Git 完整提交 / 源码摘要 / 本地修改 diff 路径：
- Python、PyTorch、CUDA、驱动版本 / `pip-freeze.txt` 路径：
- 机器绑定文件 / 指纹：
- ZIP 所在本地 Linux 文件系统 / `AIC_ARCHIVE_LOCATIONS` 路径与摘要：
- manifest / split / class_map SHA-256：
- 共同 HEAD3 的 `head.json` / `head.pt` SHA-256（须与另一台相同）：
- 官方权重 revision / 现有缓存路径：
- 是否为 B04 原失败机器 / 历史失败报告路径：
- 缺失项、阻塞项：

## 第一批：测试与六组测速

- relocation / 全套测试：实际运行数量、失败数、跳过数、日志路径：
- B04 原失败项：原机器标识、新日志中的测试名与结果、复验日期：
- compileall / pip check / git diff --check 结果与日志：
- FP32 / FP16 合成 CUDA：各两次更新、optimizer_updated、stopped_by_limit：
- `checks/checks.json` / `bundle/bundle.json` / `summary-A.json` 或 `summary-B.json`：
- 测速前 GPU 空闲情况（附 `nvidia-smi` 输出和时间）：

每项均为 train/eval 各 2 次预热、10 次测量；失败项写原因和日志路径，不能留空或改参重试。

| profile | train / eval 退出码 | 预计总轮时(s) | 冷启动(s，train/eval) | 数据等待(s，train/eval) | 峰值分配/预留显存 | 余量≥15% | worker 清理 | 失败证据 |
|---|---|---|---|---|---|---|---|---|
| fp32-m16-w4-e0 | | | | | | | | |
| fp32-m32-w4-e4 | | | | | | | | |
| fp16-m32-w4-e4 | | | | | | | | |
| fp16-m64-w4-e4 | | | | | | | | |
| fp16-m128-w4-e4 | | | | | | | | |
| fp16-m64-w2-e2 | | | | | | | | |

## 第二批：最终准入

- 负责人发回的 `selection.json` SHA-256 / 选中 profile：
- 三次独立 eval 各自日志路径 / 退出码 / 显存 / worker 全部正常退出情况：
- `acceptance.json` 路径与 SHA-256：
- 原始测速、配置、环境、源码自第一批起有无变化：
- 负责人确认两台均通过的记录（未确认则不得开跑）：

## 第三批：各三轮结果

- 对照目录 / 候选目录（必须不同且均为本轮新建）：
- control LR=`1e-4`；candidate LR=（A `3e-5` / B `3e-4`）：
- 两支各自 `completed_epochs=3`、三条 epoch `complete=true`、`paused=true`、`stopped_by_limit=false`：
- 两支各自更新数 / 训练样本访问数 / 实际 train/dev 样本数：
- 总 wall time / 中断及恢复记录（没有则写无）：

指标填写百分数，增益用百分点；每轮两支使用同一 dev 划分。

| epoch | control macro/micro | candidate macro/micro | Δmacro/Δmicro（候选－对照） | control 实际 train/dev/save(s) | candidate 实际 train/dev/save(s) |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

- 两支最佳 epoch / best.pt 和 last.pt SHA-256 / 原机保留位置：
- resolved.json、result.json、epochs.json、全部 dev-epoch-*.json、dev-predictions.json 路径：
- 逐类指标 / 逐图预测 / 数据等待 / 显存 / stdout / stderr / 退出码文件：
- 异常或偏离任务之处及处理经过：
- confirm/test 图像是否始终未访问 / 是否未提交排行榜：
- 三轮后的建议及依据（不自行续跑）：

## 本批附件

- 回传批次、包名、包 SHA-256：
- 文件 SHA256SUMS（排除校验清单自身）：
- 本批包含的文件 / 未传大文件及原机保留路径：
- 历史报告保留位置：
