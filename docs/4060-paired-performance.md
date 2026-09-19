# 两台 4060：B03 / B04 训练任务

**分工：组员 A 负责 B03，组员 B 负责 B04。各自在自己的 4060 上先验收，再训练 10 轮。**
不要操作 T4 服务器上正在运行的任务，也不要更新其代码或环境。本次结果仅用于内部对照，不提交主办方排行榜。

2026-09-19 更新：旧配置的两组 10 轮结果已经回传并完成核验，见[结果评审](research/t4-control-results-20260919.md)。无需为了补齐基线重复训练；尚未启动的组员先做[worker 专项验收](eval-worker-acceptance.md)，完整重跑是否必要由负责人结合验收结果决定。以下正式训练步骤仅适用于已经确认需要的新运行。

## 一、准备资料与环境

- 使用独立的 Linux / WSL 工作目录。代码必须包含 `2824689` 的 worker 修复；两人使用相同代码、依赖版本和 seed 17。
- 使用 Miniconda 环境 `.conda/aic-robust-clip`，禁止安装到 `base`。缺依赖、权重或数据时列出清单，由用户下载或传输，agent 不自行下载。
- 使用相同的当前赛段数据、冻结 manifest、split、class_map 和官方 CLIP ViT-B/32 权重。不得重新划分数据或改写冻结记录。
- 从负责人取得服务器已完成的**同一份 HEAD3**：`head.pt` 和 `head.json`，核对两份文件的 SHA-256。不要分别重建，不用测速产物替代，不恢复旧 T4 训练断点。
- B03/B04 在线训练不需要重新生成特征缓存。不要重跑审计、划分、缓存或 HEAD3。
- 每台机器单独登记训练资格，不能复制其他机器的绑定文件。执行 agent 自行向组员询问 SSH 地址、工作目录和获准使用的 GPU，不索要或记录私钥。

以下命令均在项目根目录执行；示例 GPU `0` 必须替换为本机获准使用的编号。

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# B04 负责人改成 B04；后续命令使用此变量。
AIC_RECIPE=B03
```

若数据移动了位置，创建本机路径映射 JSON，并在每个实际任务进程中设置：

```bash
export AIC_ARCHIVE_LOCATIONS=/绝对路径/archive-locations.json
```

映射格式如下；旧路径来自冻结 manifest，新路径指向本机文件，摘要必须与冻结记录和实际文件同时一致。

```json
{
  "schema_version": 1,
  "reason": "迁移到组员的独立训练机器",
  "archives": {
    "/冻结记录中的/train.zip": {
      "path": "/本机实际位置/train.zip",
      "sha256": "替换为冻结记录中的64位小写SHA256"
    }
  }
}
```

以上路径和摘要均为占位符，不可照抄运行。每个新 worker 首次读取都会完整校验压缩包，冷启动可能较慢，不要重复启动任务。

## 二、测试通过后生成配置

先运行以下测试，全部通过且零跳过才能继续。缺依赖或任何失败均停止并回报。

```bash
CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -p test_loader_lifecycle.py -v
CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -p test_performance.py -v
CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -v
.conda/aic-robust-clip/bin/python -m compileall -q src tests
.conda/aic-robust-clip/bin/python -m pip check
git diff --check
```

本次代码的预期结果：生命周期测试 10 项、性能测试 20 项、完整测试 90 项。开发工作树另有 1 项未提交测试，因此本机为 91 项；干净代码的 90 项不是漏测。后续新增测试时以对应版本回执为准。

保存环境信息，完成最多两次更新的 CUDA 启动检查；成功后登记本机。下列输出文件必须不存在；已有有效绑定则复用，不覆盖或手改。

```bash
mkdir -p outputs/4060-acceptance
nvidia-smi
.conda/aic-robust-clip/bin/aic-doctor \
  --output outputs/4060-acceptance/environment.json \
  --record-lock outputs/4060-acceptance/pip-freeze.txt
.conda/aic-robust-clip/bin/aic-check-model \
  --weights checkpoints/openai-clip-vit-b32 \
  --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --recipe "$AIC_RECIPE" --device cuda \
  --output outputs/4060-acceptance/startup.json
.conda/aic-robust-clip/bin/aic-doctor --bind-training machine-training.json
```

启动报告须显示 CUDA、`optimizer_updated=true`、`stopped_by_limit=true`，没有读取比赛图像；这不是正式训练结果。

复制对应的 `configs/formal/B03.json` 或 `B04.json` 为本机配置，填好本机的权重、manifest、split、class_map、HEAD3、机器绑定和输出路径，建议使用绝对路径。保持 FP32、普通 CE、seed 17、10 轮及原有优化器、调度器和增强方式不变。

从已填好的本机配置生成候选，目标目录必须不存在。只使用本人负责模型的 `m16-w4`，不要运行其他候选。

```bash
.conda/aic-robust-clip/bin/python -m aic_robust_clip.performance_configs \
  --config /绝对路径/本机正式配置.json \
  --output-dir outputs/4060-candidates
AIC_CONFIG="$PWD/outputs/4060-candidates/$AIC_RECIPE-m16-w4.json"
```

确认配置：训练 microbatch 16、有效 batch 128、梯度累积 8、训练 workers 4；验证 batch 64、workers 0。生成器同时设置 prefetch 2、pin_memory=true、缓存 batch 64、HEAD3 batch 128，但本任务不运行缓存或 HEAD3 生成。

## 三、有界测速通过后，分别训练

每人做一次训练短窗，以及三次验证短窗。每次均为 2 次预热加 10 次测量，不自动接着正式训练。

```bash
.conda/aic-robust-clip/bin/python -m aic_robust_clip.benchmark \
  --config "$AIC_CONFIG" --phase train --warmup-steps 2 --measure-steps 10 \
  --output outputs/4060-acceptance/train-01
.conda/aic-robust-clip/bin/python -m aic_robust_clip.benchmark \
  --config "$AIC_CONFIG" --phase eval --warmup-steps 2 --measure-steps 10 \
  --output outputs/4060-acceptance/eval-01
```

首个验证短窗成功后，将输出目录分别改为 `eval-02`、`eval-03` 再各执行一次。每条命令检查退出码；任何一次失败都停止，不覆盖输出、不自动重试。每次保留标准输出、标准错误和生成的 JSON。

验收要求：

- 训练 4 workers、验证 0 workers，batch 与配置一致；没有 OOM、SIGABRT、卡死或退出清理错误。
- 保留吞吐、数据等待时间、冷启动时间及显存占用；显存建议至少留 15% 余量，不能直接套用 T4 测速结果。
- 两人回传验收结果，由负责人核对 HEAD3、数据身份、代码、依赖和训练设置，确认共同配置后才开始正式训练。

```bash
# 仅在验收通过、共同配置获确认后执行。
# 正式输出目录必须是新目录，不能指向测速产物或旧训练目录。
.conda/aic-robust-clip/bin/aic-train --config "$AIC_CONFIG"
```

各自只运行被分配的模型，完成 10 轮。记录第一整轮含验证和保存检查点的真实耗时，再估计总时间；目前不承诺一小时完成。发生错误立即保留日志并回报，不自行降精度、减轮数、换 batch 或恢复其他任务的检查点。

验证 workers=0 是隔离配置，不代表原多 worker 崩溃已经根治。只有另行安排诊断时，才复制配置改成 `eval_num_workers=4`，用新目录最多复测三次，首次失败即停止并回传 `failure.json`。诊断不阻塞已通过验收的隔离路线；4060 通过也不能代替原 T4 环境复验。

## 四、回传材料

每人按本人的模型名称归档，回传：

- 代码提交及源摘要、最终配置、环境报告、依赖清单、GPU 信息。
- 数据身份摘要及 `head.pt`、`head.json` 的 SHA-256，便于核对配对一致性。
- 全部测试、启动检查和测速日志、退出码、JSON；每轮训练/验证指标及耗时。
- 最终模型与元数据、检查点 SHA-256、完成或失败说明。模型通过团队约定渠道传输，不上传 GitHub。

数据、权重、机器绑定、密钥均不得上传 GitHub。新结果与 T4 在跑结果分别命名；B03/B04 即使训练完成也不具备排行榜提交资格，只有经过充分优化并留存选择证据的叶子方案才能提交。
