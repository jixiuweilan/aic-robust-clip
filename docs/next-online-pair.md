# 下一步：完成 B04 / B03 公平对照

> 2026-09-19：本页保留历史 B04/B03 对照流程，不是本轮 4060 执行指令。当前按 [4060 学习率任务包](4060-next-round.md) 操作：不重复旧十轮基线，不重建已有共同 HEAD3，不操作正在运行的 T4。

更新：2026-09-17。状态：用户已决定布置本任务，尚未收到本轮执行结果。
这是交给一位组员的一项完整任务，不展开其他方法或参数搜索。
本机只做合成数据正确性测试；下文正式命令仅供独立训练机器执行。

## 目标与当前证据

回答一个问题：相同数据、初始化、增强和训练预算下，视觉 Q/V LoRA
是否优于冻结编码器？B04 是在线增强、冻结 CLIP、只训练分类头；B03
在同样条件下增加 Q/V LoRA。B01 是固定缓存特征的 CACHE20，不能直接
作为 LoRA 效果的公平对照。

已收到 B01 的 20 轮日志和 best/last 检查点，检查点实物哈希、累计
12,920 次更新及记录的一致性已核对。收到的最佳 dev micro Top-1 为
60.7535%，macro recall 为 60.7200%（第 19 轮）。这是有噪声验证标签的
代理指标，不是官方测试分数，也尚未在接收端复现推理。

更新后的 `B01-deliverable-20260917.zip` 的 SHA-256：
`2c432859e36ae8e5d400ca17422545091a032535ba3b78b220ddcf3348483a2f`。
补交的最佳轮次预测和 train/dev 缓存 index 已通过指标重算、身份及覆盖
检查；best/last 实物哈希未变。详细接收审查保存在本机忽略目录
`outputs/reviews/B01-deliverable-20260917-review-v2-2c432859e36a.md`。
不上传原包到公开仓库，不重训 B01、重分数据或重建缓存。

组员声明旧原始测试 stdout、startup-checks JSON 和 pip freeze 输出已丢失；
在相同 Conda 环境（Python 3.11.16、PyTorch 2.7.0+cu126）重跑得到
45/45、workflow 9/9、compileall、git diff、pip check 通过。此条仅为
组员报告，未收到新日志，不等于恢复旧证据；此前收到并检查的环境库存和
B03 CUDA 报告仍作为已有交付保存。用户已决定不再追补旧日志，进入本任务。
这个决定不代表旧证据完整或独立复现了 GPU 运行。本轮须留存新的实际日志。

## 固定不变的实验约定

| 项目 | B04 和 B03 共用约定 |
| --- | --- |
| 配置 | `configs/formal/B04.json`、`configs/formal/B03.json` |
| 数据 | 原 preliminary manifest、split-v2、class-map；不重新审计生成或划分 |
| 样本数 | train 82,586；dev 10,378；confirm 10,254 暂不使用 |
| 模型 | 官方 OpenAI CLIP ViT-B/32，固定 revision 与权重哈希 |
| 初始化 | 同一个 seed 17 的 HEAD3 文件，不用 B01 best 替代 |
| 主训练 | ONLINE10，各 10 轮；microbatch 1，有效 batch 128，float32 |
| 优化 | CE；头 LR 1e-3，LoRA LR 1e-4；AdamW，warmup 1 轮 + cosine |
| 唯一方法差异 | B04 编码器冻结；B03 增加视觉 Q/V LoRA，rank=4、alpha=4 |
| 选模 | dev macro recall 优先，micro Top-1 次之，再取较早轮次 |

两份配置只允许 recipe 和输出目录不同。使用同一版本代码和环境先后
运行，不在两次之间修改增强、学习率、精度、batch、seed 或训练轮数。
沿用原训练机器与数据绝对路径；manifest 包含路径，擅自改写会改变身份。
不启用 W/P/I、新损失、重标注、confirm、test 推理或模型融合。

冻结 split digest：
`6faf643b751abbfd83514ce4eabe1a6629f524020d67aab5ad8d91295117d325`。
冻结 class-map digest：
`0a9454efba76698a829731485838aabcb80b6726ad7f3645d703119d4812010a`。

## 执行顺序（仅独立训练机器）

在仓库根目录激活已有 `.conda/aic-robust-clip`，保留现有机器绑定、权重、
缓存和 B01 输出。无需重新安装或下载；缺少文件先报告。
记录本轮实际代码版本，不要求与 B01 的旧提交完全相同，但本轮两支必须相同。
已有未提交修改时先保存并说明，不执行强制 reset/pull。

本轮不需要修改训练代码或正式配置。可沿用 B01 使用的提交
`6aa8660b3e5f75f6194ec35ee6b5a0cabe185819`，其原套件为 45 项、workflow 9 项。
本地新增对照回归与本文尚未推送；若只收到本文，不要为了凑 46 项而安装
依赖或改代码。若同步了新版 `tests/test_workflow.py`，应为 46 项、workflow
10 项。两种版本均须明确记录且零失败、零跳过；两支运行代码必须一致。

```bash
conda activate "$PWD/.conda/aic-robust-clip"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p outputs/online-pair
AIC_PAIR_DIR="$(mktemp -d "$PWD/outputs/online-pair/acceptance-XXXXXX")"
set -o pipefail
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v 2>&1 | tee "$AIC_PAIR_DIR/tests.txt"
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_workflow.py -v 2>&1 | tee "$AIC_PAIR_DIR/workflow-tests.txt"
python -m compileall -q src tests 2>&1 | tee "$AIC_PAIR_DIR/compileall.txt"
git diff --check 2>&1 | tee "$AIC_PAIR_DIR/diff-check.txt"
python -m pip check 2>&1 | tee "$AIC_PAIR_DIR/pip-check.txt"
```

每条命令退出码为 0 才执行下一条；失败就停止。无输出的检查记录退出码，
不能仅以空日志声称通过。不要把专项测试数与全套相加，它是全套的子集。

```bash
git rev-parse HEAD | tee "$AIC_PAIR_DIR/git-head.txt"
git status --short | tee "$AIC_PAIR_DIR/git-status.txt"
git diff --binary | tee "$AIC_PAIR_DIR/source.diff"
python -c 'from aic_robust_clip.runtime import current_code_revision; print(current_code_revision())' | tee "$AIC_PAIR_DIR/source-identity.txt"
aic-doctor --output "$AIC_PAIR_DIR/environment.json" --record-lock "$AIC_PAIR_DIR/installed-lock.txt"
aic-doctor --config configs/formal/B04.json --output "$AIC_PAIR_DIR/B04-preflight.json"
aic-doctor --config configs/formal/B03.json --output "$AIC_PAIR_DIR/B03-preflight.json"
aic-check-model --weights checkpoints/openai-clip-vit-b32 \
  --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --recipe B04 --device cuda --output "$AIC_PAIR_DIR/B04-startup.json"
aic-check-model --weights checkpoints/openai-clip-vit-b32 \
  --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --recipe B03 --device cuda --output "$AIC_PAIR_DIR/B03-startup.json"
```

两个 startup 都应为生成图片、2 次更新、有限终止、零比赛图片读取。
`aic-doctor` 不证明 HEAD3 已存在；继续检查下一步。若机器绑定报错，不绕过
限制，按 [环境交接](handoff.md) 确认确实是独立训练机器。

HEAD3 必须单独训练一次：仅用已有 train 缓存，3 轮，取最后分类头，
不使用 dev 选初始化。B01 不需要 HEAD3，所以缺少它不构成 B01 失败。
**仅当 `outputs/preliminary/shared-head-seed17` 不存在时**执行：

```bash
aic-init-head --config configs/formal/B04.json 2>&1 | tee "$AIC_PAIR_DIR/HEAD3.log"
```

如果目录已存在，先验证下面的身份、哈希和完成量；不覆盖、不删除、不
自动重跑。若是残缺输出或 HEAD-SMOKE，停止并反馈。

```bash
python - <<'PY'
from pathlib import Path
from aic_robust_clip.configuration import prepare
from aic_robust_clip.contracts import read_json
from aic_robust_clip.workflow import load_head
digests = []
for recipe in ('B04', 'B03'):
    ctx = prepare(f'configs/formal/{recipe}.json')
    _, digest = load_head(ctx)
    descriptor = read_json(Path(ctx.config['head']) / 'head.json')
    result = descriptor['result']
    assert descriptor['identity']['profile'] == 'HEAD3'
    assert (result['updates'], result['samples']) == (1938, 247758)
    assert not result['stopped_by_limit']
    assert len(result['epoch_logs']) == 3 and all(e['complete'] for e in result['epoch_logs'])
    digests.append(digest)
    print(recipe, descriptor['identity'], digest)
assert digests[0] == digests[1]
PY
```

然后顺序执行，上一条成功且输出完整后才执行下一条：

```bash
aic-train --config configs/formal/B04.json 2>&1 | tee "$AIC_PAIR_DIR/B04.log"
aic-train --config configs/formal/B03.json 2>&1 | tee "$AIC_PAIR_DIR/B03.log"
```

每支预期完整 10 轮、6,460 次更新、825,860 次样本访问，保留每轮 dev
预测与 best/last。不能仅看 `status=complete` 就宣告成功，仍须核对
`stopped_by_limit=false`、完整轮数与累计更新量。
不要用 `result.json.dev_metrics` 代替 best 指标——它是最后一轮。

发生 OOM、非有限 loss 或异常时，保存错误和当前目录，停止反馈；不要
静默改参数或自动重试。若仅进程中断，负责人确认后可用相同配置从对应
`last.pt` 续跑，例如：

```bash
aic-train --config configs/formal/B03.json --resume outputs/preliminary/runs/B03-seed17/last.pt
```

训练时间不预估为 B01 的 1 小时 44 分钟：B01 只训练缓存特征上的分类头，
该耗时不包含编码器前向/反向。记录 HEAD3 和两支实际耗时、峰值显存；
若发生续跑，逐段保存计时，最终一次调用的 elapsed 不等于全部耗时。

## 一次性交付与判断标准

通过团队私有传输渠道交付，不提交到公开 GitHub：

- 本轮 acceptance 目录、完整命令/终端日志、Git 提交与 source digest；有
  本地修改则附对应 diff，两支之间不能修改运行代码。
- `shared-head-seed17/` 内的 head.pt、head.json、resolved.json、epochs.json；
  原 train/dev 缓存的 index.json，无需重复传输大缓存和比赛原图。
- 两个 run 目录中的 result.json、resolved.json、epochs.json、所有
  `dev-epoch-*.json`、best.pt、last.pt；包含异常/续跑说明（如有）。
- 一张结果表：recipe、最佳轮次、dev macro/micro、head/mid/tail、训练时间、
  峰值显存、可训练参数数、初始化哈希、best 检查点哈希、split/配置/source 身份。

打包为新的 `B04-B03-deliverable-YYYYMMDD.zip`，附整个压缩包 SHA-256；
不要覆盖 B01 交付包。先完成两支启动检查和 HEAD3 后发一句开跑状态，
最后一次性交付全部结果，异常则立即反馈。不要求额外中途实验或调参。

我们接收后检查身份一致性、检查点哈希、完成预算，并从最佳轮次逐样本
预测重算指标；对比 **B03 − B04 的百分点差值**。单 seed 17 只做筛选，
不能宣称统计稳定提升。如果 B03 有一致收益，再决定多 seed 或噪声鲁棒模块；
若无收益，先分析误差和训练曲线，不立即展开全部矩阵。confirm 继续封存。

## 本机准备验证

新增合成数据回归覆盖：共享分类头/身份、在线采样顺序和图像张量一致、
零初始化 LoRA 下初始 logits 一致、B04 编码器不更新、B03 LoRA 有更新，
两支都在 2 次更新后终止。正式配置不变；未执行真实数据训练或下载。

2026-09-17 本机使用项目 Miniconda、CPU、离线环境完成：workflow 专项
10/10，全套 46/46，均无跳过；`git diff --check` 通过。原始 B01 ZIP
当时计算 SHA-256 与首版收据一致，随后用户更新同名包；新版审查见上文。
已加入根目录交付包忽略规则，agent 未改动收到的压缩包。
这些仅为软件正确性证据，不代表 B04/B03 已训练、收敛或优于 B01。
