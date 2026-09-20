# 复赛开发与验证记录

2026-09-20：本轮交付代码、配置工具和合成正确性测试。用户最新要求为组员休息，
**不准备或发布组员任务，不安排其验收或训练**。本页是实现说明，不是执行任务书。
T4 执行指令没有上传 GitHub，也未远程执行任何操作。

当前 `blocked_on_round2_assets`：未取得真实复赛数据，未完成 HEAD20-GCE 初始化、
CUDA 测速或新机器准入，八个 run 尚不能开跑。本机没有下载、真实数据特征提取或
正式训练。旧初赛目录、HEAD3、检查点、旧配方摘要与恢复入口保持原样。

## 实现范围

独立入口 `python -m aic_robust_clip.round2`，独立身份版本 `round2-v1`，不扩展
旧 T4 调度器或 B03/B04 的 4060 准入。`--help` 可查看参数。

| 子命令 | 行为与边界 |
| --- | --- |
| prepare / check | 元数据及摘要检查；生成八份配置，缺资产/准入则 blocked；不加载模型、训练、提取特征或下载 |
| audit | 显式审计用户取得的复赛训练 ZIP；完整解码、类别表、精确重复组、覆盖报告和固定 seed17 划分 |
| cache / init-head | 受训练机器权限约束，显式生成复赛 train 冻结特征及新的 HEAD20-GCE，不由 prepare 隐式调用 |
| startup-check | 随机微型 CLIP 结构的四张合成图、两次更新；纯正确性检查，不自动继续 |
| checks | 全套测试、compileall、pip check、diff check 与各方法 FP32/FP16 两次合成 CUDA 更新；零跳过才可准入 |
| profile / choose | 固定工程网格、保留失败记录，按预计总轮时选共同配置 |
| concurrent / admit | 四任务同步短窗及新身份回执；绑定源码、GPU UUID、环境、资产、配置和原始证据摘要 |
| run | 单模型正式入口；30 轮调度，第十轮评分、验证、保存完成后暂停；4060 候选必须已有匹配 CE 十轮对照 |
| replay-dev | 使用带标签 dev 比较轮末验证 logits 与导出/重载后的学生推理；不将 dev 伪装为 test |

规划配方在 [plan.json](../../configs/second_round/plan.json)，不是可运行配置。
以下仅生成缺资产的阻塞配置，可在开发机执行：

```bash
PYTHONPATH=src .conda/aic-robust-clip/bin/python -m aic_robust_clip.round2 prepare --output outputs/second_round/blocked-v1
PYTHONPATH=src .conda/aic-robust-clip/bin/python -m aic_robust_clip.round2 check --config outputs/second_round/blocked-v1/T4-0-CE.json
```

配置和产物目录需包含 `second_round`，拒绝混入 preliminary/semifinal 路径及
不同赛段记录；唯一可复用的是合法官方预训练权重。真实配置绑定完整解码资产摘要、
公共头实际哈希以及新的准入回执。配置准备保持独立，不构建任何缺失前置产物。

公共资产协议：仅复赛训练包，按字节/像素精确重复组约 80/10/10 划分
train/dev/confirm、seed17；保留每类覆盖与冲突标签组报告。此为项目协议，若收到
官方独立验证集或新规则，应先修订协议。收到的资产版本需记录来源 URL、日期和
版本，不强行改成预期数量。配置和数据 API 本轮仅允许 train/dev，不开放 confirm/test。

不同机器可用既有 `AIC_ARCHIVE_LOCATIONS` 保留 manifest 原字节并迁移 ZIP，新目标
仍须在 second_round 目录。`AIC_ROUND2_WEIGHTS` 可覆盖官方快照的本机路径，内容
仍按官方白名单逐文件验证。新回执不能跨机器/GPU 使用，不能复用旧 HEAD3 身份。

## 方法与状态

只使用 OpenAI CLIP ViT-B/32。`full_visual` 解冻视觉编码器与视觉投影；LoRA
保持视觉 Q/V rank=4、alpha=4。文本冻结；优化器显式划分视觉、LoRA、头和辅助
模块，拒绝参数重复或无归属。bias/归一化不衰减。独立实现不会改变旧优化器摘要。

公共 HEAD20-GCE 在新 train 冻结特征上训练 20 轮，q=.7、AdamW、LR=.01、
WD=1e-4、batch128、1 轮 warmup 后 cosine；固定第20轮，不按 dev 选头。
在线配方为头/辅助 LR=.001、全视觉 LR=.00001、LoRA LR=.0001、WD=1e-4，
224 原增强、FP16，有效 batch128，FP32 评分/验证 batch64。30 轮调度以
“完成轮次＋当前轮完成比例”推进，筛选改变样本量不改变完整预算。

- TURN：新头初始评分及逐轮全 train 固定视图 CE 损失，按观测类别 GMM 低损失
  后验 ≥.6 选择下一轮 CE 视图，不裁平类别、不改原标签。
- FINE：归一化视觉特征 Gram 主方向的投影平方，逐类 GMM 高分后验 ≥.6。
  小类使用等价的小型样本 Gram 计算，避免每类做大矩阵分解。首轮来自官方冻结
  特征，随后每轮当前模型重算；不加入损失筛选或邻居投票。
- SNSCL：前五轮原标签 CE，第五轮末启用全 train 损失 GMM；第六轮起按原文
  Eq.1–3 使用阈值 .5、软标签 EMA=.99。动量系数 .999、温度 .07、每类队列32、
  投影维128；三层随机特征模块输出均值与 log 方差，采用
  `mu + exp(logvar/2)*epsilon`。损失为软标签 CE + .1 对比项 + 1e-4 KL。
  空队列/无本类正样本时对比项为零；队列只记录真实有效训练键，不填随机占位。

SNSCL 公式在线核对：[论文 Eq.1–8](https://arxiv.org/html/2303.02404v1)，
查阅日期 2026-09-20。上述实现均登记为项目变体，不宣称原论文复现或参数最优。
本轮未保存新的官方文件或论文下载。复赛通知重读超时，资产版本仍待取得后核验。

NumPy 确定性两成分 EM：分位数初始化、最多100次、容差1e-6、方差下限1e-6，
没有新增依赖。TURN/FINE 类样本少于8或恒分时记录“不筛选”，保留原监督，不
伪造可信度。非有限数、不收敛、正常拟合后空类则停止该候选。SNSCL 使用论文的
全局 GMM，退化时停止，不生成未定义软标签。无自动降阈值、补样本或参数重试。

动量与入队按成功优化器更新推进；microbatch 累积完成前不改变队列。AMP 跳过
更新时恢复方法 RNG，队列拒绝入队时内容、指针、计数均不变。检查点包含学生、
优化器、scaler、方法阶段、筛选 ID、软标签、队列、动量模块、全部 RNG、调度和
下一轮采样身份。缺字段、身份不符或非法状态拒绝恢复。中断轮从上一个完整轮末
快照重新开始；轮内未完成的评分/验证不能成为可恢复的完整轮。原子保存保留旧快照。
第十轮之后本次 `run --resume` 仍拒绝继续。学生导出不包含文本、动量模型、
队列或随机模块，无多模型融合路径。

## 工程准入实现

固定 microbatch={4,8,16,32} × workers={2,4}，每项2次预热、10次测量；按训练、
全量评分/筛选、验证、保存的预计总轮时选择，至少15%显存余量。差距不足5%时
先少 worker 再小 microbatch。方法数值失败停止该方法，其余未执行网格留记录；
单项工程失败留证，不换参数重复同项。无共同通过配置则阻塞准入。

最终配置各做三次独立 eval worker 生命周期窗口。T4 并发短窗有同步屏障，并核对
四个训练窗口确实重叠、所有子进程与 worker 正常退出。回执绑定源码、依赖、CUDA、
GPU UUID、实名、配置、真实资产及完整原始证据链。4060 必须先明确是否原 B04
失败机器；未知身份不能放行，原机器需原证据与本次失败项复验。新代码拒绝旧回执。
这些 CUDA 功能已实现，**尚未在真实训练机器验证**，不能宣称机器已经获准开跑。

逐轮输出 best/last、逐类逐图 dev 结果、筛选覆盖与变化、SNSCL 软标签及队列统计、
实际样本访问、成功更新/AMP跳过、分阶段耗时、显存及检查点哈希。失败记录保留。
耗时无绝对淘汰门槛；后续观察点需另行评审。本次不安排 confirm/test 或排行榜提交。

## 验证

21 项复赛专项测试通过；全套 **147 项通过、零失败、零错误、零跳过**，
最终全套耗时 25.800 秒。compileall、pip check、git diff --check 均通过。
测试使用生成图像及随机微型模型，仅软件正确性证据，不是模型精度或 CUDA 准入。

验证环境：项目 Miniconda，Python 3.11.16、torch 2.7.0+cpu、NumPy 2.2.6、
transformers 4.57.6、Pillow 11.3.0、huggingface-hub 0.36.2。
验证源码树摘要（沿用 current_code_revision 的 src 摘要算法）：
`aecaa7a933d8220937ee6a621f1c79298bda6bc93d1ea367d9733bf821ef1dea`。

```bash
PYTHONPATH=src .conda/aic-robust-clip/bin/python -m unittest discover -s tests -p test_round2.py -v
PYTHONPATH=src .conda/aic-robust-clip/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .conda/aic-robust-clip/bin/python -m compileall -q src tests
.conda/aic-robust-clip/bin/python -m pip check
git diff --check
```

全套 worker 回归首次在受限沙箱遇到本地 socket 权限错误；没有删除或跳过测试，
最终在允许本地进程通信的环境中重新通过。保留日志位于开发机
`/tmp/aic-round2-focused-release.log`、`/tmp/aic-round2-suite-release.log`。

新增回归覆盖：赛段/旧头/旧回执拒绝、配置准备无训练副作用、参数分组与文本冻结、
GMM/FINE/软标签/重参数化公式、空队列/拒绝/环绕/有效键、AMP跳过、动量按更新推进、
完整方法状态恢复、连续与中断恢复参数一致、完整30轮调度下十轮验证保存后暂停、
学生 dev 重载一致、准入证据篡改和匹配对照门槛。旧缓存失效、worker生命周期、
历史身份及恢复回归也全部保留并通过。
