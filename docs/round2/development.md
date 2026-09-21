# 复赛开发与验证记录

2026-09-20：本轮交付代码、配置工具和合成正确性测试。用户最新要求为组员休息，
**不准备或发布组员任务，不安排其验收或训练**。本页是实现说明，不是执行任务书。
T4 执行指令没有上传 GitHub，也未远程执行任何操作。

当前 `blocked_on_round2_assets`：未取得真实复赛数据，未完成 HEAD20-GCE 初始化、
CUDA 测速或新机器准入，八个 run 尚不能开跑。本机没有下载、真实数据特征提取或
正式训练。旧初赛目录、HEAD3、检查点、旧配方摘要与恢复入口保持原样。

## 已授权的初赛独立验证

用户已明确授权在复赛资产仍缺失时，使用现有初赛正式 B03/普通 CE 配置做一次
独立的方法验证。入口为
`python -m aic_robust_clip.preliminary_pilot`，固定为 LoRA+TURN、microbatch 32、
workers 2、有效 batch 128；它先执行全套测试、FP32/FP16 两次 CUDA 合成更新和
三次短评估窗口，再从已核验的初赛 HEAD3 重新初始化，按完整 30 轮计划运行到第
10 轮完成评分、验证和保存后暂停。profile 权重不会用于正式 run。

该入口只接受 `stage=preliminary`、seed17、B03/CE、formal 的已有配置，输出必须
位于新的 `preliminary` 目录；拒绝复赛配置、研究候选、confirm/test 和本机执行。
输出中的 `MORNING.md`、`morning-summary.json`、检查日志、筛选记录、checkpoint
摘要和 dev 回放均标记为 `preliminary_method_validation_only`，不能作为复赛资产、
HEAD20-GCE、排行榜结果或严格的 CE 因果对照。任何测试、准入、GMM、OOM 或回放
错误都会保留失败证据并停止，不自动重试、降阈值、换参或续跑。

该入口的 `--startup-check --device cpu` 模式仅使用随机微型模型、四张合成图和
两次更新，不接受真实资产，也不自动进入正式流程。正式入口依赖的 CUDA 环境必须
在独立 checkout 的项目环境中现场核验；旧机器文件中的版本和 `not_run` 字段仅是
历史快照。T4 的固定提交、具体路径与后台启动任务仍只在对话中交付。

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

NumPy 确定性两成分 EM：分位数初始化、最多1000次、容差1e-6、方差下限1e-6，
两个初始四分位点完全相同而输入不恒定时，显式记录并使用最小/最大端点初始化，
不加随机扰动、不删重复观测、不重试其他初始化。没有新增依赖。TURN/FINE 类样本少于8或恒分时记录“不筛选”，保留原监督，不
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

以下为原复赛版本 `9908199` 的验收记录：
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

2026-09-21 初赛入口增量验收：新增 8 项专项测试，包括真实初赛接口上的合成图像
加载、HEAD3 工厂、TURN 训练循环、十轮暂停、学生导出后 dev 回放，以及独立两次
更新启动模式。全部使用生成数据和随机微型模型，不是正式训练或真实 CUDA 证据。
T4 上的 `software_checks` 会重新执行全套测试，任一失败或跳过都阻止训练。

本次全套 **155 项通过、零失败、零错误、零跳过**，耗时 76.742 秒；compileall、
pip check、git diff --check 通过。多进程测试在获准支持本地进程通信的环境中运行，
日志为 `/tmp/aic-pilot-final-suite.log`；专项日志为 `/tmp/aic-pilot-final-focused.log`。
此前的只读 Git 和审批额度阻塞属于历史情况，不能代替本次发布状态核验。

## 初赛 profile GMM 边界修复（2026-09-21，4341c3e 历史记录）

服务器回传：`99d7460` 的初赛 TURN 在 profile 中报 GMM 100 次未收敛，未进入正式
epoch 训练。原始证据只有异常和运行身份，不能确认真实输入属于哪类退化。
调用链为 `preliminary_pilot.run_night → admission._profile_one → engine.scoring →
MethodState.rescore → select → gmm`；profile 的初次筛选使用完整 train 固定视图，
不是只拿测速窗口的少量样本拟合。

已复现并修复两处边界：重复值使初始四分位点相同时，两个等参分量会停留在对称
解；采用上文确定性端点规则解除该初始化退化。原循环在第100次 M 更新后立即
抛异常，未对新参数做 E 步收敛检查；现以初始化 E 步为第0次，每次 M 更新后评估，
最多100次更新。保持两分量、原样本权重、原始分数尺度、方差下限1e-6、似然容差
1e-6和TURN/FINE阈值0.6，不进行多次拟合或改参重试。

少于8样本与恒分仍按原规则明确记录“不筛选”；近常数但非恒定输入不会因此自动
放行：分量无可靠顺序、数值异常、正常拟合空类、100次后未收敛均继续停止。SNSCL
全局 GMM 退化也仍停止，不产生替代软标签。成功时记录初始化分支与迭代统计；
失败时 profile 及外层失败 JSON 保留 `method_diagnostics`，含方法/类别索引、数量、
唯一值数/重复比例、五个分位数、范围、方差、初始化、最后分量参数、实际迭代数和
平均对数似然增量。不会保存图片、样本ID、完整损失/特征/后验数组。

本修复没有声称已重现服务器的未知输入。再次失败时应回传新的诊断证据，不能改成
成功回执。旧失败目录保留；新代码需重新验收，并用新输出目录从原HEAD3开始。
服务器修复通过本地固定提交和 Git bundle 交给操作 agent 传输，不依赖 GitHub。

验证：新增 GMM 专项9项通过（重复值、近常数、少样本、短迭代窗口、正常双峰、
确定性、完整100次仍失败、真实profile错误传播和失败证据保留）；全套164项通过，
零失败、零错误、零跳过，耗时32.189秒。compileall、pip check、git diff --check
均通过。日志位于开发机 `/tmp/aic-gmm-focused.log`、`/tmp/aic-gmm-suite.log`。

## 正常数据慢收敛与固定预算修订（2026-09-21）

第二次服务器回传的外层失败诊断显示：观测类别10，170样本、167唯一值，方差
0.11615313786018694，初始四分位点不同，未触发方差下限；第100次平均对数似然
增量为1.3274247719907706e-6，仍高于1e-6。这是正常输入在原预算内未收敛，
失败判定正确，不能放宽容差、套用恒分例外或把已有结果标记为通过。

原始回传SHA256清单与checks.json摘要均已核验；服务器报告164项软件测试和
FP32/FP16各两次合成CUDA更新通过，但没有正式训练轮次。包内failure.json和
failure-001.json是相同外层失败内容，不是两份独立profile证据；没有原评分数组，
不能从聚合统计精确重建输入。

用户明确接受把固定GMM求解上限从100提高到1000。代码默认值与允许上限、配置
RECIPE、复赛规划JSON和初赛派生配置统一使用同一预算；这是显式数值求解预算
修订，不是声称原100次循环存在新错误。仍从固定初始化只拟合一次，满足原1e-6
容差即可提前结束；达到1000仍不收敛则保留诊断并停止。两分量、原始分数尺度、
方差下限、筛选阈值、方法和训练日程不变。旧源码/配置身份的回执不可用于新任务。

合成回归使用seed17生成170个值并设置3个重复观测，得到167唯一值和相同方差。
它与服务器原数组不同：范围约1.99876（服务器约2.20159）。100次增量约1.5333e-6，
必须失败；固定1000预算下在第109次达到约9.8796e-7并成功，重复调用完全确定。
测试还校验TURN只调用一次拟合、100次测试预算仍失败、旧阈值及30轮/10轮暂停不变。
这个结果验证预算修订的行为，不保证未知服务器数组或所有类别都能在1000次内收敛。

服务器重试使用新输出目录，从原HEAD3重新开始，重新完成软件/CUDA/profile验收。
回传材料应保留profile子目录，避免其failure.json与根目录同名文件相互覆盖。
代码和指令通过本机交付包由操作agent传输，不上传GitHub、不由准备agent启动训练。

接手续验：GMM专项11项、全套166项通过，零失败、零错误、零跳过；全套耗时
25.268秒。compileall、pip check、git diff --check通过；8份复赛阻塞配置均完成
纯元数据生成和加载校验，GMM预算一致为1000，未加载真实资产或解除阻塞。
环境仍为Python 3.11.16、torch 2.7.0+cpu、NumPy 2.2.6、transformers 4.57.6、
Pillow 11.3.0、huggingface-hub 0.36.2；源码树摘要为
`ed739997db4ff7fb6cafc912f74205df5890b6def596ad36a6a049f2d49e81a4`。
日志为`/tmp/aic-gmm1000-takeover-focused.log`、`/tmp/aic-gmm1000-takeover-suite.log`、
`/tmp/aic-gmm1000-takeover-pip-check.log`。这些仅为本机软件正确性证据，
新版本的服务器CUDA/profile和正式训练结果仍待执行agent验证。

本机交付工具另通过4项纯合成文件测试：嵌套同名失败文件分别保留、哈希清单验证、
权重/逐样本文件排除、拒绝覆盖和符号链接、中途快照标记及显式启动参数保护。
服务器脚本通过`bash -n`，没有在本机执行`--execute`；任务说明与工具仅随私有包交付。
