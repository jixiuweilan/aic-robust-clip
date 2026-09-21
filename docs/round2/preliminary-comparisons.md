# 初赛同期对照与机器准备

本轮仅使用已授权的初赛train/dev、已有seed17划分及已核验HEAD3。
新入口为 `python -m aic_robust_clip.preliminary_experiments`；正式复赛仍待资产，
不把这些结果作为复赛初始化或成绩。当前不安排组员工作，交付由操作agent接手。

## 配方和资源

| 资源 | 本轮用途 |
| --- | --- |
| T4-0 | LoRA＋CE，同期普通监督对照，优先开跑 |
| T4-1 | LoRA＋FINE，严格零选中/数值失败停止 |
| T4-2 | 保留已交付TURN-ABSTAIN任务；未启动时可使用新入口，不重复启动 |
| T4-3 | LoRA＋SNSCL，5轮CE后启用修正与对比学习，严格退化停止 |
| 4060-A | CE＋TURN的机器准入，暂不释放正式训练 |
| 4060-B | CE＋FINE的机器准入，暂不释放正式训练 |

均为ViT-B/32项目变体。224、原增强、AdamW、seed17、Q/V LoRA rank4 alpha4、
LoRA学习率1e-4、头1e-3、有效batch128、FP16训练、FP32评分/验证保持一致。
完整30轮、1轮warmup后cosine，运行到第10轮完成评分、dev验证、保存后暂停；
不自动续跑，不使用候选或CE训练后的权重初始化另一个任务。
T4固定microbatch32/workers2，评分和dev batch64。该初赛增量验证不另做T4参数搜索；
复赛正式的完整工程网格规则保持不变。TURN显式弃权只属于TURN，FINE/SNSCL不会继承。

优先释放CE，不让FINE/SNSCL的准入失败拖住普通监督对照。`admit-t4`支持显式方法子集：
可先CE单任务准入，再对FINE/SNSCL做并发短窗；各回执如实记录覆盖方法数及重叠时间，
不能把子集回执描述成四任务准入。全部尚未运行时，也可一次验收四任务。
方法组内任何失败都停止该组，不重试、不降阈值；独立已运行任务不被终止。
新增任务在现有负载下做实际profile，保留15%显存余量。共享CPU/内存/读取影响会进入
实际耗时与失败证据，不以绝对时间门槛淘汰方法。

## 已实现命令

以下是接口说明；训练机器的固定提交、UUID、真实路径及可直接执行脚本随私下交付包提供。
元数据与启动检查不需要真实资产，也不会自动继续训练：

```bash
python -m aic_robust_clip.preliminary_experiments prepare --output outputs/preliminary/comparison-plans
python -m aic_robust_clip.preliminary_experiments startup-check --method snscl --device cpu
python -m aic_robust_clip.preliminary_experiments --help
```

`prepare`生成4份`blocked_on_machine_admission`配置，只读取程序元数据。
`startup-check`对指定方法使用四张合成图、两次更新。
`admit-t4`按显式GPU分配分别运行全套测试及对应方法FP32/FP16合成检查，
随后执行完整train固定视图评分、2次预热＋10次测量、三次独立dev窗口、保存和学生回放。
并发组通过共同开始屏障证明训练窗口重叠，并等待worker退出；该命令自身不训练正式轮次。
`run`只消费本入口的T4回执，重新核验源码/依赖/GPU、HEAD3、配置与资产；
方法、策略或证据摘要不符即停止。恢复必须显式`--resume`且仍不能越过第10轮。

`admit-4060`只生成`4060_admission_only`回执，`run`拒绝消费它。
操作agent提供当前机器的source-config、owner、machine-history；未知身份不伪造。
machine-history包含host、gpu_uuid、original_b04_failure布尔值、reviewer和basis。
原B04失败机另提供绑定同一host/UUID的previous-failure，注明test为
`test_verification_cached_only_for_unchanged_file_and_process`及original_evidence_sha256。
代码要求全套通过，并额外执行原失败测试恰好一次，保存日志与摘要。

4060对CE和对应候选分别测试microbatch `{4,8,16,32}` × workers `{2,4}`，
每项仍有效batch128、2预热＋10测量；按预计整轮开销和15%余量选共同通过配置。
差距不足5%时选较少worker、再较小microbatch；最后两方法各做三次独立dev窗口。
工程配置失败留证，继续剩余预先声明的格点，不重试该格点；MethodError立即停止整个候选准入。
所有准入都不下载、不改标签、不读取confirm/test。

## 对照和交接证据

新入口的`comparison_digest`绑定共同资产、HEAD3、适配方式、训练日程和工程配置。
CE/FINE/SNSCL的`summary.json`记录方法、初始化/源码身份、best/last、暂停和回放结果。
逐轮`run/epoch-XX.json`记录macro/micro、样本访问量、更新数、轮时/显存、筛选报告、
SNSCL队列和软标签统计；逐类逐图dev结果及选择ID保留在服务器私有run目录。

`compare --control <CE目录> --candidate <候选目录>`核验双方均完成第10轮、
检查点摘要及学生dev回放，并要求相同源码、主机、共同配方/资产身份后计算macro增量。
独立子集准入可比较；不同源码或旧10轮CE预算不能冒充匹配对照。
已经用`bf9ad21`运行的TURN保持原样，其结果可供工程观察；它与新提交CE的差异必须保留，
本工具不会跨源码自动给出严格配对结论。需要补跑时另记录目的和预算，不自动重复跑TURN。

只交付本地固定提交和SHA256包，由操作agent上传；不修改旧checkout或运行中源码。
组员以后恢复工作且确需领取任务时，另按规则先发布相关代码和中文说明到GitHub。

## 本机验证

2026-09-21：新增入口专项11项通过，全套182项通过（29.733秒），零失败/错误/跳过。
compileall、pip check、git diff --check通过。测试覆盖纯元数据生成、四方法两次合成更新、
准入身份/策略/显存余量拒绝、并发失败不释放训练、并发回执绑定、4060固定16格点＋
两方法最终窗口、B04历史身份边界，以及CE/FINE/SNSCL生成图片的有限十轮流程与学生回放。
SNSCL合成流程检查第4轮未评分、第5轮评分、第6轮有效队列，并拒绝第10轮后恢复。
该流程的可靠性输入为测试注入，不是算法分数证据；GMM本身沿用独立真实数值专项。
日志：`/tmp/aic-preliminary-matrix-focused-v3.log`、`/tmp/aic-preliminary-matrix-suite.log`。
新版本的真实CUDA、工程网格和训练均待操作agent在对应机器执行，不宣称已完成准入。

## UUID 准入修复（2026-09-21）

8bb7fcc回传的CE profile通过，但外层准入失败：CUDA命令行传入`GPU-<完整UUID>`，
PyTorch设备属性的`str(uuid.UUID)`返回不带前缀的UUID，直接比较产生假不匹配。
这只证明profile和软件检查完成；当时正式epoch尚未启动，不能说正式训练已通过。

修复在独立的`gpu_identity.py`中严格解析完整128位UUID，只在单个身份比较处规范化。
CUDA分配仍必须带`GPU-`前缀，设备序号、缩写、缺失值、MIG名称和重复物理卡均拒绝。
原runtime字典、来源/资产摘要、回执内容及历史checkpoint身份不改写、不放宽。
初赛并发、复赛并发及准入汇总、4060当前/历史失败身份均使用相同边界规则。
复赛还逐方法核对分配卡，防止四卡集合相同但方法互换。

先将夹具改成真实PyTorch格式后，旧实现产生两个预期错误（T4并发及4060历史核对）；
随后修复。新增5项专项检查包含`uuid.UUID`对象、错卡/缺失/序号拒绝、大小写重复卡、
完整准入回执到run入口、来源/资产/配置篡改拒绝和复赛方法错卡拒绝。
原4060完整准入测试也改用真实格式，保留全部原资产/来源校验。
全套189项通过，零失败/错误/跳过；compileall、pip check、git diff --check通过。
测试日志为`/tmp/aic-uuid-before-fix.log`、`/tmp/aic-uuid-focused-v2.log`、
`/tmp/aic-uuid-full-suite.log`。服务器必须使用新提交重新准入，不能复用8bb7fcc回执。
