# 复赛两台 4060：先准入，再 CE → 候选

组员 A 固定操作 4060-A：LoRA CE 十轮完成后运行 LoRA TURN。
组员 B 固定操作 4060-B：LoRA CE 十轮完成后运行 LoRA FINE。
两台机器独立准入；每支都从公共 HEAD20-GCE 开始，候选不加载 CE 权重。
保持 seed17、有效 batch128、30轮完整调度；第10轮评分/验证/保存后停止。
不读取 confirm/test，不最终重训、不提交排行榜、不自动继续20/30轮。

## 1. 获取代码与先行检查

源码和本说明先发布 GitHub，通知中固定提交号。使用新 checkout，不能更新运行中的目录。
仅在干净的新工作目录使用 `git fetch origin` 后 `git checkout --detach <通知中的完整提交>`；
记录 `git rev-parse HEAD`。不得使用初赛启动包、HEAD3、旧划分/缓存或旧准入回执。
正式实验仅在单独训练机进行。环境固定为项目 `.conda/aic-robust-clip`，
参照 [环境说明](../team-setup.md)；缺包由用户自行下载/安装，脚本不会下载。

在新 checkout 中：

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nvidia-smi --query-gpu=uuid,name,memory.total --format=csv
.conda/aic-robust-clip/bin/aic-doctor --help
```

机器绑定使用既有命令 `.conda/aic-robust-clip/bin/aic-doctor --bind-training machine-training.json`。
若该文件已存在则先检查内容和机器指纹，不覆盖。每个任务保持相同线程环境。
在私有 `second_round` 目录创建 `machine-history.json`：

```json
{
  "host": "填写 machine-training.json 的 fingerprint",
  "gpu_uuid": "GPU-填写完整UUID",
  "original_b04_failure": false,
  "reviewer": "填写核对人",
  "basis": "填写原B03/B04回传和本机身份核对依据，不能猜测"
}
```

若确为原 B04 失败机器，将布尔值改为 true，并准备 `previous-failure.json`：
`host`、`gpu_uuid`、`test="test_verification_cached_only_for_unchanged_file_and_process"`、
`original_evidence_sha256`。保留旧失败原件；未知身份时不准入。

复制 [A模板](../../configs/second_round/jobs/4060-a-group.json) 或
[B模板](../../configs/second_round/jobs/4060-b-group.json) 到私有目录，填实际绝对路径和完整GPU UUID。
原失败机器加上 `previous_failure` 字段指向上述文件。
公共头未完成时，将副本的 `action` 改为 `checks`，使用独立输出目录，先完成全套软件及
各方法 FP32/FP16 各两次合成 CUDA 更新，不要求资产。之后 `group` 会重新检查当前环境。

## 2. 收到公共资产后进行真实准入

私下接收服务器生成的 `assets/`：manifest、split、class-map、coverage、duplicates、
decode-failures、assets.json、status.json，以及 `HEAD20-GCE/head.pt` 和 head.json。
核对发送方 SHA256 清单。无需接收服务器 train-cache；不得混入初赛文件。
训练 ZIP 保存在本机独立 `second_round` 目录。
按 [路径迁移说明](../archive-relocation.md) 创建本机映射并设置：

```bash
export AIC_ARCHIVE_LOCATIONS=/实际私有目录/second_round/archive-locations.json
export AIC_ROUND2_WEIGHTS=/实际本机官方ViT-B-32快照目录
bash scripts/round2-job.sh validate --job /实际私有目录/second_round/group-job.json
bash scripts/round2-job.sh launch --job /实际私有目录/second_round/group-job.json
bash scripts/round2-job.sh status --output /实际私有目录/second_round/本组-admission-v1
```

不修改公共清单来适配路径。`launch` 创建独立会话，可关闭SSH；仅 `succeeded` 且
`exit_code=0` 表示此阶段完成。查看任务目录 `console.log`、`events.jsonl`、
`status.json` 和编号步骤日志。进程 PID、目录存在、`running` 均不表示成功；
断电/SIGKILL 后旧状态可能停在 running，保留证据，用新任务目录排查。

`group` 按顺序完成全套检查、所有方法的全train初始评分和两次真实有界更新、
固定 `{4,8,16,32} × {2,4}` 网格、共同配置选择、三次独立eval窗口、准入和配置生成。
有效 batch128、显存余量至少15%。数值异常、不收敛、正常拟合零选中停止整个组；
工程网格的单项OOM留证，由剩余共同通过项决定是否准入，不原地重试或改配方。
所有原始回执/日志需保留在原位置，正式入口会复核哈希和身份。

## 3. CE → 候选各十轮

复制本机的 [CE/A](../../configs/second_round/jobs/4060-A-CE.json)、
[TURN/A](../../configs/second_round/jobs/4060-A-TURN.json)，或
[CE/B](../../configs/second_round/jobs/4060-B-CE.json)、[FINE/B](../../configs/second_round/jobs/4060-B-FINE.json)，
填准入生成的配置路径、机器绑定和UUID。先 launch CE；CE任务 succeeded 且
run 的 result.json 为 `paused_at_epoch10` 后，再 launch 候选。候选入口再次强制检查匹配同机CE。

```bash
bash scripts/round2-job.sh launch --job /实际私有目录/second_round/CE-job.json
bash scripts/round2-job.sh status --output /实际私有目录/second_round/CE-task-v1
# 确认 CE 已完成第十轮及 dev 回放后，才执行：
bash scripts/round2-job.sh launch --job /实际私有目录/second_round/candidate-job.json
```

交付程序首先运行至首轮完整保存，核对指标、完整状态、best/last摘要和更新数，
再按原配置恢复至十轮，最后用学生导出做 dev 回放。不会以脚本启动/profile通过代替训练完成。
中断失败不得盲目重试：先定位原因，保留失败证据。代码/资产/配方不变且故障已修复时，
新任务JSON可设 `resume=true`，指向同一配置和本run的last.pt；task输出必须用新目录。
源码有任何变化须新版本及受影响准入重做；旧身份检查点不可强行恢复。

## 4. 回传与停止条件

全套测试必须零失败、零错误、零跳过，compileall/pip check/diff check通过。
错误赛段、旧HEAD/旧回执、源码/环境/GPU/资产身份不一致、worker异常、非有限数、
方法失败、检查点不完整、学生回放不一致均停止所属任务；失败目录不删除。
运行慢用于排查，不是自动淘汰标准。每组可独立推进，不等另一组解除阻塞。

私下回传固定提交、依赖、机器历史及原失败复验、完整任务日志/状态、准入原始证据、
公共资产/HEAD摘要、resolved配置、逐轮指标/逐类指标、筛选统计、样本访问/更新数、
AMP跳过、分阶段轮时/显存、best/last摘要与失败记录；附文件和压缩包SHA256。
逐图 dev 预测和 best/last 原件留在对应私有run目录，完整结果通过私有渠道传输。
不能放入GitHub；不把图片、凭据、特征缓存和checkpoint夹入源码包。
八支到十轮后统一评审候选相对同机CE增益；本轮无后续续训授权。
