# 复赛单张 4090：首批十轮

首批最多运行 `C4090-LORA-CE`、`C4090-FULL-CE`、`C4090-FULL-TURN`。
仅当全视觉 TURN 未准入、LoRA TURN 已准入时，用 `C4090-LORA-TURN` 代替第三支。
所有 run 从同一复赛 HEAD20-GCE 开始，30 轮日程，第 10 轮验证和保存后暂停。
FINE、SNSCL 和旧 T4/4060 任务不在本批执行。代码、数据、权重和日志各自放在
新目录，不更新运行中的 checkout。

## 1. 部署与环境

将固定提交的离线源码包传到服务器，核对 SHA256 后放到新 checkout；
将已有 `train.zip` 放到私有 `second_round` 目录，核对摘要
`11b70e8b86e7b093cfeb117f5a8422b38c0f7011e41344cc14ae80bc29f3d319`。
官方 OpenAI CLIP ViT-B/32 快照也经私有渠道传入，其目录须含
`official-weight-manifest.json`。不读取或传输 `test.zip`。

在 checkout 中用项目本地 `.conda/aic-robust-clip`。若包未缓存，由用户执行下载与安装；
目标为 Python 3.11、PyTorch 2.7.0+cu126、`requirements/common.in` 和项目本身，
不能安装到 Conda base。服务器预装的 PyTorch 2.5.1 不满足项目依赖。
确认 `torch.cuda.is_available()`、`pip check` 后登记机器：

```bash
.conda/aic-robust-clip/bin/aic-doctor --bind-training machine-training.json
nvidia-smi --query-gpu=uuid,name,memory.total --format=csv
```

机器文件和 GPU UUID 只给该 checkout 使用。任务执行前固定
`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`，并设置
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`；入口不会下载依赖或模型。

## 2. 公共资产

在私有目录填写 `round2-delivery-v1` 的 `public-assets` 任务：

```json
{
  "schema": "round2-delivery-v1",
  "stage": "second_round",
  "action": "public-assets",
  "archive": "/私有目录/second_round/train.zip",
  "expected_sha256": "11b70e8b86e7b093cfeb117f5a8422b38c0f7011e41344cc14ae80bc29f3d319",
  "member_prefix": "train",
  "source_url": "https://www.aicomp.cn/tracks/tracks-1/3714.html",
  "retrieved_at": "2026-09-21",
  "organizer_version": "2026-09-21-received-second-round-train",
  "weights": "/私有目录/openai-clip-vit-b32",
  "weight_revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
  "machine": "/新checkout/machine-training.json",
  "gpu_uuid": "GPU-填写完整UUID",
  "output": "/私有目录/second_round/public-assets-v1"
}
```

执行 `bash scripts/round2-job.sh validate --job /私有目录/public-job.json`，再执行
`launch --job`，用 `status --output` 查看结果。只在 `status=succeeded`、
`exit_code=0` 且 `public-assets.json` 存在时使用其 `assets/assets.json` 和 HEAD20。
`launch` 的 PID 不是完成证明。审计失败时保留日志和原 ZIP，在新目录排查。

## 3. 单卡准入和训练

分别准备 `cloud4090-lora`、`cloud4090-full` 的 `group` 任务，填写同一 GPU UUID、
机器文件、公共 `assets.json`、执行人和独立输出目录。两组均先填
`"methods": ["ce", "turn"]`，完成全 train 初始评分、两次有界更新、固定
`{4,8,16,32} × {2,4}` 测速、共同配置选择、三次验证窗口和准入。
任务结构如下；LoRA 组将 `group` 改为 `cloud4090-lora` 并使用另一输出目录：

```json
{
  "schema": "round2-delivery-v1", "stage": "second_round", "action": "group",
  "group": "cloud4090-full", "methods": ["ce", "turn"], "owner": "服务器执行人",
  "gpu_uuids": ["GPU-填写完整UUID"], "machine": "/新checkout/machine-training.json",
  "assets": "/私有目录/second_round/public-assets-v1/assets/assets.json",
  "output": "/私有目录/second_round/full-pair-admission-v1"
}
```

每组只使用一张卡，不做 T4 并发短窗，也不要求 4060 B04 历史记录。
如果 TURN 的数值/筛选准入失败，保留该任务证据，另建新任务目录并把
`methods` 改为 `["ce"]`，只为 CE 准入；不能将此 CE 回执配给 TURN。

每组 `group` 成功后，其 `configs/` 中只有回执覆盖的方法为 `ready`。
按本页首段顺序，为每支 ready 的 run 填写 `train` 任务（`run_id`、对应配置绝对路径、
同一机器文件和 UUID、新输出目录），依次 `validate`、`launch`、`status`。
例如全视觉 CE：

```json
{
  "schema": "round2-delivery-v1", "stage": "second_round", "action": "train",
  "run_id": "C4090-FULL-CE",
  "config": "/私有目录/second_round/full-pair-admission-v1/configs/C4090-FULL-CE.json",
  "machine": "/新checkout/machine-training.json", "gpu_uuid": "GPU-填写完整UUID",
  "output": "/私有目录/second_round/full-ce-task-v1"
}
```

入口核验第 1 轮完整检查点后恢复至第 10 轮，随后执行学生 dev 回放。
只有 `status=succeeded` 且逐轮产物完整才算完成；候选还要求同组 CE 已完成十轮。
正式结果回传 best/last、逐类/逐图 dev、筛选覆盖、更新数与样本量、轮时、显存、
检查点 SHA256、日志及失败记录。十轮后统一评审，不自动续训或提交排行榜。
