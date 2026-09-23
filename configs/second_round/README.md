# 复赛配方

`plan.json` 是单张 4090 首批实验的规划表，不是可运行配置。旧 T4/4060 配置
保留作历史回归，本轮不执行。

使用 `python -m aic_robust_clip.round2 prepare --output outputs/second_round/blocked-v1`
生成阻塞配置。真实复赛资产和新机准入齐备后，在新目录重新运行 prepare，
显式传入 `--assets` 和 `--receipt`；只有该准入组的配置会变为 ready。

执行说明：[单卡 4090](../../docs/round2/cloud4090-execution.md)。当前不下发组员任务。

2026-09-22 新任务模板位于 `jobs/`。模板中的路径/UUID必须由执行人填入私有副本，
使用 `bash scripts/round2-job.sh validate --job ...` 校验；模板本身不可直接运行。
当前组员说明见 [team-execution.md](../../docs/round2/team-execution.md)。
`group` 生成配置，但只放行本组已准入的方法；其他方法保持 blocked。
