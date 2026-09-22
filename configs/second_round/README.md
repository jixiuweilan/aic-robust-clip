# 复赛配方

`plan.json` 是八个 run 的固定配方规划表，当前为 `blocked_on_round2_assets`。
它不是可运行的资产绑定配置。

使用 `python -m aic_robust_clip.round2 prepare --output outputs/second_round/blocked-v1`
生成八份阻塞配置。真实复赛资产和本机新准入齐备后，在新目录重新运行 prepare，
显式传入 `--assets` 和 `--receipt`；只有该准入组的配置会变为 ready。

开发说明：[复赛开发与验证](../../docs/round2/development.md)。当前组员休息，本表不下发任务。

2026-09-22 新任务模板位于 `jobs/`。模板中的路径/UUID必须由执行人填入私有副本，
使用 `bash scripts/round2-job.sh validate --job ...` 校验；模板本身不可直接运行。
当前组员说明见 [team-execution.md](../../docs/round2/team-execution.md)。
`group` 生成八个配置，但只放行本组；其他组仍保持 blocked。
