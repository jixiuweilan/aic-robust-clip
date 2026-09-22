# Operational Scripts

The package exposes explicit preparation and packaging commands. Installed
entry points are `aic-audit-archive`, `aic-build-class-map`, `aic-make-split`,
`aic-package-submission`, and the bounded `aic-startup-check`;
`aic-validate-submission` remains available.

The new workflow also exposes `aic-doctor`, `aic-provision-weights`,
`aic-check-model`, `aic-cache-features`, `aic-init-head`, `aic-train`,
`aic-evaluate`, `aic-lock-selection`, and `aic-predict`. See
[the current handoff](../docs/handoff.md) for exact arguments and acceptance
status. Weight provisioning is user-operated; agents must not invoke downloads.

Use Miniconda with `.conda/aic-robust-clip`, never `base` or another project's
environment. `prepare-wsl.ps1` reports WSL status without installing anything.
`setup-training-env.sh` is a user-operated offline installer requiring cached
Conda packages and a user-supplied wheelhouse; it does not download or train.

Full-data auditing, feature caching, and formal training commands belong on
the separate training machine. Keep the end-to-end workflow reproducible and
ensure no script can accidentally include test data in a training data loader.

复赛任务使用 `bash scripts/round2-job.sh {validate,launch,status} ...`。
它调用独立 `round2.delivery`，不接受初赛任务格式，不安装或下载任何资产。
详见 [组员复赛说明](../docs/round2/team-execution.md)。
