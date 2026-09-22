#!/usr/bin/env bash
# 仅复赛；不安装依赖、不下载、不绕过机器准入。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec .conda/aic-robust-clip/bin/python -m aic_robust_clip.round2.delivery "$@"
