#!/usr/bin/env bash
# USER-OPERATED offline installer. Downloads are a separate user action.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo 'Usage: bash scripts/setup-training-env.sh /path/to/miniconda3/bin/conda /absolute/path/to/wheelhouse' >&2
  exit 2
fi
training_conda="$1"
training_wheels="$2"
test -d "$training_wheels"
test ! -e .conda/aic-robust-clip
"$training_conda" create --offline --prefix "$PWD/.conda/aic-robust-clip" python=3.11 pip --yes
.conda/aic-robust-clip/bin/python -m pip install --no-index --find-links "$training_wheels" -r requirements/gpu-wsl.in
.conda/aic-robust-clip/bin/python -m pip install --no-index --no-deps --no-build-isolation -e .
.conda/aic-robust-clip/bin/aic-doctor --output outputs/environment/preflight.json --record-lock outputs/environment/installed-gpu-lock.txt
echo 'Installed lock is an inventory, not evidence of real-model startup or formal training.'
echo 'Review the report, then explicitly bind the separate machine using aic-doctor --bind-training.'
