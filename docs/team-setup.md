# Team checkout and startup acceptance

The shared repository is public:
`https://github.com/jixiuweilan/aic-robust-clip`.
Anyone can clone it without an invitation. Never share the owner's token, SSH
key, or authenticated CLI configuration.

## Get the code

The teammate runs these commands in WSL; HTTPS Git cloning needs no GitHub
authentication for this public repository:

```bash
git clone https://github.com/jixiuweilan/aic-robust-clip.git
cd aic-robust-clip
git switch -c work/your-task
```

Public visibility grants read access, not push access. Without collaborator
write permission, use a fork, task branch and pull request for changes.
Commit source, configuration,
tests and documentation only. Do not force-push shared history. Read `AGENTS.md`
before working; cloning the repository does not authorize formal experiments.

## Recreate the environment

Use Miniconda inside WSL with the project-local prefix below. Do not copy the
development machine's `.conda/`, use `base`, or install the CPU wheel on the
CUDA acceptance machine. Downloads are performed by the teammate, not agents.

```bash
# Fresh environment only; skip creation if it already exists.
conda create --prefix "$PWD/.conda/aic-robust-clip" python=3.11 pip
conda activate "$PWD/.conda/aic-robust-clip"
python -m pip install -r requirements/gpu-wsl.in
python -m pip install --no-index --no-deps --no-build-isolation -e .
python -m pip check
```

For CPU-only development, use `requirements/cpu-test.in` instead of
`requirements/gpu-wsl.in`. GPU pins are candidate bootstrap versions, not proof
of target-driver compatibility. See [the handoff](handoff.md) for WSL/driver and
offline installation guidance.

## Assets are separate from Git

Copy the verified `checkpoints/openai-clip-vit-b32/` directory separately using
an approved team transfer method. Include all four files:

- `config.json`
- `preprocessor_config.json`
- `pytorch_model.bin`
- `official-weight-manifest.json`

Do not upload these files, competition data, generated predictions, environment
directories, or machine bindings to GitHub. The startup check below needs only
these weights, not competition archives, split manifests or feature caches.
The loader verifies the pinned revision and file hashes locally before loading.

## First CUDA acceptance

In the new environment, confirm the intended GPU is visible:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

If unavailable, stop and report the error; do not substitute CPU for CUDA
acceptance. Under WSL, `nvidia-smi` may be at `/usr/lib/wsl/lib/nvidia-smi`.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v

mkdir -p outputs/cuda-acceptance
AIC_CHECK_DIR="$(mktemp -d "$PWD/outputs/cuda-acceptance/run-XXXXXX")"
aic-doctor --output "$AIC_CHECK_DIR/environment.json" \
  --record-lock "$AIC_CHECK_DIR/installed-gpu-lock.txt"
python -c 'from aic_robust_clip.runtime import current_code_revision; print(current_code_revision())'

aic-check-model --weights checkpoints/openai-clip-vit-b32 \
  --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --recipe B03 --device cuda --output "$AIC_CHECK_DIR/B03.json"
```

Run each step only after the preceding check succeeds. With the migration
regressions, the current local suite has 61 tests, no skips; see the
[documented server baseline difference](archive-relocation.md). The B03 check
uses generated images, batch 1 and at most two optimizer
updates. The report must show `device=cuda`, two updates, `optimizer_updated`
and `stopped_by_limit` true, positive peak GPU allocation, zero competition-image
reads and `formal_training=not_run`. It does not write a trained checkpoint.

Return the environment report, installed dependency inventory, source identity,
B03 report and test summary through the team's approved channel. Do not commit
generated outputs. If OOM or another failure occurs, stop and retain its full
error; do not retry as formal training or silently change precision/settings.

This startup check does not require `aic-doctor --bind-training`. It does not
run full audits, caches, HEAD3 or `aic-train`. Target CUDA acceptance is separate
from [the already-recorded local CPU evidence](validation-20260916.md), and a
passing two-step check is not evidence of convergence or a competition score.

After the first B01 delivery, the next cohesive assignment is the
[B04/B03 matched comparison](next-online-pair.md). It preserves the original
data split and requires a shared HEAD3; do not rerun B01 to prepare it.
