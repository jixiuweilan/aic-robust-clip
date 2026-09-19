# N01–N05 Engineering Handoff

Prepared 2026-09-16. **Code implementation is available; acceptance is not yet
complete.** [Current CPU validation](validation-20260916.md) passes all 45 tests
without skips; official-weight B03 CPU startup also passed. The separate 4060's
CUDA startup checks remain pending. No formal
training, full-data cache, scored experiment, or official submission had run
at that historical checkpoint.

Update 2026-09-17: the received B01 package supports completion of CACHE20 and
includes a B03 CUDA startup report. Supplementary prediction/cache-index checks
pass. Old test logs are unavailable; the teammate reports a successful rerun,
and the user elected to proceed. This is not local CUDA reproduction.
The current next assignment and evidence boundary are in
[B04/B03 matched comparison](next-online-pair.md). The commands below remain
setup reference, not instructions to recreate existing data/caches or rerun B01.

## Machine and download boundaries

- Development machine: project-local Miniconda environment
  `.conda/aic-robust-clip`, Python 3.11.16. Created offline from existing cached
  Conda packages. Do not install into `base` or another project environment.
- Separate experiment machine: RTX 4060, Windows + WSL2, Ubuntu 24.04, Miniconda
  Python 3.11. Actual memory, driver and CUDA support must be measured; no
  capacity or throughput claim is inferred from the GPU name.
- **All downloads are performed by the user**, including package wheels,
  Miniconda/WSL installation, drivers and pretrained weights. Agents use local
  inputs and offline commands only. Generated tests never access the network.
- Windows supplies the GPU driver. Do not install a Linux NVIDIA display driver
  inside WSL. Keep project/data on WSL's Linux filesystem rather than `/mnt/c`.
  Sources: [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html),
  [Microsoft filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems).

The tracked `requirements/*.in` files are **candidate bootstrap pins**, not
verified complete locks. CPU and GPU PyTorch wheels are intentionally separate.
Record the installed transitive lock after user setup using `aic-doctor`; a
dependency inventory alone does not certify CUDA or real-model behavior.
The user-installed CPU environment has now passed the local suite; its observed
versions are recorded in `requirements/cpu-tested-20260916.txt`.

## 1. User setup and downloads

On this development machine the Conda environment and CPU dependencies are
already installed. The following user-operated command is retained for setup
reference; there is no need to repeat it for the current verified environment:

```bash
cd /home/t/projects/new
.conda/aic-robust-clip/bin/python -m pip install -r requirements/cpu-test.in
```

On the separate experiment machine, the user installs the Windows NVIDIA driver,
WSL2 Ubuntu and Miniconda. `scripts/prepare-wsl.ps1` checks WSL and prints setup
steps without doing downloads. Within the repository in WSL:

```bash
# USER-operated downloads; agents must not execute these commands.
conda create --prefix "$PWD/.conda/aic-robust-clip" python=3.11 pip
.conda/aic-robust-clip/bin/python -m pip install -r requirements/gpu-wsl.in
```

For an offline install, the user can download the GPU wheelhouse with a Python
3.11 Linux environment first (`python -m pip download -r requirements/gpu-wsl.in
--dest /path/to/wheelhouse`). With Python/pip already cached in Miniconda,
`scripts/setup-training-env.sh` creates the project environment using Conda
`--offline` and installs that wheelhouse with pip `--no-index`. It never trains.

In either environment, register the local code without downloading build tools:

```bash
conda activate "$PWD/.conda/aic-robust-clip"
python -m pip install --no-index --no-deps --no-build-isolation -e .
aic-doctor --output outputs/environment/preflight.json \
  --record-lock outputs/environment/installed-lock.txt
```

Only on the separate approved CUDA-visible RTX 4060 or Tesla T4, explicitly
enroll the machine. See [T4 enrollment and archive relocation](archive-relocation.md)
for migration without changing the frozen input identities:

```bash
aic-doctor --bind-training machine-training.json
```

The binding is ignored by Git and matched to the local machine fingerprint;
the known development host is rejected even if the JSON requests training.
There is no `--allow-formal` experiment flag or CPU fallback to bypass this rule.

## 2. User provisions original weights; startup remains bounded

The pinned OpenAI HF revision is
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Repository metadata and small config
files were initially checked on 2026-09-16; the user subsequently downloaded
the large model file, and its actual hash and bounded B03 CPU startup now pass.
File digests are recorded in `models/official_weights.py` and the
[validation receipt](validation-20260916.md).
Source: [official pinned snapshot](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268).

On a fresh machine, the **user** executes the explicit weight download below.
On this development machine, the verified directory already exists; do not
repeat provisioning. Copy the entire directory, including its manifest, when
transferring it to the experiment machine.

```bash
HF_HUB_DISABLE_XET=1 aic-provision-weights --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --output checkpoints/openai-clip-vit-b32
```

Ordinary loading is offline and requires this directory's manifest, the pinned
revision, actual file hashes and ViT-B/32 configuration. It rejects fabricated
digests, branches/placeholders and modified files. No arbitrary external weights
are accepted. Failed partial provisioning is retained for diagnosis; use a new
output directory for an explicit retry rather than deleting anything silently.
The local interrupted attempt is retained at
`checkpoints/openai-clip-vit-b32-interrupted-20260916`; it is not the active
weight directory. HTTP recovery was validated, but the provisioning CLI still
requires a fresh output directory and has no explicit resume option.

On the experiment machine, test one path at a time:

```bash
aic-check-model --weights checkpoints/openai-clip-vit-b32 \
  --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --recipe B03 --device cuda --output outputs/startup-checks/B03-official.json
```

Repeat independently for B01/B04/R01/F100/F010/F001 only as correctness checks.
Each fresh invocation generates four images, makes at most two optimizer
updates and bounds scoring/reference passes. It does not read competition
archives or initialize a formal head. OOM/nonfinite loss stops the check; never
automatically switch precision, retry with full epochs, or treat loss as a score.
Record actual peak allocated/reserved GPU memory and weight identity.

## 3. Explicit prerequisites on the experiment machine

The following full-data operations are **future user-scheduled execution**, not
part of local code validation. Confirm the actual stage and provenance before
using the preliminary templates. Earlier stages' learned artifacts are never
reused in another stage. Test inventory remains separate and cannot filter train.

```bash
aic-audit-archive data/train.zip --stage preliminary --role train \
  --output outputs/preliminary/train-audit.json
aic-audit-archive data/test.zip --stage preliminary --role test --no-decode \
  --output outputs/preliminary/test-audit.json
aic-build-class-map outputs/preliminary/train-audit.json \
  --output outputs/preliminary/class-map.json
aic-make-split outputs/preliminary/train-audit.json \
  --output outputs/preliminary/split-v2.json \
  --report outputs/preliminary/split-report.json
aic-doctor --config configs/formal/B03.json
aic-cache-features --config configs/formal/B03.json --partition train
aic-cache-features --config configs/formal/B03.json --partition dev
aic-init-head --config configs/formal/B03.json
```

Inspect all audit failures; no silent drops, relabeling or automatic split repair.
Use strict-decoded manifests. Truncated-recovery reports require an explicitly
reviewed policy change before formal split acceptance. v1 splits/caches are not
compatible. Exact grouping is implemented; this is not proof of near-duplicate
or scene independence. Do not claim such screening has been completed.

Caches contain normalized float32 fixed-view features in hashed `.npy` shards;
readers use memory maps, not a GPU/full-RAM concatenation. Cache keys include
stage, mode, split, class map, weights, preprocessing and partition, and exact
sample coverage is checked. Confirm/test caches are forbidden. Smoke caches
and HEAD-SMOKE artifacts cannot satisfy formal prerequisites.

HEAD3 fits a fresh head for exactly three train-only epochs, with no dev selection.
Online methods share its last head for the same seed/split. Their main optimizer
and scheduler start fresh; the head is not selected from the B01 twenty-epoch run.

## 4. Train/resume/evaluate/predict command contract

These formal examples are prepared for a later explicit experiment decision;
**do not launch them merely because engineering handoff is complete**.

```bash
aic-train --config configs/formal/B03.json
aic-train --config configs/formal/B03.json \
  --resume outputs/preliminary/runs/B03-seed17/last.pt
aic-evaluate --config configs/formal/B03.json \
  --checkpoint outputs/preliminary/runs/B03-seed17/best.pt --partition dev \
  --output outputs/preliminary/B03-dev.json
```

The seven complete templates live under `configs/formal/`; input paths are
resolved relative to the config file. Defaults: seed 17, float32, microbatch 1,
effective batch 128, zero-worker stateful loading; CACHE20 for B01 and ONLINE10
for the other six. Head LR is 1e-3, visual LoRA LR 1e-4, weight decay 1e-4 excluding
bias/norm, AdamW betas .9/.999 and epsilon 1e-8. One LR warm-up epoch precedes
cosine decay. W's two-epoch reliability warm-up is distinct. For incomplete
effective batches, CE/GCE and P use actual sample count, W uses actual weight sum.

Checkpoint ranking is macro recall, then micro Top-1, then earliest epoch.
All held-out metrics are `noisy_proxy`. Augmentation is addressed by sample ID,
seed and epoch, independent of model/global RNG consumption. Ordinary dev reads
cannot touch confirm. New runs refuse an existing output directory; resume
requires the same resolved config and a checkpoint in that run directory.
Smoke update/sample caps are cumulative across resume. Legacy scalar-selection
or missing-lineage checkpoints are rejected, not heuristically migrated.

After the recipe decision is explicitly frozen:

```bash
aic-lock-selection --config configs/formal/B03.json \
  --checkpoint outputs/preliminary/runs/B03-seed17/best.pt \
  --output outputs/preliminary/selection.json
aic-evaluate --config configs/formal/B03.json \
  --checkpoint outputs/preliminary/runs/B03-seed17/best.pt --partition confirm \
  --selection-record outputs/preliminary/selection.json \
  --output outputs/preliminary/B03-confirm.json
aic-predict --config configs/formal/B03.json \
  --checkpoint outputs/preliminary/runs/B03-seed17/best.pt \
  --test-manifest outputs/preliminary/test-audit.json \
  --selection-record outputs/preliminary/selection.json \
  --output outputs/preliminary/prediction-B03
```

Prediction rebuilds one model and its exact class map/preprocessing. B01 attaches
the original frozen encoder to the saved classifier; no second model is fitted.
The test manifest supplies exact filename coverage. CSV and ZIP are validated
before/after packaging. Local prediction checks require an explicitly synthetic
fixture manifest and at most eight samples, not official test images. This
command does not implement Q03 final refitting or authorize official submission.

Each run writes `resolved.json`, `epochs.json`, per-epoch dev predictions,
`last.pt`, dev-selected `best.pt`, and `result.json`. Checkpoints persist the
resolved config, code identity, dependencies, optimizer/scheduler/RNG/cursor,
prior and reliability history. The result records hashes and resource evidence.
Failures write `failure.json` and stop without automatic retries.

## 5. Historical acceptance ledger (2026-09-16)

For the 2026-09-17 received B01 evidence and 46-test local preparation check,
see [the next task](next-online-pair.md); the rows below retain the earlier state.

| Item | Evidence / status |
| --- | --- |
| Existing 36 regressions before current changes | Previously passed in isolated CPU environment; historical evidence only |
| Current project-Conda suite (2026-09-16) | 45/45 passed, zero skipped; focused workflow 9/9; syntax, shell and whitespace checks passed |
| New tensor, NumPy RNG, Hugging Face structural and full CLI fixture tests | Passed on CPU with generated data and explicit small-model mocks; see validation record |
| Synthetic startup / installed CPU versions | Stopped at 2 samples / 2 optimizer updates; pip check passed; CPU version snapshot recorded |
| Published official revision/hash metadata | Retrieved and pinned; original weights not downloaded by agent |
| Real ViT-B/32 weight loading and bounded startup | User-provided files verified; B03 CPU passed 2 samples / 2 updates; target GPU and other real-weight recipe checks remain pending |
| Exact installed GPU dependency lock and CUDA compatibility | Pending separate 4060 setup; candidate pins are not a verified lock |
| Full audit/cache/head fitting and seven formal experiments | Not run; separate execution stage |

After dependency setup, run:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
PYTHONPATH=src python -m compileall -q src tests
git diff --check
```

Tests use temporary generated PNGs and explicit small-model mocks. The tiny
Hugging Face structural test is not a real pretrained-backbone verification or
a research result. Do not mark N01–N05 fully accepted until official-weight and
target-machine startup have actual reports. Do not commit data, environments,
weight files, checkpoints, machine bindings or generated predictions.
