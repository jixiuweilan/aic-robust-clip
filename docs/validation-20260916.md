# Local CPU validation — 2026-09-16

Result: **45 tests passed, zero skipped**, after the user installed dependencies
in `.conda/aic-robust-clip`. Focused workflow suite: **9/9 passed**. The separate
synthetic startup command stopped at **2 samples and 2 optimizer updates**,
with `optimizer_updated=true` and `stopped_by_limit=true`. After the user supplied
official weights, a separate bounded B03 CPU check also passed (see below).
No formal training, competition-image processing or GPU testing occurred.

## Provenance

- Python 3.11.16 in the project-local Miniconda prefix, not `base`.
- Torch 2.7.0+cpu, Transformers 4.57.6, NumPy 2.2.6, Pillow 11.3.0,
  huggingface-hub 0.36.2; `pip check` passed.
- Installed versions: [CPU snapshot](../requirements/cpu-tested-20260916.txt).
  This is a version inventory, not a wheel-hash lock or GPU environment.
- Git HEAD: `cb38714c0ed53cd12ce93529fe2c7917a59b0ecf`, with uncommitted changes.
  Tests validated the working tree, not just this commit. Runtime code identity:
  `cb38714c0ed53cd12ce93529fe2c7917a59b0ecf+src.f5d060554a3da7a8cd322249417276edaddc49604976d72f9a358438368b3f10`.
- Local diagnostics: `outputs/environment/cpu-preflight-20260916.json`;
  raw pip freeze: `outputs/environment/cpu-installed-20260916.txt` (both ignored).

## Checks performed

The agent registered this checkout with `pip install --no-index --no-deps
--no-build-isolation -e .`; this built the local editable package without
downloading dependencies. Tests and startup ran with `CUDA_VISIBLE_DEVICES=''`,
`HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .conda/aic-robust-clip/bin/python -m unittest discover -s tests -v
PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .conda/aic-robust-clip/bin/python -m unittest test_workflow -v
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .conda/aic-robust-clip/bin/aic-startup-check
.conda/aic-robust-clip/bin/python -m compileall -q src tests
bash -n scripts/setup-training-env.sh
git diff --check
```

The suite covers synthetic feature caches, shared-head initialization, bounded
training and resume, locked confirmation, single-model prediction and validated
ZIP packaging. It also checks weighted accumulation, optimizer groups,
sample-addressed transforms, NumPy/Python/Torch RNG restoration, and Q/V LoRA
forward/backward on a tiny randomly initialized Hugging Face CLIP structure.
That structure is a software fixture, not the pretrained competition backbone.
Expected rejection tests print failure messages for existing run directories,
unlocked confirmation and mismatched head identities; those tests passed.

## Official-weight CPU startup — subsequent verification

The user completed the fixed-revision download using HTTP with Xet disabled.
The manifest records retrieval at `2026-09-16T13:52:32.935957+00:00` from
[the official snapshot](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268).
The agent performed only local checks; it did not initiate any download.

All three file hashes were independently checked against the pinned allowlist.
`pytorch_model.bin` is 605,247,071 bytes, SHA-256
`a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`.
The manifest's file-map digest (not the model-file hash) is
`03f536d76e72a796d211d2e316f6e426321f1f00e32dac1dcbd72dcfc0070cc5`.

The completed directory was moved without overwrite from
`checkpoints/openai-clip-vit-b32-http` to the configuration's existing default,
`checkpoints/openai-clip-vit-b32`. The interrupted directory remains intact at
`checkpoints/openai-clip-vit-b32-interrupted-20260916`. Nothing was deleted.

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
.conda/aic-robust-clip/bin/aic-check-model \
  --weights checkpoints/openai-clip-vit-b32 \
  --revision 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 \
  --recipe B03 --device cpu \
  --output outputs/startup-checks/B03-official-cpu-20260916.json
```

Observed: official model loaded offline; seed 17, batch 1, four generated image
fixtures available, **two samples consumed, two optimizer updates**, finite
losses, `optimizer_updated=true`, `stopped_by_limit=true`. The real visual Q/V
LoRA forward/backward/update path passed. There was no dev evaluation,
competition-image read or saved trained checkpoint. This is startup correctness,
not evidence of convergence, accuracy or target-GPU memory capacity. Other
recipes have fixture coverage but were not run with official weights here.

The report is `outputs/startup-checks/B03-official-cpu-20260916.json` (ignored),
SHA-256 `2eb914ab84625110ebb9d18c2e0f3df6c173484cc8860ea7ce3442ef8af746b7`.
Source identity and dependency versions are the same as recorded above. The
older environment preflight's `real_model_startup=not_run` describes its earlier
observation; this new report supplies subsequent evidence without overwriting it.

## Remaining acceptance

The CPU dependency and official-weight B03 startup blockers are resolved.
Bounded startup on the separate experiment machine, its CUDA compatibility and
installed GPU lock remain unverified. Full-data audit/cache/head initialization,
formal experiments and official submission have not run. See
[the handoff](handoff.md) for user-operated provisioning and future commands.
