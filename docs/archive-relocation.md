# Approved GPU enrollment and immutable archive relocation

Implemented 2026-09-17. This is a local code/fixture validation result, not
server CUDA acceptance or permission to train on the development machine.

## Enroll the separate training server

`aic-doctor --bind-training` accepts CUDA-visible RTX 4060 and Tesla T4 family
names (including `NVIDIA T4`). It checks the first visible GPU, so use the
server's approved device allocation. The known development fingerprint remains
forbidden both at enrollment and at formal runtime validation. A binding copied
from another host is rejected; smoke limits are unchanged. Existing valid 4060
bindings remain compatible. The informational policy name is now
`bound-approved-experiment`.

Only on the approved server, after dependency and startup checks:

```bash
.conda/aic-robust-clip/bin/aic-doctor --bind-training machine-training.json
```

This refuses to overwrite an existing binding. Do not hand-edit or delete a
binding to bypass a failure: inspect its origin and retain the evidence.

## Mapping contract

Set `AIC_ARCHIVE_LOCATIONS` to a local JSON file. The file has exactly these
fields; `reason` must be a nonempty string, paths must be absolute, and digests
must be 64 lowercase hex characters. Duplicate JSON keys are rejected.

```json
{
  "schema_version": 1,
  "reason": "Move the frozen preliminary archive to the approved T4 server",
  "archives": {
    "/home/li177/2026-噪声鲁棒分析数据集/train.zip": {
      "path": "/home/ucu_u03/aic-robust-clip/data/train.zip",
      "sha256": "c0a3cb643691930581eea00c5c6003d999be85cbaad04384035f07973b559fac"
    }
  }
}
```

The example is the recorded preliminary archive identity. Verify the transferred
file against it; do not replace this digest with the hash of a different file.
Keep this machine-specific JSON under ignored `outputs/`, and include it and its
file hash in the private run delivery. Do not commit local paths or inventories.

On the server, after creating that JSON with the reviewed values:

```bash
export AIC_ARCHIVE_LOCATIONS="$PWD/outputs/relocation/archive-locations.json"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.conda/aic-robust-clip/bin/python - <<'PY'
from pathlib import Path
from aic_robust_clip.data.relocation import resolve_archive
print(resolve_archive(
    Path('/home/li177/2026-噪声鲁棒分析数据集/train.zip'),
    archive_identity='c0a3cb643691930581eea00c5c6003d999be85cbaad04384035f07973b559fac',
))
PY
```

Run this full-archive verification only on the training server, not as part of
local fixture tests. The first mapped read in EACH process hashes the entire
archive (about 20 GB for this data), so initial silence can be expected. A
standalone verification process does not share its cache with training jobs.
Export the variable in every actual job shell/scheduler wrapper, not only the
interactive SSH session. Missing packages/data remain owner-operated downloads.

## Failure and identity behavior

- Without the variable, or for an unlisted archive path, reading is unchanged.
- An enabled malformed/missing mapping, missing/nonregular target or wrong
  hash fails closed. There is no fallback even if the old path still exists.
- The mapping digest MUST match `SampleRecord.archive_identity`; then actual
  target bytes MUST match that digest. Changing both mapping path and hash to
  another archive does not authorize changing the audited input.
- Both `ManifestDataset._bytes_for` and `_read_record_bytes` use the resolver.
  Cache generation, online training, dev scoring and inference therefore use
  the same relocation checks. This does not broaden their stage/role access.
  Auditing operates on the path explicitly supplied to the audit command and
  is deliberately not redirected.
- Records are never rewritten. `to_dict()`, manifest/split/class-map digests,
  cache record order/identity and HEAD3 identity remain unchanged.
- Successful hashes are cached per process, path, digest and file stat identity
  (device/inode/size/mtime/ctime). Changes/removal invalidate the verification;
  workers with a new process ID rehash. Mapping-file changes also invalidate
  its parsed cache. There is no persistent trusted sidecar marker.
- Keep mapping files and archives immutable during runs. These checks are not
  filesystem locks against concurrent writers. Relocation of extracted image
  directories is not supported; mapped targets must be regular archive files.

## Delivery and validation scope

Changed runtime files: `environment.py`, `pipeline_cli.py`, `data/dataset.py`,
plus new `data/relocation.py`. New `tests/test_relocation.py` adds 15 tests:
host/GPU enrollment, copied-binding/development/smoke protections, both read
paths, wrong/missing/malformed inputs, audited-identity binding, cache invalidation
and physically moved synthetic archives with identical cache and HEAD-SMOKE
hashes. Each synthetic head fit stops at two updates; no real-data fit runs here.

The uncommitted earlier paired-method regression makes this development tree
46 tests before the migration patch, so the current local suite is 61. Applying
only this migration patch to the original 45-test server tree gives 60 tests
(workflow remains 9 there, versus 10 in the local tree). Do not mistake this
documented baseline difference for a skipped test. All applicable tests must
actually pass with zero skips.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -p test_relocation.py -v
CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -p test_workflow.py -v
CUDA_VISIBLE_DEVICES='' .conda/aic-robust-clip/bin/python -m unittest discover -s tests -v
.conda/aic-robust-clip/bin/python -m compileall -q src tests
.conda/aic-robust-clip/bin/python -m pip check
git diff --check
```

The server agent still creates new B04/B03 T4 configurations and verifies its
actual CUDA/driver environment. No formal config or persisted artifact is
modified by this patch. The original B03 is complete and immutable; rebuild
HEAD3 and a new paired comparison as already authorized in `t4-ssh-handoff.md`.
