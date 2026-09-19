# SSH handoff: T4 migration and completion of the B04/B03 comparison

Prepared 2026-09-17. The owner confirmed the original HEAD3 cannot be recovered
and authorized a new HEAD3 plus a new B04/B03 pair. This is an assignment for
another execution agent, not a
claim that this server has been accessed, enrolled or tested. It supersedes
the old 4060/path assumptions in `next-online-pair.md` for this migration.
Read `AGENTS.md` and that document for experiment settings and deliverables.

## Inputs the execution agent must ask the owner for

The owner explicitly wants the execution agent, not the preparation agent,
to request SSH access details. Do not ask again whether HEAD3 is recoverable:
that decision is settled; use the rebuild procedure below.

- SSH host alias or reachable address and port, using account `ucu_u03`.
  The displayed hostname `master` is not proof of a reachable SSH address.
- An approved absolute project/storage directory and GPU allocation. Confirm
  whether direct GPU use on `master` is permitted or a scheduler job is required.
- Authentication through the owner's existing SSH configuration/agent. Never
  request passwords/private-key contents in chat or put them in command lines.

Reported server: Ubuntu 24.04.2 x86_64, kernel 6.8.0-138, Xeon Silver 4310,
48 logical CPUs, about 128 GiB RAM, four Tesla T4s. GPU driver, available disk,
GPU allocation and installed Conda are not yet verified. Pending updates and
the reboot notice are administrator matters, not authorization to change them.

## One cohesive assignment

Migrate the existing project safely, implement the minimum T4/path support,
verify the environment, and complete the approved matched comparison. Do not
launch the full experiment matrix, tune hyperparameters, add DDP, or change
precision/batch settings for speed in this assignment. One agent owns shared
setup and HEAD3; do not let multiple agents concurrently edit shared files.

All sustained training/cache creation runs on the allocated server GPUs only.
The development machine remains correctness/startup-only. No agent-initiated
downloads, including pip/Conda/model downloads or repository fetches. Missing
dependencies must be listed as owner-run commands, then wait for the owner.
Use `.conda/aic-robust-clip` on the server; do not copy the CPU Conda environment,
install in base, upgrade drivers, use sudo, reboot, kill others' jobs or weaken
SSH host-key checking. Existing artifacts must not be overwritten or deleted.

## Read-only SSH preflight

The owner first verifies the server host-key fingerprint out of band and
establishes normal SSH access. Then the execution agent can use this local
command after replacing the host alias. It deliberately creates no files.

```bash
export AIC_SSH_TARGET='REPLACE_WITH_VERIFIED_SSH_ALIAS'
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 \
  "$AIC_SSH_TARGET" 'bash -s' <<'REMOTE'
set -euo pipefail
hostname
id
uname -a
sed -n '1,8p' /etc/os-release
nvidia-smi
free -h
df -h . /tmp
command -v conda || true
command -v tmux || true
command -v sbatch || true
command -v srun || true
printf 'Visible GPU allocation: %s\n' "${CUDA_VISIBLE_DEVICES-unset}"
REMOTE
```

Do not interpret absent Slurm commands or an idle `nvidia-smi` as permission to
claim all GPUs. Obtain the owner's/admin's allocation. If jobs are scheduled,
run GPU diagnostics and training inside the allocation, respecting its device
mapping. Stop on unavailable authentication rather than disabling verification.

## Code and artifact transfer

The development worktree has unpushed tests/docs; a GitHub checkout alone is
not the complete current handoff. The owner transfers an allowlisted snapshot
of `AGENTS.md`, `.gitignore`, `README.md`, `pyproject.toml`, `src/`, `tests/`,
`configs/`, `requirements/`, `scripts/`, `docs/`, and `references/`, together with
the base commit, diff and snapshot checksums. Alternatively use an owner-pushed,
explicitly identified commit. Do not blindly copy `.git`, `.conda`, credentials,
all ignored outputs, or this machine's host binding. Preserve source provenance
even if a transferred snapshot has no Git metadata; return the exact final
source snapshot/patch with the delivery.

Transfer data/weights separately into new, owner-approved directories:

- Original `data/train.zip` and complete `checkpoints/openai-clip-vit-b32/`.
  This task needs no test-image archive; do not transfer/use it unnecessarily.
- Updated B01 delivery, SHA-256
  `2c432859e36ae8e5d400ca17422545091a032535ba3b78b220ddcf3348483a2f`.
- B04/B03 partial delivery, SHA-256
  `ccaa1d6c5b6a9b87113e16a89bc8c942d842c9ecf2bc9487d8095dca0354ab92`.
- Feature shards if already available; the delivered cache indexes alone do
  not contain the actual feature arrays. Do not block on recovering the lost
  HEAD3. Generate new train features on the server when shards are unavailable.

Check hashes, safe ZIP names and target nonexistence before extraction. Extract
received deliveries into separate immutable receipt directories, not directly
over active `outputs/`. Selectively promote verified inputs. B01 supplies the
frozen train manifest, split and class map. Preserve partition assignments,
labels, duplicate groups and stage; do not call the splitter again.

## Required implementation before formal execution

Update: the two required code changes are now implemented locally. Deploy the
reviewed migration patch and follow [archive-relocation.md](archive-relocation.md)
for the exact mapping contract, tests and server commands. Do not apply the old
unverified `.new` drafts over the implementation. Server acceptance is still
required; the list below states the invariants it must preserve.

1. Replace the hardcoded RTX 4060 enrollment restriction in `environment.py`
   and related CLI wording with explicit support for the approved T4 server.
   Preserve host-bound enrollment, rejection of copied host bindings, the known
   development-host prohibition and bounded-smoke enforcement. Do not enable
   formal execution globally or forge `machine-training.json` by hand.
2. Resolve archive relocation explicitly. Old manifests refer to
   `/home/li177/2026-噪声鲁棒分析数据集/train.zip`. Do not create directories under
   another user's home or silently rewrite hashed manifest/split records. Prefer
   a narrowly scoped, recorded runtime archive-location mapping with actual
   archive SHA-256 verification, keeping content identities and sample order
   unchanged. Apply it to all relevant dataset readers. Add rejection tests
   for a wrong archive and positive tests for unchanged IDs/splits/head identity.
   If another migration changes identities, stop for review before using it.
3. Keep received B03 outputs/resolved configuration immutable. New configs and
   runs use new server-specific filenames/directories. Existing resume demands
   exact resolved configuration and HEAD3; do not relax checks to force a load.
4. Run focused and full tests, compileall, pip check and git diff --check, saving
   stdout/stderr, exact commands and exit codes. Before new implementation the
   transferred suite is 46 tests / workflow 10; original commit is 45 / 9.
   Count new regression tests honestly; all required tests must run, not skip.
5. Record server environment and installed dependency inventory. Explicitly
   bind this server using the corrected enrollment command. Save two-update
   official-weight B04/B03 CUDA startup reports on the GPU(s) intended for use.
   Use offline environment flags and a single visible GPU per process.

After these pass, report readiness with source identity, installed lock, device
allocation, input hashes, startup results and planned new HEAD3 output. Do not
substitute an environment report for evidence of startup or test execution.

## Approved rebuild and resource use

The returned B03 is COMPLETE: ten epochs, 6,460 updates, 825,860 sample visits.
Its best and last checkpoint SHA-256 are both
`df6abe819cefd84c823f6516f0b81c643e5654062d8209bcf73cf28a1194789c`.
Verified dev micro/macro are 65.3883% / 65.3729%. Its scheduled learning rate
has reached zero. Do not resume/extend it. No B04 result was delivered; the
HEAD3 and acceptance directories are empty, and both terminal logs are empty.

The owner confirmed HEAD3 is lost and explicitly approved rebuilding. Build
one new HEAD3 from the frozen training partition (three epochs, no dev selection),
using a new verified train cache if shards are unavailable. Then train new B04
and B03 from that exact same head hash, in new output directories. Preserve the
old B03 as a separate historical result. Never use its trained head as HEAD3.
Do not request another approval for the same rebuild or expand to extra methods.
Once input, implementation, allocation and startup checks pass, proceed through
HEAD3 and both main runs; the readiness report is a progress update, not another
approval gate. Stop only for a concrete failure, missing asset/access/allocation,
or a material deviation from this assignment.

Use seed 17, float32, microbatch 1, effective batch 128, ONLINE10 and existing
optimizer/schedule/augmentation settings. Expected counts remain train 82,586,
dev 10,378; confirm 10,254 stays sealed. Each new main run must complete exactly
ten epochs / 6,460 updates / 825,860 visits. New HEAD3 has 1,938 updates /
247,758 visits. No confirm evaluation, test inference, ensembles or new methods.

One GPU suffices per run. If rebuilding the pair and two GPUs are allocated,
launch B04 and B03 independently on different devices only AFTER HEAD3 completes
and both configs verify the same initializer. Keep the other cards unused;
more seeds/methods need a separate assignment. Shared inputs are read-only;
each process gets its own run/log directory. Limit CPU worker/thread use to
the allocation rather than consuming all 48 logical CPUs.

## SSH durability, monitoring and handback

Use the site's scheduler when required; otherwise use an already installed
tmux or a tested nohup wrapper. A wrapper must enter the exact project directory,
use the project Conda executable, preserve the allocated GPU visibility, capture
stdout/stderr, record PID/start/end/exit status, and retain a nonzero exit code
through any tee pipeline. Verify that it actually survives an SSH disconnect.
Do not install tmux automatically or rely on an SSH foreground shell for hours.

Training stdout may stay empty until the command returns. Monitor process
liveness, GPU allocation and new epochs/checkpoint files together; an empty log
alone is neither failure nor success. Do not repeatedly launch a second process
against the same output directory. On OOM/NaN/error, retain evidence and stop;
no automatic retries, batch changes or checkpoint deletion. Resume only a truly
incomplete run with compatible configuration after identifying the interruption.

Deliver one new `T4-B04-B03-deliverable-YYYYMMDD.zip` with SHA-256, exact source
snapshot/patch, server/allocation information, tests and exit statuses, installed
lock, startup reports, migration mapping/hash receipt, configs and shared HEAD3,
cache indexes, and each new run's result/resolved/epochs/per-epoch dev predictions/
best/last files. Include a best-epoch metric table and report any remaining gaps.
Do not include credentials, the environment, competition images or big caches.
Transfer privately, never commit these generated artifacts to public GitHub.
