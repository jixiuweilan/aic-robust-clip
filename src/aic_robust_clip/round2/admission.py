"""Independent CUDA checks, common engineering selection and bound receipts."""
from __future__ import annotations
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter
from importlib.metadata import version
import torch
from ..contracts import read_json, write_json, sha256_json
from ..environment import machine_fingerprint
from ..gpu_identity import normalize_gpu_uuid, cuda_gpu_uuids
from ..models.provision import file_sha256
from ..runtime import current_code_revision, seed_everything
from .config import VERSION, RUNS, RECIPE, sealed, verify_seal, stage_path, load_assets, head_descriptor

GRID = [(b, w) for b in (4, 8, 16, 32) for w in (2, 4)]


def runtime_identity():
    result = {"host": machine_fingerprint(), "source": current_code_revision(), "python": sys.version,
              "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
              "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "dependencies": {p: version(p) for p in ("torch", "transformers", "numpy", "Pillow", "huggingface-hub")}}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        uuid = getattr(p, "uuid", None)
        if uuid is None:
            raise ValueError("CUDA device UUID unavailable; do not substitute a device index")
        result["gpu"] = {"uuid": normalize_gpu_uuid(str(uuid)), "name": p.name, "total_bytes": p.total_memory}
    return result


def methods_for(group):
    return [method for _, (g, _, method) in RUNS.items() if g == group]


def feasibility(assets_path, *, method, adaptation, machine, output):
    """Full initial train scoring, then two bounded updates; no checkpoint."""
    from contextlib import ExitStack
    from . import engine
    from .methods import MethodState
    engine.require_machine(machine)
    root = stage_path(output)
    root.mkdir(parents=True, exist_ok=False)
    try:
        assets = load_assets(assets_path, verify_archives=True)
        seed_everything(17)
        student, processor, head = engine.student_for(assets, adaptation)
        with ExitStack() as stack:
            train = assets.dataset("train", processor, online=True)
            score = assets.dataset("train", processor, purpose="scoring")
            for dataset in (train, score):
                stack.callback(dataset.close)
            state = MethodState(method, [r.sample_id for r in train.records],
                                [train.class_to_index[r.class_id] for r in train.records], len(assets.class_map.id_to_index))
            trainer = engine.Trainer(student, state, identity={"purpose": "feasibility_only"}, device="cuda", precision="fp16")
            with engine.loader_for(score, batch=64, workers=2) as loader:
                scored = engine.scoring(student, loader, device="cuda", features_required=method == "fine")
            if method != "ce":
                state.rescore(*scored, completed_epochs=5 if method == "snscl" else 0)
            write_json(root / "selection.json", state.report)
            selected = copy.copy(train)
            selected.records = tuple(r for r, keep in zip(train.records, state.selected) if keep)
            if len(selected) < 256:
                raise ValueError("feasibility requires two full batch128 updates")
            with engine.loader_for(selected, batch=4, workers=2, shuffle=True) as loader:
                loader.limit_dispatch(64)
                trainer.train_epoch(loader, epoch=5 if method == "snscl" else 0, max_updates=2)
            queue = int(trainer.auxiliary.queue.count.sum()) if trainer.auxiliary else None
            if trainer.updates != 2 or trainer.samples != 256 or trainer.skipped:
                raise ValueError("real feasibility requires two successful updates without AMP skips")
            if method == "snscl" and (not queue or not torch.isfinite(state.soft).all()):
                raise ValueError("SNSCL active soft-label/momentum/queue path failed")
        value = sealed({"version": VERSION, "kind": "feasibility", "status": "passed", "runtime": runtime_identity(),
                        "asset_digest": assets.descriptor["digest"], "head_sha256": head["sha256"],
                        "method": method, "adaptation": adaptation, "recipe": RECIPE, "scoring_samples": len(score),
                        "updates": trainer.updates, "samples": trainer.samples, "queue_count": queue,
                        "checkpoint_created": False, "completed_at": time.time()})
        write_json(root / "feasibility.json", value)
        return value
    except BaseException as exc:
        engine.record_failure(root, exc)
        raise


def tiny_student(adaptation, *, image_size=32):
    """Random tiny HF CLIP structure only for bounded software validation."""
    from transformers import CLIPConfig, CLIPModel
    from ..models.clip import FrozenCLIPEncoder
    from .model import Student
    config = CLIPConfig(projection_dim=8,
        text_config={"vocab_size": 16, "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2},
        vision_config={"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
                       "num_attention_heads": 2, "image_size": image_size, "patch_size": 16})
    return Student(FrozenCLIPEncoder(CLIPModel(config)), 3, adaptation)


def synthetic_check(method, adaptation, *, device, precision, zero_selection_policy="error"):
    from .engine import Trainer
    from .methods import MethodState, MethodError
    seed_everything(17)
    student = tiny_student(adaptation)
    state = MethodState(method, ["synthetic-0", "synthetic-1", "synthetic-2", "synthetic-3"], [0, 1, 2, 0], 3,
                        zero_selection_policy=zero_selection_policy)
    trainer = Trainer(student, state, identity={"purpose": "synthetic_startup"}, device=device, precision=precision, effective_batch=2)
    frozen = {k: p.detach().clone() for k, p in student.named_parameters() if not p.requires_grad}
    trainable = {k: p.detach().clone() for k, p in student.named_parameters() if p.requires_grad}
    for i in range(2):
        indices = [2 * i, 2 * i + 1]
        batch = {"image": torch.randn(2, 3, 32, 32), "sample_id": [state.ids[j] for j in indices],
                 "label_index": state.labels[indices]}
        trainer.update_window([batch], epoch=5 if method == "snscl" else 0, fraction=(i + 1) / 2)
    if trainer.updates != 2:
        raise ValueError("synthetic check requires two successful optimizer updates")
    if any(not torch.equal(dict(student.named_parameters())[k], v) for k, v in frozen.items()):
        raise ValueError("frozen/text parameter changed")
    if not any(not torch.equal(dict(student.named_parameters())[k], v) for k, v in trainable.items()):
        raise ValueError("no student parameter update")
    return {"method": method, "adaptation": adaptation, "precision": precision, "device": device,
            "zero_selection_policy": zero_selection_policy,
            "updates": 2, "samples": 4, "status": "passed", "evidence": "synthetic_correctness_only"}


def checks(output, *, group, device="cuda", previous_failure=None, machine_history=None):
    root = stage_path(output)
    root.mkdir(parents=True, exist_ok=False)
    if group not in {"t4", "4060-a", "4060-b"}:
        raise ValueError("unknown group")
    # A subprocess isolates unittest RNG and CUDA initialization from checks.
    script = '''import json,unittest,sys
suite=unittest.defaultTestLoader.discover("tests")
r=unittest.TextTestRunner(verbosity=2).run(suite)
with open(sys.argv[1],"w") as f: json.dump({"tests":r.testsRun,"failures":len(r.failures),"errors":len(r.errors),"skips":len(r.skipped)},f)
sys.exit(not r.wasSuccessful() or bool(r.skipped))
'''
    commands = [[sys.executable, "-c", script, str(root / "suite.json")],
                [sys.executable, "-m", "compileall", "-q", "src", "tests"],
                [sys.executable, "-m", "pip", "check"], ["git", "diff", "--check"]]
    results, startup = [], []
    try:
        for i, command in enumerate(commands):
            with (root / f"check-{i}.log").open("w") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            results.append(result.returncode)
            if result.returncode:
                raise ValueError(f"acceptance command {i} failed; preserve log")
        adaptation = "full_visual" if group == "t4" else "lora"
        for method in methods_for(group):
            for precision in (("fp32", "fp16") if device == "cuda" else ("fp32",)):
                startup.append(synthetic_check(method, adaptation, device=device, precision=precision))
        runtime = runtime_identity()
        history = read_json(machine_history) if machine_history else None
        if group != "t4" and device == "cuda":
            if (not history or history.get("host") != runtime["host"]
                    or normalize_gpu_uuid(history.get("gpu_uuid")) != normalize_gpu_uuid(runtime["gpu"]["uuid"])
                    or type(history.get("original_b04_failure")) is not bool
                    or not history.get("reviewer") or not history.get("basis")):
                raise ValueError("4060 machine history must identify original B04 failure status; unknown is blocked")
        prior = read_json(previous_failure) if previous_failure else None
        if prior and (prior.get("host") != runtime["host"] or normalize_gpu_uuid(prior.get("gpu_uuid")) != normalize_gpu_uuid(runtime.get("gpu", {}).get("uuid"))
                      or prior.get("test") != "test_verification_cached_only_for_unchanged_file_and_process" or not prior.get("original_evidence_sha256")):
            raise ValueError("original B04 failure machine/evidence identity is incomplete")
        if history and history["original_b04_failure"] and prior is None:
            raise ValueError("original B04 failed machine requires original evidence and successful retest")
        value = sealed({"version": VERSION, "kind": "checks", "group": group, "runtime": runtime,
                        "device": device, "suite": read_json(root / "suite.json"), "returncodes": results,
                        "startup": startup, "previous_failure": prior, "machine_history": history,
                        "logs": {f"check-{i}.log": file_sha256(root / f"check-{i}.log") for i in range(4)}})
        write_json(root / "checks.json", value)
        return value
    except BaseException as exc:
        write_json(root / "failure.json", {"error": repr(exc), "auto_retry": False})
        raise


def _profile_one(assets_path, *, method, adaptation, microbatch, workers, machine, output, eval_windows=1, barrier=None,
                 _assets=None, _student_factory=None, _stage="second_round", _zero_selection_policy="error"):
    """Real loaders/model, 2 warmup + 10 measured updates; never resumable."""
    from contextlib import ExitStack
    from .engine import require_machine, student_for, Trainer, loader_for, scoring, clock, atomic_save
    from .methods import MethodState, MethodError
    require_machine(machine, stage=_stage)
    if _zero_selection_policy != "error" and _stage != "preliminary":
        raise ValueError("abstention profile is restricted to the preliminary pilot")
    root = stage_path(output, stage=_stage)
    root.mkdir(parents=True, exist_ok=False)
    if (microbatch, workers) not in GRID or eval_windows not in {1, 3}:
        raise ValueError("invalid fixed engineering window")
    runtime = runtime_identity()
    try:
        assets = _assets if _assets is not None else load_assets(assets_path)
        seed_everything(17)
        student, processor, head = (_student_factory or student_for)(assets, adaptation)
        with ExitStack() as stack:
            train = assets.dataset("train", processor, online=True)
            score = assets.dataset("train", processor, purpose="scoring")
            dev = assets.dataset("dev", processor)
            for d in (train, score, dev):
                stack.callback(d.close)
            ids = [r.sample_id for r in train.records]
            state = MethodState(method, ids, [train.class_to_index[r.class_id] for r in train.records], len(assets.class_map.id_to_index),
                                zero_selection_policy=_zero_selection_policy)
            trainer = Trainer(student, state, identity={"purpose": "profile_only"}, device="cuda", precision="fp16")
            score_loader = stack.enter_context(loader_for(score, batch=64, workers=workers))
            torch.cuda.reset_peak_memory_stats()
            start = clock("cuda")
            if method != "ce":
                state.rescore(*scoring(student, score_loader, device="cuda", features_required=method == "fine",
                                      expected_stage=_stage), completed_epochs=5 if method == "snscl" else 0)
            scoring_seconds = clock("cuda") - start
            write_json(root / "selection.json", state.report)
            selected = copy.copy(train)
            selected.records = tuple(r for r, keep in zip(train.records, state.selected) if keep)
            loader = stack.enter_context(loader_for(selected, batch=microbatch, workers=workers, shuffle=True))
            loader.limit_dispatch(12 * (128 // microbatch))
            if len(loader.order) < 12 * 128:
                raise ValueError("insufficient selected train samples for fixed profile")
            if barrier:
                barrier = stage_path(barrier, stage=_stage)
                write_json(barrier / f"{method}.ready.json", {"uuid": runtime["gpu"]["uuid"]})
                while not (barrier / "go.json").is_file():
                    time.sleep(.1)
            window_started = time.time()
            durations, pending, count = [], [], 0
            start = clock("cuda")
            for batch in loader:
                pending.append(batch)
                count += len(batch["sample_id"])
                if count == 128:
                    _, success = trainer.update_window(pending, epoch=5 if method == "snscl" else 0,
                                                       fraction=loader.position / len(loader.order))
                    if not success:
                        raise ValueError("AMP skipped engineering update")
                    end = clock("cuda")
                    durations.append(end - start)
                    pending, count = [], 0
                    start = clock("cuda")
            if len(durations) != 12:
                raise ValueError("profile did not complete 12 updates")
            window_ended = time.time()
            windows = []
            student.eval()
            for window in range(eval_windows):
                # Independent lifecycle per eval window, including shutdown.
                with loader_for(dev, batch=64, workers=workers) as evaluation:
                    evaluation.limit_dispatch(12)
                    if len(evaluation.order) < 12 * 64:
                        raise ValueError("insufficient dev samples for fixed eval profile")
                    timings = []
                    start = clock("cuda")
                    with torch.no_grad():
                        for batch in evaluation:
                            logits = student(batch["image"].to("cuda")).float()
                            if not torch.isfinite(logits).all():
                                raise ValueError("nonfinite profile dev logits")
                            end = clock("cuda")
                            timings.append(end - start)
                            start = clock("cuda")
                if len(timings) != 12:
                    raise ValueError("incomplete eval window")
                windows.append(sum(timings[2:]) / 10)
            start = clock("cuda")
            atomic_save(trainer.checkpoint(), root / "profile-only.pt")
            save_seconds = clock("cuda") - start
            reserved = torch.cuda.max_memory_reserved()
            free, total = torch.cuda.mem_get_info()
            # Account for memory owned outside this PyTorch allocator as well.
            external = total - free - torch.cuda.memory_reserved()
            occupied_peak = reserved + max(0, external)
            if occupied_peak > .85 * total:
                raise ValueError("less than 15 percent GPU memory headroom")
            projected = sum(durations[2:]) / 10 * math.ceil(len(selected) / 128) + scoring_seconds + max(windows) * math.ceil(len(dev) / 64) + save_seconds
            # Initializer dev export replay uses fresh student and saved/reloaded
            # visual state, and explicit dev inputs before releasing any run.
            batch = next(iter(loader_for(dev, batch=4)))
            with torch.no_grad():
                expected = student(batch["image"].to("cuda")).float().cpu()
            replay_state = {k: v.cpu().clone() for k, v in student.state_dict().items()
                            if k.startswith(("encoder.clip_model.vision_model.", "encoder.clip_model.visual_projection.", "classifier."))}
            atomic_save(replay_state, root / "student-only.pt")
            restored = torch.load(root / "student-only.pt", weights_only=True, map_location="cpu")
            # Reuse allocation, destroy current visual values before loading.
            with torch.no_grad():
                for name, p in student.named_parameters():
                    if name in restored:
                        p.zero_()
            student.load_state_dict({**student.state_dict(), **restored}, strict=True)
            with torch.no_grad():
                actual = student(batch["image"].to("cuda")).float().cpu()
            if not torch.allclose(expected, actual, atol=1e-6, rtol=1e-5) or not torch.equal(expected.argmax(1), actual.argmax(1)):
                raise ValueError("pre-run explicit dev student replay failed")
            stack.close()
        value = sealed({"version": VERSION, "kind": "profile", "status": "passed", "runtime": runtime, "stage": _stage,
                        "zero_selection_policy": _zero_selection_policy, "selection": state.report,
                        "asset_digest": assets.descriptor["digest"], "head_sha256": head["sha256"],
                        "method": method, "adaptation": adaptation, "microbatch": microbatch, "workers": workers,
                        "warmup": 2, "measured": 10, "train_update_seconds": durations[2:], "eval_seconds": windows,
                        "scoring_seconds": scoring_seconds, "save_seconds": save_seconds,
                        "peak_occupied_bytes": occupied_peak, "total_bytes": total, "projected_epoch_seconds": projected,
                        "dev_student_replay": True, "workers_closed": True})
        value = sealed({**verify_seal(value), "train_window_started": window_started, "train_window_ended": window_ended})
        write_json(root / "profile.json", value)
        return value
    except BaseException as exc:
        write_json(root / "failure.json", sealed({"version": VERSION, "kind": "profile", "status": "failed",
            "runtime": runtime, "method": method, "adaptation": adaptation, "microbatch": microbatch,
            "workers": workers, "error": repr(exc), "auto_retry": False,
            "zero_selection_policy": _zero_selection_policy,
            "method_failure": isinstance(exc, MethodError),
            "method_diagnostics": exc.diagnostics if isinstance(exc, MethodError) else None}))
        raise


def profile(assets, *, group, machine, output, method=None, engineering=None, eval_windows=1):
    from .engine import require_machine
    require_machine(machine)
    root = stage_path(output)
    root.mkdir(parents=True, exist_ok=False)
    methods = [method] if method else methods_for(group)
    if not methods or any(m not in methods_for(group) for m in methods):
        raise ValueError("method not assigned to this group")
    rows = []
    for m in methods:
        method_failed = False
        for b, w in ([engineering] if engineering else GRID):
            target = root / f"{m}-b{b}-w{w}"
            if method_failed:
                record = target / "failure.json"
                write_json(record, sealed({"version": VERSION, "status": "failed", "method": m, "microbatch": b,
                    "workers": w, "error": "not executed after earlier method numerical/selection failure", "auto_retry": False}))
                rows.append({"path": str(record), "sha256": file_sha256(record)})
                continue
            command = [sys.executable, "-m", "aic_robust_clip.round2", "profile-one", "--assets", str(assets),
                "--method", m, "--adaptation", "full_visual" if group == "t4" else "lora", "--microbatch", str(b),
                "--workers", str(w), "--machine", str(machine), "--output", str(target), "--eval-windows", str(eval_windows)]
            with (root / f"{target.name}.log").open("w") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            record = target / ("profile.json" if result.returncode == 0 else "failure.json")
            if not record.is_file():
                write_json(record, sealed({"version": VERSION, "status": "failed", "method": m, "microbatch": b,
                    "workers": w, "error": f"child exit {result.returncode}; see parent log", "auto_retry": False}))
            rows.append({"path": str(record), "sha256": file_sha256(record)})
            method_failed = read_json(record).get("method_failure", False)
    value = sealed({"version": VERSION, "kind": "profile_grid", "group": group, "rows": rows})
    write_json(root / "profiles.json", value)
    return value


def choose_common(rows, methods):
    if any(r.get("method_failure") for r in rows):
        raise ValueError("method failure stops the entire admission group")
    eligible = []
    for b, w in GRID:
        matching = [r for r in rows if r.get("microbatch") == b and r.get("workers") == w]
        if len(matching) != len(methods) or {r["method"] for r in matching} != set(methods):
            continue
        if any(r.get("status") != "passed" or r["peak_occupied_bytes"] > .85 * r["total_bytes"]
               or not math.isfinite(r["projected_epoch_seconds"]) or r["projected_epoch_seconds"] <= 0 for r in matching):
            continue
        eligible.append({"microbatch": b, "workers": w, "seconds": max(r["projected_epoch_seconds"] for r in matching)})
    if not eligible:
        raise ValueError("no common passing engineering configuration; stop group admission")
    best = min(r["seconds"] for r in eligible)
    candidates = [r for r in eligible if r["seconds"] < best * 1.05 or r["seconds"] == best]
    choice = min(candidates, key=lambda r: (r["workers"], r["microbatch"]))
    return {k: choice[k] for k in ("microbatch", "workers")}


def read_profiles(paths):
    rows = []
    for path in paths:
        value = read_json(path)
        verify_seal(value)
        if value.get("kind") != "profile_grid":
            raise ValueError("new round2 profile index required")
        for entry in value["rows"]:
            if file_sha256(entry["path"]) != entry["sha256"]:
                raise ValueError("profile evidence hash changed")
            row = read_json(entry["path"])
            verify_seal(row)
            rows.append(row)
    return rows


def choose(paths, *, group, output):
    value = sealed({"version": VERSION, "group": group, "engineering": choose_common(read_profiles(paths), methods_for(group)),
                    "profile_indices": [{"path": str(Path(p).resolve()), "sha256": file_sha256(p)} for p in paths]})
    if Path(output).exists():
        raise FileExistsError(output)
    write_json(output, value)
    return value


def validate_receipt(receipt, assets, *, live):
    verify_seal(receipt)
    if assets is None or receipt.get("version") != VERSION or receipt.get("kind") != "admission" or receipt.get("recipe") != RECIPE:
        raise ValueError("new round2 admission required; legacy receipts forbidden")
    if receipt.get("asset_digest") != assets.descriptor["digest"] or receipt.get("head_sha256") != head_descriptor(assets)["sha256"]:
        raise ValueError("admission asset/head mismatch")
    if receipt.get("group") not in {"t4", "4060-a", "4060-b"} or not receipt.get("owner"):
        raise ValueError("admission group/owner missing")
    if receipt["source"] != current_code_revision():
        raise ValueError("new code requires new admission")
    if live:
        runtime = runtime_identity()
        matches = [r for r in receipt["runtimes"] if r == runtime]
        if not matches:
            raise ValueError("machine/GPU/source/dependencies differ from admission")
    for entry in receipt["evidence"]:
        if file_sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("admission evidence changed")
    if receipt.get("status") != "passed":
        raise ValueError("admission not passed")


def admit(assets_path, *, group, owner, check_paths, profile_paths, final_paths, output,
          concurrency=None, previous_failure_required=False, feasibility_paths=()):
    assets = load_assets(assets_path, verify_archives=True)
    head = head_descriptor(assets)
    rows = read_profiles(profile_paths)
    expected_grid = {(m, b, w) for m in methods_for(group) for b, w in GRID}
    if len(rows) != len(expected_grid) or {(r["method"], r["microbatch"], r["workers"]) for r in rows} != expected_grid:
        raise ValueError("retain every fixed-grid success and failure, with no retries")
    engineering = choose_common(rows, methods_for(group))
    finals = read_profiles(final_paths)
    selected = [r for r in finals if r.get("status") == "passed" and (r.get("microbatch"), r.get("workers")) == (engineering["microbatch"], engineering["workers"])]
    if len(finals) != len(selected) or len(selected) != len(methods_for(group)) or {r["method"] for r in selected} != set(methods_for(group)):
        raise ValueError("final windows do not cover every assigned method")
    if any(len(r["eval_seconds"]) != 3 or not r["dev_student_replay"] or not r["workers_closed"]
           or r["peak_occupied_bytes"] > .85 * r["total_bytes"] for r in selected):
        raise ValueError("three independent eval windows and dev replay required")
    checks_values = [read_json(p) for p in check_paths]
    runtimes = []
    for check in checks_values:
        verify_seal(check)
        if check.get("version") != VERSION or check.get("kind") != "checks" or check["group"] != group or check["device"] != "cuda":
            raise ValueError("current CUDA checks required")
        if check["suite"]["tests"] <= 0 or any(check["suite"][k] for k in ("failures", "errors", "skips")) or any(check["returncodes"]):
            raise ValueError("checks not clean")
        if group != "t4":
            history = check.get("machine_history")
            if (not history or history.get("host") != check["runtime"]["host"]
                    or normalize_gpu_uuid(history.get("gpu_uuid")) != normalize_gpu_uuid(check["runtime"]["gpu"]["uuid"])
                    or type(history.get("original_b04_failure")) is not bool
                    or not history.get("reviewer") or not history.get("basis")
                    or history["original_b04_failure"] and not check.get("previous_failure")):
                raise ValueError("unresolved 4060 original failure machine identity")
        expected = {(m, p) for m in methods_for(group) for p in ("fp32", "fp16")}
        if {(r["method"], r["precision"]) for r in check["startup"] if r["status"] == "passed" and r["updates"] == 2} != expected:
            raise ValueError("missing FP32/FP16 two-update method checks")
        runtimes.append(check["runtime"])
    if previous_failure_required and not any(c["previous_failure"] for c in checks_values):
        raise ValueError("original B04 failure identity/retest evidence required")
    if not owner or len(runtimes) != (4 if group == "t4" else 1):
        raise ValueError("owner and one checks receipt per assigned GPU required")
    if len({normalize_gpu_uuid(r["gpu"]["uuid"]) for r in runtimes}) != len(runtimes):
        raise ValueError("duplicate GPU UUIDs")
    if group == "t4" and (len({normalize_gpu_uuid(r["runtime"]["gpu"]["uuid"]) for r in selected}) != 4
                          or any("T4" not in r["gpu"]["name"] for r in runtimes)):
        raise ValueError("one method on each of four T4 GPUs required")
    if group != "t4" and any("4060" not in r["gpu"]["name"] for r in runtimes):
        raise ValueError("4060 machine required")
    for row in [r for r in rows if r.get("status") == "passed"] + selected:
        assigned = next(r for r in selected if r["method"] == row["method"])
        if normalize_gpu_uuid(row["runtime"]["gpu"]["uuid"]) != normalize_gpu_uuid(assigned["runtime"]["gpu"]["uuid"]):
            raise ValueError("method GPU assignment changed between grid and final windows")
        if row["runtime"] not in runtimes or row["runtime"]["source"] != current_code_revision() or row["asset_digest"] != assets.descriptor["digest"] or row["head_sha256"] != head["sha256"]:
            raise ValueError("profile/checks/asset source identity mismatch")
        if row["adaptation"] != ("full_visual" if group == "t4" else "lora"):
            raise ValueError("wrong adaptation in profile")
    feasible = [read_json(p) for p in feasibility_paths]
    if len(feasible) != len(methods_for(group)) or {r.get("method") for r in feasible} != set(methods_for(group)):
        raise ValueError("initial real train feasibility required for every method")
    for row in feasible:
        verify_seal(row)
        assigned = next(r for r in selected if r["method"] == row["method"])
        if (row.get("version") != VERSION or row.get("kind") != "feasibility" or row.get("status") != "passed"
                or row.get("runtime") != assigned["runtime"] or row.get("recipe") != RECIPE
                or row.get("asset_digest") != assets.descriptor["digest"] or row.get("head_sha256") != head["sha256"]
                or row.get("adaptation") != assigned["adaptation"] or row.get("updates") != 2
                or row.get("samples") != 256 or row.get("scoring_samples", 0) <= 0
                or not 0 < row.get("completed_at", 0) <= min(
                    r.get("train_window_started", 0) for r in rows + selected
                    if r.get("method") == row["method"] and r.get("status") == "passed")
                or row.get("checkpoint_created") is not False or row["method"] == "snscl" and not row.get("queue_count")):
            raise ValueError("real train feasibility identity or active update evidence mismatch")
    evidence = list(check_paths) + list(profile_paths) + list(final_paths) + list(feasibility_paths)
    for path, check in zip(check_paths, checks_values):
        for name, digest in check["logs"].items():
            log = Path(path).parent / name
            if log.parent != Path(path).parent or file_sha256(log) != digest:
                raise ValueError("checks log hash mismatch")
            evidence.append(str(log))
    for path in list(profile_paths) + list(final_paths):
        evidence.extend(entry["path"] for entry in read_json(path)["rows"])
    if group == "t4":
        if not concurrency:
            raise ValueError("four-task concurrent short window required")
        value = read_json(concurrency)
        verify_seal(value)
        if value.get("kind") != "concurrency" or value.get("status") != "passed" or value.get("overlap_seconds", 0) <= 0 or value["engineering"] != engineering or value["asset_digest"] != assets.descriptor["digest"] or value["source"] != current_code_revision() or {normalize_gpu_uuid(u) for u in value["gpu_uuids"]} != {normalize_gpu_uuid(r["gpu"]["uuid"]) for r in runtimes}:
            raise ValueError("invalid four-T4 concurrency evidence")
        evidence.append(concurrency)
        for method, digest in zip(methods_for("t4"), value["profiles"]):
            path = Path(concurrency).parent / method / "profile.json"
            row = read_json(path)
            verify_seal(row)
            if row["digest"] != digest:
                raise ValueError("concurrency child profile digest mismatch")
            evidence.append(str(path))
    value = sealed({"version": VERSION, "kind": "admission", "status": "passed", "group": group, "owner": owner,
                    "source": current_code_revision(), "runtimes": runtimes, "recipe": RECIPE,
                    "assignments": {r["method"]: r["runtime"]["gpu"]["uuid"] for r in selected},
                    "asset_digest": assets.descriptor["digest"], "head_sha256": head["sha256"], "engineering": engineering,
                    "evidence": [{"path": str(Path(p).resolve()), "sha256": file_sha256(p)} for p in evidence],
                    "previous_failure_required": previous_failure_required})
    if Path(output).exists():
        raise FileExistsError(output)
    write_json(output, value)
    return value


def concurrent(assets, *, machine, choice_path, gpu_uuids, output):
    from .engine import require_machine
    gpu_uuids = cuda_gpu_uuids(gpu_uuids)
    require_machine(machine)
    if len(gpu_uuids) != 4 or len(set(gpu_uuids)) != 4:
        raise ValueError("four distinct T4 GPU UUIDs required")
    choice = read_json(choice_path)
    verify_seal(choice)
    if choice["group"] != "t4":
        raise ValueError("T4 common choice required")
    engineering = choice["engineering"]
    root = stage_path(output)
    root.mkdir(parents=True, exist_ok=False)
    processes, handles, starts = [], [], []
    barrier = root / "barrier"
    barrier.mkdir()
    try:
        for method, uuid in zip(methods_for("t4"), gpu_uuids):
            target = root / method
            command = [sys.executable, "-m", "aic_robust_clip.round2", "profile-one", "--assets", str(assets),
                "--method", method, "--adaptation", "full_visual", "--microbatch", str(engineering["microbatch"]),
                "--workers", str(engineering["workers"]), "--machine", str(machine), "--output", str(target),
                "--barrier", str(barrier)]
            handle = (root / f"{method}.log").open("w")
            handles.append(handle)
            processes.append(subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": uuid}, stdout=handle, stderr=subprocess.STDOUT))
            starts.append(time.time())
        while not all((barrier / f"{m}.ready.json").is_file() for m in methods_for("t4")):
            if any(p.poll() is not None for p in processes):
                raise ValueError("concurrent child exited before common start barrier")
            time.sleep(.1)
        write_json(barrier / "go.json", {"released": time.time()})
        codes = [p.wait() for p in processes]
        rows = [read_json(root / m / "profile.json") for m in methods_for("t4")] if not any(codes) else []
        if any(codes) or len(rows) != 4 or {normalize_gpu_uuid(r["runtime"]["gpu"]["uuid"]) for r in rows} != {normalize_gpu_uuid(u) for u in gpu_uuids}:
            raise ValueError(f"concurrent tasks failed: {codes}")
        for method, uuid, row in zip(methods_for("t4"), gpu_uuids, rows):
            verify_seal(row)
            if row["method"] != method or normalize_gpu_uuid(row["runtime"]["gpu"]["uuid"]) != normalize_gpu_uuid(uuid):
                raise ValueError("concurrent method/GPU assignment mismatch")
        overlap = min(r["train_window_ended"] for r in rows) - max(r["train_window_started"] for r in rows)
        if overlap <= 0:
            raise ValueError("four training windows did not overlap")
        value = sealed({"version": VERSION, "kind": "concurrency", "status": "passed", "engineering": engineering,
                        "asset_digest": load_assets(assets).descriptor["digest"], "source": current_code_revision(),
                        "gpu_uuids": gpu_uuids, "exitcodes": codes, "started_at": starts,
                        "overlap_seconds": overlap, "profiles": [r["digest"] for r in rows]})
        write_json(root / "concurrency.json", value)
        return value
    except BaseException as exc:
        write_json(root / "failure.json", {"error": repr(exc), "auto_retry": False})
        raise
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
            p.wait()
        for handle in handles:
            handle.close()
