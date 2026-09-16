"""Explicit official-snapshot provisioning; ordinary loading is offline."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from ..contracts import read_json, sha256_json, write_json
from .official_weights import REVISION, FILES

MODEL_ID = "openai/clip-vit-base-patch32"
MANIFEST_NAME = "official-weight-manifest.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_weights(directory, revision):
    root = Path(directory)
    value = read_json(root / MANIFEST_NAME)
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or value.get("revision") != revision or value.get("model_id") != MODEL_ID:
        raise ValueError("official weight identity/revision mismatch")
    files = value.get("files", {})
    if revision != REVISION or files != FILES:
        raise ValueError("snapshot does not match the pinned official hash allowlist")
    if not {"config.json", "preprocessor_config.json"} <= set(files) or not ({"model.safetensors", "pytorch_model.bin"} & set(files)):
        raise ValueError("official snapshot is incomplete")
    allowed = {"config.json", "preprocessor_config.json", "model.safetensors", "pytorch_model.bin"}
    if set(files) - allowed:
        raise ValueError("unexpected files in official snapshot manifest")
    for name, expected in files.items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"official snapshot file hash mismatch: {name}")
    config = read_json(root / "config.json")
    vision = config.get("vision_config", {})
    if (vision.get("patch_size"), vision.get("hidden_size"), vision.get("num_hidden_layers"), config.get("projection_dim")) != (32, 768, 12, 512):
        raise ValueError("snapshot is not CLIP ViT-B/32")
    return {"model_id": MODEL_ID, "revision": revision, "digest": sha256_json(files),
            "preprocessing_digest": files["preprocessor_config.json"]}


def provision_weights(directory, revision):
    """Network access only here; download from the allowlisted HF repository."""
    from huggingface_hub import HfApi, hf_hub_download
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("resolve and explicitly supply a 40-character official commit SHA")
    if revision != REVISION:
        raise ValueError("update/review the official allowlist before provisioning another revision")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    info = HfApi().model_info(MODEL_ID, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("Hugging Face did not resolve the requested immutable revision")
    siblings = {item.rfilename: item for item in info.siblings}
    weights = "model.safetensors" if "model.safetensors" in siblings else "pytorch_model.bin"
    files = {}
    for name in ("config.json", "preprocessor_config.json", weights):
        path = hf_hub_download(MODEL_ID, filename=name, revision=revision, local_dir=root)
        files[name] = file_sha256(path)
        lfs = siblings[name].lfs
        if lfs is not None and files[name] != lfs.sha256:
            raise ValueError(f"official LFS hash mismatch: {name}")
    write_json(root / MANIFEST_NAME, {"model_id": MODEL_ID, "revision": revision, "files": files,
        "source_url": f"https://huggingface.co/{MODEL_ID}/tree/{revision}",
        "retrieved_at": datetime.now(timezone.utc).isoformat()})
    return inspect_weights(root, revision)
