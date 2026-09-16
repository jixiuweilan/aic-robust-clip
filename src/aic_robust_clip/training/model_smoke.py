"""Official-weight startup checks on generated images, never data archives."""
from __future__ import annotations

import copy

from ..configuration import RECIPES
from ..contracts import RunConfig
from ..data.dataset import SampleItem
from ..data.loading import StatefulBatchLoader
from ..models.clip import load_openai_clip, torch
from ..models.lora import VisualLoRAModel
from ..runtime import resolve_run_config, seed_everything
from .baseline import FrozenFeatureBaseline, OnlineFrozenBaseline, TrainConfig, train_baseline


def check_official_model(weights, revision, recipe, *, device="cpu"):
    from PIL import Image
    if recipe not in RECIPES or device not in {"cpu", "cuda"}:
        raise ValueError("unknown smoke recipe/device")
    run = RunConfig(stage="preliminary", parameters=RECIPES[recipe], max_samples=4, max_updates=2)
    resolve_run_config(run)
    seed_everything(run.seed)
    bundle = load_openai_clip(revision=revision, local_path=weights)
    reference = copy.deepcopy(bundle.encoder) if recipe == "F010" else None
    images = [bundle.preprocess(Image.new("RGB", (224, 224), (i * 40, 80, 120))) for i in range(4)]
    if recipe == "B01":
        bundle.encoder.eval()
        with torch.no_grad():
            images = list(bundle.encoder(torch.stack(images)))
        model = FrozenFeatureBaseline(bundle.encoder.output_dim, 2)
    elif recipe == "B04":
        model = OnlineFrozenBaseline(bundle.encoder, bundle.encoder.output_dim, 2)
    else:
        model = VisualLoRAModel(bundle.encoder, 2)
        model.assert_qv_only()
    items = [SampleItem(image, f"generated-{i}", f"{i % 2:04}", i % 2) for i, image in enumerate(images)]
    ids = [item.sample_id for item in items]
    def loader(shuffle):
        return StatefulBatchLoader(items, sample_ids=ids, max_samples=4, batch_size=1, shuffle=shuffle)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = train_baseline(model, loader(True), config=TrainConfig.from_run(run), class_count=2,
        training_labels={item.sample_id: item.label_index for item in items},
        scoring_loader=loader(False) if recipe == "F100" else None, reference_encoder=reference, device=device)
    return {"recipe": recipe, "evidence": "official_weight_startup_only", "weight_identity": bundle.identity.to_dict(),
        "device": device, "result": result.to_dict(),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
        "formal_training": "not_run", "competition_images_read": 0}
