"""Workflow acceptance with generated images and explicit tiny-model mocks.

These tests never download weights, read competition data, or enable formal
execution. The Hugging Face structural test uses random tiny configuration,
not another research backbone or a measured experiment.
"""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import random
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aic_robust_clip.contracts import RunConfig, read_json, write_json
from aic_robust_clip.configuration import prepare
from aic_robust_clip.data.audit import audit_archive, class_map_from_records, write_audit_report
from aic_robust_clip.data.splits import make_grouped_split, write_split_manifest
from aic_robust_clip.environment import DEVELOPMENT_HOST, machine_policy
from aic_robust_clip.models.clip import FrozenCLIPEncoder, torch
from aic_robust_clip.models.provision import inspect_weights
from aic_robust_clip.runtime import RuntimeLimitError
from aic_robust_clip.training.baseline import TrainConfig, train_baseline
from aic_robust_clip.training.optimization import optimizer_groups, selection_key, warmup_cosine_factor

STACK = torch is not None and all(importlib.util.find_spec(name) is not None for name in ("PIL", "numpy", "transformers"))


class BoundaryTests(unittest.TestCase):
    def test_formal_permission_fails_before_missing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            write_json(path, {"schema_version": 2, "recipe": "B03", "stage": "preliminary", "execution_mode": "formal"})
            with patch("aic_robust_clip.configuration.load_manifest", side_effect=AssertionError("read input before gate")):
                with self.assertRaises(RuntimeLimitError):
                    prepare(path)

    def test_bound_machine_file_cannot_enable_development_host(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "machine.json"
            write_json(path, {"role": "training", "fingerprint": DEVELOPMENT_HOST})
            with patch("aic_robust_clip.environment.machine_fingerprint", return_value=DEVELOPMENT_HOST):
                with self.assertRaises(RuntimeLimitError):
                    machine_policy(path)

    def test_fabricated_weight_manifest_is_not_an_official_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory) / "official-weight-manifest.json", {"model_id": "openai/clip-vit-base-patch32",
                "revision": "a" * 40, "files": {"config.json": "b" * 64}})
            with self.assertRaisesRegex(ValueError, "allowlist"):
                inspect_weights(directory, "a" * 40)

    def test_selection_and_schedule_reference_values(self):
        macro_winner = SimpleNamespace(macro_recall=.8, micro_top1=.6)
        micro_winner = SimpleNamespace(macro_recall=.7, micro_top1=.9)
        self.assertGreater(selection_key(macro_winner, 3), selection_key(micro_winner, 1))
        self.assertGreater(selection_key(macro_winner, 1), selection_key(macro_winner, 2))
        factors = [warmup_cosine_factor(i, warmup_updates=2, total_updates=6) for i in range(7)]
        self.assertEqual(factors[:3], [.5, 1., 1.])
        self.assertAlmostEqual(factors[4], .5)
        self.assertEqual(factors[-1], 0.)


@unittest.skipUnless(STACK, "CPU torch/transformers/Pillow/numpy required")
class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixtures(self, root):
        from PIL import Image
        archive_path = root / "train.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for index in range(20):
                data = io.BytesIO()
                Image.new("RGB", (10 + index, 12), (index * 9, 20, 255 - index * 8)).save(data, format="PNG")
                archive.writestr(f"{index % 2:04}/{index}.png", data.getvalue())
        report = audit_archive(archive_path, stage="preliminary", role="train")
        write_audit_report(root / "manifest.json", report)
        write_json(root / "class-map.json", class_map_from_records(report.records).to_dict())
        write_split_manifest(root / "split.json", make_grouped_split(report.records))
        return {"schema_version": 2, "recipe": "B03", "stage": "preliminary", "execution_mode": "smoke",
            "seed": 17, "device": "cpu", "manifest": "manifest.json", "split": "split.json",
            "class_map": "class-map.json", "weights": "unused-mocked-weights", "weight_revision": "a" * 40,
            "train_cache": "train-cache", "dev_cache": "dev-cache", "head": "shared-head", "output": "run",
            "limits": {"max_samples": 8, "max_updates": 2, "max_eval_batches": 2}}

    def bundle(self, ctx):
        from transformers import CLIPImageProcessor
        class TinyCLIP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(projection_dim=4)
                self.vision_model = torch.nn.Module()
                self.vision_model.encoder = torch.nn.Module()
                layer = torch.nn.Module()
                layer.self_attn = torch.nn.Module()
                for name in ("q_proj", "k_proj", "v_proj"):
                    setattr(layer.self_attn, name, torch.nn.Linear(4, 4))
                self.vision_model.encoder.layers = torch.nn.ModuleList([layer])
            def get_image_features(self, pixel_values):
                features = torch.cat((pixel_values.mean(dim=(-1, -2)), torch.ones(len(pixel_values), 1)), dim=1)
                attention = self.vision_model.encoder.layers[0].self_attn
                return attention.q_proj(features) + attention.k_proj(features) + attention.v_proj(features)
        # Fixed original weights, independent of recipe/global RNG.
        with torch.random.fork_rng():
            torch.manual_seed(3)
            encoder = FrozenCLIPEncoder(TinyCLIP())
        return SimpleNamespace(encoder=encoder, processor=CLIPImageProcessor())

    def test_end_to_end_cache_head_resume_evaluate_predict_cli(self):
        from aic_robust_clip import pipeline_cli as cli
        from aic_robust_clip.submission import validate_submission
        identity = {"model_id": "openai/clip-vit-base-patch32", "revision": "a" * 40,
                    "digest": "b" * 64, "preprocessing_digest": "c" * 64}
        with tempfile.TemporaryDirectory() as directory, patch("aic_robust_clip.configuration.inspect_weights", return_value=identity), \
                patch("aic_robust_clip.workflow.load_bundle", side_effect=self.bundle):
            root = Path(directory)
            config = self.fixtures(root)
            path = root / "config.json"
            write_json(path, config)
            common = ["--config", str(path)]
            self.assertEqual(cli.cache_main(common + ["--partition", "train"]), 0)
            self.assertEqual(cli.cache_main(common + ["--partition", "dev"]), 0)
            self.assertEqual(cli.init_head_main(common), 0)
            head_hash = read_json(root / "shared-head/head.json")["sha256"]
            self.assertEqual(cli.train_main(common + ["--stop-after-updates", "1"]), 0)
            self.assertEqual(cli.train_main(common + ["--resume", str(root / "run/last.pt")]), 0)
            result = read_json(root / "run/result.json")
            self.assertEqual((result["updates"], result["samples"]), (2, 2))
            checkpoint = root / "run/best.pt"
            state = torch.load(checkpoint, weights_only=False)
            self.assertEqual(state["metadata"]["initialization_digest"], head_hash)
            self.assertEqual(cli.train_main(common), 1)  # never overwrite a run
            self.assertEqual(cli.evaluate_main(common + ["--checkpoint", str(checkpoint), "--partition", "confirm",
                                                        "--output", str(root / "confirm.json")]), 1)
            self.assertEqual(cli.lock_selection_main(common + ["--checkpoint", str(checkpoint), "--output", str(root / "selection.json")]), 0)
            self.assertEqual(cli.evaluate_main(common + ["--checkpoint", str(checkpoint), "--partition", "confirm",
                "--selection-record", str(root / "selection.json"), "--output", str(root / "confirm.json")]), 0)
            with zipfile.ZipFile(root / "test.zip", "w") as archive, zipfile.ZipFile(root / "train.zip") as source:
                archive.writestr("fixture.png", source.read("0000/0.png"))
            test_report = audit_archive(root / "test.zip", stage="preliminary", role="test").to_dict()
            test_report["synthetic_fixture"] = True
            write_json(root / "test.json", test_report)
            self.assertEqual(cli.predict_main(common + ["--checkpoint", str(checkpoint), "--test-manifest", str(root / "test.json"),
                                                        "--output", str(root / "prediction")]), 0)
            self.assertEqual(validate_submission(root / "prediction/submission.zip"), 1)
            # B01 exercises cache consumption and attaching its frozen encoder
            # for pixel inference; no second classifier fit occurs at prediction.
            config.update(recipe="B01", output="b01")
            write_json(path, config)
            self.assertEqual(cli.train_main(common), 0)
            self.assertEqual(cli.predict_main(common + ["--checkpoint", str(root / "b01/best.pt"), "--test-manifest", str(root / "test.json"),
                                                        "--output", str(root / "prediction-b01")]), 0)
            # An existing cache/head must not be silently reused across identities.
            config["seed"] = 29
            config["recipe"] = "B03"
            config["output"] = "wrong-init"
            write_json(path, config)
            self.assertEqual(cli.train_main(common), 1)
            self.assertFalse((root / "wrong-init").exists())

    def test_addressed_augmentation_and_numpy_rng_resume(self):
        import numpy as np
        from PIL import Image
        from transformers import CLIPImageProcessor
        from aic_robust_clip.data.transforms import ClipTransform
        from aic_robust_clip.training.checkpoint import _rng_state, _restore_rng_state
        raw = io.BytesIO()
        array = np.arange(40 * 60 * 3, dtype=np.uint8).reshape(40, 60, 3)
        Image.fromarray(array).save(raw, format="PNG")
        transform = ClipTransform(CLIPImageProcessor(), seed=17, online=True)
        a = transform.apply(raw.getvalue(), sample_id="x", epoch=1)
        torch.rand(20)
        random.random()
        np.random.random()
        b = transform.apply(raw.getvalue(), sample_id="x", epoch=1)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, transform.apply(raw.getvalue(), sample_id="x", epoch=2)))
        state = _rng_state()
        expected = (random.random(), np.random.random(), torch.rand(3))
        _restore_rng_state(state)
        actual = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_online_pair_shared_initialization_views_and_update_scope(self):
        from aic_robust_clip.workflow import cache_command, init_head_command, stream, training_components
        # Freeze the actual handoff configs as a matched pair, without enabling
        # formal execution or inspecting any competition assets.
        configs = Path(__file__).resolve().parents[1] / "configs/formal"
        pair = [read_json(configs / f"{recipe}.json") for recipe in ("B04", "B03")]
        self.assertEqual([config["recipe"] for config in pair], ["B04", "B03"])
        self.assertNotEqual(pair[0]["output"], pair[1]["output"])
        self.assertEqual(*[{k: v for k, v in config.items() if k not in {"recipe", "output"}} for config in pair])
        identity = {"model_id": "openai/clip-vit-base-patch32", "revision": "a" * 40,
                    "digest": "b" * 64, "preprocessing_digest": "c" * 64}
        with tempfile.TemporaryDirectory() as directory, patch("aic_robust_clip.configuration.inspect_weights", return_value=identity), \
                patch("aic_robust_clip.workflow.load_bundle", side_effect=self.bundle):
            root = Path(directory)
            config = self.fixtures(root)
            path = root / "config.json"
            write_json(path, config)
            cache_command(path, "train")
            head = init_head_command(path)  # HEAD-SMOKE, explicitly two updates
            components = []
            try:
                for recipe in ("B04", "B03"):
                    config.update(recipe=recipe, output=recipe)
                    write_json(path, config)
                    ctx = prepare(path)
                    model, train, dev, scoring, reference, initial = training_components(ctx)
                    components.append((ctx, model, train, dev))
                    self.assertEqual(initial, head["sha256"])
                    self.assertIsNone(scoring)
                    self.assertIsNone(reference)
                frozen, lora = components[0][1], components[1][1]
                for name, tensor in frozen.classifier.state_dict().items():
                    self.assertTrue(torch.equal(tensor, lora.classifier.state_dict()[name]), name)
                lora.assert_qv_only()
                # Sample order and transforms must not depend on RNG consumed
                # during construction of LoRA; inspect only two fixture batches.
                loaders = [stream(ctx, train) for ctx, _, train, _ in components]
                for loader in loaders:
                    loader.reset(1)
                for _ in range(2):
                    left = next(loaders[0])
                    torch.rand(11)
                    right = next(loaders[1])
                    self.assertEqual(left["sample_id"], right["sample_id"])
                    self.assertTrue(torch.equal(left["label_index"], right["label_index"]))
                    self.assertTrue(torch.equal(left["image"], right["image"]))
                for index in range(2):
                    left, right = [item[3][index] for item in components]
                    self.assertEqual(left.sample_id, right.sample_id)
                    self.assertTrue(torch.equal(left.image, right.image))
                frozen.eval()
                lora.eval()
                with torch.no_grad():
                    pixels = left.image.unsqueeze(0)
                    self.assertTrue(torch.allclose(frozen(pixels), lora(pixels), atol=1e-7, rtol=1e-6))
                # Both branches must start, update the intended parameters only,
                # and terminate at the same explicit two-update smoke bound.
                for ctx, model, train, _ in components:
                    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                    result = train_baseline(model, stream(ctx, train), config=ctx.train, class_count=2, device="cpu")
                    self.assertEqual((result.updates, result.samples), (2, 2))
                    self.assertTrue(result.stopped_by_limit)
                    changed = {name for name, parameter in model.named_parameters()
                               if not torch.equal(before[name], parameter)}
                    allowed = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
                    self.assertTrue(changed <= allowed)
                    self.assertTrue(any(name.startswith("classifier.") for name in changed))
                    if model is frozen:
                        self.assertTrue(all(name.startswith("classifier.") for name in allowed))
                    else:
                        self.assertTrue(any("lora_B" in name for name in changed))
            finally:
                for _, _, train, dev in components:
                    train.close()
                    dev.close()

    def test_huggingface_structural_qv_forward_backward(self):
        from transformers import CLIPConfig, CLIPModel
        from aic_robust_clip.models.lora import VisualLoRAModel
        from aic_robust_clip.data.dataset import SampleItem
        from aic_robust_clip.data.loading import StatefulBatchLoader
        configuration = CLIPConfig(projection_dim=8,
            text_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2, "vocab_size": 20},
            vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
                           "image_size": 224, "patch_size": 32})
        model = VisualLoRAModel(FrozenCLIPEncoder(CLIPModel(configuration)), 2)
        items = [SampleItem(torch.rand(3, 224, 224), "fixture", "0000", 0)]
        loader = StatefulBatchLoader(items, sample_ids=["fixture"], max_samples=1)
        result = train_baseline(model, loader, config=TrainConfig(RunConfig(stage="preliminary", max_updates=1)),
                                class_count=2, device="cpu")
        self.assertEqual(result.updates, 1)
        model.assert_qv_only()

    def test_uneven_accumulation_matches_full_batch(self):
        from aic_robust_clip.data.dataset import SampleItem
        from aic_robust_clip.data.loading import StatefulBatchLoader
        from aic_robust_clip.training.reliability import ReliabilityState
        items = [SampleItem(torch.tensor([float(i), 1.]), f"s{i}", str(i % 2), i % 2) for i in range(3)]
        labels = {item.sample_id: item.label_index for item in items}
        state = ReliabilityState(tuple(labels), {key: str(value) for key, value in labels.items()},
                                 weights={"s0": 1., "s1": .2, "s2": .5}, completed_epochs=2)
        initial = torch.nn.Linear(2, 2)
        outputs = []
        for micro, accumulation in ((2, 2), (3, 1)):
            model = copy.deepcopy(initial)
            loader = StatefulBatchLoader(items, batch_size=micro, shuffle=False, max_samples=3, sample_ids=list(labels))
            scoring = StatefulBatchLoader(items, batch_size=micro, shuffle=False, max_samples=3, sample_ids=list(labels))
            config = TrainConfig(RunConfig(stage="preliminary", batch_size=micro, max_updates=1),
                                 accumulation_steps=accumulation, weighting=True, weight_decay=0.)
            train_baseline(model, loader, config=config, class_count=2, device="cpu", training_labels=labels,
                           scoring_loader=scoring, reliability=copy.deepcopy(state))
            outputs.append(model.state_dict())
        for key in outputs[0]:
            self.assertTrue(torch.allclose(outputs[0][key], outputs[1][key], atol=1e-7, rtol=1e-6), key)

    def test_optimizer_parameter_groups(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(2, 2)
                self.norm = torch.nn.LayerNorm(2)
                self.lora_A = torch.nn.Parameter(torch.ones(2, 2))
        model = Model()
        groups = optimizer_groups(model, head_lr=1e-3, lora_lr=1e-4, weight_decay=1e-4)
        by_id = {id(parameter): group for group in groups for parameter in group["params"]}
        self.assertEqual(by_id[id(model.lora_A)]["lr"], 1e-4)
        self.assertEqual(by_id[id(model.classifier.weight)]["lr"], 1e-3)
        self.assertEqual(by_id[id(model.classifier.bias)]["weight_decay"], 0.)
        self.assertEqual(by_id[id(model.norm.weight)]["weight_decay"], 0.)


if __name__ == "__main__":
    unittest.main()
