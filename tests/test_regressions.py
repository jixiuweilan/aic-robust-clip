"""Small synthetic regressions only: no official data, weights or benchmarks."""
from __future__ import annotations

import copy
import io
import json
import random
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from aic_robust_clip.contracts import CheckpointMetadata, RunConfig, SampleRecord
from aic_robust_clip.data.audit import audit_archive
from aic_robust_clip.data.dataset import DatasetError, ManifestDataset, SampleItem
from aic_robust_clip.data.loading import StatefulBatchLoader, collate_samples
from aic_robust_clip.data.splits import assert_no_group_crossing, make_grouped_split
from aic_robust_clip.models.clip import torch
from aic_robust_clip.training.baseline import FrozenFeatureBaseline, TrainConfig, TrainingError, train_baseline
from aic_robust_clip.training.reliability import ReliabilityState


def record(index: int, class_id: str = "0000") -> SampleRecord:
    return SampleRecord(stage="preliminary", role="train", archive_identity="a" * 64,
        archive_path="fixture.zip", member_path=f"{class_id}/{index}.jpg", sample_id=f"s{index:04}",
        byte_size=1, crc32=0, byte_sha256=f"{index:064x}", pixel_sha256=f"{index:064x}",
        decode_status="decoded", class_id=class_id)


class DataRegressions(unittest.TestCase):
    def test_recipe_presets_resolve_without_changing_smoke_mode(self):
        presets = json.loads((Path(__file__).resolve().parents[1] / "configs/research-methods.json").read_text())
        self.assertEqual(set(presets), {"B01", "B03", "B04", "R01", "F100", "F010", "F001"})
        for preset in presets.values():
            config = TrainConfig.from_run(RunConfig(stage="preliminary", parameters=preset["parameters"]))
            self.assertEqual(config.run.execution_mode, "smoke")
        with self.assertRaisesRegex(ValueError, "disabled"):
            TrainConfig.from_run(RunConfig(stage="preliminary", parameters={"objective": "gce", "weighting": True}))

    def test_split_exact_ratio_for_unique_balanced_classes(self):
        records = [record(i, f"{i // 200:04}") for i in range(600)]
        split = make_grouped_split(records)
        self.assertEqual(split.counts(), {"train": 480, "dev": 60, "confirm": 60})
        for counts in split.report()["per_class"].values():
            self.assertEqual(counts, {"train": 160, "dev": 20, "confirm": 20})
        self.assertEqual(split.digest, make_grouped_split(reversed(records)).digest)

    def test_scarce_and_conflicting_groups_keep_training_support(self):
        for size in (1, 2, 3):
            self.assertGreater(make_grouped_split([record(i) for i in range(size)]).counts()["train"], 0)
        records = [record(i) for i in range(20)]
        records[1] = replace(records[1], class_id="0001", byte_sha256=records[0].byte_sha256)
        split = make_grouped_split(records)
        assert_no_group_crossing(split)
        self.assertEqual(len(split.report()["conflicting_label_groups"]), 1)
        self.assertEqual(len({item.sample_id for item in split.records}), 20)

    def test_stale_parent_and_cross_partition_ids_rejected(self):
        records = [record(i) for i in range(10)]
        split = make_grouped_split(records)
        changed = [replace(records[0], pixel_sha256="f" * 64), *records[1:]]
        with self.assertRaisesRegex(DatasetError, "digest mismatch"):
            ManifestDataset.from_split(changed, split.records, stage="preliminary", partition="train",
                                       purpose="train", class_to_index={"0000": 0})
        with self.assertRaisesRegex(DatasetError, "duplicate sample"):
            ManifestDataset.from_split(records, [*split.records, replace(split.records[0], partition="dev")],
                stage="preliminary", partition="train", purpose="train", class_to_index={"0000": 0})

    def test_encrypted_first_and_later_member_have_own_labels(self):
        for encrypted_index in (0, 1):
            with self.subTest(index=encrypted_index), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "fixture.zip"
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("0000/a.jpg", b"a")
                    archive.writestr("0001/b.jpg", b"b")
                original = zipfile.ZipFile.infolist
                def marked(archive):
                    infos = original(archive)
                    infos[encrypted_index].flag_bits |= 1
                    return infos
                with patch.object(zipfile.ZipFile, "infolist", marked):
                    report = audit_archive(path, stage="preliminary", role="train", decode=False)
                encrypted = next(item for item in report.records if item.decode_status == "encrypted")
                self.assertEqual(encrypted.class_id, f"{encrypted_index:04}")
                self.assertEqual(report.failures[0].reason, "encrypted_member")


def tensor_image(raw):
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as image:
        return torch.tensor(list(image.convert("RGB").tobytes()), dtype=torch.float32) / 255


class SyntheticDataset:
    """On-access randomness detects unwanted replay or RNG resets on resume."""
    def __init__(self, count=5):
        self.count, self.reads = count, []
    def __len__(self):
        return self.count
    def __getitem__(self, index):
        self.reads.append(index)
        return SampleItem(torch.tensor([float(index), 1., 0., -1.]) + torch.rand(4) * .1 + random.random() * .01,
                          f"s{index}", str(index % 2), index % 2)


def stream(count=5, *, shuffle=True):
    dataset = SyntheticDataset(count)
    return StatefulBatchLoader(dataset, batch_size=1, seed=17, shuffle=shuffle,
                               max_samples=count, sample_ids=[f"s{i}" for i in range(count)])


def metadata(config):
    return CheckpointMetadata(model_family="synthetic-regression", stage=config.run.stage,
        class_map_digest=config.run.class_map_digest, manifest_digest=config.run.manifest_digest,
        split_digest=config.run.split_digest, official_weight_id=config.run.official_weight_id,
        official_weight_revision=config.run.official_weight_revision, configuration_digest=config.digest,
        code_revision="synthetic-test", progress={}, optimizer_state_present=True,
        scheduler_state_present=True, rng_state_present=True, sampler_state_present=True)


def run_config(**kwargs):
    return RunConfig(stage="preliminary", max_updates=3, max_samples=8,
        class_map_digest="a" * 64, manifest_digest="b" * 64, split_digest="c" * 64,
        official_weight_revision="d" * 40, **kwargs)


@unittest.skipIf(torch is None, "CPU torch required for tensor regressions")
class TensorRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_manifest_dataloader_and_bounded_training(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.zip"
            raw = io.BytesIO()
            Image.new("RGB", (2, 2), (100, 20, 3)).save(raw, format="PNG")
            with zipfile.ZipFile(path, "w") as archive:
                for i in range(3):
                    archive.writestr(f"0000/{i}.png", raw.getvalue())
            report = audit_archive(path, stage="preliminary", role="train")
            dataset = ManifestDataset(report.records, stage="preliminary", role="train", partition="train",
                purpose="train", class_to_index={"0000": 0}, transform=tensor_image)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_samples)
            model = FrozenFeatureBaseline(12, 2)
            before = copy.deepcopy(model.state_dict())
            result = train_baseline(model, loader, config=TrainConfig(RunConfig(stage="preliminary")),
                                    class_count=2, device="cpu")
            self.assertEqual((result.updates, result.samples), (2, 2))
            self.assertTrue(any(not torch.equal(before[key], value) for key, value in model.state_dict().items()))
            dataset.close()

    def test_no_extra_sample_fetch_at_update_limit(self):
        loader = stream()
        result = train_baseline(FrozenFeatureBaseline(4, 2), loader,
            config=TrainConfig(run_config()), class_count=2, device="cpu")
        self.assertEqual(len(loader.dataset.reads), 3)
        self.assertEqual(result.updates, 3)

    def test_resume_matches_uninterrupted_with_rng_optimizer_and_scheduler(self):
        torch.manual_seed(7)
        initial = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(4, 2))
        config = TrainConfig(run_config(), scheduler="step")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = copy.deepcopy(initial)
            full_loader = stream()
            train_baseline(full, full_loader, config=config, class_count=2, device="cpu",
                checkpoint_dir=root / "full", checkpoint_metadata=metadata(config))
            paused = copy.deepcopy(initial)
            first_loader = stream()
            train_baseline(paused, first_loader, config=config, class_count=2, device="cpu",
                checkpoint_dir=root / "resume", checkpoint_metadata=metadata(config),
                stop_after_updates=1)
            next_loader = stream()
            resumed = copy.deepcopy(initial)
            result = train_baseline(resumed, next_loader, config=config, class_count=2, device="cpu",
                checkpoint_dir=root / "resume", checkpoint_metadata=metadata(config),
                resume_from=root / "resume/last.pt")
            self.assertEqual(result.updates, 3)
            self.assertEqual(first_loader.dataset.reads + next_loader.dataset.reads, full_loader.dataset.reads)
            for key, value in full.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
            a = torch.load(root / "full/last.pt", weights_only=False)
            b = torch.load(root / "resume/last.pt", weights_only=False)
            self.assertEqual(a["scheduler_state"], b["scheduler_state"])
            for index, state in a["optimizer_state"]["state"].items():
                for key, value in state.items():
                    self.assertTrue(torch.equal(value, b["optimizer_state"]["state"][index][key]))
            completed = stream()
            train_baseline(resumed, completed, config=config, class_count=2, device="cpu",
                resume_from=root / "resume/last.pt", checkpoint_metadata=metadata(config))
            self.assertEqual(completed.dataset.reads, [])
            with self.assertRaisesRegex(Exception, "configuration digest"):
                changed = replace(config, learning_rate=.1)
                train_baseline(copy.deepcopy(initial), stream(), config=changed, class_count=2, device="cpu",
                    checkpoint_metadata=metadata(changed), resume_from=root / "resume/last.pt")

    def test_research_paths_and_auxiliary_bounds(self):
        class Adapted(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(4, 4)
                self.classifier = torch.nn.Linear(4, 2)
            def forward_with_features(self, images):
                features = self.encoder(images)
                return self.classifier(features), features
            def forward(self, images):
                return self.forward_with_features(images)[0]
        labels = {f"s{i}": i % 2 for i in range(5)}
        for name, options in (("R01", {"objective": "gce"}), ("F100", {"weighting": True}),
                ("F010", {"lambda_preserve": .1}), ("F001", {"prior_tau": 1.}),
                ("F111", {"weighting": True, "lambda_preserve": .1, "prior_tau": 1.})):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                config = TrainConfig.from_run(run_config(parameters=options))
                model = Adapted()
                reference = torch.nn.Linear(4, 4)
                before = copy.deepcopy(reference.state_dict())
                result = train_baseline(model, stream(), config=config, class_count=2, device="cpu",
                    training_labels=labels, scoring_loader=stream(shuffle=False), reference_encoder=reference,
                    checkpoint_dir=directory, checkpoint_metadata=metadata(config))
                self.assertEqual(result.updates, 3)
                self.assertEqual(result.reference_samples, 3 if config.lambda_preserve else 0)
                self.assertEqual(result.scoring_samples, 5 if config.weighting else 0)
                for key, value in reference.state_dict().items():
                    self.assertTrue(torch.equal(before[key], value))
                self.assertTrue(all(parameter.grad is None for parameter in reference.parameters()))
                saved = torch.load(Path(directory) / "last.pt", weights_only=False)
                if config.weighting:
                    self.assertEqual(len(saved["module_state"]["reliability"]["ema_losses"]), 5)
                    self.assertEqual(saved["module_state"]["reliability"]["completed_epochs"], 0)

    def test_resume_accumulation_tail_and_lineage_rejection(self):
        from aic_robust_clip.training.checkpoint import load_checkpoint
        config = TrainConfig(run_config(), accumulation_steps=2)
        initial = FrozenFeatureBaseline(4, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full, paused, resumed = (copy.deepcopy(initial) for _ in range(3))
            train_baseline(full, stream(3), config=config, class_count=2, device="cpu")
            train_baseline(paused, stream(3), config=config, class_count=2, device="cpu",
                checkpoint_dir=root, checkpoint_metadata=metadata(config), stop_after_updates=1)
            result = train_baseline(resumed, stream(3), config=config, class_count=2, device="cpu",
                checkpoint_metadata=metadata(config), resume_from=root / "last.pt")
            self.assertEqual((result.updates, result.samples), (2, 3))
            self.assertTrue(all(torch.equal(value, resumed.state_dict()[key]) for key, value in full.state_dict().items()))
            requested = {key: getattr(metadata(config), key) for key in (
                "stage", "class_map_digest", "manifest_digest", "split_digest",
                "official_weight_id", "official_weight_revision")}
            for key in ("stage", "class_map_digest", "split_digest"):
                with self.subTest(key=key), self.assertRaisesRegex(Exception, "incompatible checkpoint"):
                    changed = dict(requested, **{key: "semifinal" if key == "stage" else "f" * 64})
                    load_checkpoint(root / "last.pt", model=copy.deepcopy(initial), requested=changed)

    def test_gce_and_all_off_objective_values(self):
        from aic_robust_clip.training.objectives import combined_wpi_loss, gce_loss
        logits = torch.tensor([[1., 2.], [3., 1.]], requires_grad=True)
        labels = torch.tensor([0, 1])
        ce = torch.nn.functional.cross_entropy(logits, labels)
        self.assertTrue(torch.equal(ce, combined_wpi_loss(logits, labels)))
        self.assertTrue(torch.allclose(ce, gce_loss(logits, labels, q=0)))
        loss = gce_loss(logits, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_weighting_prior_and_preservation_known_values(self):
        from aic_robust_clip.training.objectives import combined_wpi_loss
        logits = torch.tensor([[1., 2.], [3., 1.]], requires_grad=True)
        labels = torch.tensor([0, 1])
        weights = torch.tensor([1., .2])
        prior = torch.tensor([.75, .25])
        features = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        teacher = torch.tensor([[0., 1.], [0., 1.]], requires_grad=True)
        actual = combined_wpi_loss(logits, labels, sample_weights=weights, prior=prior, tau=1.,
            adapted_features=features, frozen_features=teacher, lambda_preserve=.1)
        expected = (torch.nn.functional.cross_entropy(logits + prior.log(), labels, reduction="none") * weights).sum() / weights.sum() + .05
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(features.grad)

    def test_lora_zero_start_allowed_updates_and_reload(self):
        from types import SimpleNamespace
        from aic_robust_clip.models.clip import FrozenCLIPEncoder
        from aic_robust_clip.models.lora import VisualLoRAModel
        class FakeCLIP(torch.nn.Module):
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
                attention = self.vision_model.encoder.layers[0].self_attn
                return attention.q_proj(pixel_values) + attention.k_proj(pixel_values) + attention.v_proj(pixel_values)
        base = FrozenCLIPEncoder(FakeCLIP())
        x = torch.tensor([[1., 2., 3., 4.]])
        expected = base(x)
        model = VisualLoRAModel(base, class_count=2)
        self.assertTrue(torch.equal(expected, model.encoder(x)))
        model.assert_qv_only()
        original = copy.deepcopy(model.state_dict())
        frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
        train_baseline(model, stream(2), config=TrainConfig(RunConfig(stage="preliminary", max_updates=1)),
                       class_count=2, device="cpu")
        self.assertTrue(all(torch.equal(original[name], model.state_dict()[name]) for name in frozen))
        self.assertTrue(any(not torch.equal(original[name], value) for name, value in model.state_dict().items() if "lora_B" in name))
        restored = VisualLoRAModel(FrozenCLIPEncoder(FakeCLIP()), class_count=2)
        restored.load_state_dict(model.state_dict())
        model.eval()
        restored.eval()
        self.assertTrue(torch.equal(model(x), restored(x)))

    def test_research_resume_restores_w_and_prior_without_replaying(self):
        labels = {f"s{i}": i % 2 for i in range(5)}
        state = ReliabilityState(tuple(labels), {key: str(value) for key, value in labels.items()},
                                 weights={key: .2 + index * .1 for index, key in enumerate(labels)},
                                 completed_epochs=2)
        config = TrainConfig(run_config(), weighting=True, prior_tau=1.)
        initial = FrozenFeatureBaseline(4, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full, paused, resumed = (copy.deepcopy(initial) for _ in range(3))
            for model, subdir, pause, resume in ((full, "full", None, None), (paused, "resume", 1, None),
                    (resumed, "resume", None, root / "resume/last.pt")):
                train_baseline(model, stream(), config=config, class_count=2, device="cpu",
                    training_labels=labels, reliability=copy.deepcopy(state), scoring_loader=stream(shuffle=False),
                    checkpoint_dir=root / subdir, checkpoint_metadata=metadata(config),
                    stop_after_updates=pause, resume_from=resume)
            self.assertTrue(all(torch.equal(value, resumed.state_dict()[key]) for key, value in full.state_dict().items()))
            a = torch.load(root / "full/last.pt", weights_only=False)
            b = torch.load(root / "resume/last.pt", weights_only=False)
            self.assertEqual(a["module_state"], b["module_state"])

    def test_caps_and_loader_roles_fail_before_reading(self):
        from aic_robust_clip.runtime import RuntimeLimitError
        from aic_robust_clip.training.startup import run_fixture_startup_check
        with self.assertRaises(RuntimeLimitError):
            run_fixture_startup_check(run=RunConfig(stage="preliminary", max_samples=1000))
        loader = stream()
        loader.dataset.purpose, loader.dataset.partition = "dev", "dev"
        loader.dataset.role, loader.dataset.stage = "train", "preliminary"
        with self.assertRaises(TrainingError):
            train_baseline(FrozenFeatureBaseline(4, 2), loader, config=TrainConfig(run_config()), class_count=2, device="cpu")
        self.assertEqual(loader.dataset.reads, [])
        with self.assertRaises(RuntimeLimitError):
            train_baseline(FrozenFeatureBaseline(4, 2), stream(),
                config=TrainConfig(RunConfig(stage="preliminary", execution_mode="formal")), class_count=2, device="cpu")

    def test_partial_accumulation_and_dev_best_checkpoint(self):
        config = TrainConfig(run_config(), accumulation_steps=2)
        with tempfile.TemporaryDirectory() as directory:
            result = train_baseline(FrozenFeatureBaseline(4, 2), stream(3), config=config,
                dev_loader=stream(2, shuffle=False), class_count=2, device="cpu",
                checkpoint_dir=directory, checkpoint_metadata=metadata(config))
            self.assertEqual((result.updates, result.samples), (2, 3))
            self.assertTrue((Path(directory) / "best.pt").is_file())
            self.assertEqual(result.dev_metrics.class_support[0] + result.dev_metrics.class_support[1], 2)


if __name__ == "__main__":
    unittest.main()
