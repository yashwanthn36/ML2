import csv
import json
import math
import sys
import wave

import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ave_mm.splits import source_video_disjoint_split
from ave_mm.dataset import AVEDataset
from ave_mm.integrity import audit_manifest
from multimodal.data_adapters.media import read_video_frames
from multimodal.contribution.fcd import InfoNCEFCD
from multimodal.contribution.signatures import estimate_reference_fcd, estimate_signatures
from multimodal.grouping.profiles import cluster_profiles
from multimodal.losses.balanced import class_balanced_ce, group_balanced_ce
from multimodal.optim.moo import min_norm_combine, moo_optimizer_step
from multimodal.models.fusion import TwoModalityFusion
from multimodal.training.splits import grouped_reference_holdout
from multimodal.contribution.targets import build_ucd_targets
from multimodal.training.reference import train_reference
from multimodal.training.joint import train_joint
from scripts.training import evaluate_test


class SignatureDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"modality_1": torch.tensor([1.]), "modality_2": torch.tensor([1.]),
                "label": torch.tensor(0), "sample_id": "s", "source_id": "v",
                "label_name": "class", "split": "train"}


class ControlledModel(nn.Module):
    def forward(self, x1, x2, modality_mask=None):
        if modality_mask is None:
            return torch.tensor([[0., 1.]], device=x1.device)
        mask = modality_mask[0].tolist()
        # [0,1] keeps modality 2; [1,0] keeps modality 1.
        if mask == [0., 1.]:
            return torch.tensor([[2., 0.]], device=x1.device)
        return torch.tensor([[0., 2.]], device=x1.device)


def test_signature_mask_columns_and_signed_deltas(tmp_path):
    path = estimate_signatures(ControlledModel(), DataLoader(SignatureDataset()), tmp_path / "sig.csv",
                               device=torch.device("cpu"))
    with path.open() as f:
        row = next(csv.DictReader(f))
    full = float(row["full_loss"])
    without_1 = float(row["loss_without_1"])
    without_2 = float(row["loss_without_2"])
    assert without_1 == pytest.approx(float(nn.functional.cross_entropy(torch.tensor([[2., 0.]]), torch.tensor([0]))))
    assert without_2 == pytest.approx(float(nn.functional.cross_entropy(torch.tensor([[0., 2.]]), torch.tensor([0]))))
    assert float(row["delta_1"]) == pytest.approx(without_1 - full)
    assert float(row["delta_2"]) == pytest.approx(without_2 - full)
    assert float(row["delta_1"]) < 0 < float(row["delta_2"])


def test_fcd_valid_gradients_queue_and_invalid_inputs():
    estimator = InfoNCEFCD(queue_size=3)
    fused = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    x1 = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
    x2 = torch.tensor([[0., 1.], [0., 1.]], requires_grad=True)
    fcd = estimator(fused, x1, x2)
    assert fcd.shape == (2,)
    assert torch.isfinite(fcd).all() and (fcd >= 0).all()
    assert fcd.sum().item() == pytest.approx(1.0)
    info = estimator.per_sample_information(fused.detach(), x1.detach(), x2.detach())
    expected_positive = 1.0 + math.log(2.) - math.log(math.exp(1.) + math.exp(0.))
    expected_orthogonal = math.log(2.) - math.log(math.exp(0.) + math.exp(1.))
    assert info[0, 0].item() == pytest.approx(expected_positive)
    assert info[0, 1].item() == pytest.approx(expected_orthogonal)
    fcd[0].backward()
    assert fused.grad is not None and torch.isfinite(fused.grad).all()
    same = estimator.distribution(torch.zeros(4, 2))
    assert torch.allclose(same, torch.tensor([.5, .5]))
    estimator.per_sample_information(fused.detach(), x1.detach(), x2.detach(), update_queue=True)
    assert estimator.queue_count == 2
    estimator.per_sample_information(fused[:1].detach(), x1[:1].detach(), x2[:1].detach())
    with pytest.raises(ValueError, match="fewer than two"):
        InfoNCEFCD(queue_size=0).per_sample_information(fused[:1], x1[:1], x2[:1])
    with pytest.raises(ValueError, match="shape"):
        estimator.distribution(torch.ones(2, 3))
    with pytest.raises(ValueError, match="non-finite"):
        estimator.distribution(torch.tensor([[float("nan"), 0.]]))
    identity_queue = InfoNCEFCD(queue_size=4)
    identity_queue._remember(torch.tensor([[1., 0.]]), ["same"])
    with pytest.raises(ValueError, match="fewer than two"):
        identity_queue.per_sample_information(fused[:1].detach(), x1[:1].detach(), x2[:1].detach(),
                                               sample_ids=["same"])


def test_disjoint_split_keeps_each_source_together_and_is_reproducible():
    rows = [{"video_id": f"v{i}", "label_id": i % 2, "sample_id": f"{i}-{j}"}
            for i in range(20) for j in range(2)]
    first, summary = source_video_disjoint_split(rows, seed=3, n_splits=5)
    again, _ = source_video_disjoint_split(rows, seed=3, n_splits=5)
    assert [r["split"] for r in first] == [r["split"] for r in again]
    by_video = {}
    for row in first:
        by_video.setdefault(row["video_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_video.values())
    assert all(not overlap for overlap in summary["source_id_overlap"].values())
    assert set(summary["split_summary"]) == {"train", "val", "test"}


def test_group_fit_assignment_coverage_and_signed_values(tmp_path):
    fitted, assigned = tmp_path / "fit.csv", tmp_path / "assign.csv"
    headers = ["sample_id", "source_id", "label", "delta_1", "delta_2"]
    rows = [{"sample_id": f"s{i}", "source_id": f"v{i}", "label": "0",
             "delta_1": str((-1) ** i * .1), "delta_2": str(i * .01)} for i in range(6)]
    for path, subset in ((fitted, rows[:4]), (assigned, rows)):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers); w.writeheader(); w.writerows(subset)
    result = cluster_profiles(fitted, tmp_path / "groups.json", min_support_floor=2,
                              assignment_signatures_csv=assigned)
    assert set(result["assignments"]) == {r["sample_id"] for r in rows}
    assert result["fit_sample_ids"] == [r["sample_id"] for r in rows[:4]]
    assert result["assignment_sample_ids"] == [r["sample_id"] for r in rows]


def test_hierarchical_losses_exact_present_scope_means():
    logits = torch.tensor([[2., 0.], [0., 2.], [0., 0.]])
    labels = torch.tensor([0, 1, 1])
    raw = nn.functional.cross_entropy(logits, labels, reduction="none")
    expected_class = (raw[labels == 0].mean() + raw[labels == 1].mean()) / 2
    assert class_balanced_ce(logits, labels, {0: 1, 1: 2}) == pytest.approx(expected_class)
    groups = torch.tensor([0, 0, 1])
    expected_group = (raw[0] + (raw[1] + raw[2]) / 2) / 2
    assert group_balanced_ce(logits, labels, groups, {(0, 0): 1, (1, 0): 1, (1, 1): 1},
                             {0: 1, 1: 2}) == pytest.approx(expected_group)


def test_moo_conflicting_degenerate_and_optimizer_update():
    alpha, combined = min_norm_combine((torch.tensor([1.]),), (torch.tensor([-1.]),))
    assert alpha.item() == pytest.approx(.5)
    assert combined[0].item() == pytest.approx(0.)
    alpha, combined = min_norm_combine((torch.zeros(1),), (torch.zeros(1),))
    assert torch.isfinite(alpha) and torch.isfinite(combined[0]).all()
    parameter = nn.Parameter(torch.tensor([1.]))
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad(): model.weight.fill_(1.)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    before = model.weight.detach().clone()
    x = torch.tensor([[1.]])
    diagnostics = moo_optimizer_step(model(x).square().mean(), 2 * model(x).square().mean(),
                                     model, optimizer)
    assert not torch.equal(before, model.weight)
    assert math.isfinite(diagnostics["cosine"])


def test_dataset_full_clip_ignores_invalid_annotation_but_event_policy_rejects_it(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    with wave.open(str(audio), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(np.ones(16000, dtype="<i2").tobytes())
    manifest = tmp_path / "m.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "video_id", "label", "label_id", "split",
            "time_start", "time_end", "video_path", "audio_path"])
        writer.writeheader(); writer.writerow({"sample_id":"s", "video_id":"v", "label":"c", "label_id":0,
            "split":"train", "time_start":0, "time_end":0, "video_path":"v.mp4", "audio_path":"a.wav"})
    class Capture:
        seeks = []
        def __init__(self, path): pass
        def isOpened(self): return True
        def get(self, prop): return 10 if prop == 5 else 20
        def set(self, prop, value): self.seeks.append(value)
        def read(self):
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            frame[:, :, 0] = 255
            return True, frame
        def release(self): pass
    monkeypatch.setattr("ave_mm.dataset.cv2.VideoCapture", Capture)
    ds = AVEDataset(manifest, "train", root=tmp_path, num_frames=2, image_size=8, audio_samples=16000,
                    temporal_window_policy="full_clip")
    original_row = dict(ds.rows[0])
    item = ds[0]
    assert item["modality_1"].shape == (16000,)
    assert item["modality_2"].shape == (2, 3, 8, 8)
    assert item["modality_2"][0, 0].max() == 0
    assert item["modality_2"][0, 2].max() == 1
    assert max(Capture.seeks) > 1000  # Full-clip video sampling uses its 2 s duration, not the 1 s WAV duration.
    assert ds.rows[0] == original_row
    assert (item["sample_id"], item["label"].item(), ds.rows[0]["time_start"], ds.rows[0]["time_end"]) == (
        "s", 0, "0", "0")
    assert item["annotation_time_start"] == "0" and item["annotation_time_end"] == "0"
    assert audit_manifest(ds.rows)["invalid_intervals"][0]["sample_id"] == "s"
    event = AVEDataset(manifest, "train", root=tmp_path, num_frames=2, image_size=8, audio_samples=16000,
                       temporal_window_policy="annotated_event")
    with pytest.raises(ValueError, match="Invalid annotated event interval.*s.*time_end > time_start"):
        event[0]


def test_video_reader_resizes_decoded_frames(monkeypatch, tmp_path):
    class Capture:
        def __init__(self, path):
            self.reads = 0
        def isOpened(self): return True
        def get(self, prop): return 2 if prop == 5 else 20
        def set(self, *args): pass
        def read(self):
            self.reads += 1
            return True, np.zeros((24, 32, 3), dtype=np.uint8)
        def release(self): pass

    monkeypatch.setattr("multimodal.data_adapters.media.cv2.VideoCapture", Capture)
    frames = read_video_frames(tmp_path / "fixture.mp4", start=None, end=None,
                               num_frames=2, image_size=12)
    assert frames.shape == (2, 3, 12, 12)


@pytest.mark.parametrize(("start", "end"), [("nan", "1"), ("", "1"), ("2", "1")])
def test_annotated_event_rejects_nonfinite_missing_and_reversed_intervals(tmp_path, monkeypatch, start, end):
    audio = tmp_path / "a.wav"
    with wave.open(str(audio), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(np.ones(32000, dtype="<i2").tobytes())
    manifest = tmp_path / "m.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "video_id", "label", "label_id", "split",
            "time_start", "time_end", "video_path", "audio_path"])
        writer.writeheader(); writer.writerow({"sample_id": "bad", "video_id": "v", "label": "c",
            "label_id": 0, "split": "train", "time_start": start, "time_end": end,
            "video_path": "v.mp4", "audio_path": "a.wav"})
    class Capture:
        def __init__(self, path): pass
        def isOpened(self): return True
        def get(self, prop): return 10 if prop == 5 else 20
        def set(self, *args): pass
        def read(self): return True, np.zeros((8, 8, 3), dtype=np.uint8)
        def release(self): pass
    monkeypatch.setattr("ave_mm.dataset.cv2.VideoCapture", Capture)
    event = AVEDataset(manifest, "train", root=tmp_path, num_frames=2, image_size=8, audio_samples=16000,
                       temporal_window_policy="annotated_event")
    with pytest.raises(ValueError, match="Invalid annotated event interval.*bad"):
        event[0]
    event.rows[0]["time_start"], event.rows[0]["time_end"] = "0", "0.1"
    with pytest.raises(ValueError, match="fills only"):
        event[0]


class TokenEncoder(nn.Module):
    def __init__(self, width=3):
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        return self.proj(x).unsqueeze(1)


def test_fusion_mask_full_path_and_absent_raw_input_independence():
    torch.manual_seed(4)
    model = TwoModalityFusion(TokenEncoder(), TokenEncoder(), 3, dim=4, heads=2).eval()
    x1, x2 = torch.randn(2, 3), torch.randn(2, 3)
    full = model(x1, x2)
    explicit_full = model(x1, x2, modality_mask=torch.ones(2, 2))
    assert full.shape == (2, 3)
    assert torch.allclose(full, explicit_full)
    missing_1 = torch.tensor([[0., 1.], [0., 1.]])
    before = model(x1, x2, modality_mask=missing_1)
    after = model(x1 + torch.randn_like(x1) * 100, x2, modality_mask=missing_1)
    assert torch.allclose(before, after, atol=1e-6)
    mixed = model(x1, x2, modality_mask=torch.tensor([[0., 1.], [1., 0.]]))
    assert torch.allclose(mixed[0], model(x1[:1], x2[:1], modality_mask=missing_1[:1])[0])
    assert torch.allclose(mixed[1], model(x1[1:], x2[1:], modality_mask=torch.tensor([[1., 0.]]))[0])


def test_grouped_holdout_source_separation_and_repeatability():
    labels = [i % 2 for i in range(40)]
    sample_ids = [f"s{i}" for i in range(40)]
    sources = [f"v{i // 2}" for i in range(40)]
    fit_a, estimate_a, metadata = grouped_reference_holdout(labels, sample_ids, source_ids=sources,
                                                             estimation_fraction=.25, seed=9)
    fit_b, estimate_b, _ = grouped_reference_holdout(labels, sample_ids, source_ids=sources,
                                                       estimation_fraction=.25, seed=9)
    assert fit_a == fit_b and estimate_a == estimate_b
    assert not (set(sources[i] for i in fit_a) & set(sources[i] for i in estimate_a))
    assert metadata["fit_sample_ids"] == [sample_ids[i] for i in fit_a]
    assert metadata["estimation_sample_ids"] == [sample_ids[i] for i in estimate_a]


def test_ucd_cluster_bootstrap_and_parent_fallback(tmp_path):
    signatures = tmp_path / "signatures.csv"
    with signatures.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "source_id", "label", "delta_1", "delta_2"])
        writer.writeheader()
        writer.writerows([{"sample_id":f"s{i}", "source_id":f"v{i//2}", "label":0,
                           "delta_1":.3, "delta_2":.1} for i in range(8)])
    one = build_ucd_targets(signatures, tmp_path / "u1.json", class_ids=[0, 1], min_support=3,
                            bootstrap_replicates=10, seed=4)
    two = build_ucd_targets(signatures, tmp_path / "u2.json", class_ids=[0, 1], min_support=3,
                            bootstrap_replicates=10, seed=4)
    assert one == two
    assert one["classes"]["1"]["fallback"] == "global"
    assert one["global"]["uncertainty"]["resampling_unit"] == "source_id_cluster"
    assert "ci_95_low" not in one["global"]["uncertainty"]


class TinyDataset(Dataset):
    def __init__(self, offset=0):
        self.labels = [0, 1, 0, 1]
        self.sample_ids = [f"tiny{offset}_{i}" for i in range(4)]
        self.source_ids = [f"source{offset}_{i}" for i in range(4)]
        self.num_classes = 2
        self.class_names = ["0", "1"]
        self.class_to_id = {"0": 0, "1": 1}
        self.split = "train"
        self.split_protocol = "synthetic"
        self.metadata = {"dataset_name": "tiny", "adapter": "tests.TinyDataset"}

    def __len__(self):
        return 4

    def __getitem__(self, index):
        label = self.labels[index]
        torch.manual_seed(index + 90)
        return {"modality_1": torch.randn(3), "modality_2": torch.randn(3),
                "label": torch.tensor(label), "sample_id": self.sample_ids[index],
                "source_id": self.source_ids[index], "label_name": str(label)}


def make_tiny_model(num_classes, **kwargs):
    return TwoModalityFusion(TokenEncoder(), TokenEncoder(), num_classes, dim=4, heads=2)


def make_tiny_test_dataset(**kwargs):
    return TinyDataset(20)


def make_tiny_training_datasets(**kwargs):
    return TinyDataset(), TinyDataset(10)


def test_reference_to_joint_checkpoint_smoke(tmp_path):
    train = TinyDataset()
    validation = TinyDataset(10)
    model = TwoModalityFusion(TokenEncoder(), TokenEncoder(), 2, dim=4, heads=2)
    training = {"learning_rate": .005, "weight_decay": 0., "epochs": 1,
                "modality_dropout": .1, "grad_clip": 1., "early_stopping_patience": 2}
    reference_path = train_reference(model, DataLoader(train, batch_size=2),
                                     DataLoader(validation, batch_size=2), training=training,
                                     run_dir=tmp_path / "ref", seed=2,
                                     config={"dataset": {"split_protocol": "synthetic"}, "device": "cpu"},
                                     data_metadata={"split_protocol": "synthetic"},
                                     device=torch.device("cpu"))
    checkpoint = torch.load(reference_path, map_location="cpu", weights_only=False)
    assert checkpoint["split_protocol"] == "synthetic"
    signature_path = estimate_signatures(model, DataLoader(train, batch_size=2),
                                         tmp_path / "signatures.csv", device=torch.device("cpu"))
    fcd_reference = estimate_reference_fcd(model, DataLoader(train, batch_size=2), InfoNCEFCD(queue_size=0),
                                           device=torch.device("cpu"))
    assert len(fcd_reference["distribution"]) == 2
    grouping = cluster_profiles(signature_path, tmp_path / "groups.json", min_support_floor=1,
                                assignment_signatures_csv=signature_path)
    targets_from_signatures = build_ucd_targets(signature_path, tmp_path / "targets.json",
                                               class_ids=[0, 1], group_ids=grouping["assignments"],
                                               min_support=1, bootstrap_replicates=4, seed=2)
    assert len(targets_from_signatures["global"]["target"]) == 2
    fcd = InfoNCEFCD(queue_size=4)
    targets = {"global": {"target": [.5, .5]}, "classes": {"0": {"target": [.5, .5]},
               "1": {"target": [.5, .5]}}, "groups": {}}
    joint_training = {**training, "optimization_mode": "fixed_weight", "ci_objective": "class_balanced",
                      "lambda_mi": .1}
    joint_path = train_joint(TwoModalityFusion(TokenEncoder(), TokenEncoder(), 2, dim=4, heads=2),
                             DataLoader(train, batch_size=2), DataLoader(validation, batch_size=2),
                             training=joint_training, targets=targets, group_assignments=None,
                             fcd_estimator=fcd, run_dir=tmp_path / "joint", seed=2, config={
                                 "dataset": {"split_protocol": "synthetic"}, "device": "cpu",
                                 "contribution": {"target_granularity": "global", "epsilon": 1e-8}},
                             reference_checkpoint=checkpoint)
    joint = torch.load(joint_path, map_location="cpu", weights_only=False)
    assert joint["epoch"] == 1
    config_path = tmp_path / "smoke.yaml"
    config = {"seed": 2, "device": "cpu", "output_dir": str(tmp_path / "eval"),
              "dataset": {"name": "tiny", "adapter": "synthetic", "split_protocol": "synthetic", "factory": "tests.test_core:make_tiny_training_datasets", "kwargs": {}},
              "test_dataset": {"name": "tiny", "adapter": "synthetic", "split_protocol": "synthetic", "factory": "tests.test_core:make_tiny_test_dataset", "kwargs": {}},
              "model": {"factory": "tests.test_core:make_tiny_model", "kwargs": {}},
              "training": {"batch_size": 2, "num_workers": 0, "epochs": 1,
                           "learning_rate": .001, "weight_decay": 0.},
              "contribution": {"fcd_factory": "multimodal.contribution.fcd:InfoNCEFCD",
                               "fcd_kwargs": {"queue_size": 0}}}
    config_path.write_text(yaml.safe_dump(config))
    previous_argv = sys.argv
    try:
        sys.argv = ["evaluate_test.py", "--config", str(config_path), "--checkpoint", str(joint_path),
                    "--allow-legacy-artifacts"]
        evaluate_test.main()
    finally:
        sys.argv = previous_argv
    metrics_path = tmp_path / "eval" / "seed_2" / "test" / "test_metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["split_protocol"] == "synthetic"
    assert metrics["seed"] == 2 and metrics["test_sample_count"] == 4
    assert metrics["audio_video"]["macro_precision"] >= 0
    assert metrics["audio_video"]["per_class_average_precision"]
    prediction_path = metrics_path.with_name("test_predictions.csv")
    with prediction_path.open(newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert len(prediction_rows) == 4
    for row in prediction_rows:
        for condition in ("audio_video", "audio_only", "video_only"):
            probs = [float(row[f"prob_{condition}_class_{i}"]) for i in range(2)]
            assert sum(probs) == pytest.approx(1.)
            assert int(row[f"prediction_{condition}"]) == int(np.argmax(probs))
        assert int(row["true_class_id"]) == int(row["label"])
