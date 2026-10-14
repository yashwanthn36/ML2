import json
import csv

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ave_mm.integrity import audit_manifest
from ave_mm.dataset import AVEDataset
from multimodal.contribution.fcd import InfoNCEFCD
from multimodal.training.device import select_device
from multimodal.models.fusion import TwoModalityFusion
from multimodal.training.provenance import (
    ensure_protocol_output,
    file_sha256,
    make_run_provenance,
    validate_checkpoint,
    validate_downstream_artifact,
)
from multimodal.training.classification_metrics import (
    classification_metrics, identify_minority_classes, validate_probabilities,
)
from multimodal.training.reference import train_reference
from multimodal.training.joint import train_joint
from multimodal.training.selection import select_validation_macro_f1


class _ProtocolTinyDataset(Dataset):
    def __init__(self, protocol, suffix):
        self.split_protocol = protocol
        self.labels = [0, 1, 0, 1]
        self.sample_ids = [f"{protocol}-{suffix}-{i}" for i in range(4)]
        self.source_ids = [f"{protocol}-{suffix}-source-{i}" for i in range(4)]
        self.num_classes = 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index + 31)
        return {"modality_1": torch.randn(3, generator=generator),
                "modality_2": torch.randn(3, generator=generator),
                "label": torch.tensor(self.labels[index]), "sample_id": self.sample_ids[index],
                "source_id": self.source_ids[index]}


class _TokenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4)

    def forward(self, value):
        return self.projection(value).unsqueeze(1)


def _tiny_model():
    return TwoModalityFusion(_TokenEncoder(), _TokenEncoder(), 2, dim=4, heads=2)


@pytest.mark.parametrize(
    ("config_name", "protocol", "output"),
    [
        ("ave_official.yaml", "official", "outputs/ave/official"),
    ],
)
def test_protocol_configs_and_synthetic_artifact_paths(config_name, protocol, output, tmp_path):
    config = yaml.safe_load((__import__("pathlib").Path("configs") / config_name).read_text())
    assert config["dataset"]["split_protocol"] == protocol
    assert config["test_dataset"]["split_protocol"] == protocol
    assert config["output_dir"] == output
    assert config["contribution"]["fcd_kwargs"]["exclude_same_source_negatives"] is True

    # Exercise the real artifact checks with a synthetic run for each protocol.
    training_ids = [f"{protocol}-train-1", f"{protocol}-train-2"]
    config["model"] = {"factory": "synthetic"}
    ref = make_run_provenance(split_protocol=protocol, seed=17, run_id=f"run-{protocol}",
                              training_sample_ids=training_ids, config=config)
    checkpoint = {"run_provenance": ref}
    assert validate_checkpoint(checkpoint, split_protocol=protocol, seed=17, config=config,
                               training_sample_ids=training_ids) == ref
    digest = "a" * 64
    downstream = {**ref, "reference_checkpoint_sha256": digest}
    validate_downstream_artifact(downstream, reference_provenance=ref, reference_sha256=digest,
                                 split_protocol=protocol, seed=17, artifact="synthetic targets")

    protocol_root = tmp_path / protocol
    ensure_protocol_output(protocol_root, split_protocol=protocol, seed=17)
    assert json.loads((protocol_root / "protocol_output.json").read_text())["split_protocol"] == protocol


@pytest.mark.parametrize("config_name,protocol,output", [
    ("ave_official.yaml", "official", "outputs/ave/official"),
])
def test_synthetic_reference_to_joint_path_for_each_protocol(config_name, protocol, output, tmp_path):
    config = yaml.safe_load((__import__("pathlib").Path("configs") / config_name).read_text())
    config["output_dir"] = str(tmp_path / output)
    config["device"] = "cpu"
    train, validation = _ProtocolTinyDataset(protocol, "train"), _ProtocolTinyDataset(protocol, "validation")
    reference_training = {"learning_rate": .01, "weight_decay": 0., "epochs": 1,
                         "modality_dropout": 0., "grad_clip": 1., "early_stopping_patience": 1}
    provenance = make_run_provenance(split_protocol=protocol, seed=17,
        run_id=f"synthetic-{protocol}", training_sample_ids=train.sample_ids, config=config,
        estimation_sample_ids=train.sample_ids[:2], reference_fit_sample_ids=train.sample_ids[2:])
    run_dir = tmp_path / output / "seed_17"
    ensure_protocol_output(run_dir, split_protocol=protocol, seed=17)
    checkpoint_path = train_reference(_tiny_model(), DataLoader(train, batch_size=2),
        DataLoader(validation, batch_size=2), training=reference_training, run_dir=run_dir,
        seed=17, config=config, data_metadata={"split_protocol": protocol, "provenance": provenance},
        device=torch.device("cpu"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checked_reference = validate_checkpoint(checkpoint, split_protocol=protocol, seed=17,
        config=config, training_sample_ids=train.sample_ids)

    digest = file_sha256(checkpoint_path)
    artifact_provenance = {**checked_reference, "reference_checkpoint_sha256": digest}
    targets = {"global": {"target": [.5, .5]}, "classes": {}, "groups": {},
               "provenance": artifact_provenance}
    validate_downstream_artifact(targets["provenance"], reference_provenance=checked_reference,
        reference_sha256=digest, split_protocol=protocol, seed=17, artifact="synthetic targets")
    joint_training = {**reference_training, "optimization_mode": "fixed_weight",
                      "ci_objective": "class_balanced", "lambda_mi": .1}
    joint_config = {**config, "contribution": {"target_granularity": "global", "epsilon": 1e-8}}
    joint_path = train_joint(_tiny_model(), DataLoader(train, batch_size=2),
        DataLoader(validation, batch_size=2), training=joint_training, targets=targets,
        group_assignments=None, fcd_estimator=InfoNCEFCD(queue_size=2),
        run_dir=run_dir / "joint", seed=17, config=joint_config,
        reference_checkpoint=checkpoint, reference_checkpoint_sha256=digest,
        device=torch.device("cpu"))
    joint_checkpoint = torch.load(joint_path, map_location="cpu", weights_only=False)
    assert joint_checkpoint["split_protocol"] == protocol
    assert joint_checkpoint["run_provenance"]["reference_checkpoint_sha256"] == digest


def test_provenance_rejects_checkpoint_protocol_seed_and_config_mismatch():
    config = {"dataset": {"split_protocol": "official"}, "seed": 17}
    provenance = make_run_provenance(split_protocol="official", seed=17, run_id="r1",
                                     training_sample_ids=["a", "b"], config=config)
    checkpoint = {"run_provenance": provenance}
    with pytest.raises(ValueError, match="split_protocol.*expected 'source_video_disjoint'.*got 'official'"):
        validate_checkpoint(checkpoint, split_protocol="source_video_disjoint", seed=17, config=config)
    with pytest.raises(ValueError, match="seed.*expected 18.*got 17"):
        validate_checkpoint(checkpoint, split_protocol="official", seed=18, config=config)
    with pytest.raises(ValueError, match="training_sample_ids_sha256"):
        validate_checkpoint(checkpoint, split_protocol="official", seed=17, config=config,
                             training_sample_ids=["a", "different"])


@pytest.mark.parametrize("artifact", ["contribution target file", "profile group file"])
def test_provenance_rejects_target_or_group_mismatches(artifact):
    config = {"dataset": {"split_protocol": "official"}}
    ref = make_run_provenance(split_protocol="official", seed=17, run_id="r1",
                              training_sample_ids=["a", "b"], config=config)
    downstream = {**ref, "reference_checkpoint_sha256": "expected-checkpoint"}
    bad = {**downstream, "reference_checkpoint_sha256": "different-checkpoint"}
    with pytest.raises(ValueError, match="reference_checkpoint_sha256"):
        validate_downstream_artifact(bad, reference_provenance=ref,
                                     reference_sha256="expected-checkpoint", split_protocol="official",
                                     seed=17, artifact=artifact)


def test_protocol_output_root_collision_is_rejected(tmp_path):
    root = tmp_path / "run"
    ensure_protocol_output(root, split_protocol="official", seed=17)
    with pytest.raises(ValueError, match="belongs to split_protocol='official'.*expects 'source_video_disjoint'"):
        ensure_protocol_output(root, split_protocol="source_video_disjoint", seed=17)
    assert not (root / "source_video_disjoint" / "protocol_output.json").exists()


def test_manifest_audit_reports_duplicates_conflicts_intervals_and_split_leaks():
    rows = [
        {"sample_id": "s1", "video_id": "v1", "label": "A", "label_id": 0,
         "time_start": 0, "time_end": 2, "split": "train"},
        {"sample_id": "s2", "video_id": "v1", "label": "A", "label_id": 0,
         "time_start": 0, "time_end": 2, "split": "train"},  # exact duplicate
        {"sample_id": "s3", "video_id": "v1", "label": "B", "label_id": 1,
         "time_start": 1, "time_end": 3, "split": "val"},  # legitimate distinct but overlaps/conflicts
        {"sample_id": "s4", "video_id": "v2", "label": "C", "label_id": 2,
         "time_start": 4, "time_end": 4, "split": "test"},  # invalid interval
    ]
    report = audit_manifest(rows)
    assert report["row_count"] == 4
    assert report["exact_duplicate_annotations"][0]["sample_ids"] == ["s1", "s2"]
    assert report["duplicate_source_label_interval_combinations"][0]["sample_ids"] == ["s1", "s2"]
    assert report["cross_label_source_conflicts"][0]["sample_ids"] == ["s1", "s2", "s3"]
    assert any(x["sample_ids"] == ["s1", "s3"] for x in report["overlapping_or_identical_intervals"])
    assert report["invalid_intervals"][0]["sample_id"] == "s4"
    assert report["source_id_overlap"]["train_val"] == ["v1"]
    assert "No rows are removed" in report["duplicate_training_policy"]


def test_manifest_audit_preserves_legitimate_nonoverlapping_segments_and_checks_summary():
    rows = [
        {"sample_id": "s1", "video_id": "v1", "label": "A", "label_id": 0,
         "time_start": 0, "time_end": 1, "split": "train"},
        {"sample_id": "s2", "video_id": "v1", "label": "A", "label_id": 0,
         "time_start": 2, "time_end": 3, "split": "train"},
        {"sample_id": "s3", "video_id": "v2", "label": "B", "label_id": 1,
         "time_start": 0, "time_end": 1, "split": "val"},
        {"sample_id": "s4", "video_id": "v3", "label": "B", "label_id": 1,
         "time_start": 0, "time_end": 1, "split": "test"},
    ]
    summary = {"protocol": "source_video_disjoint", "split_summary": {
        "train": {"segment_count": 2, "source_video_count": 1, "class_counts": {"0": 2, "1": 0}, "missing_classes": ["1"]},
        "val": {"segment_count": 1, "source_video_count": 1, "class_counts": {"0": 0, "1": 1}, "missing_classes": ["0"]},
        "test": {"segment_count": 1, "source_video_count": 1, "class_counts": {"0": 0, "1": 1}, "missing_classes": ["0"]},
    }, "source_id_overlap": {"train_val": [], "train_test": [], "val_test": []}}
    report = audit_manifest(rows, summary=summary)
    assert report["exact_duplicate_annotations"] == []
    assert report["overlapping_or_identical_intervals"] == []
    assert report["summary_consistent"]
    assert all(not overlap for overlap in report["source_id_overlap"].values())
    summary["split_summary"]["train"]["segment_count"] = 99
    assert not audit_manifest(rows, summary=summary)["summary_consistent"]


def test_ave_label_id_mapping_is_identical_across_splits(tmp_path):
    manifest = tmp_path / "mapping.csv"
    fields = ["sample_id", "video_id", "label", "label_id", "split", "time_start", "time_end",
              "video_path", "audio_path"]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([
            {"sample_id": "a", "video_id": "v1", "label": "alpha", "label_id": 0,
             "split": "train", "time_start": 0, "time_end": 1, "video_path": "v1.mp4", "audio_path": "v1.wav"},
            {"sample_id": "b", "video_id": "v2", "label": "beta", "label_id": 1,
             "split": "val", "time_start": 0, "time_end": 1, "video_path": "v2.mp4", "audio_path": "v2.wav"},
            {"sample_id": "c", "video_id": "v3", "label": "alpha", "label_id": 0,
             "split": "test", "time_start": 0, "time_end": 1, "video_path": "v3.mp4", "audio_path": "v3.wav"},
        ])
    datasets = [AVEDataset(manifest, split, root=tmp_path) for split in ("train", "val", "test")]
    assert [dataset.class_names for dataset in datasets] == [["alpha", "beta"]] * 3
    assert [dataset.class_to_id for dataset in datasets] == [{"alpha": 0, "beta": 1}] * 3


def test_fcd_same_source_and_sample_id_exclusions_cover_current_batch_and_queue():
    estimator = InfoNCEFCD(queue_size=4, exclude_same_source_negatives=True)
    fused = torch.eye(2)
    x1 = torch.eye(2)
    x2 = torch.eye(2)
    # Distinct sample/source IDs make a valid negative.
    information = estimator.per_sample_information(fused, x1, x2,
        sample_ids=["a", "b"], source_ids=["v1", "v2"])
    assert torch.isfinite(information).all()
    with pytest.raises(ValueError, match="fewer than two valid candidates"):
        estimator.per_sample_information(fused, x1, x2,
            sample_ids=["same", "same"], source_ids=["v1", "v2"])
    # Same-source sibling is not a negative; with a singleton batch no negatives remain.
    with pytest.raises(ValueError, match="fewer than two valid candidates"):
        estimator.per_sample_information(fused[:1], x1[:1], x2[:1], sample_ids=["c"], source_ids=["v3"])
    estimator._remember(torch.tensor([[0., 1.]]), ["queued"], ["v1"])
    with pytest.raises(ValueError, match="fewer than two valid candidates"):
        estimator.per_sample_information(fused[:1], x1[:1], x2[:1], sample_ids=["new"], source_ids=["v1"])
    estimator.reset_queue()
    assert estimator.queue_count == 0


def test_fcd_fallback_without_source_ids_and_device_selection(monkeypatch):
    estimator = InfoNCEFCD(queue_size=0, exclude_same_source_negatives=True)
    values = torch.eye(2)
    assert torch.isfinite(estimator.per_sample_information(values, values, values,
                                                           sample_ids=["a", "b"])).all()
    assert select_device("cpu", log=False).type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was explicitly requested.*false"):
        select_device("cuda", log=False)


def test_minority_classes_use_training_counts_only_and_include_quantile_ties():
    result = identify_minority_classes([0] * 10 + [1] * 5 + [2] + [3],
                                       class_names=["major", "mid", "minor-a", "minor-b"])
    assert result["minority_class_ids"] == [2, 3]
    assert result["training_class_counts"] == {"0": 10, "1": 5, "2": 1, "3": 1}
    assert result["minority_class_names"] == ["minor-a", "minor-b"]
    assert "training count" in result["rule"]


def test_metrics_per_class_ap_confusion_and_undefined_missing_positive():
    names = ["cat", "dog", "bird"]
    y_true = [0, 0, 1, 1]
    probs = torch.tensor([[.8, .1, .1], [.4, .5, .1], [.2, .7, .1], [.6, .3, .1]]).numpy()
    metrics = classification_metrics(y_true, probs, class_names=names, minority_class_ids=[1, 2])
    assert metrics["accuracy"] == pytest.approx(.5)
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(.5)
    assert metrics["macro_precision"] == pytest.approx(1 / 3)
    assert metrics["macro_recall"] == pytest.approx(1 / 3)
    assert metrics["weighted_f1"] == pytest.approx(.5)
    assert metrics["per_class"][0]["class_name"] == "cat"
    assert metrics["per_class"][0]["precision"] == pytest.approx(.5)
    assert metrics["per_class"][0]["recall"] == pytest.approx(.5)
    assert metrics["per_class"][0]["f1"] == pytest.approx(.5)
    assert metrics["per_class"][0]["support"] == 2
    assert metrics["per_class"][0]["average_precision"] == pytest.approx(5 / 6)
    assert metrics["per_class"][2]["average_precision"] is None
    assert metrics["macro_average_precision"] is not None
    assert metrics["confusion_matrix"] == [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
    assert metrics["confusion_matrix_row_normalized"] == [[.5, .5, 0.], [.5, .5, 0.], [0., 0., 0.]]
    assert metrics["minority_macro_f1"] == pytest.approx(.25)
    assert metrics["minority_macro_recall"] == pytest.approx(.25)
    with pytest.raises(ValueError, match="sum to one"):
        validate_probabilities(torch.tensor([[.2, .2]]).numpy(), num_classes=2)
    with pytest.raises(ValueError, match="non-finite"):
        validate_probabilities(torch.tensor([[float("nan"), 0.]]).numpy(), num_classes=2)


def test_ci_only_training_never_calls_mi_estimator_and_updates_ci_parameters(tmp_path):
    train, validation = _ProtocolTinyDataset("official", "train"), _ProtocolTinyDataset("official", "val")
    reference_model = _tiny_model()
    reference_model(torch.randn(2, 3), torch.randn(2, 3))
    reference = {"model_state_dict": reference_model.state_dict(), "split_protocol": "official"}
    before = {key: value.clone() for key, value in reference["model_state_dict"].items()}

    class ForbiddenMI:
        def reset_queue(self):
            raise AssertionError("CI-only training must not reset or use an MI estimator")
        def per_sample_information(self, *args, **kwargs):
            raise AssertionError("CI-only training must not estimate MI")
        def distribution(self, *args, **kwargs):
            raise AssertionError("CI-only training must not build contribution distributions")

    run_config = {"seed": 17, "device": "cpu", "dataset": {"split_protocol": "official"},
                  "contribution": {"target_granularity": "global"}}
    run_dir = tmp_path / "ci_only"
    checkpoint_path = train_joint(_tiny_model(), DataLoader(train, batch_size=2),
        DataLoader(validation, batch_size=2), training={"learning_rate": .02, "weight_decay": 0.,
            "epochs": 1, "modality_dropout": 0., "grad_clip": 1., "early_stopping_patience": 1,
            "optimization_mode": "fixed_weight", "ci_objective": "class_balanced",
            "training_objective": "ci_only"}, targets=None, group_assignments=None,
        fcd_estimator=ForbiddenMI(), run_dir=run_dir, seed=17, config=run_config,
        reference_checkpoint=reference, device=torch.device("cpu"))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["model_condition"] == "ci_only"
    assert payload["training_objective"] == "ci_only"
    assert payload["lambda_mi"] is None
    assert any(not torch.equal(before[key], value) for key, value in payload["model_state_dict"].items())
    record = json.loads((run_dir / "training_log.jsonl").read_text().splitlines()[0])
    assert record["mi_objective_applied"] is False
    assert record["train_loss_mi"] is None


def test_lambda_selection_uses_validation_macro_f1_only():
    selected = select_validation_macro_f1([
        {"lambda_mi": .01, "validation_macro_f1": .72, "test_macro_f1": .1},
        {"lambda_mi": 10., "validation_macro_f1": .61, "test_macro_f1": 1.},
    ])
    assert selected["lambda_mi"] == .01

