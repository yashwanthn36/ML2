"""Generic entry point; dataset and model adapters are selected by config."""
from __future__ import annotations
import argparse
import importlib
import json
import sys
import uuid
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from multimodal.training import set_seed, train_reference
from multimodal.training.reference import seed_data_worker
from multimodal.data_adapters.contract import validate_split_datasets
from multimodal.training.splits import grouped_reference_holdout
from multimodal.training.provenance import (make_run_provenance, configured_split_protocol,
    effective_split_protocol, ensure_protocol_output, requires_source_disjoint)
from multimodal.training.device import select_device
from multimodal.training.configuration import validate_pipeline_config


def load_factory(spec: str):
    module_name, function_name = spec.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), function_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a paired-modality reference model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_pipeline_config(config, path=config_path)
    configured_protocol = configured_split_protocol(config)
    if args.device:
        config["device"] = args.device
    selected_device = select_device(str(config.get("device", "auto")), log=False)
    seed = int(config["seed"])
    set_seed(seed)

    dataset_cfg = config["dataset"]
    dataset_kwargs = dict(dataset_cfg.get("kwargs", {}))
    if "split_protocol" in dataset_cfg:
        dataset_kwargs["split_protocol"] = dataset_cfg["split_protocol"]
    train_set, validation_set = load_factory(dataset_cfg["factory"])(**dataset_kwargs)
    effective_split_protocol(train_set, configured_protocol)
    contract = validate_split_datasets(train_set, validation_set,
        require_source_disjoint=requires_source_disjoint(config, configured_protocol))
    training_cfg = config["training"]
    estimation_fraction = float(training_cfg.get("contribution_estimation_fraction", 0.2))
    if not 0.0 <= estimation_fraction < 1.0:
        raise ValueError("contribution_estimation_fraction must be in [0, 1)")
    if estimation_fraction:
        if not hasattr(train_set, "labels") or not hasattr(train_set, "sample_ids"):
            raise ValueError("Dataset adapter must expose training-only labels and sample_ids for the holdout split")
        source_ids = getattr(train_set, "source_ids", None)
        reference_indices, estimation_indices, contribution_metadata = grouped_reference_holdout(
            train_set.labels, train_set.sample_ids, source_ids=source_ids,
            estimation_fraction=estimation_fraction, seed=seed,
        )
        reference_train = Subset(train_set, reference_indices)
        estimation_ids = [train_set.sample_ids[i] for i in estimation_indices]
    else:
        if not training_cfg.get("allow_in_sample_contributions", False):
            raise ValueError("Set contribution_estimation_fraction > 0 for independent signatures; "
                             "in-sample estimation requires allow_in_sample_contributions: true")
        reference_train = train_set
        estimation_ids = list(train_set.sample_ids)
        estimation_indices = list(range(len(train_set)))
        contribution_metadata = {"protocol": "explicit_in_sample_fallback", "fit_sample_ids": list(train_set.sample_ids),
                                 "estimation_sample_ids": list(train_set.sample_ids),
                                 "assignment_sample_ids": list(train_set.sample_ids),
                                 "limitations": ["Signatures are in-sample and may be optimistic; do not treat "
                                                 "them as independent estimates."]}
    loader_options = {"batch_size": int(training_cfg["batch_size"]),
                      "num_workers": int(training_cfg["num_workers"]),
                      "pin_memory": selected_device.type == "cuda"}
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(reference_train, shuffle=True, worker_init_fn=seed_data_worker,
                              generator=generator, **loader_options)
    validation_loader = DataLoader(validation_set, shuffle=False, worker_init_fn=seed_data_worker,
                                   generator=torch.Generator().manual_seed(seed + 1), **loader_options)

    model_cfg = config["model"]
    model = load_factory(model_cfg["factory"])(
        num_classes=train_set.num_classes, **model_cfg.get("kwargs", {})
    )
    split_protocol = getattr(train_set, "split_protocol", dataset_cfg.get("split_protocol", "unknown"))
    output = ensure_protocol_output(PROJECT_ROOT / config["output_dir"] / f"seed_{seed}",
                                    split_protocol=split_protocol, seed=seed)
    run_provenance = make_run_provenance(
        split_protocol=split_protocol, seed=seed, run_id=uuid.uuid4().hex,
        training_sample_ids=list(train_set.sample_ids), config=config,
        estimation_sample_ids=estimation_ids,
        reference_fit_sample_ids=contribution_metadata.get("fit_sample_ids", []),
        dataset_metadata=getattr(train_set, "metadata", {}), class_to_id=train_set.class_to_id,
    )
    run_provenance["dataset_contract"] = contract
    training_class_counts = torch.bincount(torch.as_tensor(train_set.labels, dtype=torch.long),
                                           minlength=train_set.num_classes).tolist()
    (output / "training_estimation_ids.json").write_text(
        json.dumps({**contribution_metadata, "indices": estimation_indices, "sample_ids": estimation_ids,
                    "split_protocol": split_protocol, "seed": seed,
                    "training_sample_ids": list(train_set.sample_ids),
                    "provenance": run_provenance},
                   indent=2),
        encoding="utf-8",
    )
    checkpoint = train_reference(model, train_loader, validation_loader,
                                 training=training_cfg, run_dir=output, seed=seed, config=config,
                                 data_metadata={**contribution_metadata, "provenance": run_provenance,
                                                "dataset_metadata": getattr(train_set, "metadata", {}),
                                                "dataset_contract": contract,
                                                "class_to_id": train_set.class_to_id,
                                                "training_class_counts": training_class_counts},
                                 device=selected_device)
    print(f"selected checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
