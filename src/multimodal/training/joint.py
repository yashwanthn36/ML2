"""Second-stage CI/MI optimization with fixed-weight or minimum-norm MOO."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from multimodal.losses.alignment import target_to_current_kl
from multimodal.losses.balanced import class_balanced_ce, group_balanced_ce
from multimodal.optim import min_norm_combine, gradient_diagnostics, normalize_gradients
from multimodal.training.metadata import experiment_metadata
from multimodal.training.device import assert_module_device, select_device, on_device, log_device
from multimodal.training.provenance import effective_split_protocol


def _scope_alignment(information: torch.Tensor, labels: torch.Tensor,
                     group_ids: torch.Tensor | None, targets: dict[str, Any],
                     granularity: str, epsilon: float, fcd_estimator: Any,
                     diagnostic_rows: list[dict[str, Any]] | None = None) -> torch.Tensor:
    if granularity == "global":
        scopes = [(torch.ones_like(labels, dtype=torch.bool), targets["global"]["target"])]
    elif granularity == "class":
        scopes = [(labels == label, targets["classes"][str(int(label))]["target"])
                  for label in torch.unique(labels)]
    elif granularity == "group":
        if group_ids is None:
            raise ValueError("group granularity requires group IDs for every batch item")
        scopes = []
        for label in torch.unique(labels):
            class_mask = labels == label
            for group_id in torch.unique(group_ids[class_mask]):
                mask = class_mask & (group_ids == group_id)
                key = f"{int(label)}:{int(group_id)}"
                scoped = targets["groups"].get(key, targets["classes"][str(int(label))])
                scopes.append((mask, scoped["target"]))
    elif granularity == "reliability":
        if group_ids is None:
            raise ValueError("reliability granularity requires group IDs for every batch item")
        scopes = []
        global_target = targets["global"]["target"]
        for label in torch.unique(labels):
            class_mask = labels == label
            class_entry = targets["classes"].get(str(int(label)), {})
            for group_id in torch.unique(group_ids[class_mask]):
                mask = class_mask & (group_ids == group_id)
                key = f"{int(label)}:{int(group_id)}"
                group_entry = targets.get("groups", {}).get(key, {})
                if group_entry.get("stable") and not group_entry.get("fallback"):
                    selected = group_entry
                    selected_level = "group"
                elif class_entry.get("stable") and not class_entry.get("fallback"):
                    selected = class_entry
                    selected_level = "class"
                else:
                    selected = targets["global"]
                    selected_level = "global"
                scopes.append((mask, selected["target"], selected_level))
    else:
        raise ValueError(f"Unsupported target granularity: {granularity}")
    losses = []
    for scope in scopes:
        mask, target = scope[:2]
        if mask.any():
            current_distribution = fcd_estimator.distribution(information[mask])
            target_tensor = torch.as_tensor(target, dtype=information.dtype, device=information.device)
            losses.append(target_to_current_kl(target_tensor, current_distribution, epsilon))
            if diagnostic_rows is not None:
                diagnostic_rows.append({"target": target_tensor.detach(),
                    "current": current_distribution.detach(), "scope": scope[2] if len(scope) > 2 else granularity})
    return torch.stack(losses).mean()


def train_joint(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader,
                *, training: dict[str, Any], targets: dict[str, Any] | None,
                group_assignments: dict[str, int] | None,
                fcd_estimator: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None,
                run_dir: Path, seed: int, config: dict[str, Any],
                reference_checkpoint: dict[str, Any] | None = None,
                reference_checkpoint_sha256: str | None = None,
                device: torch.device | None = None) -> Path:
    """Optimize selected CI/MI objectives; the FCD implementation is injected explicitly."""
    from multimodal.training.reference import evaluate

    if device is None:
        device = select_device(str(config.get("device", "auto")))
    else:
        log_device(device)
    model = model.to(device)
    assert_module_device(model, device)
    if reference_checkpoint is None:
        raise ValueError("Second-stage optimization must initialize from the reference checkpoint")
    model.load_state_dict(reference_checkpoint["model_state_dict"])
    # The architecture's second stage optimizes projections, attention, fusion,
    # and the classifier while keeping both reference encoders fixed.
    if not hasattr(model, "encoder_1") or not hasattr(model, "encoder_2"):
        raise ValueError("The paired-modality model must expose encoder_1 and encoder_2 for stage-two freezing")
    model.encoder_1.requires_grad_(False)
    model.encoder_2.requires_grad_(False)
    model.freeze_encoders = True
    model.encoder_1.eval()
    model.encoder_2.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable fusion/projection parameters remain for stage-two optimization")
    optimizer = torch.optim.AdamW(parameters, lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    mode = str(training.get("optimization_mode", "fixed_weight"))
    ci_mode = str(training.get("ci_objective", "class_balanced"))
    objective = str(training.get("training_objective", "ci_mi"))
    if objective not in {"ci_only", "ci_mi"}:
        raise ValueError("training_objective must be 'ci_only' or 'ci_mi'")
    if objective == "ci_mi" and targets is None:
        raise ValueError("The proposed CI + MI objective requires contribution targets")
    if objective == "ci_mi" and fcd_estimator is None:
        raise ValueError("The proposed CI + MI objective requires an FCD estimator")
    if objective == "ci_only" and ci_mode != "class_balanced":
        raise ValueError("The controlled CI-only baseline requires ci_objective='class_balanced'")
    model_condition = "ci_only" if objective == "ci_only" else "proposed_ci_mi"
    if objective == "ci_mi" and hasattr(fcd_estimator, "reset_queue"):
        fcd_estimator.reset_queue()
    granularity = str(config.get("contribution", {}).get("target_granularity", "global"))
    split_protocol = effective_split_protocol(train_loader.dataset,
        config.get("dataset", {}).get("split_protocol"))
    epsilon = float(config.get("contribution", {}).get("epsilon", 1e-8))
    lambda_mi = float(training.get("lambda_mi", 0.1)) if objective == "ci_mi" else None
    run_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = experiment_metadata(config)
    run_metadata["effective_config"] = config
    run_metadata["device"] = str(device)
    run_metadata["model_condition"] = model_condition
    run_metadata["dataset_metadata"] = reference_checkpoint.get("data_metadata", {}).get("dataset_metadata", {})
    run_metadata["class_to_id"] = reference_checkpoint.get("data_metadata", {}).get("class_to_id", {})
    (run_dir / "experiment_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    run_provenance = {**(reference_checkpoint.get("run_provenance") or {}),
                      "reference_checkpoint_sha256": reference_checkpoint_sha256}
    checkpoint_path = run_dir / "joint_best.pt"
    best_f1, stale = -1.0, 0
    group_assignments = group_assignments or {}
    dataset = train_loader.dataset
    class_counts: dict[int, int] = {}
    group_counts: dict[tuple[int, int], int] = {}
    groups_per_class: dict[int, int] = {}
    if hasattr(dataset, "labels"):
        for label in dataset.labels:
            class_counts[int(label)] = class_counts.get(int(label), 0) + 1
        if group_assignments and hasattr(dataset, "sample_ids"):
            for label, sample_id in zip(dataset.labels, dataset.sample_ids):
                group_id = group_assignments.get(str(sample_id))
                if group_id is not None:
                    key = (int(label), int(group_id))
                    group_counts[key] = group_counts.get(key, 0) + 1
            for label, _ in group_counts:
                groups_per_class[label] = groups_per_class.get(label, 0) + 1
    history_path = run_dir / "training_log.jsonl"
    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(1, int(training["epochs"]) + 1):
            model.train()
            sums = {"loss_ci": 0.0, "loss_mi": 0.0, "steps": 0, "negative_cosine": 0,
                    "gradient_cosine": 0.0, "gradient_norm_ci": 0.0, "gradient_norm_mi": 0.0,
                    "gradient_count": 0, "weighted_alignment_norm": 0.0, "moo_ci_weight": 0.0,
                    "fcd_1": 0.0, "fcd_2": 0.0, "target_1": 0.0, "target_2": 0.0,
                    "scope_count": 0, "mi_bound_1": 0.0, "mi_bound_2": 0.0,
                    "negative_bound_count": 0, "bound_value_count": 0, "queue_count_max": 0}
            for batch in train_loader:
                x1 = batch["modality_1"].to(device, non_blocking=True)
                x2 = batch["modality_2"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                if not all(on_device(tensor, device) for tensor in (x1, x2, labels)):
                    raise RuntimeError(f"Training batch is not on selected device {device}")
                fused, modality_1, modality_2 = model.fused_features(x1, x2)
                logits = model.classifier(fused)
                ids = batch.get("sample_id", [])
                group_ids = None
                if ci_mode == "group_balanced" or granularity in {"group", "reliability"}:
                    if not ids or not group_assignments:
                        raise ValueError("Group objectives require saved training group assignments")
                    missing_ids = [str(sample_id) for sample_id in ids
                                   if str(sample_id) not in group_assignments]
                    if missing_ids:
                        raise ValueError(f"Missing group assignments for {len(missing_ids)} batch samples; "
                                         f"examples: {missing_ids[:5]}")
                    group_ids = torch.tensor([group_assignments[str(sample_id)] for sample_id in ids],
                                             device=device, dtype=torch.long)
                if ci_mode == "group_balanced":
                    loss_ci = group_balanced_ce(logits, labels, group_ids,
                                                group_counts or None, groups_per_class or None)
                elif ci_mode == "class_balanced":
                    loss_ci = class_balanced_ce(logits, labels, class_counts or None)
                else:
                    loss_ci = nn.functional.cross_entropy(logits, labels)
                loss_mi = None
                scope_diagnostics: list[dict[str, Any]] = []
                if objective == "ci_mi":
                    if not hasattr(fcd_estimator, "per_sample_information") or not hasattr(fcd_estimator, "distribution"):
                        raise TypeError("FCD estimator must provide per_sample_information and distribution methods")
                    information = fcd_estimator.per_sample_information(
                        fused, modality_1, modality_2, update_queue=True,
                        sample_ids=[str(sample_id) for sample_id in ids] if ids else None,
                        source_ids=[str(source_id) for source_id in batch.get("source_id", [])]
                        if batch.get("source_id") else None,
                    )
                    if information.shape != (labels.shape[0], 2):
                        raise ValueError("FCD information estimator must return signed [batch, 2] MI bounds")
                    loss_mi = _scope_alignment(information, labels, group_ids, targets,
                                               granularity, epsilon, fcd_estimator,
                                               diagnostic_rows=scope_diagnostics)
                    sums["mi_bound_1"] += float(information[:, 0].detach().mean())
                    sums["mi_bound_2"] += float(information[:, 1].detach().mean())
                    sums["negative_bound_count"] += int((information.detach() < 0).sum())
                    sums["bound_value_count"] += information.numel()
                    for scope_record in scope_diagnostics:
                        sums["fcd_1"] += float(scope_record["current"][0])
                        sums["fcd_2"] += float(scope_record["current"][1])
                        sums["target_1"] += float(scope_record["target"][0])
                        sums["target_2"] += float(scope_record["target"][1])
                        sums["scope_count"] += 1
                    sums["queue_count_max"] = max(sums["queue_count_max"],
                        int(getattr(fcd_estimator, "queue_count", 0)))
                optimizer.zero_grad(set_to_none=True)
                diagnostics = {}
                if objective == "ci_only":
                    loss_ci.backward()
                elif mode == "fixed_weight":
                    diagnostic_interval = int(training.get("gradient_diagnostics_interval", 0))
                    if (objective == "ci_mi" and diagnostic_interval > 0 and
                            sums["steps"] % diagnostic_interval == 0):
                        grad_ci = torch.autograd.grad(loss_ci, parameters, retain_graph=True, allow_unused=True)
                        grad_mi = torch.autograd.grad(loss_mi, parameters, retain_graph=True, allow_unused=True)
                        diagnostics = gradient_diagnostics(grad_ci, grad_mi)
                        sums["gradient_cosine"] += diagnostics["cosine"]
                        sums["gradient_norm_ci"] += diagnostics["norm_ci"]
                        sums["gradient_norm_mi"] += diagnostics["norm_mi"]
                        sums["weighted_alignment_norm"] += lambda_mi * diagnostics["norm_mi"]
                        sums["negative_cosine"] += int(diagnostics["cosine"] < 0.0)
                        sums["gradient_count"] += 1
                    (loss_ci + lambda_mi * loss_mi).backward()
                elif mode in {"moo", "normalized_moo"}:
                    gradients_ci = torch.autograd.grad(loss_ci, parameters, retain_graph=True, allow_unused=True)
                    gradients_mi = torch.autograd.grad(loss_mi, parameters, retain_graph=False, allow_unused=True)
                    diagnostics = gradient_diagnostics(gradients_ci, gradients_mi)
                    sums["gradient_cosine"] += diagnostics["cosine"]
                    sums["gradient_norm_ci"] += diagnostics["norm_ci"]
                    sums["gradient_norm_mi"] += diagnostics["norm_mi"]
                    sums["weighted_alignment_norm"] += diagnostics["norm_mi"]
                    sums["gradient_count"] += 1
                    if mode == "normalized_moo":
                        gradients_ci = normalize_gradients(gradients_ci)
                        gradients_mi = normalize_gradients(gradients_mi)
                    alpha, combined = min_norm_combine(gradients_ci, gradients_mi)
                    sums["moo_ci_weight"] += float(alpha.detach())
                    for parameter, gradient in zip(parameters, combined):
                        parameter.grad = gradient
                    sums["negative_cosine"] += int(diagnostics["cosine"] < 0.0)
                else:
                    raise ValueError("optimization_mode must be 'fixed_weight', 'moo' or 'normalized_moo'")
                torch.nn.utils.clip_grad_norm_(parameters, float(training["grad_clip"]))
                optimizer.step()
                sums["loss_ci"] += float(loss_ci.detach())
                if loss_mi is not None:
                    sums["loss_mi"] += float(loss_mi.detach())
                sums["steps"] += 1
            training_counts = torch.bincount(torch.as_tensor(dataset.labels, dtype=torch.long),
                                             minlength=getattr(dataset, "num_classes", 0)).tolist() \
                if hasattr(dataset, "labels") else []
            minority_ids = []
            if training_counts:
                import numpy as np
                threshold = float(np.quantile(training_counts,
                    float(config.get("evaluation", {}).get("minority_bottom_quantile", 0.25))))
                minority_ids = [index for index, count in enumerate(training_counts) if count <= threshold]
            metrics = evaluate(model, validation_loader, device, minority_class_ids=minority_ids)
            record = {"epoch": epoch, "seed": seed, "mode": mode, "ci_objective": ci_mode,
                      "training_objective": objective, "model_condition": model_condition,
                      "split_protocol": split_protocol,
                      "device": str(device), "run_provenance": run_provenance,
                      "granularity": granularity, "lambda_mi": lambda_mi,
                      "mi_objective_applied": objective == "ci_mi",
                      "train_loss_ci": sums["loss_ci"] / max(sums["steps"], 1),
                      "train_loss_mi": (sums["loss_mi"] / max(sums["steps"], 1)
                                        if objective == "ci_mi" else None),
                      "weighted_alignment_loss": (lambda_mi * sums["loss_mi"] / max(sums["steps"], 1)
                                                  if objective == "ci_mi" and mode == "fixed_weight" else None),
                      "fcd_proportion_mean": ([sums["fcd_1"] / max(sums["scope_count"], 1),
                                               sums["fcd_2"] / max(sums["scope_count"], 1)]
                                              if sums["scope_count"] else None),
                      "ucd_target_proportion_mean": ([sums["target_1"] / max(sums["scope_count"], 1),
                                                       sums["target_2"] / max(sums["scope_count"], 1)]
                                                      if sums["scope_count"] else None),
                      "mi_lower_bound_mean": ([sums["mi_bound_1"] / max(sums["steps"], 1),
                                               sums["mi_bound_2"] / max(sums["steps"], 1)]
                                              if objective == "ci_mi" else None),
                      "negative_mi_bound_fraction": (sums["negative_bound_count"] /
                          max(sums["bound_value_count"], 1)
                          if objective == "ci_mi" else None),
                      "alignment_scope_count": sums["scope_count"],
                      "queue_count_max": sums["queue_count_max"],
                      "weighted_alignment_gradient_norm_mean": (sums["weighted_alignment_norm"] /
                          max(sums["gradient_count"], 1) if sums["gradient_count"] else None),
                      "gradient_cosine_mean": (sums["gradient_cosine"] / max(sums["gradient_count"], 1)
                                                if sums["gradient_count"] else None),
                      "gradient_norm_ci_mean": (sums["gradient_norm_ci"] / max(sums["gradient_count"], 1)
                                                 if sums["gradient_count"] else None),
                      "gradient_norm_mi_mean": (sums["gradient_norm_mi"] / max(sums["gradient_count"], 1)
                                                 if sums["gradient_count"] else None),
                      "moo_ci_weight_mean": (sums["moo_ci_weight"] / max(sums["gradient_count"], 1)
                                             if sums["gradient_count"] else None),
                      "negative_cosine_fraction": (sums["negative_cosine"] / max(sums["gradient_count"], 1)
                                                     if sums["gradient_count"] else None),
                      "checkpoint_selection_metric": "val_macro_f1",
                      **{f"val_{key}": value for key, value in metrics.items()}}
            if metrics["macro_f1"] > best_f1:
                best_f1, stale = metrics["macro_f1"], 0
                record["checkpoint_selected"] = True
                torch.save({"model_state_dict": model.state_dict(), "config": config,
                            "seed": seed, "epoch": epoch, "validation_metrics": metrics,
                            "optimization_mode": mode, "split_protocol": split_protocol,
                            "experiment_metadata": run_metadata, "device": str(device),
                            "run_provenance": run_provenance, "model_condition": model_condition,
                            "training_objective": objective, "lambda_mi": lambda_mi,
                            "checkpoint_selection_metric": "val_macro_f1"}, checkpoint_path)
            else:
                record["checkpoint_selected"] = False
                stale += 1
            history.write(json.dumps(record) + "\n")
            history.flush()
            print(json.dumps(record))
            if not record["checkpoint_selected"] and stale >= int(training["early_stopping_patience"]):
                break
    return checkpoint_path
