"""Create protocol-specific tables and figures from completed evaluation artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from multimodal.training.classification_metrics import validate_probabilities
from multimodal.training.provenance import file_sha256

MODELS = ("reference", "ci_only", "proposed_ci_mi")
CONDITIONS = ("audio_video", "audio_only", "video_only")
DISPLAY = {"reference": "Reference", "ci_only": "CI-only", "proposed_ci_mi": "Proposed CI + MI"}


def resolve(path: str | Path, *, base: Path = ROOT) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else base / value).resolve()


def parse_results(values: list[str], protocol: str, selected_seed: int | None):
    collected: dict[str, list[dict]] = defaultdict(list)
    for item in values:
        if "=" not in item:
            raise ValueError(f"--result must be MODEL=METRICS_JSON, got {item!r}")
        model, raw_path = item.split("=", 1)
        if model not in MODELS:
            raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")
        path = resolve(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation result is missing: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        actual_protocol = metrics.get("split_protocol")
        if actual_protocol != protocol:
            raise ValueError(f"{path} uses split_protocol={actual_protocol!r}; requested {protocol!r}")
        declared_condition = metrics.get("model_condition")
        if declared_condition is not None and declared_condition != model:
            raise ValueError(f"{path} records model_condition={declared_condition!r}; supplied as {model!r}")
        seed = int(metrics.get("experiment_seed", metrics.get("seed", -1)))
        if seed < 0:
            raise ValueError(f"{path} has no valid seed metadata")
        if selected_seed is not None and seed != selected_seed:
            raise ValueError(f"{path} uses seed={seed}; requested seed={selected_seed}")
        prediction_path = path.parent / "test_predictions.csv"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Predictions required for plots are missing: {prediction_path}")
        item_data = {"model": model, "metrics_path": path, "metrics": metrics,
                     "predictions_path": prediction_path, "seed": seed}
        item_data["predictions"] = read_predictions(prediction_path, metrics)
        collected[model].append(item_data)
    missing = [model for model in MODELS if not collected[model]]
    if missing:
        raise ValueError(f"Missing completed evaluation results for model conditions: {missing}")
    seed_sets = {model: {item["seed"] for item in collected[model]} for model in MODELS}
    if len({tuple(sorted(seeds)) for seeds in seed_sets.values()}) != 1:
        raise ValueError(f"Model conditions must have results for the same seed set: {seed_sets}")
    for model in MODELS:
        collected[model].sort(key=lambda item: item["seed"])
        model_seeds = [item["seed"] for item in collected[model]]
        if len(model_seeds) != len(set(model_seeds)):
            raise ValueError(f"Duplicate evaluation result supplied for {model} seed(s): {model_seeds}")
    return collected


def read_predictions(path: Path, metrics: dict) -> dict:
    class_names = list(map(str, metrics["class_names"]))
    num_classes = len(class_names)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(metrics.get("test_sample_count", metrics.get("test_size", -1))):
        raise ValueError(f"Prediction count in {path} does not match its metric metadata")
    labels, ids, sources = [], [], []
    probabilities = {condition: [] for condition in CONDITIONS}
    for row in rows:
        label_key = "true_class_id" if "true_class_id" in row else "label"
        name_key = "true_class_name" if "true_class_name" in row else "label_name"
        label = int(row[label_key])
        if not 0 <= label < num_classes:
            raise ValueError(f"Invalid class ID {label} in {path}")
        if str(row[name_key]) != class_names[label]:
            raise ValueError(f"Class name/ID mismatch in {path} for sample {row['sample_id']!r}")
        labels.append(label)
        ids.append(str(row["sample_id"]))
        sources.append(str(row.get("source_id", "")))
        for condition in CONDITIONS:
            keys = [f"prob_{condition}_class_{class_id}" for class_id in range(num_classes)]
            if any(key not in row for key in keys):
                raise ValueError(f"{path} lacks probability columns for {condition}; re-run test evaluation")
            values = np.asarray([float(row[key]) for key in keys], dtype=np.float64)
            probabilities[condition].append(values)
            prediction_key = f"prediction_{condition}"
            if prediction_key not in row or int(row[prediction_key]) != int(values.argmax()):
                raise ValueError(f"Saved prediction does not match probabilities for {row['sample_id']} ({condition})")
    return {"rows": rows, "labels": np.asarray(labels, dtype=np.int64), "ids": ids, "sources": sources,
            "probabilities": {key: validate_probabilities(np.asarray(value), num_classes=num_classes)
                              for key, value in probabilities.items()}}


def mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    valid = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if not len(valid):
        return None, None
    return float(valid.mean()), float(valid.std(ddof=1)) if len(valid) > 1 else None


def aggregate_metrics(records: list[dict], condition: str) -> dict:
    keys = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
            "balanced_accuracy", "minority_macro_f1", "minority_macro_recall", "macro_average_precision")
    return {key: mean_std([record["metrics"][condition].get(key) for record in records]) for key in keys}


def check_comparable(results: dict[str, list[dict]], protocol: str) -> tuple[list[str], list[int], list[int]]:
    first = results[MODELS[0]][0]["metrics"]
    names, minority, train_counts = (list(map(str, first["class_names"])),
        list(first["minority_class_definition"]["minority_class_ids"]),
        first["class_counts"]["train"])
    dataset_identity = (first.get("dataset_name"), first.get("dataset_adapter"),
                        (first.get("run_provenance") or {}).get("dataset_manifest_sha256"),
                        (first.get("run_provenance") or {}).get("split_assignments_sha256"))
    for model in MODELS:
        for item in results[model]:
            metrics = item["metrics"]
            current_identity = (metrics.get("dataset_name"), metrics.get("dataset_adapter"),
                                (metrics.get("run_provenance") or {}).get("dataset_manifest_sha256"),
                                (metrics.get("run_provenance") or {}).get("split_assignments_sha256"))
            if current_identity != dataset_identity:
                raise ValueError("Dataset identity or manifest hash differs across comparisons")
            if list(map(str, metrics["class_names"])) != names:
                raise ValueError("Class-name mapping differs across model result files")
            definition = metrics.get("minority_class_definition", {})
            if definition.get("minority_class_ids") != minority or metrics.get("class_counts", {}).get("train") != train_counts:
                raise ValueError("Training-derived minority-class definition differs across comparisons")
            for condition in CONDITIONS:
                if condition not in metrics:
                    raise ValueError(f"{item['metrics_path']} lacks condition metrics {condition!r}")
    by_seed = {model: {item["seed"]: item["metrics"] for item in results[model]} for model in MODELS}
    for seed in sorted(by_seed[MODELS[0]]):
        reference = by_seed["reference"][seed]
        reference_provenance = reference.get("run_provenance") or {}
        reference_digest = reference.get("checkpoint_sha256")
        if not reference_digest or not reference_provenance.get("training_sample_ids_sha256"):
            raise ValueError(f"Reference result for seed {seed} lacks checkpoint/training-ID provenance")
        if reference_provenance.get("seed") != seed or reference_provenance.get("split_protocol") != protocol:
            raise ValueError(f"Reference result for seed {seed} has mismatched run provenance")
        for model in ("ci_only", "proposed_ci_mi"):
            actual = by_seed[model][seed]
            provenance = actual.get("run_provenance") or {}
            if provenance.get("seed") != seed or provenance.get("split_protocol") != protocol:
                raise ValueError(f"{model} seed {seed} has mismatched run protocol/seed provenance")
            if provenance.get("reference_checkpoint_sha256") != reference_digest:
                raise ValueError(f"{model} seed {seed} was not initialized from the evaluated reference checkpoint")
            for key in ("training_sample_ids_sha256", "experiment_config_sha256", "dataset_manifest_sha256",
                        "split_assignments_sha256", "class_to_id"):
                expected_value = reference_provenance.get(key)
                actual_value = provenance.get(key)
                if not expected_value or actual_value != expected_value:
                    raise ValueError(f"{model} seed {seed} has incompatible {key}: "
                                     f"expected {expected_value!r}, got {actual_value!r}")
    return names, minority, [int(train_counts[str(i)]) for i in range(len(names))]


def manifest_counts(config: dict, protocol: str, class_names: list[str]):
    dataset = config["dataset"]
    kwargs = dataset.get("kwargs", {})
    raw = kwargs.get("manifest_path")
    if not raw:
        raise ValueError("Dataset config must declare dataset.kwargs.manifest_path for class-distribution reports")
    path = resolve(raw)
    if dataset.get("adapter") == "ave" and protocol == "source_video_disjoint" and not path.stem.endswith("_source_video_disjoint"):
        path = path.with_name(path.stem + "_source_video_disjoint" + path.suffix)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest required for class-distribution figure is missing: {path}")
    columns = {"split": "split", "label": "label", "label_id": "label_id", **kwargs.get("columns", {})}
    class_mapping = kwargs.get("class_mapping")
    if class_mapping is None:
        class_mapping = {name: index for index, name in enumerate(sorted(class_names))}
    else:
        class_mapping = {str(key): int(value) for key, value in class_mapping.items()}
    split_names = {"train": kwargs.get("train_split", "train"),
                   "val": kwargs.get("validation_split", "val"),
                   "test": kwargs.get("test_split", "test")}
    counts: dict[str, dict[int, int]] = {split: defaultdict(int) for split in split_names}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        absent = [column for column in (columns["split"], columns["label"]) if column not in (reader.fieldnames or [])]
        if absent:
            raise ValueError(f"Manifest {path} lacks configured report columns {absent}")
        for row in reader:
            split = row[columns["split"]]
            split = next((key for key, value in split_names.items() if value == split), split)
            if split not in counts:
                continue
            if dataset.get("adapter") == "ave" and columns["label_id"] in row and row[columns["label_id"]].strip():
                class_id = int(row[columns["label_id"]])
            else:
                try:
                    class_id = class_mapping[row[columns["label"]]]
                except KeyError as error:
                    raise ValueError(f"Unknown manifest label {row[columns['label']]!r} in {path}") from error
            counts[split][class_id] += 1
    if not any(counts.values()):
        raise ValueError(f"Manifest contains no labeled partitions: {path}")
    return path, counts, list(class_names)


def save_figure(fig, directory: Path, stem: str):
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def training_log(item: dict) -> list[dict]:
    checkpoint = Path(item["metrics"].get("checkpoint", ""))
    model = item["model"]
    if model == "reference":
        candidates = [checkpoint.parent / "reference_training_log.jsonl"]
    else:
        candidates = [checkpoint.parent / "training_log.jsonl"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Training log for {model} seed {item['seed']} was not found beside {checkpoint}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def discover_targets(item: dict) -> Path | None:
    checkpoint = Path(item["metrics"].get("checkpoint", ""))
    for parent in checkpoint.parents:
        candidate = parent / "contributions" / "ucd_targets.json"
        if candidate.is_file():
            return candidate
    return None


def create_figures(figdir: Path, config: dict, protocol: str, seed_label: str,
                   results: dict[str, list[dict]], names: list[str], minority: list[int],
                   train_counts: list[int], targets_override: dict[int, Path]):
    manifest, split_counts, manifest_names = manifest_counts(config, protocol, names)
    if manifest_names != names:
        raise ValueError(f"Manifest class order at {manifest} differs from evaluation class names")
    for model in MODELS:
        for item in results[model]:
            recorded = item["metrics"].get("class_counts", {})
            for split in ("train", "validation", "test"):
                if split not in recorded:
                    continue
                manifest_split = "val" if split == "validation" else split
                actual = [split_counts[manifest_split].get(i, 0) for i in range(len(names))]
                expected = [int(recorded[split].get(str(i), 0)) for i in range(len(names))]
                if actual != expected:
                    raise ValueError(f"{item['metrics_path']} {split} class counts differ from protocol manifest {manifest}")
    indices = np.arange(len(names))
    colors = ["#c44e52" if i in minority else "#4c72b0" for i in indices]
    nonzero = [count for count in train_counts if count > 0]
    imbalance = max(nonzero) / min(nonzero) if nonzero else float("nan")

    fig, ax = plt.subplots(figsize=(16, 6))
    width = .25
    for offset, split in enumerate(("train", "val", "test")):
        values = [split_counts[split].get(i, 0) for i in indices]
        ax.bar(indices + (offset - 1) * width, values, width, label=split.title(), alpha=.9)
    ax.set_xticks(indices, names, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Number of segments")
    dataset_name = config.get("dataset", {}).get("name", "Dataset")
    ax.set_title(f"{dataset_name} class distribution — {protocol} — class imbalance ratio {imbalance:.2f}:1")
    ax.legend(title="Partition")
    ax.text(.99, .98, "Red class labels: minority classes selected from training counts only",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.text(.01, .98, "Imbalance ratio = largest train class count / smallest nonzero train class count",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    for tick, class_id in zip(ax.get_xticklabels(), indices):
        if class_id in minority:
            tick.set_color("#c44e52")
            tick.set_fontweight("bold")
    fig.tight_layout()
    save_figure(fig, figdir, "figure1_class_distribution")

    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    for row, model in enumerate(MODELS):
        matrices = [np.sum([np.asarray(item["metrics"]["audio_video"]["confusion_matrix"])
                            for item in results[model]], axis=0)]
        matrix = matrices[0]
        denom = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, denom, out=np.zeros_like(matrix, dtype=float), where=denom != 0)
        for col, (value, title, fmt) in enumerate(((matrix, "Counts", "d"), (normalized, "Row normalized (recall)", ".2f"))):
            ax = axes[row, col]
            image = ax.imshow(value, cmap="Blues", vmin=0, vmax=(None if col == 0 else 1))
            ax.set_title(f"{DISPLAY[model]} — {title}")
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.set_xticks(indices, names, rotation=90, fontsize=6)
            ax.set_yticks(indices, names, fontsize=6)
            fig.colorbar(image, ax=ax, fraction=.046)
            for i in indices:
                for j in indices:
                    ax.text(j, i, format(value[i, j], fmt), ha="center", va="center", fontsize=4,
                            color="white" if value[i, j] > (value.max() * .55 if col == 0 and value.max() else .55) else "black")
    fig.suptitle(f"Audio-video confusion matrices — {protocol} — seeds {seed_label}")
    fig.tight_layout()
    save_figure(fig, figdir, "figure2_confusion_matrices")

    fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharex=True, sharey=True)
    metric_keys = ("precision", "recall", "f1")
    for ax, metric in zip(axes, metric_keys):
        for model in MODELS:
            values = np.mean([[row[metric] for row in item["metrics"]["audio_video"]["per_class"]]
                              for item in results[model]], axis=0)
            ax.plot(indices, values, marker="o", linewidth=1.5, label=DISPLAY[model])
        for class_id in minority:
            ax.axvspan(class_id - .35, class_id + .35, color="#c44e52", alpha=.09)
        ax.set_title(metric.title())
        ax.set_ylim(0, 1.02)
        ax.set_xticks(indices, names, rotation=70, ha="right", fontsize=7)
        ax.set_ylabel("Score")
    axes[0].legend(fontsize=8)
    summary_text = "Minority mean F1 / recall: " + "; ".join(
        f"{DISPLAY[m]} {np.mean([r['metrics']['audio_video']['minority_macro_f1'] for r in results[m]]):.3f} / "
        f"{np.mean([r['metrics']['audio_video']['minority_macro_recall'] for r in results[m]]):.3f}" for m in MODELS)
    fig.suptitle(f"Per-class metrics — highlighted classes selected from training counts\n{summary_text}", fontsize=11)
    fig.tight_layout()
    save_figure(fig, figdir, "figure3_per_class_metrics")

    fig, axes = plt.subplots(max(1, len(minority)), 1, figsize=(8, max(4, 3.3 * len(minority))), squeeze=False)
    for ax, class_id in zip(axes[:, 0], minority):
        plotted = 0
        for model in MODELS:
            ys, scores = [], []
            for item in results[model]:
                pred = item["predictions"]
                ys.extend((pred["labels"] == class_id).astype(int).tolist())
                scores.extend(pred["probabilities"]["audio_video"][:, class_id].tolist())
            if sum(ys) == 0:
                continue
            precision, recall, _ = precision_recall_curve(ys, scores)
            ap = average_precision_score(ys, scores)
            ax.plot(recall, precision, label=f"{DISPLAY[model]} (AP={ap:.3f})")
            plotted += 1
        ax.set_title(f"{names[class_id]} — audio-video")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        if plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, "Unavailable: no positive test examples", transform=ax.transAxes,
                    ha="center", va="center")
    if not minority:
        axes[0, 0].text(.5, .5, "No minority classes selected from training counts", ha="center", va="center")
        axes[0, 0].set_axis_off()
    fig.suptitle(f"Minority-class precision–recall curves — {protocol} — audio-video condition\n"
                 "One-vs-rest AP per class; macro AP averages classes with positive test examples")
    fig.tight_layout()
    save_figure(fig, figdir, "figure4_minority_precision_recall")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    x = np.arange(len(MODELS))
    bar_width = .24
    for i, condition in enumerate(CONDITIONS):
        mf1 = [np.mean([r["metrics"][condition]["macro_f1"] for r in results[model]]) for model in MODELS]
        bacc = [np.mean([r["metrics"][condition]["balanced_accuracy"] for r in results[model]]) for model in MODELS]
        axes[0].bar(x + (i - 1) * bar_width, mf1, bar_width, label=f"{condition}: macro-F1")
        axes[0].scatter(x + (i - 1) * bar_width, bacc, marker="D", s=24, label=f"{condition}: balanced accuracy")
    axes[0].set_xticks(x, [DISPLAY[m] for m in MODELS])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Aggregate metrics by trained modality condition")
    axes[0].legend(fontsize=7, ncol=2)
    minority_values = np.asarray([[np.mean([[row["f1"] for row in item["metrics"][condition]["per_class"]]
        for item in results[model]], axis=0)[minority_i]
        for condition in CONDITIONS for minority_i in minority] for model in MODELS])
    if len(minority):
        im = axes[1].imshow(minority_values.reshape(len(MODELS), len(CONDITIONS) * len(minority)),
                            vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        axes[1].set_yticks(np.arange(len(MODELS)), [DISPLAY[m] for m in MODELS])
        axes[1].set_xticks(np.arange(len(CONDITIONS) * len(minority)),
            [f"{c}\n{names[i]}" for c in CONDITIONS for i in minority], rotation=55, ha="right", fontsize=7)
        axes[1].set_title("Minority-class F1 across modality conditions")
        fig.colorbar(im, ax=axes[1], label="F1")
    else:
        axes[1].text(.5, .5, "No minority classes selected", ha="center", va="center")
        axes[1].set_axis_off()
    fig.suptitle(f"Modality ablation — {protocol}")
    fig.tight_layout()
    save_figure(fig, figdir, "figure5_modality_ablation")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = (("train_loss_ci", "Training classification loss"), ("val_loss", "Validation loss"),
              ("val_macro_f1", "Validation macro-F1"), ("val_balanced_accuracy", "Validation balanced accuracy"))
    for model in MODELS:
        for item in results[model]:
            logs = training_log(item)
            epochs = [int(row["epoch"]) for row in logs]
            label = f"{DISPLAY[model]} seed {item['seed']}"
            for ax, (key, title) in zip(axes.flat, panels):
                actual_key = key
                if key == "train_loss_ci" and model == "reference":
                    actual_key = "train_loss"
                values = [row.get(actual_key) for row in logs]
                if any(value is not None for value in values):
                    ax.plot(epochs, values, marker=".", label=label)
                if key == "train_loss_ci" and model == "proposed_ci_mi":
                    mi = [row.get("train_loss_mi") for row in logs]
                    if any(value is not None for value in mi):
                        ax.plot(epochs, mi, linestyle="--", alpha=.7, label=f"{label} MI alignment")
            # balanced accuracy is recorded in the validation dictionary for reference/joint logs.
    for ax, (_, title) in zip(axes.flat, panels):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss" if "loss" in title.lower() else "Score")
        if "loss" not in title.lower():
            ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
    fig.suptitle(f"Training and validation curves — {protocol}")
    fig.tight_layout()
    save_figure(fig, figdir, "figure6_training_curves")

    ucd_values, fcd_values, kl_values, diagnostic_seeds = [], [], [], []
    for item in results["proposed_ci_mi"]:
        target_path = targets_override.get(item["seed"]) or discover_targets(item)
        if target_path is None:
            continue
        if not target_path.is_file():
            raise FileNotFoundError(f"UCD targets required for alignment plot are missing: {target_path}")
        target = json.loads(target_path.read_text(encoding="utf-8"))
        provenance = target.get("provenance", {})
        for key, expected in (("split_protocol", protocol), ("seed", item["seed"])):
            if key in provenance and provenance[key] != expected:
                raise ValueError(f"UCD targets {target_path} have {key}={provenance[key]!r}; expected {expected!r}")
        expected_sha = item["metrics"].get("run_provenance", {}).get("reference_checkpoint_sha256")
        actual_sha = provenance.get("reference_checkpoint_sha256")
        if expected_sha and actual_sha and expected_sha != actual_sha:
            raise ValueError(f"UCD targets {target_path} were produced from a different reference checkpoint")
        ucd = np.asarray(target["global"]["target"], dtype=float)
        fcd = np.asarray(item["metrics"].get("fcd_test"), dtype=float)
        if ucd.shape != (2,) or fcd.shape != (2,) or not np.isfinite(ucd).all() or not np.isfinite(fcd).all():
            raise ValueError(f"UCD/FCD distributions in {target_path} must be finite two-modality vectors")
        ucd_values.append(ucd)
        fcd_values.append(fcd)
        kl_values.append(item["metrics"].get("ucd_minus_fcd_kl_test"))
        diagnostic_seeds.append(item["seed"])
    if ucd_values:
        ucd, fcd = np.mean(ucd_values, axis=0), np.mean(fcd_values, axis=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        positions = np.arange(2)
        ax.bar(positions - .18, ucd, .36, label="Training UCD target")
        ax.bar(positions + .18, fcd, .36, label="Test FCD estimate")
        ax.set_xticks(positions, ["Audio", "Video"])
        ax.set_ylabel("Probability")
        kl_mean, kl_std = mean_std(kl_values)
        kl_text = "unavailable" if kl_mean is None else f"{kl_mean:.4f}" + (f" ± {kl_std:.4f}" if kl_std is not None else " (single seed)")
        ax.set_title(f"Contribution-alignment diagnostic — seeds {diagnostic_seeds}; UCD vs test FCD; KL={kl_text}")
        ax.legend()
        fig.tight_layout()
        save_figure(fig, figdir, "figure7_contribution_alignment_diagnostic")
    else:
        print("Skipping Figure 7: compatible UCD targets and saved test FCD values are not both available.")


def write_tables(outdir: Path, results: dict[str, list[dict]], names: list[str], protocol: str):
    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            aggregate = aggregate_metrics(results[model], condition)
            rows.append({"model": DISPLAY[model], "model_condition": model,
                         "evaluation_condition": condition,
                         **{key: mean for key, (mean, _) in aggregate.items()},
                         **{key + "_std": std for key, (_, std) in aggregate.items()},
                         "seed_count": len(results[model]),
                         "seeds": ",".join(str(item["seed"]) for item in results[model]),
                         "reporting": "single_seed" if len(results[model]) == 1 else "mean_and_sample_std"})
    fields = list(rows[0])
    with (outdir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    dataset_name = results[MODELS[0]][0]["metrics"].get("dataset_name", "Dataset")
    md = [f"# {dataset_name} results — {protocol}", "", "Metrics come only from the supplied completed evaluations.",
          "Single-seed values are not multi-seed averages. Multi-seed values show mean ± sample standard deviation.", "",
          "| Model | Condition | Accuracy | Macro-F1 | Balanced accuracy | Macro precision | Minority macro-F1 | Minority macro-recall | Seeds |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        model = row["model"]
        condition = row["evaluation_condition"]
        record = aggregate_metrics(results[row["model_condition"]], condition)
        values = []
        for key in ("accuracy", "macro_f1", "balanced_accuracy", "macro_precision",
                    "minority_macro_f1", "minority_macro_recall"):
            mean, std = record[key]
            values.append("NA" if mean is None else (f"{mean:.4f}" if std is None else f"{mean:.4f} ± {std:.4f}"))
        md.append(f"| {model} | {condition} | " + " | ".join(values) + f" | {row['seeds']} |")
    (outdir / "summary_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    per_class = []
    for model in MODELS:
        for condition in CONDITIONS:
            class_rows = [item["metrics"][condition]["per_class"] for item in results[model]]
            for class_id, name in enumerate(names):
                output = {"model": DISPLAY[model], "model_condition": model,
                          "evaluation_condition": condition, "class_id": class_id, "class_name": name}
                for key in ("precision", "recall", "f1", "support", "average_precision"):
                    mean, std = mean_std([run[class_id][key] for run in class_rows])
                    output[key] = mean
                    if key != "support":
                        output[key + "_std"] = std
                per_class.append(output)
    with (outdir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_class[0])); writer.writeheader(); writer.writerows(per_class)
    return rows, per_class


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, help="Configured dataset split protocol")
    parser.add_argument("--seed", type=int, help="Select one seed; omit to aggregate matching completed seeds")
    parser.add_argument("--config", type=Path, help="Experiment YAML; defaults exist for the two AVE protocols")
    parser.add_argument("--provenance-manifest", type=Path,
                        help="Optional file whose hash is recorded in evaluation provenance; "
                             "use when cached-feature runs use an index file as their dataset identity")
    parser.add_argument("--result", action="append", required=True,
                        help="Completed result as MODEL=path/to/test_metrics.json; repeat for each model/seed")
    parser.add_argument("--targets", action="append",
                        help="Optional UCD targets for Figure 7; use SEED=PATH when aggregating seeds")
    parser.add_argument("--output-dir", type=Path, help="Override protocol report directory")
    args = parser.parse_args(argv)
    default_cfg = {"official": "configs/ave_official.yaml",
                   "source_video_disjoint": "configs/ave_source_video_disjoint.yaml"}.get(args.protocol)
    if args.config is None and default_cfg is None:
        raise ValueError("--config is required for protocols without a built-in AVE configuration")
    cfg_path = resolve(args.config or default_cfg)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Protocol config does not exist: {cfg_path}")
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if (config.get("dataset", {}).get("split_protocol") != args.protocol or
            config.get("test_dataset", {}).get("split_protocol", args.protocol) != args.protocol):
        raise ValueError(f"Config {cfg_path} training/test protocols do not both match {args.protocol!r}")
    results = parse_results(args.result, args.protocol, args.seed)
    names, minority, train_counts = check_comparable(results, args.protocol)
    manifest, _, _ = manifest_counts(config, args.protocol, names)
    provenance_manifest = resolve(args.provenance_manifest) if args.provenance_manifest else manifest
    if not provenance_manifest.is_file():
        raise FileNotFoundError(f"Provenance manifest does not exist: {provenance_manifest}")
    expected_manifest_sha = file_sha256(provenance_manifest)
    for model in MODELS:
        for item in results[model]:
            metrics = item["metrics"]
            if metrics.get("dataset_name") != config["dataset"].get("name"):
                raise ValueError(f"{item['metrics_path']} dataset_name does not match report config")
            if metrics.get("dataset_adapter") != config["dataset"].get("adapter"):
                raise ValueError(f"{item['metrics_path']} dataset_adapter does not match report config")
            actual_manifest_sha = (metrics.get("run_provenance") or {}).get("dataset_manifest_sha256")
            if actual_manifest_sha != expected_manifest_sha:
                raise ValueError(f"{item['metrics_path']} manifest hash does not match configured manifest {manifest}")
            actual_split_sha = (metrics.get("run_provenance") or {}).get("split_assignments_sha256")
            if actual_split_sha != expected_manifest_sha:
                raise ValueError(f"{item['metrics_path']} split-assignment hash does not match configured manifest {manifest}")
    seeds = sorted({item["seed"] for model in MODELS for item in results[model]})
    if args.seed is not None and seeds != [args.seed]:
        raise ValueError(f"Expected only seed {args.seed}, found {seeds}")
    seed_dir = f"seed_{seeds[0]}" if len(seeds) == 1 else "seed_multi"
    outdir = resolve(args.output_dir) if args.output_dir else resolve(config["output_dir"]) / seed_dir / "reports"
    marker = outdir / "report_protocol.json"
    if marker.is_file():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("protocol") != args.protocol:
            raise ValueError(f"Report output directory {outdir} belongs to protocol={existing.get('protocol')!r}; "
                             f"requested {args.protocol!r}")
    elif outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"Report output directory {outdir} is populated but has no protocol marker")
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text(json.dumps({"protocol": args.protocol, "seeds": seeds}, indent=2), encoding="utf-8")
    target_paths = {}
    for value in args.targets or []:
        if "=" in value:
            seed_text, raw_path = value.split("=", 1)
            try:
                target_seed = int(seed_text)
            except ValueError as error:
                raise ValueError(f"--targets must be SEED=PATH for multi-seed reports, got {value!r}") from error
            target_paths[target_seed] = resolve(raw_path)
        else:
            if len(seeds) > 1:
                raise ValueError("Use --targets SEED=PATH for each seed in a multi-seed alignment report")
            target_paths[seeds[0]] = resolve(value)
    extra_target_seeds = sorted(set(target_paths) - set(seeds))
    if extra_target_seeds:
        raise ValueError(f"UCD target seed(s) have no matching evaluation results: {extra_target_seeds}")
    for target_path in target_paths.values():
        if not target_path.is_file():
            raise FileNotFoundError(f"UCD targets file does not exist: {target_path}")
    summary_rows, per_class_rows = write_tables(outdir, results, names, args.protocol)
    create_figures(figdir, config, args.protocol, ", ".join(map(str, seeds)), results,
                   names, minority, train_counts, target_paths)
    summary = {"dataset_name": config.get("dataset", {}).get("name"),
        "dataset_adapter": config.get("dataset", {}).get("adapter"),
        "protocol": args.protocol, "seeds": seeds,
        "class_names": names,
        "class_to_id": {name: class_id for class_id, name in enumerate(names)},
        "dataset_manifest_sha256": (results[MODELS[0]][0]["metrics"].get("run_provenance") or {}).get("dataset_manifest_sha256"),
        "class_distribution_manifest": str(manifest),
        "provenance_manifest": str(provenance_manifest),
        "aggregation": "single_seed" if len(seeds) == 1 else "mean and sample standard deviation across seeds",
        "model_conditions": {model: [{"seed": item["seed"],
            "metrics_path": str(item["metrics_path"]),
            "checkpoint_identity": item["metrics"].get("checkpoint_identity"),
            "audio_video": item["metrics"]["audio_video"]} for item in results[model]] for model in MODELS},
        "minority_class_definition": results[MODELS[0]][0]["metrics"]["minority_class_definition"],
        "figures_directory": str(figdir),
        "figure7_generated": (figdir / "figure7_contribution_alignment_diagnostic.png").exists(),
        "summary_rows": summary_rows, "per_class_rows": per_class_rows}
    (outdir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote protocol={args.protocol} reports to {outdir}")
    return outdir


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as error:
        raise SystemExit(f"plot_results: {error}") from error
