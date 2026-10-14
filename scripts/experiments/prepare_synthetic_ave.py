"""Create deterministic training-only long-tail AVE subsets from frozen features."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from multimodal.training.provenance import ensure_protocol_output


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def long_tail_targets(natural_counts: dict[int, int], factor: int) -> dict[int, int]:
    """Geometric class-frequency schedule with exactly the requested max/min ratio."""
    if factor not in (10, 20):
        raise ValueError("factor must be 10 or 20")
    ordered = sorted(natural_counts, key=lambda class_id: (-natural_counts[class_id], class_id))
    if len(ordered) < 2 or any(natural_counts[class_id] <= 0 for class_id in ordered):
        raise ValueError("At least two classes with positive counts are required")
    original_max = max(natural_counts.values())
    tail_count = max(1, original_max // factor)
    target_max = min(original_max, tail_count * factor)
    targets = {}
    for rank, class_id in enumerate(ordered):
        geometric = target_max * (1.0 / factor) ** (rank / (len(ordered) - 1))
        targets[class_id] = max(tail_count, min(natural_counts[class_id], int(round(geometric))))
    targets[ordered[0]] = target_max
    targets[ordered[-1]] = tail_count
    return targets


def prepare(factor: int, seed: int) -> Path:
    if factor not in (0, 10, 20):
        raise ValueError("Supported conditions are natural (0), IF=10, and IF=20")
    official = ROOT / "outputs/ave/official/seed_17/feature_cache"
    train_path = official / "train_index.csv"
    val_path = official / "validation_index.csv"
    test_path = official / "test_index.csv"
    for path in (train_path, val_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required completed official feature cache is missing: {path}")
    validation_sha = sha256_file(val_path)
    test_sha = sha256_file(test_path)
    fields, rows = read_csv(train_path)
    by_class: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["label"])].append(row)
    if len(by_class) < 2:
        raise ValueError("Need at least two AVE classes to create a long-tail subset")

    natural_counts = Counter({class_id: len(class_rows) for class_id, class_rows in by_class.items()})
    ordered = sorted(natural_counts, key=lambda class_id: (-natural_counts[class_id], class_id))
    original_max = max(natural_counts.values())
    targets: dict[int, int] = long_tail_targets(dict(natural_counts), factor) if factor else dict(natural_counts)

    retained: list[dict[str, str]] = []
    for class_id in ordered:
        class_rows = sorted(by_class[class_id], key=lambda row: row["sample_id"])
        if factor:
            rng = random.Random((seed << 16) ^ (factor << 8) ^ class_id)
            keep = set(rng.sample(range(len(class_rows)), targets[class_id]))
            retained.extend(row for i, row in enumerate(class_rows) if i in keep)
        else:
            retained.extend(class_rows)
    retained.sort(key=lambda row: row["sample_id"])

    condition_name = "natural" if factor == 0 else f"if{factor}"
    condition = ROOT / "outputs" / "ave" / condition_name / f"seed_{seed}"
    ensure_protocol_output(condition, split_protocol="official", seed=seed)
    dataset_dir = condition / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    subset_index = dataset_dir / "train_index.csv"
    # Paths in indexes remain portable relative paths within the repository.
    original_features = official / "features"
    for row in retained:
        source_feature = original_features / Path(row["feature_path"]).name
        row["feature_path"] = Path(__import__("os").path.relpath(source_feature, dataset_dir)).as_posix()
    with subset_index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained)

    actual_counts = Counter(int(row["label"]) for row in retained)
    achieved = max(actual_counts.values()) / min(actual_counts.values())
    if sha256_file(val_path) != validation_sha or sha256_file(test_path) != test_sha:
        raise RuntimeError("Official validation/test indexes changed while preparing a training subset")
    subset_manifest = dataset_dir / "train_subset_manifest.csv"
    manifest_fields = ["sample_id", "source_id", "label", "label_name", "split", "split_protocol"]
    with subset_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in manifest_fields} for row in retained)

    config_path = condition / "experiment.yaml"
    config = yaml.safe_load((ROOT / "configs/ave_official.yaml").read_text(encoding="utf-8"))
    config["seed"] = int(seed)
    config["output_dir"] = f"outputs/ave/{condition_name}"
    config["dataset"]["adapter"] = "cached_features"
    config["dataset"]["factory"] = "multimodal.data_adapters.cached_features:build_datasets"
    config["dataset"]["kwargs"] = {
        "train_index": subset_index.relative_to(ROOT).as_posix(),
        "validation_index": val_path.relative_to(ROOT).as_posix(),
        "split_protocol": "official",
    }
    config["test_dataset"]["adapter"] = "cached_features"
    config["test_dataset"]["factory"] = "multimodal.data_adapters.cached_features:build_test_dataset"
    config["test_dataset"]["kwargs"] = {
        "index_path": test_path.relative_to(ROOT).as_posix(),
        "split_protocol": "official",
    }
    config["model"]["kwargs"]["cached_features"] = True
    config["contribution"]["target_granularity"] = "global"
    config["training"]["lambda_mi_grid"] = [0.03, 0.1, 0.3] if factor == 0 and seed == 23 else [0.1]
    config["training"]["gradient_diagnostics_interval"] = 50
    config["training"]["epochs"] = 30
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    counts_by_class = {str(class_id): {"class_name": by_class[class_id][0]["label_name"],
        "natural_count": natural_counts[class_id], "target_count": targets[class_id],
        "retained_count": actual_counts[class_id]} for class_id in sorted(by_class)}
    record = {"dataset": "AVE", "split_protocol": "official",
        "condition": "natural" if factor == 0 else f"synthetic_if{factor}", "imbalance_factor_requested": factor or None,
        "imbalance_factor_achieved": achieved, "seed": seed,
        "sampling": "deterministic per-class downsampling without replacement",
        "class_order": ordered, "class_counts": counts_by_class,
        "retained_sample_count": len(retained), "train_index": subset_index.relative_to(ROOT).as_posix(),
        "subset_manifest": subset_manifest.relative_to(ROOT).as_posix(),
        "validation_index": val_path.relative_to(ROOT).as_posix(),
        "test_index": test_path.relative_to(ROOT).as_posix(),
        "validation_test_unchanged": True,
        "validation_index_sha256": validation_sha,
        "test_index_sha256": test_sha,
        "config": config_path.relative_to(ROOT).as_posix()}
    (dataset_dir / "subset_metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", choices=("natural", "10", "20"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(prepare(0 if args.factor == "natural" else int(args.factor), args.seed))


if __name__ == "__main__":
    main()
