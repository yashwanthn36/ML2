"""Training-only Ward clustering of signed two-modality loss signatures."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def cluster_profiles(signatures_csv: str | Path, output_json: str | Path, *,
                     min_support: int | None = None, min_support_floor: int = 20,
                     max_groups: int = 3, dispersion_epsilon: float = 1e-8,
                     assignment_signatures_csv: str | Path | None = None) -> dict[str, Any]:
    """Fit groups on estimation signatures and optionally assign other train examples."""
    with Path(signatures_csv).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No training signatures were supplied")
    values = np.asarray([[float(r["delta_1"]), float(r["delta_2"])] for r in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Training signatures contain non-finite contribution values")
    mean, std = values.mean(axis=0), values.std(axis=0)
    scaled = (values - mean) / np.maximum(std, dispersion_epsilon)
    labels = np.asarray([int(r["label"]) for r in rows])
    ids = [r["sample_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Fitted signature sample_id values must be unique")
    fit_source_ids = [r.get("source_id", "") for r in rows]
    groups: dict[str, int] = {}
    stats: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    class_centroids: dict[int, dict[int, np.ndarray]] = {}
    for class_id in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == class_id)
        count = len(indices)
        required = min_support or max(min_support_floor, math.ceil(0.02 * count))
        class_values = scaled[indices]
        class_key = str(class_id)
        best_labels = np.zeros(count, dtype=np.int64)
        chosen = 1
        best_score: float | None = None
        reason = "single_group_default"
        if count < 2 * required:
            reason = "insufficient_support"
        elif float(np.max(np.var(class_values, axis=0))) <= dispersion_epsilon:
            reason = "negligible_dispersion"
        else:
            upper = min(max_groups, count // required)
            for candidate_count in range(2, upper + 1):
                if candidate_count >= count:
                    continue  # silhouette is undefined when every sample is its own cluster
                candidate = AgglomerativeClustering(n_clusters=candidate_count, linkage="ward").fit_predict(class_values)
                support = np.bincount(candidate, minlength=candidate_count)
                if int(support.min()) < required:
                    continue
                score = float(silhouette_score(class_values, candidate, metric="euclidean"))
                if best_score is None or score > best_score:
                    best_score, chosen, best_labels = score, candidate_count, candidate
            reason = "best_supported_silhouette" if chosen > 1 else "no_supported_stable_candidate"
        for local, sample_index in enumerate(indices):
            groups[ids[sample_index]] = int(best_labels[local])
        for group_id in range(chosen):
            members = indices[best_labels == group_id]
            raw = values[members]
            covariance = np.cov(raw.T).tolist() if len(members) > 1 else [[0.0, 0.0], [0.0, 0.0]]
            stats[f"{class_key}:{group_id}"] = {
                "class_id": int(class_id), "group_id": group_id, "count": int(len(members)),
                "centroid_scaled": scaled[members].mean(axis=0).tolist(),
                "mean_signed_signature": raw.mean(axis=0).tolist(), "covariance": covariance,
            }
        class_centroids[int(class_id)] = {
            group_id: np.asarray(stats[f"{class_key}:{group_id}"]["centroid_scaled"])
            for group_id in range(chosen)
        }
        diagnostics[class_key] = {"class_count": count, "min_support": required,
                                  "groups": chosen, "silhouette": best_score, "decision": reason}
    assignment_fallbacks: dict[str, str] = {}
    if assignment_signatures_csv is not None:
        with Path(assignment_signatures_csv).open("r", newline="", encoding="utf-8") as handle:
            assignment_rows = list(csv.DictReader(handle))
        assignment_ids = [row["sample_id"] for row in assignment_rows]
        if len(set(assignment_ids)) != len(assignment_ids):
            raise ValueError("Assignment signature sample_id values must be unique")
        assignment_source_ids = [row.get("source_id", "") for row in assignment_rows]
        for row in assignment_rows:
            sample_id = row["sample_id"]
            if sample_id in groups:
                fit_label = labels[ids.index(sample_id)]
                if int(row["label"]) != int(fit_label):
                    raise ValueError(f"Label mismatch for fitted sample {sample_id!r}")
                continue
            class_id = int(row["label"])
            if class_id not in class_centroids:
                # A training-only estimation subset can omit very rare classes,
                # especially after deterministic long-tail downsampling. Do not
                # fit their profile from reference-fit examples: assign the
                # single fallback group and let UCD use its parent/global target.
                groups[sample_id] = 0
                assignment_fallbacks[str(class_id)] = "no_estimation_support_global_target_fallback"
                continue
            signature = np.asarray([float(row["delta_1"]), float(row["delta_2"])])
            if not np.isfinite(signature).all():
                raise ValueError(f"Non-finite contribution signature for sample {sample_id!r}")
            standardized = (signature - mean) / np.maximum(std, dispersion_epsilon)
            nearest = min(class_centroids[class_id],
                          key=lambda group_id: float(np.sum((standardized - class_centroids[class_id][group_id]) ** 2)))
            groups[sample_id] = int(nearest)
    else:
        assignment_ids = ids
        assignment_source_ids = fit_source_ids
    if assignment_signatures_csv is not None and set(assignment_ids) - set(groups):
        missing = sorted(set(assignment_ids) - set(groups))
        raise ValueError(f"No contribution group assignment for {len(missing)} samples: {missing[:5]}")
    result = {"assignments": groups, "group_statistics": stats, "diagnostics": diagnostics,
              "assignment_fallbacks": assignment_fallbacks,
              "fit_sample_ids": ids, "fit_source_ids": fit_source_ids,
              "assignment_sample_ids": assignment_ids,
              "assignment_source_ids": assignment_source_ids,
              "assignment_signature_status": ("reference_model_in_sample_for_fit_rows"
                                                if assignment_signatures_csv else "fit_rows_only"),
              "scaler": {"mean": mean.tolist(), "std": std.tolist()},
              "linkage": "ward", "distance": "euclidean", "candidate_groups": list(range(1, max_groups + 1)),
              "min_support_floor": min_support_floor}
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
