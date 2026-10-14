# Contribution-profile-aware multimodal learning

This repository contains a reusable two-modality learning pipeline and an AVE adapter. It implements contribution signatures, contribution-profile grouping, risk-based UCD targets, Utopia-style mutual-information FCD estimates, class/group-balanced losses, and fixed-weight or minimum-norm two-objective training. The architecture reference is `references/contribution_profile_multimodal_architecture_redesigned.pdf`; the FCD equations are checked against `references/1-s2.0-S1566253525004567-main.pdf`.

## Setup

The project uses Python 3.13. From PowerShell at the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Alternatively, run `.venv\Scripts\python.exe` without activating. `requirements.txt` includes PyTorch, torchvision, torchaudio, PANNs inference, OpenCV, scikit-learn, PyYAML, imageio-ffmpeg, and pytest.

For an editable package install from the repository root, use `python -m pip install -e ".[dev]"`. `requirements.txt` pins this workstation's CUDA 12.8 stack; `pyproject.toml` declares portable dependency ranges for editable installs. There is no cross-platform lock file because the supported PyTorch wheel differs by operating system and accelerator.

The AVE media and archive are excluded from Git because they require several gigabytes. Place the archive at `data/raw/AVE/ave.zip` and extracted tracks at `data/raw/AVE/ave/audio/` and `data/raw/AVE/ave/video/`. The manifest and split metadata are tracked. Rebuild manifests from archive metadata with:

```powershell
.\.venv\Scripts\python.exe scripts/data/prepare_ave_manifest.py
.\.venv\Scripts\python.exe scripts/data/extract_ave_audio.py
```


## Dataset and split protocols

`official` preserves the published AVE train/validation/test assignments, including their observed overlap in source-video IDs. Use it as a named benchmark baseline; do not describe its test result as source-video-independent.

The active AVE experiment uses the published `official` assignments. Shared utilities can audit source IDs, but no source-video-disjoint experiment is part of the active workflow. The AVE reader also accepts `temporal_window_policy: full_clip` or `annotated_event` and `min_audio_coverage`.

The configured AVE policy is `full_clip`: audio and video are sampled over the available clip, with 16 kHz mono 16-bit PCM, fixed waveform length, RGB frames, and encoder-side image normalization. The `annotated_event` option synchronously crops both streams to the annotation interval. Short/empty intervals fail with a descriptive error when valid audio would fall below the configured coverage threshold; padding cannot silently dominate. The AVE adapter supplies tensor inputs, while PANNs and ResNet adapters perform their own required preprocessing.

## Model and encoders

`TwoModalityFusion` is dataset-independent: both encoders return token sequences, trainable projections align feature widths, bidirectional cross-attention contextualizes them, a learned sigmoid gate fuses them, and one classifier produces logits. The gate is not an uncertainty estimator. Modality dropout trains learned missing-token paths. A missing token can still attend to the available modality through cross-attention, so the ablations represent the trained missing-modality path rather than deleting an encoder. Changing an absent raw input cannot change its masked representation or the resulting prediction.

The AVE config is the reproducible PANNs CNN14 + ImageNet ResNet-18 baseline. The backbones are frozen; projections, attention, gate, and classifier remain trainable. PANNs handles its audio sample-rate requirements; ResNet handles ImageNet normalization. BEATs and VideoMAE V2 ViT-Giant are not configured or claimed. Encoders remain configurable through model factories/adapters; weights are not downloaded as part of tests.

## Contribution and optimization method

For modality 1 (audio) and modality 2 (video), the saved signed signatures are:

```text
delta_1 = loss_without_1 - full_loss   # mask [0, 1], modality 2 remains
delta_2 = loss_without_2 - full_loss   # mask [1, 0], modality 1 remains
```

Negative deltas are retained. They are used as two-dimensional within-class profile features. Ward clustering standardizes with a scaler fit only on the designated signature-fitting subset, checks group support, and uses a single-group fallback when support/dispersion is insufficient. Centroids and support statistics are fit on that subset; eligible training samples are assigned to the nearest centroid within their class. Missing assignments and label mismatches raise explicit errors.

The UCD target is risk-based: mean positive signed risk increments are normalized, with an epsilon for finite probabilities. Class and class/group targets fall back to their parent scope when support is low or bootstrap stability exceeds the configured threshold. Uncertainty uses a source-video cluster bootstrap when source IDs exist (sample IDs otherwise). Its output is labelled bootstrap spread/quantiles, not calibrated confidence intervals.

FCD is a separate MI estimate; it is not computed from signed loss signatures. The local Utopia paper defines an InfoNCE lower bound in Eqs. (7)â€“(9): L2-normalized modality and fused features, a positive paired score, a log candidate-count term, and a log-sum-exp over positive and negative candidates. `InfoNCEFCD.per_sample_information` returns signed lower bounds `[anchors, 2]`. Negatives are other current-batch examples plus a detached FIFO bank of previous training examples; matching sample IDs are excluded, and AVE sets `exclude_same_source_negatives: true` to exclude sibling source-video segments from current and queued negatives. Sample IDs and source IDs are stored in queue entries. Datasets without source IDs fall back to sample-ID exclusion. Insufficient candidate pools raise an actionable error. This configurable negative-sampling policy is an AVE experiment choice, not an exact reproduction of Utopia. Consequently batch composition and queue contents affect the estimate; batch size 4 alone is a weak contrastive pool, which is why training uses a configurable queue (1024 by default). The reference estimate uses the entire independent estimation subset as its candidate pool. Validation/test estimates do not update a queue.

The paper normalizes MI estimates to get FCD but does not specify how to turn a negative finite-sample InfoNCE bound into a probability. This implementation applies Softplus to the mean bound within each optimization scope, adds epsilon, and normalizes the two modality values. Softplus is an explicit adaptation, not an exact reproduction of Utopia. The result is a finite nonnegative distribution summing to one, and its gradients flow through current representations; queued negatives are detached. Scope-level distributions are computed separately for the selected global, class, or class/group scope. UCD-to-FCD KL uses target UCD as its first argument.

Class-balanced CE uses full-training inverse class frequencies with an N/B scale; under natural uniform minibatch sampling, its mean is an unbiased estimator of equal class risk. Hierarchical group-balanced CE uses full-training class/group counts with the corresponding N/B scale to estimate equal class weighting followed by equal group weighting within each class. It does not self-normalize weights over the current batch, which would make the effective objective composition dependent. Fallback behavior without full counts averages the scopes present in the batch and is marked as an approximation. MOO computes both objective gradients over the same trainable parameter list, handles unused gradients, combines them with the minimum-norm convex coefficient, then assigns gradients and steps once. Diagnostics log gradient norms and cosine similarity.

The optional `normalized_moo` mode normalizes each objective gradient to unit norm before the same convex combination. This is a scale-normalized sensitivity analysis, not the raw minimum-norm method in the paper. The AVE seed-23 comparison is documented in `results/summary.md` and `docs/method_notes.md`.

## Training-only data protocol

By default the reference trainer holds out 20% of the training partition for contribution estimation, using stratified grouping by source video where available. The reference model is fit on the remaining training data and selected using validation macro-F1. Signature fitting, scaling, profile fitting, UCD targets, and the reference FCD use only the independent estimation subset. The remaining training rows may also receive signatures from the held-out-trained reference model solely to assign them to already-fitted centroids; these in-sample assignments are recorded in `contribution_protocol.json` and are not pooled into signature fitting or target estimation. IDs and split protocol are saved under the reference run. Cross-validation is not implemented; this independent holdout is the explicit compute-saving protocol.

An explicit all-training in-sample fallback is available only if the config sets `contribution_estimation_fraction: 0` and `allow_in_sample_contributions: true`. It is statistically weaker and is recorded as such. Validation/test data are not passed into estimators. The test evaluator runs only after checkpoint selection.

## AVE protocol configurations and full comparison workflow

Use `configs/ave_official.yaml` for the published AVE assignments. Its official split has known source-video ID overlap, so results must not be described as source-video-independent. `configs/ave_reference.yaml` remains a backward-compatible alias for the official setup.

The reference model uses the configured natural-sampling classification objective. The CI-only baseline initializes from the exact same reference checkpoint, uses class-balanced CE, and does not construct or call an FCD estimator. The proposed CI + MI condition also initializes from that reference checkpoint, uses the same CI objective and budget, and adds the existing contribution-alignment loss. Its positive lambda grid is selected using validation macro-F1 only. CI-only and proposed checkpoints/logs are distinct and explicitly tagged; test scores do not select checkpoints or lambda.

### Official protocol

```powershell
$cfg = "configs/ave_official.yaml"
$root = "outputs/ave/official/seed_17"
python scripts/training/train_reference.py --config $cfg
python scripts/training/estimate_contributions.py --config $cfg --checkpoint "$root/reference_best.pt"
python scripts/training/train_joint.py --config $cfg --condition ci_only --reference-checkpoint "$root/reference_best.pt"
python scripts/training/train_joint.py --config $cfg --condition proposed_ci_mi --reference-checkpoint "$root/reference_best.pt" --targets "$root/contributions/ucd_targets.json" --groups "$root/contributions/profile_groups.json"
$selection = Get-Content "$root/proposed_ci_mi/lambda_grid_selection.json" -Raw | ConvertFrom-Json
$proposedCheckpoint = $selection.selected.checkpoint
python scripts/training/evaluate_test.py --config $cfg --checkpoint "$root/reference_best.pt" --output-dir "$root/evaluations/reference"
python scripts/training/evaluate_test.py --config $cfg --checkpoint "$root/ci_only/joint_best.pt" --output-dir "$root/evaluations/ci_only"
python scripts/training/evaluate_test.py --config $cfg --checkpoint $proposedCheckpoint --targets "$root/contributions/ucd_targets.json" --output-dir "$root/evaluations/proposed_ci_mi"
python scripts/analysis/plot_results.py --protocol official --seed 17 --targets "$root/contributions/ucd_targets.json" --result "reference=$root/evaluations/reference/test_metrics.json" --result "ci_only=$root/evaluations/ci_only/test_metrics.json" --result "proposed_ci_mi=$root/evaluations/proposed_ci_mi/test_metrics.json"
```


Legacy artifacts need the explicit `--allow-legacy-artifacts` option. Official AVE has known source-video overlap and must not be described as source-video-independent.

## Test metrics and minority classes

Evaluation saves accuracy, macro precision/recall/F1, weighted-F1, balanced accuracy, per-class precision/recall/F1/support, per-class one-vs-rest average precision (AP), macro AP, raw and row-normalized confusion matrices, minority-class macro-F1/recall, test FCD, and optional UCD-to-FCD KL. Macro precision/recall/F1 average over the configured full class ID range, assigning zero to a class with no test support. Balanced accuracy follows scikit-learn's definition and averages recall over classes represented in the test labels. Macro AP averages only classes with positive test examples; AP for a class with no positive test sample is `null`/unavailable.

Minority classes come only from training-set class counts, never test performance or test frequencies. The default rule selects every class whose training count is at or below the empirical 25th percentile of class counts; ties are included, so the selected set can contain more than one quarter of classes. Set `evaluation.minority_bottom_quantile` in a config to change the quantile. Metrics store counts, class names, selected class IDs/names, the quantile, and its threshold. The same training split therefore yields the same minority set for all conditions in that protocol. Validation and test rows are not changed by training balancing.

`test_predictions.csv` contains IDs, true class IDs/names, hard predictions, and a probability column for every class under audio-video, audio-only, and video-only conditions. Probabilities are checked for finite values, range, and row sum; hard predictions are their argmax and the modality conditions use the model's trained missing-modality path.

## Results reports and figures

The curated, aggregate AVE report is built from completed run artifacts by `scripts/analysis/report_experiments.py`. It writes `results/summary.md`, machine-readable tables under `results/tables/`, and primary figures under `results/figures/`; detailed metrics and predictions remain in their run directories. The current report includes matched five-seed natural, IF=10, and IF=20 comparisons and clearly identifies the official split's source-video overlap.

After all three evaluations finish, `scripts/analysis/plot_results.py` requires actual metric and prediction files for reference, CI-only, and proposed models. It rejects missing files, class-map or training-count differences, wrong protocols, duplicate seeds, and results with different seed sets. It never makes up performance values. For several completed seeds, pass repeated `--result MODEL=METRICS_JSON` values for each model/seed and omit `--seed`; the summary uses means and sample standard deviations. For one seed it explicitly labels the report single-seed. Use `--targets SEED=PATH` per seed for a multi-seed contribution diagnostic, or let the script discover each seed's compatible target artifact.

The script generates `summary_metrics.csv`, `summary_metrics.md`, `per_class_metrics.csv`, and `summary_metrics.json`, plus high-resolution PNG and vector PDF versions of:

1. Class distribution by train/validation/test, highlighting training-derived minorities and defining imbalance ratio as largest train class count divided by smallest nonzero train class count.
2. Raw and row-normalized confusion matrices for the three models.
3. Every class's precision, recall, and F1 with minority classes highlighted and a minority summary.
4. Minority-class audio-video precision-recall curves with per-class one-vs-rest AP. Classes without positive test examples have no curve/AP.
5. Modality-condition macro metrics and minority per-class F1.
6. Training loss, validation loss, validation macro-F1, and validation balanced accuracy from reference/joint logs.
7. UCD versus test FCD and UCD-to-FCD KL as a contribution-alignment diagnostic when compatible UCD artifacts and saved FCD values exist. KL is not evidence of classification improvement.

Reports are saved under `outputs/ave/<protocol>/seed_<seed>/reports/figures/` (or `seed_multi` for a multi-seed set). A protocol marker prevents a custom output directory from being reused across protocols. Figures 1?6 require the completed evaluations and matching manifest/log/prediction artifacts. Figure 7 also requires compatible contribution targets; otherwise it is explicitly skipped.

## Invalid AVE annotation interval

`full_clip` samples audio from the WAV duration and video from the available clip duration; it does not require valid annotation timestamps. `annotated_event` strictly rejects missing, nonnumeric, non-finite, zero-length, or reversed intervals. The known `train_0063` zero-length interval remains unchanged in the source manifest and integrity report. No label, sample ID, timestamp, or split is repaired or removed. Both policies continue to validate actual audio/video availability, decoding, and audio coverage.

## AVE annotation integrity and FCD sampling


The AVE configs set `contribution.fcd_kwargs.exclude_same_source_negatives: true`: other same-video segments are excluded from current/queued negatives while the paired positive remains. Datasets without source IDs fall back to sample-ID exclusion. This is an experimental negative-sampling policy, not an exact reproduction of Utopia. FCD also uses the documented Softplus adaptation for negative InfoNCE lower bounds.

## Device selection and verification status

All run commands accept `--device auto|cuda|cpu`. `auto` selects CUDA when available and CPU otherwise; startup logs show the selected device, GPU name, CUDA availability, and PyTorch CUDA version. Explicit unavailable CUDA raises an error. Model/batch placement is checked. Device selection does not imply a speed measurement.

Smoke and code checks (not performance results):

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe scripts/training/train_reference.py --help
.\.venv\Scripts\python.exe scripts/training/estimate_contributions.py --help
.\.venv\Scripts\python.exe scripts/training/train_joint.py --help
.\.venv\Scripts\python.exe scripts/training/evaluate_test.py --help
.\.venv\Scripts\python.exe scripts/training/cache_features.py --help
.\.venv\Scripts\python.exe scripts/data/prepare_ave_manifest.py --help
.\.venv\Scripts\python.exe scripts/data/extract_ave_audio.py --help
.\.venv\Scripts\python.exe scripts/analysis/plot_results.py --help
```

The current offline regression suite passes all 39 tests, including the complete synthetic three-condition integration test. Python compilation, CLI help, and `git diff --check` pass. Both local AVE manifests contain 4,140 unique segments across 28 classes; every referenced audio/video path exists, both integrity summaries agree, official split assignments match the archive's split files, and the seed-17 source-disjoint assignments reproduce exactly with no source overlap. The extracted media audit decoded 4,097 audio and 4,097 video files without errors. On the actual `train_0063` media, `full_clip` loads and `annotated_event` rejects its retained zero-length annotation. The configured production PANNs CNN14 + ResNet-18 forward/backward and AVE end-to-end training smoke remain unverified until the official PANNs checkpoint is completely available. No full 30-epoch experiment has been run and no classification improvement is claimed. Official AVE retains source overlap; interpret results under the named protocol.

## Reusing the pipeline with another dataset

Training, contribution estimation, evaluation, and reporting use the paired dataset contract documented in [docs/architecture.md](docs/architecture.md). The AVE reader remains AVE-specific; `multimodal.data_adapters.generic_audio_video` is the reusable manifest-backed adapter. It accepts arbitrary class names and counts, configurable column names, explicit train/validation/test assignments, optional source IDs, relative or absolute WAV/video paths, and preprocessing parameters. Relative manifest and media-root configuration paths resolve from the repository root. Relative media paths default to `media_root`; use `media_paths_relative_to: manifest` to resolve against the CSV directory.

The generic media implementation supports uncompressed PCM WAV (8/16/24/32-bit) and OpenCV-decodable videos. Audio is converted to mono, linearly resampled, then padded or truncated to the configured window. Video is uniformly sampled over available clip duration. Corrupt or unsupported files fail with a sample/path error. Compressed audio and non-video modalities require a dataset-specific adapter; see [docs/dataset_adapter_guide.md](docs/dataset_adapter_guide.md). The complete copyable model/experiment config is [configs/datasets/dataset_template.yaml](configs/datasets/dataset_template.yaml).

For data without official assignments, generate a new split manifest; the utility never overwrites a populated split manifest or the input:

```powershell
python scripts/data/split_manifest.py --input data/manifests/my_dataset/unsplit.csv --output data/manifests/my_dataset/split_seed_17.csv --seed 17
```

By default it uses source-grouped deterministic splitting and fails when source IDs are missing. The sidecar reports per-class support and limitations because group constraints can make stratification impossible. Explicitly pass `--allow-sample-level` to stratify by sample when source IDs are unavailable or source overlap is acceptable; that protocol can share recordings across splits. Use a dataset-specific protocol name and preserve official split assignments.

Once an external dataset has its manifest/media and a config based on the template, the same `train_reference.py`, `estimate_contributions.py`, `train_joint.py`, `evaluate_test.py`, and `plot_results.py` commands apply. For example, set `$cfg = "configs/datasets/my_dataset.yaml"` and `$root = "outputs/my_dataset/official/seed_17"`, then use the official workflow above with `--config $cfg`; pass the configured protocol string and config explicitly to the report command. Outputs are organized as `outputs/<dataset>/<protocol>/seed_<seed>/` with reference, contributions, CI-only, proposed, evaluations, and reports. Dataset identity, manifest/split hash, class mapping, seed, checkpoint identity, and config provenance are checked before results can be compared.

The tiny offline synthetic integration test exercises all three conditions, contribution artifacts, all three evaluation masks, reports, protocol mismatch rejection, and deterministic split metadata without pretrained downloads:

```powershell
python -m pytest -q -p no:cacheprovider tests/integration/test_generic_pipeline.py
```

This uses locally generated PCM WAV/MJPEG fixtures and lightweight `audio_cnn`/`frame_cnn` encoders. It tests pipeline compatibility, not the AVE pretrained PANNs/ResNet encoders or production classification performance. The repository audit checklist is [docs/audit_checklist.md](docs/audit_checklist.md).

## Repository layout

```text
configs/       AVE protocols, dataset template, and dataset configs
data/          documented raw/processed/manifests locations (ignored by Git)
docs/          architecture, adapter onboarding, experiment, audit, reproducibility
references/    preserved research papers and reference files
outputs/       ignored experiment artifacts
scripts/       data, training, and analysis commands
src/ave_mm/    AVE adapter, integrity audit, split policy, and compatibility models
src/multimodal/ shared adapters, encoders/fusion, objectives, training, provenance
tests/         unit/reliability/report tests and offline end-to-end integration
```
