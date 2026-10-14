# AVE Results â€” Official Split, Seed 17

This page summarizes the completed official-split test evaluations. Each model was evaluated on the same **401 test examples**. The main audio-video result is macro-F1, which gives each AVE class equal weight.

## Audio-video performance

| Model | Accuracy | Macro-F1 | Minority macro-F1 | Macro average precision |
|---|---:|---:|---:|---:|
| Reference | 90.0% | 89.5% | 87.1% | 95.6% |
| CI-only | 89.5% | 89.3% | 89.3% | 94.7% |
| Proposed CI + MI | 89.0% | 88.8% | 88.5% | 95.1% |

On this seed and split, the proposed model did not improve overall macro-F1 over the reference. Its minority-class macro-F1 was higher. These single-seed results do not establish a reliable difference between methods.

## Modality-specific macro-F1

| Model | Audio-video | Audio-only | Video-only |
|---|---:|---:|---:|
| Reference | 89.5% | 82.3% | 48.0% |
| CI-only | 89.3% | 80.5% | 36.8% |
| Proposed CI + MI | 88.8% | 79.8% | 41.6% |

## Run and checkpoint selection

- Dataset: AVE, official split; seed: 17.
- The proposed CI + MI model selected **lambda = 0.1** using validation macro-F1 (**93.31%**, epoch 4). Test scores above were measured after that validation-based selection.
- The checkpoints are in `reference_best.pt`, `ci_only/joint_best.pt`, and `proposed_ci_mi/lambda_0.1/joint_best.pt`.
- Detailed metrics and per-example predictions are in `evaluations/<model>/test_metrics.json` and `evaluations/<model>/test_predictions.csv`.
- The selected lambda and validation candidate scores are in `proposed_ci_mi/lambda_grid_selection.json`.

## Scope and report status

These are results for the **official split**, which has known source-video ID overlap and is not source-video-independent. The separate source-video-disjoint workflow was removed from the active experiment setup.

The automated report has since been generated. Its summary tables are in `reports/summary_metrics.md` and `reports/summary_metrics.csv`; per-class results are in `reports/per_class_metrics.csv`; figures are in `reports/figures/` as PNG and PDF files. The report uses the AVE manifest for split class counts and the cached training index for run-provenance validation.

