# AVE experiment summary

## Protocol and method

All AVE runs use the published official train/validation/test assignments. The official split has known source-video ID overlap; results are not source-video-independent.
Audio/video features use frozen pretrained PANNs CNN14 and ImageNet-pretrained ResNet-18 encoders. Training fits projections, bidirectional cross-attention, a learned fusion gate, and classifier. Validation Macro-F1 selects checkpoints and proposed-model lambda; test metrics are reported after selection. Per-run device, target granularity, and selected lambda are in `tables/main_results.csv`; total runtime is not consistently recorded by legacy runs.
Synthetic IF=10/20 runs downsample training examples only without replacement. Their validation/test indexes are the unchanged official indexes. See each run's `dataset/subset_metadata.json` for exact counts and subset manifest.

## Test results

Values are mean ± sample standard deviation over completed seeds; a single completed run is labeled n=1. Metrics are read from saved evaluation artifacts.

| Condition | Model | Seeds | Validation Macro-F1 | Test Macro-F1 | Accuracy | Minority Macro-F1 | Minority Recall | Balanced accuracy | Macro AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| natural | reference | 5 | 0.9160 ± 0.0075 | 0.9033 ± 0.0081 | 0.9077 ± 0.0061 | 0.8840 ± 0.0144 | 0.8841 ± 0.0321 | 0.9040 ± 0.0091 | 0.9550 ± 0.0042 |
| natural | ci_only | 5 | 0.9182 ± 0.0064 | 0.8980 ± 0.0076 | 0.9002 ± 0.0064 | 0.8872 ± 0.0097 | 0.9044 ± 0.0258 | 0.9006 ± 0.0080 | 0.9525 ± 0.0042 |
| natural | proposed_ci_mi | 5 | 0.9238 ± 0.0067 | 0.9020 ± 0.0151 | 0.9052 ± 0.0140 | 0.8894 ± 0.0214 | 0.8933 ± 0.0382 | 0.9039 ± 0.0148 | 0.9527 ± 0.0023 |
| if10 | reference | 5 | 0.8835 ± 0.0176 | 0.8466 ± 0.0172 | 0.8603 ± 0.0164 | 0.7825 ± 0.0303 | 0.7716 ± 0.0603 | 0.8450 ± 0.0208 | 0.9101 ± 0.0050 |
| if10 | ci_only | 5 | 0.8903 ± 0.0109 | 0.8601 ± 0.0086 | 0.8723 ± 0.0054 | 0.8132 ± 0.0474 | 0.7925 ± 0.0518 | 0.8591 ± 0.0086 | 0.9204 ± 0.0127 |
| if10 | proposed_ci_mi | 5 | 0.8943 ± 0.0075 | 0.8512 ± 0.0153 | 0.8673 ± 0.0129 | 0.7771 ± 0.0358 | 0.7774 ± 0.0707 | 0.8517 ± 0.0187 | 0.9197 ± 0.0060 |
| if20 | reference | 5 | 0.8608 ± 0.0122 | 0.8246 ± 0.0075 | 0.8494 ± 0.0065 | 0.7174 ± 0.0168 | 0.6869 ± 0.0539 | 0.8215 ± 0.0048 | 0.8956 ± 0.0127 |
| if20 | ci_only | 5 | 0.8683 ± 0.0131 | 0.8306 ± 0.0106 | 0.8524 ± 0.0065 | 0.7301 ± 0.0622 | 0.7041 ± 0.0642 | 0.8293 ± 0.0109 | 0.8991 ± 0.0134 |
| if20 | proposed_ci_mi | 5 | 0.8687 ± 0.0075 | 0.8310 ± 0.0150 | 0.8439 ± 0.0115 | 0.7881 ± 0.0452 | 0.7771 ± 0.0669 | 0.8304 ± 0.0155 | 0.9018 ± 0.0132 |

## Target granularity ablation

Matched natural AVE seed-23 comparisons use the same reference initialization, training examples, contribution estimates, and validation-selected lambda. This is a single-seed ablation; values are not a multi-seed claim.

| Granularity | Seed | Lambda | Validation Macro-F1 | Test Macro-F1 | Minority Macro-F1 | Minority recall | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| global | 23 | 0.1 | 0.9273254534326301 | 0.8993483646544735 | 0.8779448621553885 | 0.8488095238095239 | complete |
| class | 23 | 0.1 | 0.9252081658728711 | 0.8892957591503731 | 0.8644148249411406 | 0.819047619047619 | complete |
| group | 23 | 0.1 | 0.9252081658728711 | 0.8892957591503731 | 0.8644148249411406 | 0.819047619047619 | complete |
| reliability | 23 | 0.1 | 0.9252081658728711 | 0.8892957591503731 | 0.8644148249411406 | 0.819047619047619 | complete |
See [`tables/granularity_results.csv`](tables/granularity_results.csv) and [`diagnostics/target_granularity.png`](diagnostics/target_granularity.png).

## Optimization mode ablation

The fixed-weight row uses the matched natural seed-23 run. The MOO row is included only after its validation-selected checkpoint has a test evaluation; this is a single-seed ablation.

| Seed | Mode | Lambda | Validation Macro-F1 | Test Macro-F1 | Minority Macro-F1 | Minority recall | Status |
|---:|---|---:|---:|---:|---:|---:|---|
| 23 | fixed_weight | 0.1 | 0.9273254534326301 | 0.8993483646544735 | 0.8779448621553885 | 0.8488095238095239 | complete |
| 23 | moo | None | 0.4600146736269484 | 0.37676040821915696 | 0.24789863920298702 | 0.34325396825396826 | complete |
| 23 | normalized_moo | None | 0.8430568253578838 | 0.8429251686441211 | 0.8273778509072628 | 0.8186507936507936 | complete |
See [`tables/optimization_results.csv`](tables/optimization_results.csv).

## Modality ablations

Per-seed audio-video/audio-only/video-only metrics are in [`tables/modality_results.csv`](tables/modality_results.csv). Per-class metrics are in [`tables/per_class_metrics.csv`](tables/per_class_metrics.csv). Exact synthetic training counts, achieved factors, and manifests are in [`tables/imbalance_results.csv`](tables/imbalance_results.csv). Matched-seed proposed-minus-baseline differences are in [`tables/paired_differences.csv`](tables/paired_differences.csv). Method assumptions and diagnostics are in [`../docs/method_notes.md`](../docs/method_notes.md).

## Completion and reproduction

Completed model evaluations: 45; pending/incomplete: 0. See [`tables/experiment_status.csv`](tables/experiment_status.csv).
Natural seed 17 artifacts remain in `outputs/ave/official/seed_17`. Resume synthetic conditions one job at a time with:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/report_experiments.py
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/experiments/run_synthetic_ave.ps1 -Factor natural -Seed 23
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/experiments/extend_validation_lambda_grid.ps1 -Factor natural -Seed 23 -AdditionalLambda "1 3"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/experiments/run_granularity_ablation.ps1 -Granularity class -Seed 23
# Repeat the granularity command with group and reliability; repeat synthetic runs with -Factor 10 or 20.
```

## Limitations

The official split can share source-video IDs across partitions. Results are descriptive for this protocol. Reliability of an apparent method advantage requires matched multi-seed comparisons; incomplete/single-seed conditions are not evidence of a robust gain.
