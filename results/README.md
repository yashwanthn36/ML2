# Curated AVE results

`summary.md` is generated from completed run artifacts. The CSV files in `tables/` contain per-run metrics, paired seed differences, modality ablations, per-class results, synthetic class counts, and experiment status. Primary figures appear in PNG and vector PDF under `figures/`; detailed training diagnostics are in `diagnostics/`.

Refresh the tables and plots from saved outputs with:

```powershell
.\.venv\Scripts\python.exe scripts/analysis/report_experiments.py
```

All AVE results use the published official split, which has known source-video ID overlap. Validation Macro-F1 selects checkpoints and hyperparameters. Test metrics are reported only after that selection. Incomplete and single-seed conditions are identified in the status table and figure labels.
