"""Build curated AVE result tables and figures from saved run artifacts only."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("reference", "ci_only", "proposed_ci_mi")
METRICS = ("macro_f1", "accuracy", "minority_macro_f1", "minority_macro_recall", "balanced_accuracy", "macro_average_precision")

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def test_path(run: Path, model: str) -> Path:
    return run / "evaluations" / model / "test_metrics.json"

def validation(run: Path, model: str):
    checkpoint = run / ("reference_best.pt" if model == "reference" else "ci_only/joint_best.pt")
    if model == "proposed_ci_mi":
        selection = run / "proposed_ci_mi/lambda_grid_selection.json"
        if selection.exists():
            item = load(selection).get("selected", {})
            checkpoint = ROOT / item.get("checkpoint", "")
    if not checkpoint.exists():
        return {}
    import torch
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return obj.get("validation_metrics", {})

def modalities(metrics: dict, condition: str, model: str, seed: int):
    rows=[]
    for modality, item in metrics.items():
        if modality not in ("audio_video", "audio_only", "video_only") or not isinstance(item, dict):
            continue
        rows.append({"condition": condition, "model": model, "seed": seed,
                     "modality": modality, **{key: item.get(key) for key in METRICS}})
    return rows

def fmt(values):
    vals=[v for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
    if not vals: return "n/a"
    return f"{mean(vals):.4f} ± {stdev(vals):.4f}" if len(vals)>1 else f"{vals[0]:.4f} (n=1)"

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",default="results")
    args=parser.parse_args()
    out=ROOT/args.results
    (out/"tables").mkdir(parents=True,exist_ok=True)
    (out/"figures").mkdir(parents=True,exist_ok=True)
    (out/"diagnostics").mkdir(parents=True,exist_ok=True)
    rows=[]; modality_rows=[]; per_class_rows=[]; imbalance_rows=[]; paired_rows=[]; granularity_rows=[]; optimization_rows=[]; statuses=[]
    conditions={"natural":[ROOT/"outputs/ave/official/seed_17", *sorted((ROOT/"outputs/ave/natural").glob("seed_*"))],
                "if10":sorted((ROOT/"outputs/ave/if10").glob("seed_*")),
                "if20":sorted((ROOT/"outputs/ave/if20").glob("seed_*"))}
    for condition,runs in conditions.items():
        for run in runs:
            if not run.is_dir(): continue
            seed=int(run.name.removeprefix("seed_"))
            subset=run/"dataset/subset_metadata.json"
            if subset.exists():
                subset_meta=load(subset)
                for class_id,counts in subset_meta.get("class_counts",{}).items():
                    imbalance_rows.append({"condition":condition,"seed":seed,"class_id":class_id,
                        "class_name":counts.get("class_name"),"natural_count":counts.get("natural_count"),
                        "retained_count":counts.get("retained_count"),
                        "imbalance_factor_requested":subset_meta.get("imbalance_factor_requested"),
                        "imbalance_factor_achieved":subset_meta.get("imbalance_factor_achieved"),
                        "subset_manifest":subset_meta.get("subset_manifest")})
            for model in MODELS:
                path=test_path(run,model)
                if not path.is_file():
                    statuses.append({"condition":condition,"seed":seed,"model":model,"status":"incomplete"})
                    continue
                metrics=load(path)
                av=metrics.get("audio_video",{})
                val=validation(run,model)
                selection_path=run/"proposed_ci_mi/lambda_grid_selection.json"
                selected_lambda=(load(selection_path).get("selected",{}).get("lambda_mi")
                                 if model=="proposed_ci_mi" and selection_path.exists() else None)
                config_file=run/"experiment.yaml"
                run_config=(__import__("yaml").safe_load(config_file.read_text(encoding="utf-8-sig"))
                            if config_file.exists() else {})
                if not run_config:
                    legacy_checkpoint=run/"reference_best.pt"
                    if legacy_checkpoint.exists():
                        import torch
                        run_config=torch.load(legacy_checkpoint,map_location="cpu",weights_only=False).get("config",{})
                run_metadata=load(run/"experiment_metadata.json") if (run/"experiment_metadata.json").exists() else {}
                row={"condition":condition,"seed":seed,"model":model,
                     "selected_lambda_mi":selected_lambda,
                     "target_granularity":run_config.get("contribution",{}).get("target_granularity"),
                     "device":run_metadata.get("device"),
                     "validation_macro_f1":val.get("macro_f1"),
                     "test_sample_count":metrics.get("test_sample_count"),
                     "test_split_protocol":metrics.get("split_protocol")}
                row.update({f"test_{key}":av.get(key) for key in METRICS})
                rows.append(row); statuses.append({"condition":condition,"seed":seed,"model":model,"status":"complete"})
                modality_rows.extend(modalities(metrics,condition,model,seed))
                per_class_rows.extend({"condition":condition,"seed":seed,"model":model,**item}
                                      for item in av.get("per_class",[]))
    # Compare target granularities on the matched seed-23 natural split.
    global_row=next((row for row in rows if row["condition"]=="natural" and row["seed"]==23 and row["model"]=="proposed_ci_mi"),None)
    if global_row:
        granularity_rows.append({"seed":23,"target_granularity":"global","selected_lambda_mi":global_row["selected_lambda_mi"],
            "validation_macro_f1":global_row["validation_macro_f1"],"test_macro_f1":global_row["test_macro_f1"],
            "test_minority_macro_f1":global_row["test_minority_macro_f1"],"test_minority_macro_recall":global_row["test_minority_macro_recall"],"status":"complete"})
    for variant_root in sorted((ROOT/"outputs/ave/granularity").glob("*/seed_*")):
        level=variant_root.parent.name
        selection=variant_root/"proposed_ci_mi/lambda_grid_selection.json"
        metrics_path=variant_root/"evaluations/proposed_ci_mi/test_metrics.json"
        if not selection.exists() or not metrics_path.exists():
            granularity_rows.append({"seed":int(variant_root.name.removeprefix("seed_")),"target_granularity":level,"status":"incomplete"})
            continue
        chosen=load(selection).get("selected",{})
        checkpoint=Path(chosen.get("checkpoint",""))
        if not checkpoint.is_absolute(): checkpoint=ROOT/checkpoint
        import torch
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        val=payload.get("validation_metrics",{})
        test=load(metrics_path).get("audio_video",{})
        granularity_rows.append({"seed":int(variant_root.name.removeprefix("seed_")),"target_granularity":level,
            "selected_lambda_mi":chosen.get("lambda_mi"),"validation_macro_f1":val.get("macro_f1"),
            "test_macro_f1":test.get("macro_f1"),"test_minority_macro_f1":test.get("minority_macro_f1"),
            "test_minority_macro_recall":test.get("minority_macro_recall"),"status":"complete"})
    with (out/"tables/granularity_results.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["seed","target_granularity","selected_lambda_mi","validation_macro_f1","test_macro_f1","test_minority_macro_f1","test_minority_macro_recall","status"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(granularity_rows)
    fixed_row=next((row for row in rows if row["condition"]=="natural" and row["seed"]==23 and row["model"]=="proposed_ci_mi"),None)
    if fixed_row:
        optimization_rows.append({"seed":23,"optimization_mode":"fixed_weight","selected_lambda_mi":fixed_row["selected_lambda_mi"],
            "validation_macro_f1":fixed_row["validation_macro_f1"],"test_macro_f1":fixed_row["test_macro_f1"],
            "test_minority_macro_f1":fixed_row["test_minority_macro_f1"],
            "test_minority_macro_recall":fixed_row["test_minority_macro_recall"],"status":"complete"})
    optimization_roots=sorted((ROOT/"outputs/ave/optimization").glob("*/seed_*"))
    for mode_name in ("moo", "normalized_moo"):
        if not any(root.parent.name==mode_name and root.name=="seed_23" for root in optimization_roots):
            optimization_rows.append({"seed":23,"optimization_mode":mode_name,"status":"pending"})
    for variant_root in optimization_roots:
        selection_path=variant_root/"proposed_ci_mi/lambda_grid_selection.json"
        metrics_path=variant_root/"evaluations/proposed_ci_mi/test_metrics.json"
        if not selection_path.exists() or not metrics_path.exists():
            optimization_rows.append({"seed":int(variant_root.name.removeprefix("seed_")),
                "optimization_mode":variant_root.parent.name,"status":"incomplete"})
            continue
        selected=load(selection_path).get("selected",{})
        checkpoint=Path(selected.get("checkpoint",""))
        if not checkpoint.is_absolute(): checkpoint=ROOT/checkpoint
        import torch
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        val=payload.get("validation_metrics",{})
        test=load(metrics_path).get("audio_video",{})
        mode=payload.get("config",{}).get("training",{}).get("optimization_mode",variant_root.parent.name)
        optimization_rows.append({"seed":int(variant_root.name.removeprefix("seed_")),"optimization_mode":mode,
            "selected_lambda_mi":selected.get("lambda_mi"),"validation_macro_f1":val.get("macro_f1"),
            "test_macro_f1":test.get("macro_f1"),"test_minority_macro_f1":test.get("minority_macro_f1"),
            "test_minority_macro_recall":test.get("minority_macro_recall"),"status":"complete"})
    with (out/"tables/optimization_results.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["seed","optimization_mode","selected_lambda_mi","validation_macro_f1","test_macro_f1",
                "test_minority_macro_f1","test_minority_macro_recall","status"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(optimization_rows)
    fields=["condition","seed","model","selected_lambda_mi","target_granularity","device","validation_macro_f1","test_sample_count","test_split_protocol",*[f"test_{m}" for m in METRICS]]
    with (out/"tables/main_results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    modality_fields=["condition","model","seed","modality",*METRICS]
    with (out/"tables/modality_results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=modality_fields);writer.writeheader();writer.writerows(modality_rows)
    with (out/"tables/per_class_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["condition","seed","model","class_id","class_name","precision","recall","f1","support","average_precision"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(per_class_rows)
    with (out/"tables/imbalance_results.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["condition","seed","class_id","class_name","natural_count","retained_count","imbalance_factor_requested","imbalance_factor_achieved","subset_manifest"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(imbalance_rows)
    for condition in conditions:
        seeds=sorted({row["seed"] for row in rows if row["condition"]==condition})
        for seed in seeds:
            by_model={row["model"]:row for row in rows if row["condition"]==condition and row["seed"]==seed}
            if not all(model in by_model for model in MODELS): continue
            proposed=by_model["proposed_ci_mi"]
            for baseline in ("reference","ci_only"):
                base=by_model[baseline]
                paired_rows.append({"condition":condition,"seed":seed,"comparison":f"proposed_ci_mi_minus_{baseline}",
                    "validation_macro_f1_difference":proposed["validation_macro_f1"]-base["validation_macro_f1"],
                    "test_macro_f1_difference":proposed["test_macro_f1"]-base["test_macro_f1"],
                    "test_minority_macro_f1_difference":proposed["test_minority_macro_f1"]-base["test_minority_macro_f1"],
                    "test_minority_recall_difference":proposed["test_minority_macro_recall"]-base["test_minority_macro_recall"]})
    with (out/"tables/paired_differences.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["condition","seed","comparison","validation_macro_f1_difference","test_macro_f1_difference","test_minority_macro_f1_difference","test_minority_recall_difference"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(paired_rows)
    with (out/"tables/experiment_status.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["condition","seed","model","status"]);writer.writeheader();writer.writerows(statuses)
    summary=["# AVE experiment summary", "", "## Protocol and method", "",
      "All AVE runs use the published official train/validation/test assignments. The official split has known source-video ID overlap; results are not source-video-independent.",
      "Audio/video features use frozen pretrained PANNs CNN14 and ImageNet-pretrained ResNet-18 encoders. Training fits projections, bidirectional cross-attention, a learned fusion gate, and classifier. Validation Macro-F1 selects checkpoints and proposed-model lambda; test metrics are reported after selection. Per-run device, target granularity, and selected lambda are in `tables/main_results.csv`; total runtime is not consistently recorded by legacy runs.",
      "Synthetic IF=10/20 runs downsample training examples only without replacement. Their validation/test indexes are the unchanged official indexes. See each run's `dataset/subset_metadata.json` for exact counts and subset manifest.", "",
      "## Test results", "", "Values are mean ± sample standard deviation over completed seeds; a single completed run is labeled n=1. Metrics are read from saved evaluation artifacts.", "",
      "| Condition | Model | Seeds | Validation Macro-F1 | Test Macro-F1 | Accuracy | Minority Macro-F1 | Minority Recall | Balanced accuracy | Macro AP |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition in conditions:
        for model in MODELS:
            selected=[r for r in rows if r["condition"]==condition and r["model"]==model]
            if not selected: continue
            test_values=" | ".join(fmt([row.get("test_"+metric) for row in selected]) for metric in METRICS)
            summary.append(f"| {condition} | {model} | {len(selected)} | " + fmt([row.get("validation_macro_f1") for row in selected]) + " | " + test_values + " |")
    if granularity_rows:
        summary += ["", "## Target granularity ablation", "", "Matched natural AVE seed-23 comparisons use the same reference initialization, training examples, contribution estimates, and validation-selected lambda. This is a single-seed ablation; values are not a multi-seed claim.", "", "| Granularity | Seed | Lambda | Validation Macro-F1 | Test Macro-F1 | Minority Macro-F1 | Minority recall | Status |", "|---|---:|---:|---:|---:|---:|---:|---|"]
        for item in granularity_rows:
            summary.append("| " + " | ".join(str(item.get(key,"n/a")) for key in ("target_granularity","seed","selected_lambda_mi","validation_macro_f1","test_macro_f1","test_minority_macro_f1","test_minority_macro_recall","status")) + " |")
        summary.append("See [`tables/granularity_results.csv`](tables/granularity_results.csv) and [`diagnostics/target_granularity.png`](diagnostics/target_granularity.png).")
    if optimization_rows:
        summary += ["", "## Optimization mode ablation", "",
            "The fixed-weight row uses the matched natural seed-23 run. Raw MOO and normalized MOO use the same reference initialization, training data, targets, and validation-selected checkpoint protocol. Both are single-seed sensitivity ablations; test metrics were recorded after validation selection.", "",
            "| Seed | Mode | Lambda | Validation Macro-F1 | Test Macro-F1 | Minority Macro-F1 | Minority recall | Status |",
            "|---:|---|---:|---:|---:|---:|---:|---|"]
        for item in optimization_rows:
            summary.append("| " + " | ".join(str(item.get(key,"n/a")) for key in ("seed","optimization_mode","selected_lambda_mi","validation_macro_f1","test_macro_f1","test_minority_macro_f1","test_minority_macro_recall","status")) + " |")
        summary.append("See [`tables/optimization_results.csv`](tables/optimization_results.csv).")
    summary += ["", "## Modality ablations", "", "Per-seed audio-video/audio-only/video-only metrics are in [`tables/modality_results.csv`](tables/modality_results.csv). Per-class metrics are in [`tables/per_class_metrics.csv`](tables/per_class_metrics.csv). Exact synthetic training counts, achieved factors, and manifests are in [`tables/imbalance_results.csv`](tables/imbalance_results.csv). Matched-seed proposed-minus-baseline differences are in [`tables/paired_differences.csv`](tables/paired_differences.csv). Method assumptions and diagnostics are in [`../docs/method_notes.md`](../docs/method_notes.md).", "",
      "## Completion and reproduction", "", f"Completed model evaluations: {sum(s['status']=='complete' for s in statuses)}; pending/incomplete: {sum(s['status']=='incomplete' for s in statuses)}. See [`tables/experiment_status.csv`](tables/experiment_status.csv).",
      "Natural seed 17 artifacts remain in `outputs/ave/official/seed_17`. Resume synthetic conditions one job at a time with:", "", "```powershell", ".\\.venv\\Scripts\\python.exe scripts/analysis/report_experiments.py", "powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/experiments/run_synthetic_ave.ps1 -Factor natural -Seed 23", "powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/experiments/extend_validation_lambda_grid.ps1 -Factor natural -Seed 23 -AdditionalLambda \"1 3\"", "powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/experiments/run_granularity_ablation.ps1 -Granularity class -Seed 23", "# Repeat the granularity command with group and reliability; repeat synthetic runs with -Factor 10 or 20.", "```", "", "## Limitations", "", "The official split can share source-video IDs across partitions. Results are descriptive for this protocol. Reliability of an apparent method advantage requires matched multi-seed comparisons; incomplete/single-seed conditions are not evidence of a robust gain.", ""]
    (out/"summary.md").write_text("\n".join(summary),encoding="utf-8")
    # Main figure uses saved test metrics; label each bar with actual seed count.
    complete=[r for r in rows if r.get("test_macro_f1") is not None]
    if complete:
        fig,ax=plt.subplots(figsize=(8,4.8))
        positions=[];labels=[];x=0
        for condition in conditions:
            for model in MODELS:
                vals=[r["test_macro_f1"] for r in complete if r["condition"]==condition and r["model"]==model]
                if vals:
                    positions.append((x,mean(vals),stdev(vals) if len(vals)>1 else 0.0));labels.append(f"{condition}\n{model}\nn={len(vals)}")
                x+=1
            x+=0.5
        ax.bar([p[0] for p in positions],[p[1] for p in positions],yerr=[p[2] for p in positions],capsize=3,color=["#607d8b" if "reference" in label else "#2a9d8f" if "ci_only" in label else "#e9c46a" for label in labels])
        ax.set_xticks([p[0] for p in positions],labels,rotation=25,ha="right",fontsize=8);ax.set_ylabel("Test Macro-F1");ax.set_ylim(0,1);ax.set_title("AVE performance by training condition (official split)");fig.tight_layout();fig.savefig(out/"figures/overall_performance.png",dpi=220);fig.savefig(out/"figures/overall_performance.pdf");plt.close(fig)
        # Minority performance uses exactly the same completed test rows.
        fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True)
        for axis,metric,title in zip(axes,("test_minority_macro_f1","test_minority_macro_recall"),
                                     ("Minority Macro-F1","Minority recall")):
            labels=[];values=[];errors=[]
            for condition in conditions:
                for model in MODELS:
                    vals=[r[metric] for r in complete if r["condition"]==condition and r["model"]==model and r.get(metric) is not None]
                    if vals:
                        labels.append(f"{condition}\n{model}\nn={len(vals)}");values.append(mean(vals));errors.append(stdev(vals) if len(vals)>1 else 0.)
            axis.bar(range(len(values)),values,yerr=errors,capsize=3);axis.set_xticks(range(len(values)),labels,rotation=35,ha="right",fontsize=7);axis.set_title(title);axis.set_ylim(0,1)
        axes[0].set_ylabel("Test score");fig.suptitle("Minority-class performance (official split)");fig.tight_layout();fig.savefig(out/"figures/minority_performance.png",dpi=220);fig.savefig(out/"figures/minority_performance.pdf");plt.close(fig)
    if modality_rows:
        fig,ax=plt.subplots(figsize=(8,4.8));labels=[];values=[];errors=[]
        for condition in conditions:
            for model in MODELS:
                for modality in ("audio_video","audio_only","video_only"):
                    vals=[r["macro_f1"] for r in modality_rows if r["condition"]==condition and r["model"]==model and r["modality"]==modality and r.get("macro_f1") is not None]
                    if vals:
                        labels.append(f"{condition}\n{model}\n{modality}\nn={len(vals)}");values.append(mean(vals));errors.append(stdev(vals) if len(vals)>1 else 0.)
        ax.bar(range(len(values)),values,yerr=errors,capsize=2);ax.set_xticks(range(len(values)),labels,rotation=40,ha="right",fontsize=7);ax.set_ylabel("Test Macro-F1");ax.set_ylim(0,1);ax.set_title("Audio-video and single-modality evaluation");fig.tight_layout();fig.savefig(out/"figures/modality_ablation.png",dpi=220);fig.savefig(out/"figures/modality_ablation.pdf");plt.close(fig)
    if any(r["condition"] in ("if10","if20") for r in complete):
        fig,ax=plt.subplots(figsize=(7,4.5))
        for model in MODELS:
            xs=[];ys=[];es=[]
            for condition,x in (("natural",0),("if10",10),("if20",20)):
                vals=[r["test_macro_f1"] for r in complete if r["condition"]==condition and r["model"]==model]
                if vals: xs.append(x);ys.append(mean(vals));es.append(stdev(vals) if len(vals)>1 else 0.)
            if xs: ax.errorbar(xs,ys,yerr=es,marker="o",capsize=3,label=model)
        ax.set_xticks([0,10,20],["Natural", "IF=10", "IF=20"]);ax.set_ylabel("Test Macro-F1");ax.set_ylim(0,1);ax.set_title("Performance under synthetic training imbalance");ax.legend();fig.tight_layout();fig.savefig(out/"figures/imbalance_sensitivity.png",dpi=220);fig.savefig(out/"figures/imbalance_sensitivity.pdf");plt.close(fig)
    completed_granularity=[item for item in granularity_rows if item.get("status")=="complete"]
    if completed_granularity:
        fig,axes=plt.subplots(1,2,figsize=(8,4.2))
        for axis,key,title in zip(axes,("validation_macro_f1","test_macro_f1"),("Validation Macro-F1","Test Macro-F1")):
            vals=[item.get(key) for item in completed_granularity]
            labels=[f"{item['target_granularity']}\nn=1" for item in completed_granularity]
            axis.bar(range(len(vals)),vals);axis.set_xticks(range(len(vals)),labels,rotation=25,ha="right");axis.set_ylim(0,1);axis.set_title(title)
        fig.suptitle("Target granularity ablation (seed 23)");fig.tight_layout();fig.savefig(out/"diagnostics/target_granularity.png",dpi=200);plt.close(fig)
    diagnostic_series=[]
    for root in (ROOT/"outputs/ave/official",ROOT/"outputs/ave/natural",ROOT/"outputs/ave/if10",ROOT/"outputs/ave/if20"):
        for log in root.glob("seed_*/proposed_ci_mi/**/training_log.jsonl"):
            try:
                values=[json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            except (OSError,json.JSONDecodeError):
                continue
            if values: diagnostic_series.append((log,values))
    if diagnostic_series:
        fig,axes=plt.subplots(3,1,figsize=(8,8),sharex=True)
        for log,values in diagnostic_series:
            name=f"{log.parents[1].name}/{log.parent.name}"
            epochs=[v["epoch"] for v in values]
            for axis,key in zip(axes,("val_macro_f1","weighted_alignment_loss","gradient_cosine_mean")):
                data=[v.get(key) for v in values]
                if any(value is not None for value in data): axis.plot(epochs,data,marker=".",label=name)
        axes[0].set_ylabel("Validation Macro-F1");axes[1].set_ylabel("Weighted alignment loss");axes[2].set_ylabel("Gradient cosine");axes[2].set_xlabel("Epoch")
        axes[0].legend(fontsize=6,ncol=2);fig.suptitle("Proposed-model training diagnostics");fig.tight_layout();fig.savefig(out/"diagnostics/alignment_training.png",dpi=200);plt.close(fig)
    print(f"Wrote {len(rows)} completed result rows and {len(statuses)} status rows to {out}")

if __name__=="__main__": main()


