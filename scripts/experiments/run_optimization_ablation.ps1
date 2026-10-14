param([int]$Seed = 23,[ValidateSet('moo','normalized_moo')][string]$Mode = 'moo')
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path '.').Path
$python = (Resolve-Path '.venv\Scripts\python.exe').Path
$sourceRun = Join-Path $project "outputs\ave\natural\seed_$Seed"
$config = Join-Path $sourceRun 'experiment.yaml'
$reference = Join-Path $sourceRun 'reference_best.pt'
$targets = Join-Path $sourceRun 'contributions\ucd_targets.json'
$groups = Join-Path $sourceRun 'contributions\profile_groups.json'
$selection = Get-Content -LiteralPath (Join-Path $sourceRun 'proposed_ci_mi\lambda_grid_selection.json') -Raw | ConvertFrom-Json
$lambda = [string]$selection.selected.lambda_mi
$outputRoot = "outputs/ave/optimization/$Mode"
$runRoot = Join-Path $project "$outputRoot\seed_$Seed"
$selectionPath = Join-Path $runRoot 'proposed_ci_mi\lambda_grid_selection.json'
$evalDir = Join-Path $runRoot 'evaluations\proposed_ci_mi'
if ((Test-Path -LiteralPath $selectionPath) -and (Test-Path -LiteralPath (Join-Path $evalDir 'test_metrics.json'))) {
    Write-Host "MOO ablation already complete for seed $Seed."
    exit 0
}
$log = Join-Path $project "outputs\logs\optimization_${Mode}_seed_$Seed.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
$utf8 = [System.Text.UTF8Encoding]::new($false)
function Invoke-Logged([string]$Name,[string[]]$StepArguments) {
    [System.IO.File]::AppendAllText($log,"===== START $Name $(Get-Date -Format o) =====`n",$utf8)
    $stdout="$log.stdout.tmp"; $stderr="$log.stderr.tmp"
    $quoted=@($StepArguments | ForEach-Object { '"' + $_.Replace('"','\"') + '"' })
    $process=Start-Process -FilePath $python -ArgumentList $quoted -WorkingDirectory $project -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    foreach($file in @($stdout,$stderr)) {
        if(Test-Path -LiteralPath $file) {
            $content=Get-Content -LiteralPath $file -Raw -Encoding utf8
            [System.IO.File]::AppendAllText($log,$content,$utf8); Write-Host $content
            Remove-Item -LiteralPath $file
        }
    }
    if($process.ExitCode -ne 0) { throw "$Name failed with exit code $($process.ExitCode); see $log" }
    [System.IO.File]::AppendAllText($log,"===== DONE $Name $(Get-Date -Format o) =====`n",$utf8)
}
$trainArgs=@('scripts/training/train_joint.py','--config',$config,'--condition','proposed_ci_mi',
    '--reference-checkpoint',$reference,'--targets',$targets,'--groups',$groups,'--device','cuda',
    '--target-granularity','global','--optimization-mode',$Mode,'--output-dir',$outputRoot,'--lambda-grid',$lambda)
Invoke-Logged "train_$Mode" $trainArgs
$selected=Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
if ($selected.selected.lambda_mi -ne $null) { throw "$Mode is expected to use an unweighted multi-objective update." }
$evalArgs=@('scripts/training/evaluate_test.py','--config',$config,'--checkpoint',[string]$selected.selected.checkpoint,
    '--targets',$targets,'--output-dir',$evalDir,'--device','cuda')
Invoke-Logged "evaluate_selected_$Mode" $evalArgs
$reproduce="powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/experiments/run_optimization_ablation.ps1 -Seed $Seed -Mode $Mode"
[System.IO.File]::WriteAllText((Join-Path $runRoot 'reproduce.txt'),$reproduce,$utf8)
Write-Host "$Mode ablation complete: seed $Seed, lambda $lambda selected by validation Macro-F1."
