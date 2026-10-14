param(
    [Parameter(Mandatory=$true)][ValidateSet('class','group','reliability')][string]$Granularity,
    [int]$Seed = 23
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path '.').Path
$python = (Resolve-Path '.venv\Scripts\python.exe').Path
$sourceRun = Join-Path $project "outputs\ave\natural\seed_$Seed"
$config = "outputs/ave/natural/seed_$Seed/experiment.yaml"
$reference = Join-Path $sourceRun 'reference_best.pt'
$targets = Join-Path $sourceRun 'contributions\ucd_targets.json'
$groups = Join-Path $sourceRun 'contributions\profile_groups.json'
$selected = Get-Content -LiteralPath (Join-Path $sourceRun 'proposed_ci_mi\lambda_grid_selection.json') -Raw | ConvertFrom-Json
$lambda = [string]$selected.selected.lambda_mi
if ([string]::IsNullOrWhiteSpace($lambda)) { throw "No validation-selected lambda_mi found for seed $Seed" }
$outputRoot = "outputs/ave/granularity/$Granularity"
$runRoot = Join-Path $project "$outputRoot\seed_$Seed"
$log = Join-Path $project "outputs\logs\granularity_${Granularity}_seed_$Seed.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
$utf8 = [System.Text.UTF8Encoding]::new($false)
$trainArgs = @('scripts/training/train_joint.py','--config',$config,'--condition','proposed_ci_mi',
    '--reference-checkpoint',$reference,'--targets',$targets,'--groups',$groups,'--device','cuda',
    '--target-granularity',$Granularity,'--output-dir',$outputRoot,'--lambda-grid',$lambda)
$quoted = @($trainArgs | ForEach-Object { '"' + $_.Replace('"','\"') + '"' })
function Invoke-Logged([string]$Name,[string[]]$StepArguments) {
    [System.IO.File]::AppendAllText($log,"===== START $Name $(Get-Date -Format o) =====`n",$utf8)
    $stdout="$log.stdout.tmp";$stderr="$log.stderr.tmp"
    if ($null -eq $StepArguments -or $StepArguments.Count -eq 0 -or $StepArguments -contains $null) {
        throw "No valid arguments supplied for step $Name"
    }
    $process=Start-Process -FilePath $python -ArgumentList $StepArguments -WorkingDirectory $project -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    foreach($file in @($stdout,$stderr)) { if(Test-Path $file) { $content=Get-Content $file -Raw -Encoding utf8;[System.IO.File]::AppendAllText($log,$content,$utf8);Write-Host $content;Remove-Item $file } }
    if($process.ExitCode -ne 0) { throw "$Name failed with exit code $($process.ExitCode); see $log" }
    [System.IO.File]::AppendAllText($log,"===== DONE $Name $(Get-Date -Format o) =====`n",$utf8)
}
Invoke-Logged 'train' $quoted
$selectionPath=Join-Path $runRoot 'proposed_ci_mi\lambda_grid_selection.json'
$selection=Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
$checkpoint=[string]$selection.selected.checkpoint
$evalDir=Join-Path $runRoot 'evaluations\proposed_ci_mi'
$evalArgs=@('scripts/training/evaluate_test.py','--config',$config,'--checkpoint',$checkpoint,
    '--targets',$targets,'--output-dir',$evalDir,'--device','cuda') | ForEach-Object { '"' + $_.Replace('"','\"') + '"' }
Invoke-Logged 'evaluate_selected' $evalArgs
$command="powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/experiments/run_granularity_ablation.ps1 -Granularity $Granularity -Seed $Seed"
[System.IO.File]::WriteAllText((Join-Path $runRoot 'reproduce.txt'),$command,$utf8)
"Completed $Granularity target granularity for seed $Seed; lambda $lambda selected by validation Macro-F1."
