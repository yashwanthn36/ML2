param(
    [Parameter(Mandatory=$true)][ValidateSet('natural','10','20')][string]$Factor,
    [Parameter(Mandatory=$true)][int]$Seed,
    [Parameter(Mandatory=$true)][string]$AdditionalLambda
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path '.').Path
$python = (Resolve-Path '.venv\Scripts\python.exe').Path
$condition = if ($Factor -eq 'natural') { 'natural' } else { "if$Factor" }
$run = Join-Path $project "outputs\ave\$condition\seed_$Seed"
$config = Join-Path $run 'experiment.yaml'
$targets = Join-Path $run 'contributions\ucd_targets.json'
$groups = Join-Path $run 'contributions\profile_groups.json'
$base = Join-Path $run 'proposed_ci_mi'
$selection = Join-Path $base 'lambda_grid_selection.json'
$attempts = Join-Path $run 'attempts'
$logs = Join-Path $run 'logs'
New-Item -ItemType Directory -Force -Path $attempts,$logs | Out-Null
if (-not (Test-Path $selection)) { throw "No completed initial validation selection exists: $selection" }
$lambdaValues = @($AdditionalLambda -split '[\s,;]+' | Where-Object { $_ } | ForEach-Object { [string][double]::Parse($_,[Globalization.CultureInfo]::InvariantCulture) })
$currentSelection = Get-Content -LiteralPath $selection -Raw | ConvertFrom-Json
$present = @($currentSelection.candidates | ForEach-Object { [double]$_.lambda_mi })
$lambdaValues = @($lambdaValues | Where-Object { $candidate=[double]$_; $candidate -notin $present })
$evalDir = Join-Path $run 'evaluations\proposed_ci_mi'
if ($lambdaValues.Count -eq 0 -and (Test-Path (Join-Path $evalDir 'test_metrics.json'))) { Write-Host 'Requested lambda candidates and selected test evaluation are already complete.'; exit 0 }
$initial = Join-Path $attempts 'lambda_grid_initial.json'
if (-not (Test-Path $initial)) { Copy-Item -LiteralPath $selection -Destination $initial }

function Invoke-Logged([string]$Name,[string[]]$Arguments) {
    $log = Join-Path $logs "$Name.log"
    $stdout = "$log.stdout.tmp"; $stderr = "$log.stderr.tmp"
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::AppendAllText($log, "===== START $Name $(Get-Date -Format o) =====`n", $utf8)
    $quoted = @($Arguments | ForEach-Object { '"' + $_.Replace('"','\"') + '"' })
    $process = Start-Process -FilePath $python -ArgumentList $quoted -WorkingDirectory $project `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    foreach ($file in @($stdout,$stderr)) {
        if (Test-Path -LiteralPath $file) {
            $content = Get-Content -LiteralPath $file -Raw -Encoding utf8
            [System.IO.File]::AppendAllText($log, $content, $utf8)
            Write-Host $content
            Remove-Item -LiteralPath $file
        }
    }
    if ($process.ExitCode -ne 0) { throw "$Name failed with exit code $($process.ExitCode); see $log" }
    [System.IO.File]::AppendAllText($log, "===== DONE $Name $(Get-Date -Format o) =====`n", $utf8)
}

$lambdaArgs = @('scripts/training/train_joint.py','--config',$config,'--condition','proposed_ci_mi',
    '--reference-checkpoint',(Join-Path $run 'reference_best.pt'),'--targets',$targets,'--groups',$groups,'--device','cuda','--lambda-grid')
$lambdaArgs += $lambdaValues
Invoke-Logged 'proposed_high_lambda_grid' $lambdaArgs
$high = Get-Content -LiteralPath $selection -Raw | ConvertFrom-Json
$allCandidates = @($currentSelection.candidates) + @($high.candidates)
$candidateMap = @{}
foreach ($candidate in $allCandidates) { $key=([double]$candidate.lambda_mi).ToString('G',[Globalization.CultureInfo]::InvariantCulture); $candidateMap[$key]=$candidate }
$candidates = @($candidateMap.Values)
$selected = $candidates | Sort-Object validation_macro_f1 -Descending | Select-Object -First 1
$combined = [ordered]@{ model_condition='proposed_ci_mi'; seed=$Seed; split_protocol='official';
    selection_metric='validation_macro_f1'; candidate_lambda_grid=@($candidates | ForEach-Object { $_.lambda_mi });
    candidates=$candidates; selected=$selected }
$json = $combined | ConvertTo-Json -Depth 30
[System.IO.File]::WriteAllText($selection,$json,[System.Text.UTF8Encoding]::new($false))

$oldEval = Join-Path $run 'evaluations\proposed_ci_mi_before_extended_grid'
if (Test-Path -LiteralPath $evalDir) {
    if (-not (Test-Path -LiteralPath $oldEval)) { Move-Item -LiteralPath $evalDir -Destination $oldEval }
    else {
        $repeatEval = Join-Path $attempts ('proposed_ci_mi_evaluation_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
        Move-Item -LiteralPath $evalDir -Destination $repeatEval
    }
}
Invoke-Logged 'evaluate_proposed_extended_grid' @('scripts/training/evaluate_test.py','--config',$config,
    '--checkpoint',[string]$selected.checkpoint,'--targets',$targets,'--output-dir',$evalDir,'--device','cuda')
"Selected lambda $($selected.lambda_mi) by validation Macro-F1=$($selected.validation_macro_f1). Test evaluation saved under $evalDir."


