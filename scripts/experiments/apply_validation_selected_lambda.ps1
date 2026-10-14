param(
    [Parameter(Mandatory=$true)][ValidateSet('natural','10','20')][string]$Factor,
    [Parameter(Mandatory=$true)][int[]]$Seed,
    [Parameter(Mandatory=$true)][string]$Lambda
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path '.').Path
$python = (Resolve-Path '.venv\Scripts\python.exe').Path
$condition = if ($Factor -eq 'natural') { 'natural' } else { "if$Factor" }
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $project 'src'
if ($previousPythonPath) { $env:PYTHONPATH += ";$previousPythonPath" }

foreach ($currentSeed in $Seed) {
    $run = Join-Path $project "outputs\ave\$condition\seed_$currentSeed"
    $config = Join-Path $run 'experiment.yaml'
    $reference = Join-Path $run 'reference_best.pt'
    $targets = Join-Path $run 'contributions\ucd_targets.json'
    $groups = Join-Path $run 'contributions\profile_groups.json'
    $base = Join-Path $run 'proposed_ci_mi'
    $selectionPath = Join-Path $base 'lambda_grid_selection.json'
    $selection = if (Test-Path -LiteralPath $selectionPath) {
        Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
    } else { $null }
    $previousEvaluation = Join-Path $run 'evaluations\proposed_ci_mi'
    $requested = [double]::Parse($Lambda, [Globalization.CultureInfo]::InvariantCulture)
    $isSelected = $selection -and [math]::Abs([double]$selection.selected.lambda_mi - $requested) -lt 1e-12
    $checkpoint = Join-Path $base "lambda_$Lambda\joint_best.pt"
    $metrics = Join-Path $previousEvaluation 'test_metrics.json'
    if ($isSelected -and (Test-Path -LiteralPath $checkpoint) -and (Test-Path -LiteralPath $metrics)) {
        Write-Host "Seed $currentSeed already has selected lambda $Lambda and its test evaluation."
        continue
    }
    if (Test-Path -LiteralPath $previousEvaluation) {
        $archive = Join-Path $run ("evaluations\proposed_ci_mi_before_lambda_" +
            ([string]$selection.selected.lambda_mi).Replace('.', '_') + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
        Move-Item -LiteralPath $previousEvaluation -Destination $archive
    }
    $trainLog = Join-Path $run "logs\selected_lambda_$Lambda.log"
    New-Item -ItemType Directory -Force -Path (Split-Path $trainLog) | Out-Null
    Write-Host "Seed ${currentSeed}: training validation-selected lambda $Lambda."
    & $python 'scripts/training/train_joint.py' --config $config --condition proposed_ci_mi `
        --reference-checkpoint $reference --targets $targets --groups $groups --device cuda --lambda-grid $Lambda *>&1 |
        Tee-Object -FilePath $trainLog -Append
    if ($LASTEXITCODE -ne 0) { throw "Seed $currentSeed selected-lambda training failed; see $trainLog" }
    $selection = Get-Content -LiteralPath $selectionPath -Raw | ConvertFrom-Json
    $selectedLambda = [double]$selection.selected.lambda_mi
    if ([math]::Abs($selectedLambda - $requested) -ge 1e-12) {
        throw "Seed $currentSeed did not select the requested fixed lambda $Lambda"
    }
    $evalDir = Join-Path $run 'evaluations\proposed_ci_mi'
    $evalLog = Join-Path $run "logs\selected_lambda_$Lambda`_test.log"
    & $python 'scripts/training/evaluate_test.py' --config $config --checkpoint ([string]$selection.selected.checkpoint) `
        --targets $targets --output-dir $evalDir --device cuda *>&1 | Tee-Object -FilePath $evalLog -Append
    if ($LASTEXITCODE -ne 0) { throw "Seed $currentSeed selected-lambda test evaluation failed; see $evalLog" }
    Write-Host "Seed $currentSeed complete: validation selected lambda $Lambda and final test metrics are under $evalDir."
}
$env:PYTHONPATH = $previousPythonPath
