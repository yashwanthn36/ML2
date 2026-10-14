param(
    [Parameter(Mandatory=$true)][ValidateSet('natural','10','20')][string]$Factor,
    [Parameter(Mandatory=$true)][int]$Seed,
    [string]$LambdaMi
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path '.').Path
$python = (Resolve-Path '.venv\Scripts\python.exe').Path
$condition = if ($Factor -eq 'natural') { 'natural' } else { "if$Factor" }
$factorArg = if ($Factor -eq 'natural') { 'natural' } else { $Factor }
$config = "outputs\ave\$condition\seed_$Seed\experiment.yaml"
$run = "outputs\ave\$condition\seed_$Seed"
$root = Join-Path $project $run
$runRelative = "outputs/ave/$condition/seed_$Seed"
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $project 'src'
if ($previousPythonPath) { $env:PYTHONPATH += ";$previousPythonPath" }
& $python -c "from multimodal.training.provenance import ensure_protocol_output; ensure_protocol_output('$runRelative', split_protocol='official', seed=$Seed)"
$env:PYTHONPATH = $previousPythonPath
if ($LASTEXITCODE -ne 0) { throw "Could not initialize protocol marker for $run" }
$logDir = Join-Path $run 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$progressPath = Join-Path $project 'outputs\experiment_progress.jsonl'
$env:TORCH_HOME = Join-Path $project '.tmp\ave-smoke\torch'
$env:USERPROFILE = Join-Path $project '.tmp\ave-smoke\panns-home'
$env:HOME = $env:USERPROFILE

function Write-ProgressRecord([string]$State, [string]$Step) {
    $record = [ordered]@{ timestamp=(Get-Date -Format o); factor=$Factor; seed=$Seed;
        condition=$condition; state=$State; step=$Step; run_dir=$run }
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::AppendAllText($progressPath, (($record | ConvertTo-Json -Compress) + "`n"), $utf8)
}
Write-ProgressRecord 'running' 'run_started'

function Invoke-Step([string]$Name, [string[]]$Arguments) {
    $log = Join-Path $logDir ($Name + '.log')
    $start = Get-Date -Format o
    Write-ProgressRecord 'running' $Name
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::AppendAllText($log, "===== START $Name $start =====`n", $utf8)
    $stdout = "$log.stdout.tmp"
    $stderr = "$log.stderr.tmp"
    $quotedArguments = @($Arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' })
    $process = Start-Process -FilePath $python -ArgumentList $quotedArguments -WorkingDirectory $project `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if (Test-Path -LiteralPath $stdout) {
        $content = Get-Content -LiteralPath $stdout -Raw -Encoding utf8
        [System.IO.File]::AppendAllText($log, $content, $utf8)
        Write-Host $content
        Remove-Item -LiteralPath $stdout
    }
    if (Test-Path -LiteralPath $stderr) {
        $content = Get-Content -LiteralPath $stderr -Raw -Encoding utf8
        [System.IO.File]::AppendAllText($log, $content, $utf8)
        Write-Host $content
        Remove-Item -LiteralPath $stderr
    }
    if ($process.ExitCode -ne 0) {
        Write-ProgressRecord 'failed' $Name
        throw "Step failed ($($process.ExitCode)): $Name; inspect $log"
    }
    [System.IO.File]::AppendAllText($log, "===== DONE $Name $(Get-Date -Format o) =====`n", $utf8)
    Write-ProgressRecord 'completed' $Name
}

Invoke-Step 'prepare' @('scripts/experiments/prepare_synthetic_ave.py','--factor',$factorArg,'--seed',"$Seed")
$reference = Join-Path $root 'reference_best.pt'
$targets = Join-Path $root 'contributions\ucd_targets.json'
$groups = Join-Path $root 'contributions\profile_groups.json'
if (-not (Test-Path -LiteralPath $reference)) { Invoke-Step 'reference' @('scripts/training/train_reference.py','--config',$config,'--device','cuda') }
if (-not (Test-Path -LiteralPath $targets)) { Invoke-Step 'contributions' @('scripts/training/estimate_contributions.py','--config',$config,'--checkpoint',$reference,'--device','cuda') }
if (-not (Test-Path -LiteralPath (Join-Path $root 'ci_only\joint_best.pt'))) {
    Invoke-Step 'ci_only' @('scripts/training/train_joint.py','--config',$config,'--condition','ci_only','--reference-checkpoint',$reference,'--device','cuda')
}
if (-not (Test-Path -LiteralPath (Join-Path $root 'proposed_ci_mi\lambda_grid_selection.json'))) {
    $proposedArguments = @('scripts/training/train_joint.py','--config',$config,'--condition','proposed_ci_mi',
        '--reference-checkpoint',$reference,'--targets',$targets,'--groups',$groups,'--device','cuda')
    if ($LambdaMi) { $proposedArguments += @('--lambda-grid',$LambdaMi) }
    Invoke-Step 'proposed_ci_mi' $proposedArguments
}

$selection = Get-Content -LiteralPath (Join-Path $root 'proposed_ci_mi\lambda_grid_selection.json') -Raw | ConvertFrom-Json
$checkpoints = @{
    reference = $reference
    ci_only = Join-Path $root 'ci_only\joint_best.pt'
    proposed_ci_mi = [string]$selection.selected.checkpoint
}
foreach ($condition in @('reference','ci_only','proposed_ci_mi')) {
    $metrics = Join-Path $root "evaluations\$condition\test_metrics.json"
    if (-not (Test-Path -LiteralPath $metrics)) {
        Invoke-Step "evaluate_$condition" @('scripts/training/evaluate_test.py','--config',$config,'--checkpoint',$checkpoints[$condition],'--output-dir',(Join-Path $root "evaluations\$condition"),'--device','cuda')
    }
}
"===== EXPERIMENT COMPLETE factor=$Factor seed=$Seed $(Get-Date -Format o) =====" | Add-Content -LiteralPath (Join-Path $logDir 'run.log')
Write-ProgressRecord 'completed' 'run_completed'
