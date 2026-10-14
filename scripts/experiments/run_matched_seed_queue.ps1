param(
    [string]$Seeds = '73 123',
    [string]$Factors = 'natural 10 20'
)
$ErrorActionPreference = 'Stop'
$seedValues = @($Seeds -split '[\s,;]+' | Where-Object { $_ } | ForEach-Object { [int]$_ })
$factorValues = @($Factors -split '[\s,;]+' | Where-Object { $_ })
if (-not $seedValues.Count -or -not $factorValues.Count) { throw 'Provide at least one seed and factor.' }
if ($factorValues | Where-Object { $_ -notin @('natural','10','20') }) {
    throw 'Factors must be natural, 10, or 20.'
}
$project = (Resolve-Path '.').Path
$log = Join-Path $project 'outputs\logs\matched_seed_queue.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
$utf8 = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::AppendAllText($log, "===== QUEUE START seeds=$($seedValues -join ',') factors=$($factorValues -join ',') $(Get-Date -Format o) =====`n", $utf8)
foreach ($seedValue in $seedValues) {
    foreach ($factorValue in $factorValues) {
        Write-Host "Starting matched AVE condition factor=$factorValue seed=$seedValue"
        [System.IO.File]::AppendAllText($log, "===== START factor=$factorValue seed=$seedValue $(Get-Date -Format o) =====`n", $utf8)
        $runArguments = @('-NoProfile','-ExecutionPolicy','Bypass','-File',
            'scripts\experiments\run_synthetic_ave.ps1','-Factor',$factorValue,'-Seed',"$seedValue")
        if ($factorValue -eq '20') { $runArguments += @('-LambdaMi','0.03') }
        & powershell.exe @runArguments
        $runExitCode = $LASTEXITCODE
        if ($runExitCode -ne 0) {
            [System.IO.File]::AppendAllText($log, "===== FAILED factor=$factorValue seed=$seedValue exit=$runExitCode $(Get-Date -Format o) =====`n", $utf8)
            throw "Matched run failed for factor=$factorValue seed=$seedValue; see $log and outputs/experiment_progress.jsonl"
        }
    }
}
[System.IO.File]::AppendAllText($log, "===== QUEUE COMPLETE $(Get-Date -Format o) =====`n", $utf8)
Write-Host 'All requested matched seed conditions completed.'
