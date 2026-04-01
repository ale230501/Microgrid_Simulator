param(
  [string]$repoRoot = $(if ($PSScriptRoot) { Split-Path -Parent (Split-Path -Parent $PSScriptRoot) } else { (Get-Location).Path }),
  [string]$mpcRoot,
  [string]$inferenceDataset,
  [string]$forecastDataset,
  [int]$startStep = 0,
  [int]$endStep,
  [int]$maxParallel = 4
)

Set-Location -Path $repoRoot

if (-not $mpcRoot) {
  $mpcRoot = Join-Path $repoRoot "outputs/MPC"
}
if (-not $inferenceDataset) {
  $inferenceDataset = Join-Path $repoRoot "data/scenario_14_timeseries_hourly.csv"
}
if (-not $forecastDataset) {
  $forecastDataset = $inferenceDataset
}

if (-not (Test-Path $mpcRoot)) {
  Write-Error "MPC output root not found: $mpcRoot"
  exit 1
}
if (-not (Test-Path $inferenceDataset)) {
  Write-Error "Inference dataset not found: $inferenceDataset"
  exit 1
}
if (-not (Test-Path $forecastDataset)) {
  Write-Error "Forecast dataset not found: $forecastDataset"
  exit 1
}

$pythonExe = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "python" }
$mpcScript = Join-Path $repoRoot "MODEL_PREDICTIVE/ems_offline_mpc_v0.py"

$logDir = Join-Path $mpcRoot "test_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$runs = Get-ChildItem -Path $mpcRoot -Directory
if (-not $runs -or $runs.Count -eq 0) {
  Write-Host "No MPC run folders found under: $mpcRoot"
  exit 0
}

$jobs = New-Object System.Collections.ArrayList

foreach ($run in $runs) {
  $cfg = Get-ChildItem -Path $run.FullName -Filter "params_OPSD.yml" | Select-Object -First 1
  if (-not $cfg) {
    $cfg = Get-ChildItem -Path $run.FullName -Filter "*.yml" | Select-Object -First 1
  }
  if (-not $cfg) {
    Write-Warning "Skipping (no .yml found): $($run.FullName)"
    continue
  }

  $runName = $run.Name
  $logPath = Join-Path $logDir "$runName.log"

  $argsList = @(
    $mpcScript,
    "--config", $cfg.FullName,
    "--MPC_inference_dataset_path", $inferenceDataset,
    "--MPC_forecast_dataset_path", $forecastDataset,
    "--start-step", $startStep
  )
  if ($PSBoundParameters.ContainsKey("endStep")) {
    $argsList += @("--end-step", $endStep)
  }

  Write-Host "Running MPC for $runName"

  $job = Start-Job -ArgumentList $pythonExe, $argsList, $logPath, $repoRoot -ScriptBlock {
    param($pythonExe, $argsList, $logPath, $repoRoot)
    Set-Location -Path $repoRoot
    & $pythonExe @argsList *> $logPath
  }
  [void]$jobs.Add($job)

  if ($jobs.Count -ge $maxParallel) {
    $done = Wait-Job -Job $jobs -Any
    [void]$jobs.Remove($done)
    Remove-Job -Job $done
  }
}

if ($jobs.Count -gt 0) {
  Wait-Job -Job $jobs | Out-Null
  Remove-Job -Job $jobs
}

Write-Host "Done. Logs: $logDir"

