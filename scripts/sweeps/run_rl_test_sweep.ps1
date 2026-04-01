param(
  [string]$repoRoot = $(if ($PSScriptRoot) { Split-Path -Parent (Split-Path -Parent $PSScriptRoot) } else { (Get-Location).Path }),
  [string]$baseOut,
  [string]$trainRootPrimary,
  [string]$trainRootSecondary,
  [ValidateSet("primary", "secondary", "both")]
  [string]$trainSource = "primary",
  [string]$datasetPath,
  [int]$testStartStep = 0,
  [int]$testSteps = 5760,
  [int]$maxParallel = 15,
  [string]$device = "cpu",
  [double]$actionDeadbandKwh = 0.0,
  [double]$socActionGuardBand = 0.0
)

Set-Location -Path $repoRoot

if (-not $baseOut) {
  $baseOut = Join-Path $repoRoot "outputs/RL_TRAINING"
}
if (-not $trainRootPrimary) {
  $trainRootPrimary = Join-Path $repoRoot "outputs/RL_TRAINING/train"
}
if (-not $trainRootSecondary) {
  $trainRootSecondary = Join-Path $repoRoot "BC_MPC/outputs/ppo_trained"
}
if (-not $datasetPath) {
  $datasetPath = Join-Path $repoRoot "data/scenario_0_timeseries_hourly.csv"
}

$pythonExe = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "python" }
$testScript = Join-Path $repoRoot "RL_AGENT/ems_offline_RL_agent.py"

$env:OMP_NUM_THREADS = 2
$env:MKL_NUM_THREADS = 2

$trainRoots = switch ($trainSource) {
  "primary" { @($trainRootPrimary) }
  "secondary" { @($trainRootSecondary) }
  "both" { @($trainRootPrimary, $trainRootSecondary) }
}

if (-not (Test-Path $datasetPath)) {
  Write-Error "Dataset not found: $datasetPath"
  exit 1
}

$logDir = Join-Path $baseOut "test_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$models = @()
foreach ($root in $trainRoots) {
  if (-not (Test-Path $root)) {
    Write-Warning "Training output not found: $root"
    continue
  }
  $models += Get-ChildItem -Path $root -Recurse -Filter "model_final.zip"
  $models += Get-ChildItem -Path $root -Recurse -Filter "ppo_trained.zip"
}

$models = $models | Sort-Object FullName -Unique
if (-not $models -or $models.Count -eq 0) {
  Write-Host "No trained models found under: $($trainRoots -join ', ')"
  exit 0
}

Write-Host "Found $($models.Count) trained models. Max parallel tests: $maxParallel (device=$device)"
Write-Host "Train roots: $($trainRoots -join ', ')"
Write-Host "Dataset: $datasetPath"

$jobs = New-Object System.Collections.ArrayList

foreach ($model in $models) {
  $runDir = $model.Directory.FullName
  $cfg = Get-ChildItem -Path $runDir -Filter "*.yml" | Select-Object -First 1
  if (-not $cfg) {
    Write-Warning "Skipping (no .yml found): $runDir"
    continue
  }

  $runName = "$($model.Directory.Name)_test"

  $tmpCfg = Join-Path $baseOut "test_configs\$runName.yml"
  @'
import sys, pathlib, yaml
cfg_path, out_path, dataset_path, action_deadband, soc_guard = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
rl = cfg.setdefault("rl", {})
rl["dataset_path_test"] = dataset_path
rl["randomize_initial_SoC"] = False
rl["action_deadband_kwh"] = float(action_deadband)
rl["soc_action_guard_band"] = float(soc_guard)
rl["tb_reasoning_plot_every"] = 0
rl["tb_enable_figures"] = False
rl["tb_enable_histograms"] = False
pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
'@ | & $pythonExe - $cfg.FullName $tmpCfg $datasetPath $actionDeadbandKwh $socActionGuardBand
  $logPath = Join-Path $logDir "$runName.log"

  Write-Host "Testing $runName"

  $job = Start-Job -ArgumentList $pythonExe, $testScript, $tmpCfg, $model.FullName, $baseOut, $runName, $device, $env:OMP_NUM_THREADS, $env:MKL_NUM_THREADS, $logPath, $repoRoot, $testStartStep, $testSteps, $datasetPath -ScriptBlock {
    param($pythonExe, $testScript, $cfgPath, $modelPath, $baseOut, $runName, $device, $omp, $mkl, $logPath, $repoRoot, $testStartStep, $testSteps, $datasetPath)
    $env:OMP_NUM_THREADS = $omp
    $env:MKL_NUM_THREADS = $mkl
    Set-Location -Path $repoRoot
    & $pythonExe $testScript --mode test --config $cfgPath --model-path $modelPath --output-dir $baseOut --run-name $runName --device $device --test-start-step $testStartStep --test-steps $testSteps --test-dataset-path $datasetPath *> $logPath
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

