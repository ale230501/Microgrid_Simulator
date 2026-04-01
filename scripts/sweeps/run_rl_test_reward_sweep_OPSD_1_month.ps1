$repoRoot = if ($PSScriptRoot) { Split-Path -Parent (Split-Path -Parent $PSScriptRoot) } else { (Get-Location).Path }
Set-Location -Path $repoRoot

$baseOut = Join-Path $repoRoot "outputs/RL_REWARD_SWEEP_OPSD_1_MONTH"
$trainOut = Join-Path $baseOut "train"
if (-not (Test-Path $trainOut)) {
  Write-Error "Training output not found: $trainOut"
  exit 1
}

$pythonExe = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "python" }
$testScript = Join-Path $repoRoot "RL_AGENT/ems_offline_RL_agent.py"

$device = "cpu"
$env:OMP_NUM_THREADS = 2
$env:MKL_NUM_THREADS = 2
$maxParallel = 10
$testStartStep = 0
$testSteps = 5760

$datasetOPSD = Join-Path $repoRoot "data/scenario_0_timeseries_hourly.csv"
if (-not (Test-Path $datasetOPSD)) {
  Write-Error "Dataset not found: $datasetOPSD"
  exit 1
}

$logDir = Join-Path $baseOut "test_logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$models = Get-ChildItem -Path $trainOut -Recurse -Filter "model_final.zip"
if (-not $models -or $models.Count -eq 0) {
  Write-Host "No model_final.zip found under $trainOut"
  exit 0
}

Write-Host "Found $($models.Count) trained models. Max parallel tests: $maxParallel (device=$device)"

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

  @"
import sys, pathlib, yaml
cfg_path, out_path, dataset_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
rl = cfg.setdefault("rl", {})
rl["dataset_path_test"] = dataset_path
rl["eval_start_step"] = 0
rl["tb_reasoning_plot_every"] = 0
pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
"@ | & $pythonExe - $cfg.FullName $tmpCfg $datasetOPSD

  $logPath = Join-Path $logDir "$runName.log"
  Write-Host "Testing $runName"

  $job = Start-Job -ArgumentList $pythonExe, $testScript, $tmpCfg, $model.FullName, $baseOut, $runName, $device, $env:OMP_NUM_THREADS, $env:MKL_NUM_THREADS, $logPath, $repoRoot, $testStartStep, $testSteps, $datasetOPSD -ScriptBlock {
    param($pythonExe, $testScript, $cfgPath, $modelPath, $baseOut, $runName, $device, $omp, $mkl, $logPath, $repoRoot, $testStartStep, $testSteps, $datasetPath)
    $env:OMP_NUM_THREADS = $omp
    $env:MKL_NUM_THREADS = $mkl
    Set-Location -Path $repoRoot
    & $pythonExe $testScript --mode test --config $cfgPath --model-path $modelPath --output-dir $baseOut --run-name $runName --device $device --no-random-start --test-start-step $testStartStep --test-steps $testSteps --test-dataset-path $datasetPath *> $logPath
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

