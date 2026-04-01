$repoRoot = if ($PSScriptRoot) { Split-Path -Parent (Split-Path -Parent $PSScriptRoot) } else { (Get-Location).Path }
Set-Location -Path $repoRoot

$cfgRoot = Join-Path $repoRoot "configs/controllers/rl"
if (-not (Test-Path $cfgRoot)) {
  Write-Error "Config root not found: $cfgRoot"
  exit 1
}

$cfgs = Get-ChildItem -Path $cfgRoot -Recurse -Filter "*.yml" | ForEach-Object { $_.FullName }
if (-not $cfgs -or $cfgs.Count -eq 0) {
  Write-Error "No .yml configs found under $cfgRoot"
  exit 1
}
$baseOut = Join-Path $repoRoot "outputs/RL_TRAINING"
$env:OMP_NUM_THREADS = 8
$env:MKL_NUM_THREADS = 8
$maxParallel = 10
$pythonExe = if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" } else { "python" }
$trainScript = Join-Path $repoRoot "RL_AGENT/ems_offline_RL_agent.py"
$jobs = New-Object System.Collections.ArrayList
$logDir = Join-Path $baseOut "logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
Write-Host "Found $($cfgs.Count) config files. Max parallel jobs: $maxParallel"

$normVariants = @(
  @{id="n1"; obs_enabled=$true;  norm_already=$false; act_enabled=$true;  rew_enabled=$false},
  @{id="n2"; obs_enabled=$true;  norm_already=$true;  act_enabled=$true;  rew_enabled=$false},
  @{id="n3"; obs_enabled=$false; norm_already=$false; act_enabled=$false; rew_enabled=$false},
  @{id="n4"; obs_enabled=$true;  norm_already=$false; act_enabled=$true;  rew_enabled=$true},
  @{id="n5"; obs_enabled=$true;  norm_already=$true;  act_enabled=$true;  rew_enabled=$true}
)

$rewardVariants = @(
  @{id="r1"; coeff_SSR=0.0},
  @{id="r2"; coeff_SSR=2.0}
)

$socVariants = @(
  @{id="s1"; randomize=$true},
  @{id="s2"; randomize=$false}
)

foreach ($cfg in $cfgs) {
  $base = [IO.Path]::GetFileNameWithoutExtension($cfg)

  foreach ($n in $normVariants) {
    foreach ($r in $rewardVariants) {
      foreach ($s in $socVariants) {
        $nId = $n['id']
        $rId = $r['id']
        $sId = $s['id']
        $runName = "${base}_${nId}_${rId}_${sId}"
        $tmp = Join-Path $baseOut "configs\$runName.yml"

        @'
import sys, pathlib, yaml
cfg_path, out_path = sys.argv[1], sys.argv[2]
obs_enabled = sys.argv[3] == "True"
norm_already = sys.argv[4] == "True"
act_enabled = sys.argv[5] == "True"
rew_enabled = sys.argv[6] == "True"
coeff_ssr = float(sys.argv[7])
rand_soc = sys.argv[8] == "True"

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

rl = cfg.setdefault("rl", {})
norm = rl.setdefault("normalization", {})
obs = norm.setdefault("observations", {})
act = norm.setdefault("actions", {})
rew = norm.setdefault("reward", {})

obs["enabled"] = obs_enabled
obs["normalize_already_scaled"] = norm_already
act["enabled"] = act_enabled
rew["enabled"] = rew_enabled

reward = rl.setdefault("reward", {})
reward["coeff_SSR"] = coeff_ssr

rl["randomize_initial_SoC"] = rand_soc

pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
'@ | & $pythonExe - $cfg $tmp $($n["obs_enabled"]) $($n["norm_already"]) $($n["act_enabled"]) $($n["rew_enabled"]) $($r["coeff_SSR"]) $($s["randomize"])

        $logPath = Join-Path $logDir "$runName.log"
        Write-Host "Starting $runName"
        $job = Start-Job -ArgumentList $pythonExe, $trainScript, $tmp, $runName, $baseOut, $env:OMP_NUM_THREADS, $env:MKL_NUM_THREADS, $logPath, $repoRoot -ScriptBlock {
          param($pythonExe, $trainScript, $tmp, $runName, $baseOut, $omp, $mkl, $logPath, $repoRoot)
          $env:OMP_NUM_THREADS = $omp
          $env:MKL_NUM_THREADS = $mkl
          Set-Location -Path $repoRoot
          & $pythonExe $trainScript --mode train --config $tmp --run-name $runName --output-dir $baseOut *> $logPath
        }
        [void]$jobs.Add($job)

        if ($jobs.Count -ge $maxParallel) {
          $done = Wait-Job -Job $jobs -Any
          [void]$jobs.Remove($done)
          Remove-Job -Job $done
        }
      }
    }
  }
}

if ($jobs.Count -gt 0) {
  Wait-Job -Job $jobs | Out-Null
  Remove-Job -Job $jobs
}
