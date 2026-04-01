$repoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { (Get-Location).Path }
$target = Join-Path $repoRoot "scripts/sweeps/run_rl_test_reward_sweep_OPSD_1_week.ps1"
& $target @args