$repoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { (Get-Location).Path }
$target = Join-Path $repoRoot "scripts/sweeps/run_mpc_sweep.ps1"
& $target @args