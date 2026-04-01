param(
    [string]$DiffRef = "HEAD"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Normalize-Path([string]$pathValue) {
    return ($pathValue -replace "\\", "/").Trim()
}

$fileMap = @{
    "SIMULATOR/microgrid_simulator.py"     = "docs/tex/files/simulator_microgrid_simulator.tex"
    "SIMULATOR/tools.py"                   = "docs/tex/files/simulator_tools.tex"
    "RULE_BASED/ems_offline.py"            = "docs/tex/files/rbc_ems_offline.tex"
    "RULE_BASED/RBC_EMS.py"                = "docs/tex/files/rbc_core_controller.tex"
    "MODEL_PREDICTIVE/ems_offline_mpc_v0.py" = "docs/tex/files/mpc_ems_offline.tex"
    "MODEL_PREDICTIVE/mpc_MILP.py"         = "docs/tex/files/mpc_milp_core.tex"
    "RL_AGENT/ems_offline_RL_agent.py"     = "docs/tex/files/rl_runner.tex"
    "RL_AGENT/EMS_RL_agent.py"             = "docs/tex/files/rl_env_agent_core.tex"
}

$changedRaw = git diff --name-only $DiffRef
$untrackedRaw = git ls-files --others --exclude-standard
$changed = @(
    ($changedRaw + $untrackedRaw) |
    ForEach-Object { Normalize-Path $_ } |
    Where-Object { $_ }
)

if ($changed.Count -eq 0) {
    Write-Host "[OK] Nessuna modifica rispetto a $DiffRef."
    exit 0
}

$changedSet = New-Object "System.Collections.Generic.HashSet[string]" ([System.StringComparer]::OrdinalIgnoreCase)
foreach ($p in $changed) { [void]$changedSet.Add($p) }

$missing = @()
foreach ($src in $fileMap.Keys) {
    $doc = $fileMap[$src]
    if ($changedSet.Contains($src) -and -not $changedSet.Contains($doc)) {
        $missing += "$src -> $doc"
    }
}

if ($missing.Count -gt 0) {
    Write-Host "[ERROR] Documentazione non allineata per i seguenti file core:"
    $missing | ForEach-Object { Write-Host " - $_" }
    Write-Host ""
    Write-Host "Aggiorna i file .tex indicati e riesegui questo check."
    exit 1
}

Write-Host "[OK] Codice core e documentazione .tex risultano allineati."
exit 0
