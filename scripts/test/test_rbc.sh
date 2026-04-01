#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Offline RBC test runner.

CONFIG="configs/controllers/rbc/params_OPSD.yml"
START_STEP=0
END_STEP=672

python RULE_BASED/ems_offline.py \
  --config "${CONFIG}" \
  --start-step "${START_STEP}" \
  --end-step "${END_STEP}"

