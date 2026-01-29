#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$ROOT/.venv-cleanroom" ]; then
  echo "Missing .venv-cleanroom. Run scripts/setup_env.sh first."
  exit 1
fi

source "$ROOT/.venv-cleanroom/bin/activate"
cd "$ROOT"

jupyter lab notebooks/elections_analysis.ipynb
