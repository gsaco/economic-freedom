#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-cleanroom"

if [ ! -d "$VENV" ]; then
  python -m venv "$VENV"
fi

source "$VENV/bin/activate"
python -m pip install --upgrade pip
pip install -r "$ROOT/requirements.txt"

# Register kernel for notebook use
python -m ipykernel install --user --name economic-freedom-cleanroom --display-name "Python (economic-freedom-cleanroom)"

printf '\nSetup complete. Activate with:\n  source .venv-cleanroom/bin/activate\n'
