#!/usr/bin/env bash
# Local CPU-only environment; no apt, system pip, model imports or downloads.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" >/dev/null; then
    printf '%s\n' "Missing $PYTHON_BIN. Install an approved Python 3.11, or explicitly select an available interpreter via PYTHON_BIN."
    exit 1
fi
if [[ ! -d .venv ]]; then
    "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -c 'import sys; print(sys.version); print(sys.executable)'
.venv/bin/python -m pip install -r requirements-relay-dev.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest
