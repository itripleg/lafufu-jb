#!/bin/bash
# Lafufu startup: activate venv, run dynamixel.py.
# Video overlay is handled by ~/.config/labwc/autostart in kiosk mode,
# not by this script — so we don't double-spawn cvlc.
#
# Overridable via env:
#   LAFUFU_REPO_DIR   repo path (default: script's own directory)
#   LAFUFU_VENV       venv path (default: $HOME/lafufu-env)
#   LAFUFU_ARGS       extra args passed to dynamixel.py (default: --no-printer)

set -euo pipefail

REPO_DIR="${LAFUFU_REPO_DIR:-$(dirname "$(readlink -f "$0")")}"
VENV="${LAFUFU_VENV:-$HOME/lafufu-env}"
read -r -a ARGS <<< "${LAFUFU_ARGS:---no-printer}"

cd "$REPO_DIR"

if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "startup.sh: venv not found at $VENV" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

exec python -u dynamixel.py "${ARGS[@]}"
