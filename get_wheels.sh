#!/usr/bin/env bash
set -euo pipefail

# Simple helper to prefetch binary wheels for offline installs.
# Note: This add-on no longer requires Open3D or Torch.
#
# Env overrides:
#   PYTHON_BIN          (default: python)
#   PYTHON_VERSION      (default: detected from Poetry env, e.g., 3.11 -> "311")
#   PLATFORM_TAG        (default: win_amd64)
#   WHEELS_DIR          (default: wheels)

PY_BIN="${PYTHON_BIN:-python}"
PIP="poetry run ${PY_BIN} -m pip"
WHEELS_DIR="${WHEELS_DIR:-wheels}"
mkdir -p "$WHEELS_DIR"

# Detect Python tag (e.g., 311) if not provided
if [ -z "${PYTHON_VERSION:-}" ]; then
  PYTHON_VERSION=$(poetry run ${PY_BIN} - <<'PY'
import sys;print(f"{sys.version_info.major}{sys.version_info.minor}")
PY
)
fi

PLATFORM_TAG="${PLATFORM_TAG:-win_amd64}"
# echo "No Open3D/Torch required; fetching core pure-Python deps only..."

# # Explicitly download helper deps we rely on at runtime (most are pure-Python)
# ${PIP} download typing_extensions sympy networkx filelock fsspec jinja2 MarkupSafe \
#   --dest "$WHEELS_DIR" \
#   --only-binary=:all: \
#   --python-version="${PYTHON_VERSION}" \
#   --platform="${PLATFORM_TAG}" || true

echo "Wheels downloaded to: ${WHEELS_DIR}"
