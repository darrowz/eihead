#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/dev-project/eihead}"
INSTALL_ROOT="${INSTALL_ROOT:-/opt/eihead}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COMMIT="${1:-$(git -C "$REPO_DIR" rev-parse --short HEAD)}"
RELEASE_DIR="$INSTALL_ROOT/releases/$COMMIT"
CURRENT_LINK="$INSTALL_ROOT/current"

if [ ! -d "$REPO_DIR" ]; then
  echo "Repository path does not exist: $REPO_DIR" >&2
  exit 2
fi

if ! git -C "$REPO_DIR" rev-parse --verify "$COMMIT^{commit}" >/dev/null 2>&1; then
  echo "Unknown commit: $COMMIT" >&2
  exit 2
fi

mkdir -p "$INSTALL_ROOT/releases"

if [ ! -d "$RELEASE_DIR" ]; then
  mkdir -p "$RELEASE_DIR"
  git -C "$REPO_DIR" archive "$COMMIT" | tar -C "$RELEASE_DIR" -xf -
fi

if [ ! -x "$RELEASE_DIR/.venv/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$RELEASE_DIR/.venv"
fi

"$RELEASE_DIR/.venv/bin/python" -m pip install --upgrade pip
"$RELEASE_DIR/.venv/bin/python" -m pip install "$RELEASE_DIR"

mkdir -p /opt/eihead /var/lib/eihead /var/log/eihead /etc/eihead
ln -sfn "$RELEASE_DIR" "$CURRENT_LINK.next"
mv -Tf "$CURRENT_LINK.next" "$CURRENT_LINK"

echo "release=$RELEASE_DIR"
echo "current=$CURRENT_LINK"
echo "commit=$COMMIT"
echo "runtime_path=$CURRENT_LINK"
