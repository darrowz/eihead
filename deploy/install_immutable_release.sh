#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/dev-project/eihead}"
INSTALL_ROOT="${INSTALL_ROOT:-/opt/eihead}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EIPROTOCOL_DIR="${EIPROTOCOL_DIR:-}"
PIPER_MODEL_DIR="${PIPER_MODEL_DIR:-/var/lib/eihead/models/piper}"
PIPER_MODEL_NAME="${PIPER_MODEL_NAME:-zh_CN-huayan-medium}"
PIPER_MODEL_BASE_URL="${PIPER_MODEL_BASE_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium}"
INSTALL_PIPER_MODEL="${INSTALL_PIPER_MODEL:-auto}"
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
if [ -z "$EIPROTOCOL_DIR" ]; then
  for candidate in /dev-project/eiprotocol /dev-project/ei-workspace/repos/eiprotocol; do
    if [ -f "$candidate/pyproject.toml" ]; then
      EIPROTOCOL_DIR="$candidate"
      break
    fi
  done
fi
if [ -n "$EIPROTOCOL_DIR" ]; then
  "$RELEASE_DIR/.venv/bin/python" -m pip install "$EIPROTOCOL_DIR"
fi
"$RELEASE_DIR/.venv/bin/python" -m pip install "$RELEASE_DIR"

mkdir -p /opt/eihead /var/lib/eihead /var/log/eihead
if [ "$INSTALL_PIPER_MODEL" != "0" ] && [ "$INSTALL_PIPER_MODEL" != "false" ]; then
  mkdir -p "$PIPER_MODEL_DIR"
  for suffix in onnx onnx.json; do
    target="$PIPER_MODEL_DIR/$PIPER_MODEL_NAME.$suffix"
    if [ ! -s "$target" ]; then
      url="$PIPER_MODEL_BASE_URL/$PIPER_MODEL_NAME.$suffix"
      tmp="$target.tmp"
      if command -v curl >/dev/null 2>&1; then
        curl -fL "$url" -o "$tmp"
      elif command -v wget >/dev/null 2>&1; then
        wget -O "$tmp" "$url"
      else
        echo "Neither curl nor wget is available to download Piper model: $url" >&2
        exit 2
      fi
      mv "$tmp" "$target"
    fi
  done
fi

sudo mkdir -p /etc/eihead
if [ -f "$RELEASE_DIR/config/eihead.honjia.yaml" ]; then
  if sudo test -f /etc/eihead/eihead.honjia.yaml && ! sudo cmp -s "$RELEASE_DIR/config/eihead.honjia.yaml" /etc/eihead/eihead.honjia.yaml; then
    sudo cp /etc/eihead/eihead.honjia.yaml "/etc/eihead/eihead.honjia.yaml.bak.$(date -u +%Y%m%d%H%M%S)"
  fi
  sudo cp "$RELEASE_DIR/config/eihead.honjia.yaml" /etc/eihead/eihead.honjia.yaml
fi
ln -sfn "$RELEASE_DIR" "$CURRENT_LINK.next"
mv -Tf "$CURRENT_LINK.next" "$CURRENT_LINK"

INSTALLED_UNITS=()
for unit in eihead-runtime.service eihead-monitor.service eihead-vision-hailo.service; do
  if [ -f "$RELEASE_DIR/deploy/systemd/$unit" ]; then
    sudo cp "$RELEASE_DIR/deploy/systemd/$unit" "/etc/systemd/system/$unit"
    INSTALLED_UNITS+=("$unit")
  fi
done

if [ "${#INSTALLED_UNITS[@]}" -gt 0 ]; then
  sudo systemctl daemon-reload
  for unit in "${INSTALLED_UNITS[@]}"; do
    sudo systemctl enable "$unit" >/dev/null
  done
fi

for unit in eihead-runtime.service eihead-monitor.service eihead-vision-hailo.service; do
  if printf '%s\n' "${INSTALLED_UNITS[@]}" | grep -qx "$unit"; then
    sudo systemctl restart "$unit"
  fi
done

if [ "${#INSTALLED_UNITS[@]}" -gt 0 ]; then
  echo "systemd_units=${INSTALLED_UNITS[*]}"
fi

echo "release=$RELEASE_DIR"
echo "current=$CURRENT_LINK"
echo "commit=$COMMIT"
echo "runtime_path=$CURRENT_LINK"
