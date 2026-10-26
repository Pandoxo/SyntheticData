#!/usr/bin/env bash
# One-time install on a Brev instance. push.sh runs this; rerunning it is harmless -
# every step skips work that is already done.
#
#   1. system libraries headless Blender needs
#   2. a venv with BlenderProc and the helper scripts' dependencies
#   3. rclone (Google Drive), configured by push.sh copying ~/.config/rclone/rclone.conf
#   4. Blender 4.2.1, downloaded by BlenderProc into ~/blender, and its Python
#      packages installed by one serial run (brev/check_blender.py)
#   5. the environment assets and the Objaverse litter models - fetched here, on the
#      data-centre connection, rather than uploaded from a home line
#   6. spawn-zone files for any scene that does not have one yet
#
# Skip the asset download (e.g. to test the install only) with SKIP_ASSETS=1.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"

if command -v apt-get >/dev/null; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        python3-venv rsync tmux unzip curl \
        libxrender1 libxi6 libxkbcommon0 libxxf86vm1 libxfixes3 libsm6 libgl1 libegl1
fi

if [ ! -x .venv/bin/blenderproc ]; then
    python3 -m venv .venv
    .venv/bin/pip install -q --upgrade pip
fi
.venv/bin/pip install -q -r brev/requirements.txt

if ! command -v rclone >/dev/null; then
    tmp=$(mktemp -d)
    curl -fsSL -o "$tmp/rclone.zip" https://downloads.rclone.org/rclone-current-linux-amd64.zip
    unzip -q "$tmp/rclone.zip" -d "$tmp"
    mkdir -p "$HOME/.local/bin"
    install -m755 "$tmp"/rclone-*-linux-amd64/rclone "$HOME/.local/bin/rclone"
    rm -rf "$tmp"
fi

# The first `blenderproc run` downloads Blender and installs BlenderProc's packages
# into it. It must happen once, alone, before parallel render jobs start - see
# brev/check_blender.py.
.venv/bin/blenderproc pip install debugpy
.venv/bin/blenderproc run brev/check_blender.py

if [ "${SKIP_ASSETS:-0}" != "1" ]; then
    # An asset download started by hand in a tmux session called "setup" (to overlap
    # it with a slow upload) finishes first; the fetchers would skip its files anyway.
    while tmux has-session -t setup 2>/dev/null; do sleep 15; done
    .venv/bin/python fetch_assets.py --profile full --jobs 16
    .venv/bin/python fetch_objaverse.py --jobs 16
    .venv/bin/blenderproc run prepare_scenes.py -- --config config.yaml --missing-only
fi

echo
nvidia-smi -L
rclone listremotes 2>/dev/null | sed 's/^/[setup] rclone remote: /' || true
echo "[setup] done - render with: bash ~/syntheticData/brev/render.sh --help"
