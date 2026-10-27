#!/usr/bin/env bash
# Upload the project to Brev instances, then install everything there.
#
#     brev/push.sh <instance> [<instance> ...]
#     SKIP_ASSETS=1 brev/push.sh <instance>        # code only, no asset download
#
# Needs `brev login` and `brev refresh` first: refresh writes an ssh alias for every
# instance into ~/.ssh/config, which is what rsync connects through. Safe to rerun -
# rsync only sends what changed, and setup.sh skips steps that are already done.
#
# What goes up from this machine (~5.5 GB the first time):
#   - the code, configs, catalogue files and scenes/ (with their .zones.json)
#   - the hand-collected models: resources/objects, resources/objects_extra,
#     resources/haven, resources/textures, ~/blenderkit_data/{models,hdrs}
#   - ~/.config/rclone/rclone.conf, so the instance can write to Google Drive
# What does not: assets/ (setup.sh downloads it on the instance), .venv, output/,
# and the folders no config reads (resources/assets, resources/objects_needs_split,
# the trash3D-* download dumps).

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "usage: $0 <instance> [<instance> ...]" >&2
    exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_DIR="syntheticData"   # relative to the remote home directory

for host in "$@"; do
    echo "[push] $host: project -> ~/$REMOTE_DIR"
    rsync -az --info=progress2 --delete \
        --exclude '.venv/' \
        --exclude 'output/' \
        --exclude 'logs/' \
        --exclude 'qa/' \
        --exclude 'exports/' \
        --exclude '/assets/' \
        --exclude '__pycache__/' \
        --exclude '*.blend1' \
        --exclude '/resources/assets/' \
        --exclude '/resources/objects_needs_split/' \
        --exclude '/resources/trash3D-*' \
        --exclude '/scenes/assets/' \
        --exclude '/.git/' \
        "$PROJECT_ROOT/" "$host:$REMOTE_DIR/"

    for sub in models hdrs; do
        if [ -d "$HOME/blenderkit_data/$sub" ]; then
            echo "[push] $host: ~/blenderkit_data/$sub"
            ssh "$host" "mkdir -p ~/blenderkit_data"
            rsync -az --info=progress2 --delete \
                "$HOME/blenderkit_data/$sub/" "$host:blenderkit_data/$sub/"
        fi
    done

    if [ -f "$HOME/.config/rclone/rclone.conf" ]; then
        echo "[push] $host: rclone config (Google Drive access)"
        ssh "$host" "mkdir -p ~/.config/rclone && chmod 700 ~/.config/rclone"
        rsync -a --chmod=F600 "$HOME/.config/rclone/rclone.conf" "$host:.config/rclone/rclone.conf"
    fi

    echo "[push] $host: running setup"
    ssh "$host" "SKIP_ASSETS=${SKIP_ASSETS:-0} bash ~/$REMOTE_DIR/brev/setup.sh"
done
