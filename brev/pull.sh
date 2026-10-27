#!/usr/bin/env bash
# Download rendered scenes from Brev instances and merge them into one COCO file.
#
#     brev/pull.sh [--config config.yaml] <instance> [<instance> ...]
#
# Instances render disjoint scene_XXXXX directories, so their outputs drop into the
# same local output/<name>/ without clashing. The per-instance merged
# coco_annotations.json is skipped - the local merge at the end replaces it. Safe to
# run while rendering is still going; rerun later to fetch the rest.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="config.yaml"
hosts=()

while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        -*) echo "unknown argument: $1" >&2; exit 2 ;;
        *) hosts+=("$1"); shift ;;
    esac
done

if [ ${#hosts[@]} -eq 0 ]; then
    echo "usage: $0 [--config config.yaml] <instance> [<instance> ...]" >&2
    exit 2
fi

OUTPUT_DIR=$(grep -E '^output_dir:' "$PROJECT_ROOT/$CONFIG" | awk '{print $2}' | tr -d '"')
mkdir -p "$PROJECT_ROOT/$OUTPUT_DIR"

for host in "${hosts[@]}"; do
    echo "[pull] $host: $OUTPUT_DIR"
    rsync -az --info=progress2 \
        --exclude '/coco_annotations.json' \
        "$host:syntheticData/$OUTPUT_DIR/" "$PROJECT_ROOT/$OUTPUT_DIR/"
done

"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/run_batch.py" --config "$CONFIG" --merge-only
