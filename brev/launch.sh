#!/usr/bin/env bash
# Split a dataset across Brev instances and start rendering on each, in tmux.
#
#     brev/launch.sh [--config config.yaml] [--script generate.py] [--start 0]
#                    [--scenes N] [--jobs-per-gpu 2] <instance> [<instance> ...]
#
# Each instance gets a contiguous, disjoint slice of [start, start + scenes) and
# renders it on all of its GPUs with render.sh. Scene index = seed, so the slices
# need no coordination and merge cleanly afterwards (pull.sh).
#
# Rendering lives in a tmux session called `render`, so closing this terminal or
# losing the ssh connection does not stop it. Watch it with:
#     ssh <instance> -t tmux attach -t render        (detach: Ctrl-b d)
#     ssh <instance> tail -f syntheticData/logs/gpu0.log

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="config.yaml"
SCRIPT="generate.py"
START=0
SCENES=""
JOBS_PER_GPU=2
REMOTE="gdrive:syntheticData/datasets"
hosts=()

while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --script) SCRIPT="$2"; shift 2 ;;
        --start) START="$2"; shift 2 ;;
        --scenes) SCENES="$2"; shift 2 ;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        --remote) REMOTE="$2"; shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        -*) echo "unknown argument: $1" >&2; exit 2 ;;
        *) hosts+=("$1"); shift ;;
    esac
done

if [ ${#hosts[@]} -eq 0 ]; then
    echo "usage: $0 [options] <instance> [<instance> ...]" >&2
    exit 2
fi

if [ -z "$SCENES" ]; then
    SCENES=$(grep -E '^num_scenes:' "$PROJECT_ROOT/$CONFIG" | awk '{print $2}')
fi

PER_HOST=$(( (SCENES + ${#hosts[@]} - 1) / ${#hosts[@]} ))
END=$(( START + SCENES ))

for i in "${!hosts[@]}"; do
    host=${hosts[$i]}
    first=$(( START + i * PER_HOST ))
    [ "$first" -ge "$END" ] && break
    count=$(( END - first < PER_HOST ? END - first : PER_HOST ))
    cmd="bash brev/render.sh --config $CONFIG --script $SCRIPT --start $first --scenes $count --jobs-per-gpu $JOBS_PER_GPU --remote $REMOTE 2>&1 | tee logs/render.log"
    echo "[launch] $host: scenes $first..$((first + count - 1))"
    ssh "$host" "cd syntheticData && mkdir -p logs && tmux kill-session -t render 2>/dev/null; tmux new-session -d -s render '$cmd'"
done
