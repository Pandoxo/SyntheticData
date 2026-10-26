#!/usr/bin/env bash
# Render a slice of the scene index space on every GPU of this machine, then merge,
# check and ship the result to Google Drive.
#
#     bash brev/render.sh [--config config.yaml] [--script generate.py]
#                         [--start 0] [--scenes N] [--jobs-per-gpu 2]
#                         [--remote gdrive:syntheticData/datasets] [--no-ship]
#
# Runs on the Brev instance (launch.sh starts it inside tmux). The slice
# [start, start + scenes) is split evenly across the GPUs nvidia-smi reports, one
# run_batch.py per GPU pinned with CUDA_VISIBLE_DEVICES. --resume is always on, so
# rerunning after a crash or a preempted instance only renders what is missing.
#
# While the GPUs work, finished shards of 100 scenes are packed and uploaded every
# few minutes (ship_dataset.py), so a lost instance loses at most one shard of work.
# At the end: merged COCO, a QA sheet, and a final upload of everything.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"

CONFIG="config.yaml"
SCRIPT="generate.py"
START=0
SCENES=""
JOBS_PER_GPU=2
REMOTE="gdrive:syntheticData/datasets"
SHIP=1

while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --script) SCRIPT="$2"; shift 2 ;;
        --start) START="$2"; shift 2 ;;
        --scenes) SCENES="$2"; shift 2 ;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        --remote) REMOTE="$2"; shift 2 ;;
        --no-ship) SHIP=0; shift ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cfg_value() {
    .venv/bin/python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])" "$CONFIG" "$1"
}
[ -z "$SCENES" ] && SCENES=$(cfg_value num_scenes)
DATASET=$(cfg_value output_dir)
mkdir -p "$DATASET" logs
cp "$CONFIG" "$DATASET/config.yaml"

GPUS=$(nvidia-smi -L | wc -l)
if [ "$GPUS" -eq 0 ]; then
    echo "[render] nvidia-smi reports no GPUs" >&2
    exit 1
fi
if ! command -v rclone >/dev/null || ! rclone listremotes 2>/dev/null | grep -q "^${REMOTE%%:*}:"; then
    echo "[render] rclone remote ${REMOTE%%:*} not configured - results stay on this machine"
    SHIP=0
fi

# Ceiling division, so the last GPU gets the remainder rather than dropping it.
PER_GPU=$(( (SCENES + GPUS - 1) / GPUS ))
END=$(( START + SCENES ))
STARTED=$(date +%s)
echo "[render] scenes $START..$((END - 1)) on $GPUS GPU(s), $PER_GPU per GPU, $JOBS_PER_GPU job(s) each"
cat > "$DATASET/run_info.json" <<EOF
{"host": "$(hostname)", "gpus": $GPUS, "gpu_model": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)",
 "start": $START, "scenes": $SCENES, "jobs_per_gpu": $JOBS_PER_GPU, "config": "$CONFIG",
 "started": "$(date -Is)"}
EOF

# Utilisation traces for capacity planning: GPU load and memory every 5 s, host RAM
# and load average every 5 s. Cheap, and the only way to tell whether a node is
# GPU-bound or CPU-bound.
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader -l 5 > logs/gpu_util.csv 2>/dev/null &
MON_PIDS=($!)
( while true; do
      echo "$(date +%s),$(free -m | awk '/Mem:/ {print $3","$2}'),$(cut -d' ' -f1 /proc/loadavg)"
      sleep 5
  done ) > logs/host_util.csv &
MON_PIDS+=($!)

pids=()
for (( gpu = 0; gpu < GPUS; gpu++ )); do
    first=$(( START + gpu * PER_GPU ))
    [ "$first" -ge "$END" ] && break
    count=$(( END - first < PER_GPU ? END - first : PER_GPU ))
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python run_batch.py \
        --config "$CONFIG" --script "$SCRIPT" \
        --start "$first" --scenes "$count" --jobs "$JOBS_PER_GPU" --resume --quiet \
        > "logs/gpu${gpu}.log" 2>&1 &
    pids+=($!)
    echo "[render] gpu $gpu: scenes $first..$((first + count - 1)) -> logs/gpu${gpu}.log"
done

ship_loop() {
    while true; do
        sleep 600
        .venv/bin/python ship_dataset.py "$DATASET" --remote "$REMOTE" \
            --start "$START" --scenes "$SCENES" >> logs/ship.log 2>&1 || true
    done
}
SHIP_PID=""
if [ "$SHIP" -eq 1 ]; then
    ship_loop &
    SHIP_PID=$!
fi

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
[ -n "$SHIP_PID" ] && kill "$SHIP_PID" 2>/dev/null || true
kill "${MON_PIDS[@]}" 2>/dev/null || true

# Each run_batch.py merged on exit, possibly over one another; this merge is the one
# that sees every GPU's scenes.
.venv/bin/python run_batch.py --config "$CONFIG" --merge-only
.venv/bin/python qa_report.py "$DATASET" --count 48 > logs/qa.log 2>&1 || true
ELAPSED=$(( $(date +%s) - STARTED ))
.venv/bin/python - "$DATASET/run_info.json" "$ELAPSED" <<'EOF'
import json, sys
path, elapsed = sys.argv[1], int(sys.argv[2])
info = json.load(open(path))
info["elapsed_s"] = elapsed
json.dump(info, open(path, "w"), indent=1)
EOF
echo "[render] rendering finished in ${ELAPSED}s, exit status $status"

if [ "$SHIP" -eq 1 ]; then
    .venv/bin/python ship_dataset.py "$DATASET" --remote "$REMOTE" \
        --start "$START" --scenes "$SCENES" --final 2>&1 | tee -a logs/ship.log
fi
exit $status
