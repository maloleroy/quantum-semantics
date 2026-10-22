#!/bin/bash
# Run all ten focused semantic configurations concurrently on a local machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -x .venv/bin/python ]]; then
    echo 'Run uv sync --locked first.' >&2
    exit 1
fi

epochs=${SEMANTIC_EPOCHS:-50}
max_sentences=${SEMANTIC_MAX_SENTENCES:-1000}
device=${SEMANTIC_DEVICE:-cpu}
threads=${SEMANTIC_THREADS:-1}
output=${SEMANTIC_OUTPUT:-outputs/semantic-local-${epochs}}
mkdir -p logs

pids=()
for experience_id in $(seq 0 9); do
    .venv/bin/python scripts/semantic_cluster_sweep.py \
        --experience-id "$experience_id" \
        --epochs "$epochs" \
        --max-sentences "$max_sentences" \
        --device "$device" \
        --threads "$threads" \
        --output "$output" >"logs/local-semantic-${experience_id}.out" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
