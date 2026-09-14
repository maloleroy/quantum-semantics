#!/bin/bash
# Same data, split, sample budget and quantum attention; change only the encoder.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

case "${ATTENTION_ENCODING:-both}" in
    both) encodings=(qcse classical) ;;
    qcse|classical) encodings=("$ATTENTION_ENCODING") ;;
    *) echo 'ATTENTION_ENCODING must be both, qcse or classical.' >&2; exit 1 ;;
esac

for argument in "$@"; do
    case "$argument" in
        --resume|--resume=*)
            echo 'Resume individual runs with scripts/semantic_experiment.py --resume RUN.' >&2
            exit 1
            ;;
    esac
done

for encoding in "${encodings[@]}"; do
    uv run --no-sync python scripts/semantic_experiment.py "$@" \
        --model attention --encoding "$encoding" \
        --output "${ATTENTION_OUTPUT:-outputs/attention}/${encoding}"
done
