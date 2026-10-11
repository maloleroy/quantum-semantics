#!/bin/bash
# Prepare one research block. Submission remains a separate bounded operation.
set -euo pipefail
cd "$(dirname "$0")/.."
block=${1:?Specify reference, hybrid-circuits, trainability, qcse-circuits, context, window-2, window-8, cleaned, phrases, or cbow}
shift
extra=(--window 4)
case "$block" in
  reference)
    arms=(qcse-crz-chain qcse-lr-0.0003 qcse-lr-0.01
          hybrid-crz-chain hybrid-lr-0.0003 hybrid-lr-0.01
          hybrid-identity hybrid-identity-lr-0.0003 hybrid-identity-lr-0.01
          mlp mlp-lr-0.0003 mlp-lr-0.01
          bert-16 bert-lr-0.0003 bert-lr-0.01 bert-32 bert-16-depth-4 mean)
    ;;
  hybrid-circuits)
    arms=(hybrid-crz-chain hybrid-crz-ring hybrid-zz-chain hybrid-xx-chain
          hybrid-ry-cx hybrid-local hybrid-identity hybrid-frozen hybrid-depth-4 hybrid-depth-8)
    ;;
  trainability)
    arms=(hybrid-crz-chain hybrid-depth-8 hybrid-identity hybrid-q2-identity hybrid-q8-identity
          hybrid-q2-depth2-init0.05 hybrid-q2-depth2-init1
          hybrid-q2-depth8-init0.05 hybrid-q2-depth8-init1
          hybrid-q4-depth2-init1 hybrid-q4-depth8-init1
          hybrid-q8-depth2-init0.05 hybrid-q8-depth2-init1
          hybrid-q8-depth8-init0.05 hybrid-q8-depth8-init1)
    ;;
  qcse-circuits)
    arms=(qcse-crz-chain qcse-zz-chain qcse-xx-chain qcse-ry-cx qcse-local
          qcse-identity qcse-frozen qcse-permuted-ids qcse-bit-bce qcse-depth-4 qcse-depth-8)
    ;;
  context)
    arms=(qcse-crz-chain qcse-context-diagonal qcse-context-phase qcse-context-hash
          qcse-context-angular hybrid-identity mlp bert-16)
    ;;
  window-2|window-8|cleaned|phrases)
    arms=(qcse-crz-chain hybrid-crz-chain hybrid-identity mlp bert-16)
    case "$block" in
      window-2) extra=(--window 2) ;;
      window-8) extra=(--window 8) ;;
      cleaned) extra=(--datasets cleaned) ;;
      phrases) extra=(--datasets phrases) ;;
    esac
    ;;
  cbow)
    arms=(qcse-crz-chain qcse-xx-chain hybrid-crz-chain hybrid-identity mlp bert-16 bert-32)
    extra=(--objectives cbow)
    ;;
  *) echo "Unknown research block: $block" >&2; exit 2 ;;
esac
command=(.venv/bin/python scripts/robust_benchmark.py prepare
         --output "outputs/robust-screen/$block"
         --objectives causal --folds 3 --seeds 42
         --epochs 100 --fixed-epochs --samples-per-epoch 20000
         --final-eval-sentences 0
         --trainability --arms "${arms[@]}" "${extra[@]}")
if [[ ${1:-} == --dry-run ]]; then
  shift
  printf '%q ' "${command[@]}" "$@"
  printf '\n'
else
  exec "${command[@]}" "$@"
fi
