#!/bin/bash
# Run copied diagnostics against the original, hash-checked benchmark checkout.
set -euo pipefail
test -x .venv/bin/python
probe=outputs/probe-nlp-trainability-d43cc96.py
causal=$(cat outputs/robust-followup/confirm-causal-study.txt)
cbow=$(cat outputs/robust-followup/confirm-cbow-study.txt)
.venv/bin/python - "$probe" "$causal" "$cbow" <<'PY'
import hashlib
import sys
from pathlib import Path
from qcse.benchmark import load_manifest

expected = "37e565e89a92d7707db1c8c88be0a397bb04a94cbbb24c2d2adfe23943dc480d"
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() != expected:
    raise SystemExit("Copied probe differs from pinned d43cc96")
for objective, path in zip(("causal", "cbow"), sys.argv[2:], strict=True):
    manifest = load_manifest(Path(path))
    if manifest["settings"]["objectives"] != [objective]:
        raise SystemExit(f"Expected the {objective} study: {path}")
PY

# Terminal jobs may remain visible in squeue. Block pending/running/unknown states.
mkdir -p logs
exec 9> logs/benchmark-submission.lock
flock -n 9 || { echo 'Another benchmark submission is in progress.' >&2; exit 1; }
queue=$(squeue --me --partition=prod10 --states=all --noheader --format='%i|%T')
while IFS='|' read -r job state; do
    [[ -z "$job" ]] && continue
    state=${state//[[:space:]]/}
    case "$state" in
        BOOT_FAIL|CANCELLED|COMPLETED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|TIMEOUT) ;;
        *) echo "Wait for existing prod10 job $job ($state) before submitting." >&2; exit 1 ;;
    esac
done <<< "$queue"
# Inherited Slurm submission options must not introduce arrays or extra GPUs.
for name in "${!SBATCH_@}"; do unset "$name"; done
sbatch --job-name=qcse-grad --partition=prod10 \
    --gres=gpu:nvidia_a100_1g.10gb:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=4 --time=01:00:00 --export=ALL \
    --output=logs/qcse-grad-%j.out --error=logs/qcse-grad-%j.err <<'SBATCH'
#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the original repository checkout}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
run="outputs/robust-gradient-replicated/job-${SLURM_JOB_ID:?}"
mkdir -p "$run"
cp "$0" "$run/submitted.sbatch"
cp outputs/probe-nlp-trainability-d43cc96.py "$run/probe.py"
printf 'stage\tseconds\n' > "$run/timing.tsv"
SECONDS=0
.venv/bin/python scripts/check_benchmark_backend.py --device cuda --trainability \
    > "$run/backend-preflight.json"
printf 'backend-preflight\t%s\n' "$SECONDS" >> "$run/timing.tsv"
for objective in causal cbow; do
    study=$(cat "outputs/robust-followup/confirm-${objective}-study.txt")
    SECONDS=0
    time .venv/bin/python "$run/probe.py" "$study" \
        --output "$run/$objective" --device cuda \
        --qubits 2 4 8 12 --layers 2 8 --init-scales 0.05 1 \
        --seeds 32 --batch-seeds 619 620 621 --head-seeds 619 719 819
    printf '%s\t%s\n' "$objective" "$SECONDS" >> "$run/timing.tsv"
done
archive="outputs/robust-gradient-replicated-${SLURM_JOB_ID}.zip"
zip -qr "$archive" "$run"
echo "Completed both objectives: $archive"
SBATCH
