# Compact validation-selected benchmark

> Historical note. Current state and priorities: [backlog](../../backlog.md).
> Commands below describe the protocol at the time; paths in code blocks are repository-relative.


This benchmark is the shorter follow-up to the grouped two-fold production
sweep. It uses the complete active corpus (`phrases` plus `cleaned`, deduplicated)
and a fresh sentence-grouped 64/16/20 split for each repetition:

- 64% of sentence groups for training;
- 16% for validation;
- 20% held out for testing.

There is one fit per repetition and no refit. Two repetitions use seeds 42 and
43. Training draws 5,000 examples with replacement per epoch, monitors a fixed
validation sample, stops after eight non-improving validation epochs, and does
not stop before epoch 10. The maximum target is 50 epochs.

The validation score selects the checkpoint. The held-out test is scored once
after that selection, and is never used for early stopping or configuration
selection. Causal runs select by exact word top-1, then paper-similarity score
and BCE as deterministic tie-breakers. Semantic and attention runs select by
cosine top-1, then cosine top-5 and cross-entropy. BCE/cross-entropy are not the
primary retrieval metric.

## Configurations

There are 15 configurations and 30 independent fits:

| Family | Configurations |
| --- | --- |
| Causal | baseline; alpha 0.1/3; layers 2; window 2/8; batch 16/256 |
| Encoding | classical encoding + trainable ansatz; classical encoding + zero ansatz; direct no-encoding/no-ansatz baseline |
| Attention | QCSE/classical encoding × quantum/no-quantum attention |

The causal comparisons are one-factor-at-a-time around alpha 1, eight layers,
window 4 and batch 64. The deep 32/64-layer arms are omitted from the local
matrix because exact 14-qubit statevector simulation is not locally tractable.
The direct no-encoding control has no meaningful ansatz
to apply, so it is represented by the direct pathway rather than an artificial
identity circuit.

## Local smoke execution

List the exact task manifest:

```bash
uv run --no-sync python scripts/focused_sweep.py --list
```

Run one small CPU task:

```bash
uv run --no-sync python scripts/focused_sweep.py \
  --family causal --task-id 0 --device cpu \
  --epochs 3 --samples-per-epoch 32 --eval-examples 16 \
  --final-eval-examples 2 --early-stopping-patience 1 --min-epochs 1
```

## Cluster execution

Each array element runs one configuration/repetition, so no job contains five
long sequential fits and the existing four-CPU Slurm allocation is unchanged:

```bash
uv sync --locked
bash scripts/submit_focused_sweep.sh
```

The submitter creates three independent arrays: 16 causal tasks, six encoding
tasks and eight attention tasks. The batch file is
`slurm-focused-prod10.sbatch`; it performs `nvidia-smi`, the CUDA backend
preflight and then the selected fit. The batch allocation remains four CPUs;
the process-level numerical/PyTorch thread target defaults to 32 and can be
overridden with `FOCUSED_THREADS`, without changing `--cpus-per-task`.

The archived full-corpus CUDA runs suggest roughly 2 hours for a typical
50-epoch single causal fit and 5 hours for a 64-layer fit before early stopping.
Those estimates should not be used for local Apple runs: a real M2 MPS probe
with the full vocabulary measured about 4.6 seconds for 500 sampled examples
at two causal layers and 16.9 seconds at eight layers (including validation
bookkeeping). A 64-sample, 64-layer prediction alone took about 12.3 seconds.
Semantic/attention probes were 2.4--5.9 seconds for a complete 5,000-sample
epoch. Thus the exact 14-qubit causal configurations are not seconds-scale
locally; a seconds-scale local sweep requires a reduced sample budget or a
reduced/approximate causal simulator.
