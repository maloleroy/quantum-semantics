# Raw QCSE on the WikiText-2 next-token task

## Declared protocol — 17 November 2026

Purpose: give the poster's accuracy chart a QCSE bar measured on the same task as
the Hamiltonian model (44.10%) and the strong MLP (50.35%) of
[coverage-seeds](coverage-seeds.md). This is a control, not a new architecture.

Shared with both reference families, unchanged:

- Dataset `output/depth-coverage/data/full.npz`, SHA256
  `37349f5fe9a3442bdcd5cea6c4e660015f237995c805ab96fd52b5c5a06d4c28`: 512 byte-pair
  tokens, 128-token contexts, 4,096 validation targets, test not scored.
- 30 epochs of 524,288 rows drawn from a fresh permutation of the pool per epoch,
  batch 512, **30,720 updates**, Adam, NLL loss, selection on lowest validation NLL.
- Seeds 11, 22 and 33, which also fix the row schedule, as for the references.
- Learning rate .003 with a constant schedule, as for the Hamiltonian model.

QCSE-specific settings:

- `--models qcse --decoder marginal`: QCSE as published, one Pauli-Z marginal per
  qubit on 9 qubits; the token probability is the product of bit marginals, so the
  NLL equals QCSE's summed bit cross-entropy and argmax is exact-match decoding.
- `--qcse-window 16`: QCSE encodes the latest 16 tokens. Its n x n context matrix
  makes all 128 tokens cost 911 encoding layers; 16 matches the recent window
  of both references.
- `--layers 64`: ansatz depth of the best full-corpus QCSE configuration.

QCSE has 1,664 trainable angles against 1,073,900 (Hamiltonian) and 750,460 (MLP)
parameters. Capacity differs by design; data, exposure and selection do not.

## Resources

A prod10 probe measured 0.673 s per update, 1.6 s per validation pass and 2.62 GiB
peak memory, projecting about 6 h per seed. Each seed runs as its own prod10 job
(`nvidia_a100_1g.10gb`, 12 h limit) in `~/quantum-semantics-qcse-wikitext-20261117`
on the DGX, writing to `output/qcse-wikitext/s{seed}/`. Jobs 11706, 11707, 11708.

## Result of the 16-token window (jobs 11706-11708, completed)

All three fits ran the full 30,720 updates without error (about 6 h each) but did
not learn: validation NLL stayed at 6.243-6.248 nats (uniform: ln 512 = 6.238),
online training loss at 6.247 throughout, argmax accuracy 0.10-0.32% (chance
0.20%). With 16 tokens the QCSE encoding stacks 15 rotation/CNOT layers; the
encoded state is likely so entangled that local ansatz rotations cannot move the
Z marginals. The short probe with a 2-token window reached 13.06% argmax after 32
updates. These fits are retained but not used on the poster.

## Amendment — 2-token window (jobs 11770-11772)

Rerun with `--qcse-window 2`, the window of the best full-corpus QCSE
configuration (one encoding layer). Every other setting is unchanged, including
learning rate .003. Outputs go to `output/qcse-wikitext-w2/s{seed}/`.

### Result (completed 18 November 2026)

All three fits ran the full 30,720 updates (about 5.5 h each). Validation, 4,096
targets, checkpoint selected on validation NLL:

| Seed | Selected epoch | Argmax accuracy | One-draw success | NLL (nats) |
|---|---|---|---|---|
| 11 | 28 | 29.69% | 2.31% | 5.2865 |
| 22 | 1 | 29.81% | 2.41% | 5.2878 |
| 33 | 30 | 29.76% | 2.66% | 5.2873 |
| **Mean** | | **29.75%** | **2.46%** | **5.2872** |

The fits plateau after the first epoch. The space token (`hex:20`) is 30.05% of
the validation targets, so always predicting a space scores 30.05%. Reloading the
selected weights reproduces each accuracy exactly and shows that QCSE predicts a
space in 79.4-79.8% of the contexts and uses at most 4 of the 512 tokens (space,
`be`, `se` and one other). At least 99.6% of its correct answers are spaces. For
reference on the same targets: the context-free product of training bit marginals
(binary token index) gives NLL 5.421 and also predicts a space, the unigram table
gives NLL 4.294, and the bigram table gives NLL 3.089 and 34.03% argmax accuracy.

The poster bars use the three-seed mean argmax accuracy: QCSE 29.75%, Hamiltonian
44.10%, MLP 50.35%. The poster states that QCSE reads only the last 2 tokens and
scores about as well as always guessing a space.

## Probe — full 128-token context, one epoch (job 11773)

One epoch (524,288 rows, 1,024 updates) at seed 11 with no QCSE window, so QCSE
reads the same 128 tokens as both references (911 encoding layers). It measures
the real cost and whether the loss moves. Output:
`output/qcse-wikitext-w128-e1/`.

Result: the epoch-0 validation, the 1,024 updates and the epoch-1 validation took
about 3 h (job start 03:16 UTC, history written 06:15 UTC), about 10 s per update.
Thirty epochs would take about 90 h. The loss does not move: online training loss
6.2473 nats, validation NLL 6.2489 before and 6.2435 after the epoch (uniform:
6.2383), argmax accuracy 0.20% (chance). On 18 November 2026 at 07:30 UTC the job
was still running after its last validation, before writing its selection files.
