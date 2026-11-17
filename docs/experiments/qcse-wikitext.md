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

## Result

Pending. Report the three-seed mean argmax (most-likely token) validation accuracy
in `docs/poster/fig/classical.tex` (`\qcseaccuracy`), with NLL and sampling success.
