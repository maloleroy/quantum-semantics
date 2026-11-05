# Content-dependent Hamiltonian controls and MIG resource probe

5 November 2026. No predictive result is available yet.

The replicated position model learns fixed lag preferences, which cannot select
different older tokens for different contexts. `prior_content` adds a small
classical content selector while retaining count-amplitude preparation, contextual
Hamiltonians, local RZ gates and direct Born token measurement. There is no learned
classical vocabulary head in the quantum candidate. This is classical attention
preprocessing, not a claim of quantum attention or efficient quantum compilation.

Four ordered query-slot tables embed the last four context tokens in 16 dimensions.
Average their valid entries to form a query; dot it with each preceding token's
16-dimensional key and divide by sqrt(16). Add this scalar to the existing
feature-specific position logits before normalizing within each recent/older
stream. The last token remains a separate stream. Padding contributes nothing,
including an entirely padded query or stream. All inputs precede the target.
Zero-initialized keys preserve the old pooling initially; random query directions
allow the keys to learn. Initial zero spectra still reproduce the train-count
distribution exactly. `--embedding` controls query/key dimension for these kinds.

The selector adds 40,960 parameters at vocabulary 512/window 128/dimension 16.
`prior_content_mlp` uses the identical feature family with a classical logit
correction instead of quantum dynamics. The candidate has 794,998 trainable
coordinates, the matched MLP 794,316, and a position-only rank-135 quantum capacity
control 794,365. A fixed-basis content model has 270,454 trainable coordinates;
contextual spectral shifts still train, so its Hamiltonian is not fully frozen.
Fixed count-cache and amplitude-preparation costs remain unchanged and counted.

Local verification passes **101 tests**, with **41 CUDA checks pending**. Independent
scalar query/pooling and dense Hamiltonian references check values and gradients;
other tests cover initialization, padding, learning of initially zero keys,
train-only counts, serialization and locked fitting. CUDA parity tests exercise
content-sensitive parameters beyond the identity initialization. Ruff and strict
Pyright pass. No dependency or training-objective change is introduced.

## Resource probe before full fitting

Use `prod10` with one `nvidia_a100_1g.10gb` slice and four CPU support threads.
This GPU-specific validation must run on the DGX; it measures the required device,
not a new language benchmark. The isolated checkout is
`~/quantum-semantics-content-probe-20261105`, with tools/artifacts in
`output/content-probe/`. A 15-minute allocation ceiling covers the bounded probe.

Probe five configurations at seed 11: full content, position-only rank-135 capacity,
content with the older 112 positions masked, fixed bases, and matched content MLP.
Use the existing 524,288 training rows, vocabulary 512, window 128, rank 128 except
capacity, two layers, one auxiliary qubit, recent-window 16 and embedding 16.
The dataset, tokenizer, counts and all target partitions remain fixed; the full
dataset SHA256 is
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.

First run all CUDA tests. For each full-size configuration, keep all training
contexts/targets and a checkpoint snapshot resident on GPU, together with model
and Adam state. Check actual CUDA forwards, finite CUDA gradients, optimizer changes,
initial count probabilities and nonzero query/key gradients after three warmup
updates. Then time three blocks of eight updates with CUDA synchronization.
Use a fixed shuffled training order, batch 512, NLL, learning rate .003 and clipping
5, matching the intended scientific update path. No validation or test metrics
are evaluated, no selected checkpoint is saved, and these probe updates will not
be reused as training or optimizer continuation.

Save per-block timings and allocator peak memory as CSV, plus small device metadata.
Report median seconds/update and its projection to 30,720 updates. This projection
excludes validation, final training-set scoring, dataset I/O and scheduler delay;
it is a planning estimate, not measured complete-fit time. Allocator memory excludes
CUDA context/driver overhead. Successful probe execution is not proof of every
full-fit memory peak. Keep the batch/update budget unchanged when deciding resources.
No prod80 escalation is authorized by this protocol alone; any later escalation
must cite measured needs or a documented compute/time requirement.

The intended predictive comparison retains full/short, capacity, fixed-basis and
matched-MLP controls plus the strong archived position/MLP references. Final execution
budget and resource choice follow this probe, before scientific fitting. No claim
of an NLL gain, older-context benefit, quantum advantage or coherent generation
is supported yet. Independent confirmation remains necessary after reused-validation
exploration.
