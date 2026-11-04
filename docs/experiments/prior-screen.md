# Count-initialized quantum refinement

4 November 2026. Declared before research training. The
[completed cosine study](dgx-cosine.md#completed-results) leaves quantum NLL near
2.70 versus the stronger MLP's 2.41, with only 6.08% sampling success on non-space
targets. Longer pooled context has no demonstrated benefit. Test whether starting
from a useful training-count distribution helps Hamiltonians learn conditional
refinements, while keeping a direct quantum measurement endpoint.

## Model and physical endpoint

For last token x, prepare word amplitudes sqrt(q(y|x)), with the existing training-only
bigram rule q=(counts+10*unigram)/(row_count+10), and auxiliary register |0>.
A fully padded last position uses the training unigram. `prior_unigram` instead
uses that unigram for every context. The model stores fixed probability tables;
there is no learned classical encoder or vocabulary head in the quantum candidate.

Two rank-128 Hamiltonians on the 512×2 joint register have learned orthonormal
bases Q and base energies e. Shared linear, pooled context tables shift their
energies by s(C); intervening learned local RZ angles also depend on context.
The evolutions are exp(-i Q diag(e+s(C)) Q†). Direct measurement of the word
register produces the token; the auxiliary register is ignored. No temperature,
rejection, retries, renormalization or classical output correction is added.

Base energies and spectral shifts initialize to zero. The initial evolutions
are identities; intermediate RZ phases preserve the count probabilities exactly.
Energy/shift gradients can start learning immediately, followed by basis and local
angle learning. Small tests check this explicitly and compare probabilities and
all parameter gradients with dense matrix exponentials away from initialization.

The same-input `prior_mlp` receives sqrt(q), pooled local angles and spectral shifts.
Its learned logit correction is added to log(q), with a zero-initialized final layer,
so it also starts at the exact bigram distribution. Its argmax remains available.
The primary same-schedule comparison uses constant-rate MLP; a cosine-rate MLP
ensures that comparison also includes the stronger optimization option seen in the
previous study. A fixed-basis quantum refit freezes Q and zero base energies while
training context-dependent energies and local angles. Its Hamiltonian is still
context-dependent: this is not a frozen-H or dephased control.

Quantum models have 736,512 trainable parameters; the matched MLP has 735,830.
The fixed-basis model trains 211,968 of 736,512 parameters. All have the same
262,656-entry count cache, including the padding fallback; quantum also stores
10,240 fixed Z-sign entries. Unigram rows are replicated in that cache. These fixed
statistics count toward memory and preprocessing, even though they are not fitted
by gradient descent. Arbitrary count-amplitude preparation may cost O(V) gates,
and learned dense spectral bases still need compilation. This is an expressivity
screen in a simulator, not an efficient quantum hardware implementation.

## Declared comparison

Six fresh seed-11 fits, all 30 epochs, batch 512, Adam initial .003, NLL training
and validation selection, float64/complex128, two stages, rank 128, auxiliary size
2, recent window 16, phase scale 1 and Born readout:

| Label | Model | Prior | Schedule | Context |
|---|---|---|---|---|
| quantum | prior | bigram | constant | 128 |
| fixed_basis | prior, freeze-core | bigram | constant | 128 |
| unigram_input | prior_unigram | unigram | constant | 128 |
| short | prior | bigram | constant | last 16 |
| mlp_constant | prior_mlp | bigram | constant | 128 |
| mlp_cosine | prior_mlp | bigram | cosine to .0003 | 128 |

Each fit has 30,720 updates and 31 validation checks, including epoch zero. Use the
unchanged 524,288-training-position WikiText byte-BPE dataset, V512/W128, 4,096
validation and 8,192 untouched test targets. Full dataset SHA256:
`7e502ba60d20f966236c07952d8a95ce05efb7ef681fab6062a821fbd311c197`.
Short context reuses the verified dataset from the cosine study: older columns
masked to -1, identical targets/groups/tokenizer/counts/byte lengths. No source
retokenization, refitting of statistics, new positions or test scoring.

Retain the audited constant-adaptive (NLL 2.702110) and cosine-MLP (2.407749)
references as locked arrays/weights with matching archived source hashes. Do not
resample or retrain them. Include unchanged bigram/unigram/modal-space baselines.
Report 11 metric rows and 22 modal/non-modal target buckets. Compare quantum with
all ten others and cosine MLP with constant MLP: 32 paired article interval rows,
omitting the infinite-NLL modal comparison. This is a single-seed reused-validation
screen; intervals are descriptive and omit training and selection uncertainty.

Save 21 unfiltered 64-token draws from five new full-context fits plus bigram and
unigram, using the same three prompts and seeds 20261030–20261032 as the cosine
study. Omit short-context generation because the generic generator uses the full
window. Store every draw, invalid/UNK outcome and raw token ID. Sampling success,
argmax, NLL and generation quality remain separate.

Retain a candidate only if it improves validation NLL over the adaptive reference;
report non-space metrics and samples even if they worsen. A gain over bigram alone
does not establish a quantum benefit. Compare learned versus fixed bases, prior
choice, context length and the matched classical model before attributing gains.
A promising single-seed result will require fresh-seed/partition confirmation;
this screen cannot establish quantum advantage or coherent long text by itself.

## Execution and evidence

Use isolated DGX checkout `~/quantum-semantics-prior-20261104`, full A100, four CPU
support threads, CUDA dependencies and explicit CUDA model execution. Run all CUDA
parity tests and real full-size forward/backward/optimizer checks before training.
Pin source, dependencies, test fixture, datasets, references and generated tools
by SHA256 locally/remotely. Reload selected weights and reproduce validation
arrays on GPU before summary. Preserve CSV histories, NPZ arrays/weights and small
JSON metadata. Estimate 2–4 hours, with an eight-hour allocation limit; this new
model's full-fit timing has not been measured.

Generated tools/artifacts live in `output/prior-screen/`. Submit once without
holding an SSH session for the training lifetime; a detached local controller
checks the exact Slurm job and retrieves artifacts on completion. Connection loss
stops observation and leaves job identity explicit; it never triggers resubmission.
The codex-long-jobs skill provides completion delivery without model polling.
