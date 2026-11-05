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

### Submission record

Slurm job **10876** started on 4 November 2026 under controller
`qcse-dgx-prior-20261104-01`. All **119 tests passed on the DGX**, including the
37 CUDA checks. Full-size CUDA forwards, finite gradients and changed predictions
after optimizer updates passed for all five distinct model/control settings.
The first `quantum` fit entered training. This is verified startup, not a completed
comparison or evidence of improved prediction. Controller log:
`output/prior-screen/controller.log`; startup evidence:
`output/prior-screen/startup-10876/verification.txt`.

At 21:22 the original local controller stopped with SSH exit 255 (`No route to
host`). A fresh connection confirmed Slurm 10876 remained RUNNING, with `quantum`
completed and `fixed_basis` training. The first fit's source/data/weight hashes,
CUDA metadata and epoch-30 selection were checked. Full report auditing remains
pending. Controller `qcse-dgx-prior-observe-10876-01` now observes that same job
and handles retrieval; its recovery script contains no submission command and
checks that the remote job ID remains 10876. No training was restarted.

### Independent completion audit prepared

The local `output/prior-screen/audit_completed.py` will check the retrieved
Slurm completion record and artifact manifest, then recompute all report tables
from saved predictions. It checks exact label/comparison coverage, target/group/
byte-length alignment, probability/NLL consistency, independently recomputed
count baselines, selected epochs, and initial count-baseline metrics. All 21
sample renderings will be decoded from saved IDs, without resampling or rejection.
Sample diagnostics are written only after the full report audit passes.

The table auditor passed the completed cosine report and rejected changed metric,
bucket, interval and sample values, duplicated rows, changed target order and
inconsistent probabilities. Nine local checks passed in 6.13 seconds; Ruff and
strict Pyright also passed. These validate the auditor, not the pending prior
experiment. All 79 pinned experiment inputs still match their submitted hashes.

After retrieval, run:

```bash
uv run python output/prior-screen/audit_completed.py
```

## Completed results

Slurm **10876 completed with exit 0:0 in 01:40:33** on 4 November; retrieval and
independent audit completed on 5 November 2026. Both local observers had stopped
on transient SSH routing failures while training continued. Recovery retrieved
the existing completed run without submitting or restarting any fit. All submitted
input hashes and downloaded artifact hashes passed. The GPU summary reloaded every
new checkpoint and reproduced validation arrays to 1e-10. The local audit checked
six locks/selected epochs and initial baselines, two archived references, three
count baselines, 11 metric rows, 22 buckets, 32 paired intervals and 21 saved sample
renderings. No held-out test was scored.

| Fit | Validation NLL ↓ | Sampling S ↑ | Argmax ↑ | Selected epoch | Fit minutes |
|---|---:|---:|---:|---:|---:|
| Quantum, bigram input | **2.667615** | 25.382% | 40.845% | 30 | 13.75 |
| Fixed Hamiltonian bases | 3.019810 | 22.724% | 34.668% | 25 | 13.33 |
| Quantum, unigram input | 2.717756 | 24.784% | 39.917% | 29 | 13.71 |
| Quantum, final 16 tokens | 2.672954 | 25.560% | 40.283% | 27 | 34.87 |
| Same-input MLP, constant | 2.527579 | 35.409% | 45.361% | 19 | 11.74 |
| Same-input MLP, cosine | 2.472620 | 36.477% | 46.484% | 27 | 11.73 |
| Archived adaptive quantum | 2.702110 | 25.774% | 40.576% | 29 | — |
| Archived cosine MLP | **2.407749** | **39.180%** | **47.925%** | 20 | — |
| Training bigram | 3.059255 | 23.158% | 34.033% | — | — |
| Training-modal space | ∞ | 30.054% | 30.054% | — | — |

The new quantum model improves NLL over the adaptive reference by
**−0.034495**, with descriptive paired-article 95% interval
**[−0.048744, −0.021331]**. It meets the declared NLL retention rule, so retain it
as the provisional quantum NLL reference. Sampling declines by **0.392 percentage
points**, with interval [−0.532, −0.254] points. Argmax improves by 0.269 points,
but its interval includes zero. This is an NLL improvement with a sampling
tradeoff, not an improvement on every endpoint.

The gain is concentrated outside the modal space target: non-space NLL improves
from **3.6793 to 3.6126**, and sampling from **6.08% to 6.50%**. Space sampling
falls from 71.62% to 69.33%. The stronger archived MLP reaches 19.30% non-space
sampling and 3.3255 non-space NLL; the new same-input cosine MLP reaches 16.11%
and 3.4117. A context-free modal-space predictor still exceeds quantum total S,
and the classical argmax strategy remains available.

Learning the bases/base energies improves NLL over the fixed-basis refit by
**−0.352195 [−0.375104, −0.328458]**. Starting with bigram instead of unigram
amplitudes helps by **−0.050141 [−0.063412, −0.036329]**. These controls support
useful learned Hamiltonian refinement in this family; they do not isolate an
entanglement/coherence advantage or rule out a classical replacement. The fixed
count cache and arbitrary amplitude-preparation cost remain part of the resource
budget. This is not a cost-free representation change relative to the old model.

The full-context model's NLL advantage over the short refit is only
**−0.005339 [−0.019229, +0.008020]**. Useful context beyond 16 tokens remains
unproven. Short-context fitting again took longer despite its unchanged padded
tensor shape; the timing cause is not established by this comparison.

The quantum model trails the same-input constant MLP by
**+0.140036 [+0.091408, +0.182182]** NLL, its cosine variant by
**+0.194995 [+0.139611, +0.242792]**, and the stronger archived MLP by
**+0.259866 [+0.194379, +0.314715]**. The new prior-based MLP does not replace
the stronger archived classical reference. The family comparison remains a
single-seed reused-validation screen; intervals omit training/selection uncertainty.

All three quantum continuations contain zero non-text outcomes, UTF-8 replacement
characters or other control characters, without rejection or resampling. They
nevertheless remain fragmented, for example the fixed first sample begins:
“The history of its first game ream the cvelBany”. Three clean-byte samples do not
establish reliable valid-byte generation or coherent long text. Fixed-basis samples
still include six replacement and two control characters.

Retain the NLL gain without claiming quantum advantage. A promising subsequent
representation change needs confirmation across seeds or a fresh validation
partition, a convincing non-space gain and evidence of useful older context.
The current result does not meet the broader text-generation objective.

Compact evidence: [metrics](prior-screen-validation.csv),
[paired intervals](prior-screen-paired.csv), [target buckets](prior-screen-target-buckets.csv),
[unfiltered samples](prior-screen-samples.csv), and
[sample diagnostics](prior-screen-sample-diagnostics.csv).
