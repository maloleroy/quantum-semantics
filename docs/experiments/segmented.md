# Separate recent and distant context in a sector qubit

Update 29 October: [all twelve larger-study fits are complete and verified](segmented-large.md);
the two missing mixture seeds finished through the targeted recovery runner. The original
preparation and handoff record below describes the 28 October state.

28 October 2026. Validation-only WikiText experiment following the decay screen,
which found that slower forgetting worsened the coherent model's known-word
predictions. This experiment preserves the recent signal separately.

## Model and declared protocol

Prepare `( |0> |near> + |1> |far> ) / sqrt(2)` when both streams are present.
Near encodes the most recent four words with population decay .7; far encodes
older words with decay .97. Each stream uses the existing fixed position phase
pi/2 and is normalized separately. When no older words exist, prepare only
`|0>|near>`; empty context never creates a fabricated word. Decay and mixture
weights are fixed, with no learned classical preprocessing or supplied counts.

Apply two rank-64 Hamiltonian evolutions on the 1,024-dimensional joint state
and measure only the nine word bits. The extra qubit now carries input context
sector information, rather than always starting at zero as in `ancilla`.
All ten-qubit measurement outcomes contribute; there is no postselection or
classical learned decoder. The quantum models retain 262,272 real trainable
coordinates and the same evolution initialization as the earlier auxiliary
model. This changes state preparation, not Hamiltonian capacity.

`segmented_mixture` is independently retrained after dephasing only the sector
qubit, preserving coherence within each word stream. In density-matrix terms
it removes off-diagonal blocks between recent and distant sectors, not all word
coherence. A device can implement this by a nonselective sector measurement or
random preparation of the two sectors. The simulator evolves the two weighted
branch states and sums their word probabilities. An advantage over this control
would isolate usefulness of cross-sector interference within this family, not
prove general quantum or entanglement advantage.

`segmented_mlp` receives exactly the real/imaginary coordinates of that same
normalized 1,024-dimensional state. Its hidden width is chosen to approximately
match the quantum parameter count (261,734 versus 262,272 at this configuration).
It uses no extra word/context features. Classical argmax remains available.
Arbitrary state preparation and learned spectral-basis compilation still have
unresolved hardware costs; these CPU simulators are classically tractable.

Declared before fits: one coherent fit, one sector-mixture fit, one same-input
MLP; recent window four, seed 11, 30 epochs, batch 512, clipping 5, NLL training
and validation-NLL selection including epoch zero. Quantum Adam LR .001, MLP
LR .003 as in earlier references. These are fixed recipes, not equal exhaustive
LR searches. Use the identical 8,192 gradient and 4,096 validation targets,
512-word training-only vocabulary and official article splits in
`output/interleaved-screen/dataset.npz`. Count statistics still use all training
articles; test is not scored. Adaptive reuse of validation is acknowledged.

Mixture pilot: six Adam updates averaged .23425 seconds, projecting 112.4 s of
updates for the small-subset 30-epoch fit, before validation/I/O/contention.
Three one-thread workers run concurrently. Generated recipe, locked metadata,
NPZ arrays/weights and CSV histories are under `output/segmented-screen/`.

Scientific checks compare stream preparation with explicit amplitudes, handle
missing older context, and compare coherent/sector-dephased probabilities and
factor gradients with dense density-matrix evolution. Existing model tests
cover normalization and portable weights. All 36 tests, strict Pyright and Ruff
pass. Defaults of earlier model kinds retain their previous preparation.

## Completed screen

| Model | Parameters | Selected epoch | Validation NLL | One-shot success | Known-target success | Fit time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Coherent sectors | 262,272 | 30 | 3.7312 | 12.34% | 1.97% | 84.8 s |
| Sector mixture | 262,272 | 30 | 3.6785 | 12.35% | 2.11% | 144.2 s |
| Same-input MLP | 261,734 | 9 | 3.4824 | 19.39% | 3.13% | 33.0 s |
| Previous auxiliary .7 | 262,272 | 30 | 3.7892 | 13.96% | 2.07% | 80.1 s |

[Exact validation results](segmented-validation.csv) also retain argmax,
known-target NLL, training NLL and run paths. This is the same 8,192-target
training subset and 4,096-example validation partition; UNK rate is 41.80%.
Known-word probabilities are not renormalized. Coherent segmentation improves
NLL against the earlier auxiliary recipe while reducing sampling success and
known-word sampling. The mixture has better NLL and known-word scores than
coherent segmentation. The same-input MLP leads both.

Coherent-minus-mixture NLL is +.05269, article-group descriptive 95% interval
[+.03857, +.06993]; sampling difference is −.0136 percentage points, interval
[−.0527, +.0282]. Against the MLP, coherent NLL is +.24887, interval
[+.20549, +.29648]. [Paired intervals](segmented-paired.csv) condition on these
validation-selected checkpoints and one seed; they do not adjust for adaptive
architecture choice or seed uncertainty. The screen does not establish useful
interference between the streams or an advantage over classical prediction.

Selected training NLL is 2.7888 for coherent, 2.9268 for mixture and 3.0762 for
MLP, versus validation 3.7312/3.6785/3.4824. More unique training examples are a
plausible next improvement to investigate, not a guaranteed remedy. Both quantum
fits selected their final epoch; convergence has not been established.

## Prepared larger experiment: user-run handoff

Prepared `output/segmented-large/data/dataset.npz` with 131,072 fixed training
targets using the existing loader's sampling rule, retaining official article
partitions. Direct comparisons verified identical vocabulary, validation
contexts/targets/groups, unigram statistics and bigram counts against the
previous 32,768-target dataset. The larger sample is not guaranteed to contain
the earlier smaller sample. Preparation is complete; long fits have not started.

`output/segmented-large/run.sh` runs seeds 11/22/33 for four models: previous
auxiliary preparation, coherent segmentation, sector mixture and same-input MLP.
All use the same new dataset, 30 full subset epochs, batch 512, rank64/two blocks
where applicable, NLL loss/validation selection, quantum LR .001 and MLP .003.
This is a larger-data, multi-initialization validation study, not a fresh held-out
confirmation. Test remains unscored. Model recipes are fixed before running it.

The mixture pilot projects 7680 updates × .23425 s = 1799 s (30 minutes) per
fit before evaluation, I/O and contention. Coherent fits project roughly 15–25
minutes from the completed screen. These exceed the user's ten-minute boundary,
so they are delegated via Bash rather than launched autonomously. Expect roughly
1–2 hours for the whole twelve-fit study at the default six one-thread workers;
that wall-time estimate is not a measurement. `SEGMENTED_WORKERS` accepts 1–8
for this eight-physical-core laptop. Source/data/script checksums guard the run.
Each invocation creates a fresh batch directory and preserves per-fit logs,
NPZ artifacts, source/data/weight locks and CSV histories; failed fits cause
nonzero exit. There is no optimizer resume. Interrupts target only its own workers.

After successful fits the runner verifies locks and writes `validation.csv` and
`paired.csv` in its printed batch directory. Paired intervals average differences
across seeds before resampling validation article groups; this still conditions
on the fitted seeds and reused validation partition. It never calls test scoring.
Generated scripts/data are ignored and must be transferred explicitly for handoff.
Bash syntax and the summarizer CLI have been checked; the long study is unrun.

```bash
bash output/segmented-large/run.sh
```

The broader research requirements—competitive known-word prediction, meaningful
long text, useful quantum computation and justified hardware costs—remain open.
