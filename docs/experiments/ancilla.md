# Auxiliary-register Hamiltonian screen

28 October 2026. Exploratory WikiText-2 validation experiment following the
low-rank study. This hypothesis expands the word-measurement family through
joint evolution with an auxiliary register.

## Architecture and declared screen

Prepare `|0>_auxiliary tensor |psi(context)>_word`, apply two learned low-rank
Hamiltonian evolutions on the joint register, and measure only the word bits.
Probability for word y is the sum of joint Born probabilities over auxiliary
outcomes. No outcomes are rejected. No classical learned encoder, decoder,
word prior or probability renormalization is present. The simulator calculates
the exact marginal; a device needs one word-register measurement, not separate
preparations to reconstruct it. Auxiliary qubits must be initialized each time.

This is a restricted implementation of a
[Stinespring quantum channel](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/quantum-channels/representations-of-channels).
The joint evolution preserves inner products, but ignoring the auxiliary system
relaxes the word-only unitary measurement constraint. This is a possible route
to a better predictor, not an assertion that it beats classical models.
Entanglement can arise across registers; any measured gain here alone would not
isolate entanglement as its cause. The general learned spectral basis still has
unresolved state-preparation and hardware compilation costs.

The fixed amplitude input is unchanged: a 32-word context with recency decay,
512-word train-only vocabulary, official article partitions. Phase interleaving
is omitted because the earlier screen did not establish a benefit. Initial
screen: A=2 auxiliary states, rank 64 per block; A=4, rank 32; and an independently
retrained dephased-input A=2, rank-64 control. Totals are respectively 262,272,
262,208 and 262,272 real trainable coordinates, close to the previous word-only
rank-128 model's 262,400. Total registers use 10, 11 and 10 qubits. Changing rank
and auxiliary dimension jointly is a budget comparison, not a pure causal test
of auxiliary size. The parameterization is redundant; all coordinates count.

Dephasing removes input off-diagonal density-matrix entries before evolution;
it does not zero phases in a pure state. Implementation evolves every input
basis vector and averages word transitions with input populations. This control
is a restricted stochastic predictor, not all classical computation. Frozen
pure-quantum models have no other modules to retrain; epoch zero is retained.

Declared before training: Adam LR .001, clipping 5, seed 11, 30 epochs, batches
512, NLL training and validation-NLL checkpoint selection including epoch zero.
8,192 gradient targets and 4,096 validation targets from the same subset used
in the earlier screens; count/vocabulary statistics still see all training
articles. No test evaluation. Comparison references are the existing word-only
rank-128 fit, MLP and all-training bigram on identical validation examples.
The same validation partition has been reused for architectural decisions;
these scores cannot substitute for fresh confirmation.

Six pilot updates for A=2 averaged .12464 s, projecting 59.8 seconds of training
updates per 30-epoch screen before validation/I/O/concurrency overhead. Three
fits run concurrently with one Torch thread each. Tools, locked run metadata,
CSV histories and NPZ weights/arrays are under `output/ancilla-screen/` (ignored).

Implementation checks: explicit dense matrix exponentials and density-matrix
readout reproduce coherent and dephased probabilities and factor gradients;
A=1 reproduces the word-only low-rank model at identical initialization.
Both new kinds also pass normalization and portable-checkpoint tests. All 30
tests, strict Pyright and Ruff pass.

## Adaptive expanded-subset comparison

The initial screen completed in 80.1 s (A=2 coherent), 110.9 s (A=4 coherent),
and 95.8 s (A=2 dephased). A=2 coherent NLL 3.7892 versus A=4 NLL 3.7920 selects
A=2 for the expanded subset under the declared NLL criterion. A=4 has better
one-shot success, 14.53% vs 13.96%; selecting it by that score would change the
criterion. A=2 dephased gives NLL 3.7525, success 13.17%. Accordingly, before
additional fits, choose A=2/rank64 coherent and dephased on the original 32,768
training-target subset, keeping 30 epochs, LR .001, seed 11 and validation rows.
No weights are warm-started. This adaptive step increases both unique examples
and update count. Conservative fourfold projections are 320/383 s per fit,
within the user's ten-minute boundary. Two one-thread workers run concurrently.

## Results on 8,192 training targets

| Model | Selected epoch | Validation NLL | One-shot success | Known-target success | Argmax |
| --- | ---: | ---: | ---: | ---: | ---: |
| A=2, rank64, coherent | 30 | 3.7892 | 13.96% | 2.07% | 33.18% |
| A=4, rank32, coherent | 30 | 3.7920 | 14.53% | 1.98% | 31.84% |
| A=2, rank64, dephased | 17 | 3.7525 | 13.17% | 2.16% | 31.98% |
| Prior word-only rank128 | 29 | 3.9902 | 11.34% | 1.93% | 32.93% |

Auxiliary models improve over the nearly parameter-matched word-only reference
on NLL and sampling success at the same subset and epoch budget. Initialization
geometry and active rank differ, so this is an architectural recipe comparison.
The A=2 coherence-minus-dephasing NLL difference is +.03668, descriptive
article-group 95% interval [+.01806, +.05317]; sampling success difference is
+.796 percentage points, interval [+.671, +.916]. Known-target sampling success
is slightly *worse* with coherence: its aggregate success gain is driven by UNK.
UNK makes up 41.80% of validation targets. Known probabilities are never
renormalized. These are validation-selected one-seed fits, not confirmation
intervals corrected for architecture/checkpoint selection.

[Exact results](ancilla-validation.csv) retain training NLL, all metrics and
artifact paths; [paired intervals](ancilla-paired.csv) retain numerical bounds.
[Post-fit diagnostics](ancilla-diagnostics.csv) average coherent pure-state
auxiliary density matrices over the validation inputs. A=2 has mean probability
.7152 for auxiliary zero and mean linear entropy `1-Tr(rho_aux^2)` of .3603;
A=4 has .6856 and .4198. Thus nonzero auxiliary outcomes contribute and the pure
joint states are entangled. This is an activity diagnostic, not a comparison
against retrained unentangled circuits and not evidence of entanglement advantage.

## Results on 32,768 training targets

| Model | Selected epoch | Validation NLL | One-shot success | Known-target success | Argmax | Fit time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A=2/rank64 coherent | 27 | 3.6379 | 14.01% | 2.24% | 34.64% | 257.5 s |
| A=2/rank64 dephased | 28 | 3.5954 | 13.65% | 2.45% | 35.47% | 287.3 s |
| Prior word-only rank128 | 15 | 3.8744 | 11.58% | 2.04% | 33.20% | 213.4 s |
| Prior MLP, LR .003 | 7 | 3.3003 | 21.92% | 4.46% | 42.36% | 57.2 s |
| All-training bigram | 0 | 3.0300 | 22.68% | 5.94% | 43.16% | <1 s |

All rows use the identical validation examples, vocabulary and train-only data
policy. Neural/quantum rows use the same 32,768 gradient examples and maximum
30-epoch budget; counts use every training article, an explicitly larger data
exposure. MLP LR differs from quantum LR. No claim of exhaustive tuning is made.

Relative to the earlier word-only recipe, coherent auxiliary NLL improves by
.2365 and sampling success by 2.435 percentage points, at almost identical
parameter count. This is a useful recipe improvement. It remains worse than
the classical references. The auxiliary dephased control has the best quantum
implementation NLL of this screen and is equivalent at prediction time to a
restricted classical stochastic map of the input populations.

Coherent-minus-dephased NLL difference is +.04248, article-group 95% interval
[+.02525, +.06324]; sampling difference is +.356 percentage points, interval
[+.246, +.470]. As in the small screen, known-target sampling favors dephasing
(2.45% vs 2.24%), so the aggregate coherence benefit is driven by UNK. These
conditional, validation-selected intervals do not establish a confirmed quantum
advantage or include seed/model-selection uncertainty. Full coherent auxiliary
zero probability averages .7106; mean linear entropy is .3648. The register is
active and entangled, which is distinct from proving its entanglement necessary.

[Context truncation diagnostics](ancilla-context.csv) evaluate locked fits with
older input positions replaced by padding, without retraining or checkpoint
reselection. Coherent NLL for keeping 4/16/32 words is 3.7195/3.6427/3.6379;
dephased NLL is 3.7768/3.6045/3.5954. Both fits use information beyond four words
on this diagnostic; the effect beyond 16 is small. Truncation to four *increases*
sampling success while worsening NLL, again showing why the metrics must remain
separate. Truncation also changes input normalization; these are sensitivity
measurements, not a retrained context-length advantage or proof of semantic
long-range reasoning. No confidence interval was computed for this diagnostic.

Next research direction: improve the input representation so useful older words
survive without random phase noise, and investigate the vocabulary/UNK bottleneck.
This screen supports a more flexible quantum readout but does not establish that
input coherence or entanglement produces an advantage. Long-form text generation,
competitive classical performance, and hardware efficiency remain unresolved.
Test was never scored; all runs completed within ten minutes.

Reproduce the expanded coherent recipe using the installed CLI:

```bash
uv run qcse train \
  --dataset output/dataset-selection/dataset-20261026T211719Z-c96984fa/dataset.npz \
  --models ancilla --ancilla-size 2 --rank 64 --layers 2 --seeds 11 \
  --epochs 30 --batch-size 512 --lr .001 --loss nll --selection nll --threads 1 \
  --output output/ancilla-reproduction
```

Use `--models ancilla_dephased` for the independently trained control. Generated
scripts, arrays and checkpoints remain under ignored `output/ancilla-screen/`;
transfer them explicitly for a handoff. Only compact evidence is versioned here.
