# QCSE causal encoding study

14 September 2026 · 26 encoding configurations · three seeds · 102 retained training runs

The clearest improvement came from **encoding train-fitted conditional word statistics**, rather than finding a more elaborate geometry for arbitrary word IDs. `bigram_angles` reached **16.59% exact next-word accuracy**, versus **4.92%** for the retrained original encoding. However, the classical probabilities used to construct that encoding already achieved **16.77%**. This is a useful encoding improvement, but these experiments provide no evidence that the quantum ansatz improves it.

The most promising broader-context candidate is **PPMI features with jointly learned angle calibration**. It reduced bitwise loss, but did not establish an advantage over a context-free frequency baseline. Small-angle and amplitude encodings improved on the original circuit, yet were statistically indistinguishable from a trained circuit receiving no context on the primary loss metric.

My recommendation is to fix the input information loss, build an encoding from learned linguistic features, and retain strong classical controls. I would not prioritize more re-uploading depth or more aggressive spreading of IDs based on these results.

## What was tested and how

- **Data:** the existing `phrases.csv`: 1,000 phrases, 729 words, 6,235 causal next-token examples, context window four, ten qubits. Sentence boundaries are respected. Duplicate sentence texts remain together.
- **Splits:** preserve the repository's outer split, seed 42: 4,989 development examples and 1,246 held-out examples. Split development sentences again with seed 31415 into **3,990 training / 999 validation** examples. The holdout is the project's existing evaluation partition, not a new external corpus.
- **Screen:** 26 configurations × seeds 11, 22, 33; the same 640 sampled training and 320 validation examples; 20 epochs. The test partition was not scored for these candidates.
- **Confirmation:** the four lowest mean validation-BCE candidates, the original encoding, the isolated RY repair, a random-feature control, and a new input-free control. Eight configurations × three seeds; all 3,990 training and 999 validation examples; 60 epochs. The selection rule was saved before confirmation in [confirmation_selection.json](results/confirmation_selection.json).
- **Optimization:** the existing SPSA/Adam recipe, perturbation `0.1 / step**0.101`, batch size 32, ansatz L2 coefficient 0.001. Every candidate uses two ansatz layers, 58 ansatz parameters, and the same seeded uniform `[-pi, pi]` initialization. Learning rate **0.01** was fixed for this study, versus the production default 0.0003. This is a controlled exploration, not a reproduction of the paper's scores or the existing eight-layer training run.
- **Checkpoint selection:** minimum validation BCE at epoch zero or every fifth epoch; test predictions made using that checkpoint. Three seeds vary initialization and SPSA randomness, **not the sentence split**.
- **Features:** PPMI, scaling statistics, bit priors, and bigram counts are fitted on the relevant training subset only. PPMI is directional: context-token/future-target co-occurrence, weighted by inverse context distance, reduced to 20 dimensions through a deterministic randomized SVD. Recency pooling uses weights proportional to `0.5**age`. Random token vectors and input-ID permutations are deterministic and shared across optimization seeds.
- **Vocabulary caveat:** the original full-corpus, frequency-ranked vocabulary is retained for comparability. Its ID ordering uses aggregate corpus counts, including held-out phrases. These are transductive experiments; they do not demonstrate generalization to unseen vocabulary. A stricter future evaluation should construct IDs from training data and define an unknown-token policy.

The new code is isolated under this directory. The production encoding, model, trainer, and saved checkpoints were not changed. The test configuration now adds the repository root to Python's import path so the experimental tests also run with the documented `uv run pytest` command.

### Metrics

**BCE** is mean binary cross-entropy over the ten target ID bits, in nats; lower is better. **Exact word** requires all ten thresholded bits to match the target. **Top-5** and **vocabulary NLL** rank valid vocabulary IDs by the product of their predicted bit probabilities, renormalized over the 729 valid IDs. This is an explicitly constructed independent-bit vocabulary distribution, not the full quantum measurement distribution. Both threshold accuracy and valid-vocabulary top-1 are recorded in the raw results.

The paper's “at least half the bits match” score is retained in the result files but is unsuitable for selecting a GPT-like model. On the existing checkpoint's split, the saved model scores **92.38% by that metric and 3.93% exact accuracy**. A context-free bit-frequency predictor also scores **92.38%**, while reaching **9.47% exact accuracy** and lower BCE, 0.5535 versus 0.5810. This does not negate the implementation's learning curve; it changes what that curve establishes.

## Every encoding tested

These are **screening validation results**, not test results or claims of convergence. Numbers average three seeds; SD is the sample standard deviation across seeds. The short screen was deliberately broad. For example, the stronger full-data results below change the ordering of several candidates.

| Encoding idea / runner name | Validation BCE (mean ± SD) | Exact word |
|---|---:|---:|
| Train-fitted conditional next-word bit probabilities as RY angles (`bigram_angles`) | 0.5557 ± 0.0025 | 4.37% |
| Quarter-scale matrix angles with RY/RZ (`scaled_ry`) | 0.5717 ± 0.0166 | 5.21% |
| Normalized PPMI recency vector with a constant anchor (`amplitude_ppmi`) | 0.5795 ± 0.0256 | 1.15% |
| PPMI recency features with jointly learned angle gains/biases (`ppmi_affine`) | 0.5795 ± 0.0274 | 6.04% |
| Normalized raw context matrix with a constant anchor (`amplitude_raw`) | 0.5835 ± 0.0280 | 2.19% |
| Fixed random token vectors, mean pooling (`random_mean`) | 0.5901 ± 0.0062 | 5.73% |
| Paper positional phase-shift matrix (`phase`) | 0.5921 ± 0.0222 | 3.02% |
| Paper minimal angular vector (`angular`) | 0.5931 ± 0.0200 | 2.81% |
| Token basis amplitudes with age-dependent complex phases (`amplitude_complex`) | 0.5943 ± 0.0357 | 0.94% |
| Paper modular hash modulation (`hash`) | 0.5963 ± 0.0196 | 1.98% |
| Jointly trained token angle table, recency pooling (`learned_token`) | 0.5971 ± 0.0160 | 5.62% |
| Original exponential context matrix, H then RX/RZ (`exponential`) | 0.5972 ± 0.0262 | 2.50% |
| Fixed random vectors, geometric recency pooling (`random_recency`) | 0.5980 ± 0.0058 | 5.10% |
| Recency-pooled input ID bits plus last-token bits (`binary_recency`) | 0.6039 ± 0.0142 | 2.08% |
| Train-fitted PPMI/SVD vectors, mean pooling (`ppmi_mean`) | 0.6040 ± 0.0051 | 4.58% |
| Reverse input gate order: H then RZ/RX (`exponential_rzrx`) | 0.6044 ± 0.0249 | 2.92% |
| Train-fitted PPMI/SVD vectors, recency pooling (`ppmi_recency`) | 0.6072 ± 0.0038 | 5.00% |
| Keep matrix; start at zero and use RY/RZ (`exponential_ry`) | 0.6105 ± 0.0102 | 1.67% |
| Train-fitted PPMI/SVD vectors in positional slots (`ppmi_slots`) | 0.6173 ± 0.0063 | 2.29% |
| Fixed random vectors in position-specific slots (`random_slots`) | 0.6193 ± 0.0115 | 1.46% |
| Paper logarithmic diagonal matrix (`diagonal`) | 0.6197 ± 0.0147 | 1.04% |
| Right-aligned sine/cosine features at frequencies 1 and 17 (`slots_fourier`) | 0.6367 ± 0.0083 | 0.94% |
| Fixed permutation of input IDs, then RY/RZ (`permuted_ry`) | 0.6367 ± 0.0036 | 0.31% |
| Right-aligned token angles and separate presence qubits (`slots_id`) | 0.6381 ± 0.0074 | 0.83% |
| Logarithmic ID-to-angle map with RY/RZ (`logrank_ry`) | 0.6487 ± 0.0211 | 0.42% |
| Fourier features re-uploaded between ansatz layers (`fourier_reupload`) | 0.6955 ± 0.0035 | 0.00% |

`constant_zero`, an additional negative control, was tested in confirmation rather than screened: every example starts in the same all-zero state before the trainable ansatz. It has **no access to its context**.

The five original matrix/vector strategies come from the QCSE paper. RY encoding, amplitude encoding, trainable encoders, and re-uploading connect to Halil's report. The controlled gate-order repair, log-rank and permutation tests, explicit presence/position slots, multi-frequency token features, random/binary controls, directional PPMI variants, complex token amplitudes, and conditional-bit loading are the concrete additional designs explored here. “Additional” describes their role in this study, not a claim of research novelty.

## Confirmation on the held-out sentences

| Encoding | BCE ↓ (mean ± SD) | Exact word ↑ | Top-5 ↑ | Vocabulary NLL ↓ |
|---|---:|---:|---:|---:|
| `bigram_angles` | 0.5129 ± 0.0008 | 16.59% | 27.45% | 5.125 |
| `ppmi_affine` | 0.5523 ± 0.0011 | 6.26% | 17.34% | 5.515 |
| `amplitude_ppmi` | 0.5554 ± 0.0010 | 7.44% | 16.88% | 5.551 |
| `constant_zero` | 0.5560 ± 0.0016 | 9.47% | 16.35% | 5.558 |
| `scaled_ry` | 0.5566 ± 0.0011 | 7.62% | 15.57% | 5.541 |
| `exponential` | 0.5607 ± 0.0008 | 4.92% | 11.34% | 5.600 |
| `random_mean` | 0.5704 ± 0.0023 | 9.52% | 16.75% | 5.679 |
| `exponential_ry` | 0.5730 ± 0.0064 | 4.63% | 10.46% | 5.685 |

For comparison, fitted on exactly the same 3,990 training examples:

| Classical reference | Test BCE | Exact / top-1 | Top-5 | Vocabulary NLL |
|---|---:|---:|---:|---:|
| Context-free bit prior | 0.5534 | 9.47% | 17.01% | 5.530 |
| Conditional bigram bit probabilities | 0.5121 | 16.77% | 28.09% | 5.116 |
| Full categorical bigram distribution | Not a bit model | 21.51% | 47.11% | 4.299 |

The categorical bigram is a more expressive output distribution with a larger classical table, so it is a **practical language-model reference**, not a parameter-matched quantum comparison. It is smoothed with ten unigram pseudo-observations per preceding word; unigram probabilities use 0.5 pseudo-count per vocabulary item. The bit version smooths each target bit toward its training prior.

![Held-out comparison with classical controls](results/confirmation.png)

Error bars show seed SD. The numerical results, including all training and validation metrics, are in [summary.json](results/summary.json); per-run histories and predictions are retained under `results/screen/` and `results/confirmation/`.

## Which ideas worked, and why

### 1. Conditional-bit angles: strongest improvement, classical source of the gain

For the last context token `w`, estimate each next-token bit probability using only training counts:

`p_j(w) = (sum_next count(w,next) * bit_j(next) + 10 * prior_j) / (count(w) + 10)`.

Prepare qubit `j` with `RY(2 * arcsin(sqrt(p_j(w))))` on zero. Before any ansatz, its Z-basis probability is exactly `p_j(w)`. This construction has an interpretable predictive signal and distinguishes common preceding words using what follows them, rather than their arbitrary numeric distance. It only uses the **last token**, which is an explicit limitation for longer-context language modeling.

With the trained ansatz it improves BCE by **0.0477** and exact accuracy by **11.66 percentage points** over the original encoding. A paired sentence bootstrap gives a 95% interval of **10.14 to 13.18 points** for that accuracy improvement.

But the direct loading without a trained ansatz already gives 0.5121 BCE and 16.77% accuracy. Adding the ansatz makes BCE slightly worse: difference **+0.00087**, bootstrap interval **+0.00049 to +0.00122**. The inference is straightforward: the useful work here is learning the classical conditional statistics. Preserve that signal if adding a quantum residual; do not initialize an arbitrary transformation and assume it helps.

### 2. Learned PPMI angle calibration: the most useful broader-context lead

`ppmi_affine` learns one gain and one bias for each of 20 pooled PPMI features, jointly with the circuit: `angle_k = gain_k * feature_k + bias_k`. This is **40 extra trainable parameters**, for 98 total, in addition to the fitted PPMI table. It is a modest implementation of the report's trainable-encoder suggestion.

It improves test BCE from 0.5607 to **0.5523**, versus 0.5704 for the random mean-feature control. This is consistent with both linguistic features and angle calibration mattering. However, the full-data confirmation does **not** include a matched fixed `ppmi_recency` run, so it cannot cleanly isolate calibration from feature choice. The short screen supports calibration as a candidate, not a definitive causal attribution.

Its exact accuracy is only **6.26%**. Its BCE advantage over the classical bit prior is **0.00107**, with a paired interval spanning zero (**-0.00275 to +0.00054** for model minus prior). Treat it as a promising follow-up, not an established improvement over a simple classical baseline.

### 3. Small-angle RY: better than the original, no clear context gain

`scaled_ry` uses the original matrix multiplied by 0.25, RY/RZ gates, and a zero initial state. It reaches **0.5566 BCE / 7.62% exact accuracy**. Yet the input-free quantum control reaches **0.5560 / 9.47%**. Their paired BCE difference is +0.00064, with interval **-0.00241 to +0.00385**.

The likely explanation is an easier-to-fit, nearly constant input state. Shrinking angles makes the high-frequency token encodings especially similar: adjacent singleton IDs among the first 64 words have mean fidelity **0.999996**. This can help reproduce global bit frequencies without extracting useful context. The control supports that interpretation; it does not prove every small-angle encoding is unhelpful.

### 4. Amplitude encoding: modest gain, state preparation and inactive qubits matter

`amplitude_ppmi` reaches **0.5554 BCE / 7.44% accuracy**. It loads `[1; pooled_features]`, normalized, so the constant anchor preserves scale information that plain normalization would remove. `amplitude_raw` applies the same idea to the flattened context matrix. `amplitude_complex` instead sums token-basis amplitudes with weights `0.7**age * exp(i * age * pi/2)`, encoding word identity in the basis index and order in phase.

PPMI amplitude loading is indistinguishable from the input-free circuit on BCE here: model-minus-control difference **-0.00058**, interval **-0.00160 to +0.00054**. Its 21 occupied components fit in five qubits before the ten-qubit ansatz. The remaining initial qubits are always zero, which is a potentially favorable bias for frequency-ranked target IDs. The results therefore do not establish amplitude encoding as a superior semantic representation.

A concrete preparation check on context IDs `[0,149,106,332]` found:

| Preparation | Active qubits used by support-aware construction | Single-qubit U gates | CNOTs | Depth |
|---|---:|---:|---:|---:|
| Raw matrix amplitudes plus anchor | 5 | 31 | 26 | 53 |
| PPMI amplitudes plus anchor | 5 | 31 | 26 | 53 |
| Complex token amplitudes | 9 | 511 | 502 | 1005 |

These are Qiskit `StatePreparation` circuits transpiled to `u,cx`, optimization level one, on one representative context. They are not lower bounds; specialized sparse preparation can do better. Naively preparing the entire ten-qubit register instead cost 987–1,013 CNOTs. In comparison, one original angle-encoding layer has 20 rotations and nine CNOTs, plus the initial Hadamards. The experiment uses exact state initialization for amplitude methods, so its runtime **does not charge physical state preparation**. See [amplitude_costs.json](results/amplitude_costs.json).

### 5. Repairing information loss is necessary, but insufficient

All contexts in this corpus fit into one encoding layer. Every first-layer RX acts on `|+>` and changes only a global phase. For a one-word prefix, the flattened feature enters only RX, so **every word produces the same physical state**. This affects all five paper encodings and **1,000 of 6,235 examples (16.0%)**. No downstream shared ansatz can recover information that was discarded.

Both H→RZ→RX and zero→RY→RZ distinguish all 64 tested singleton words, versus one state for the original schemes. But the isolated RY repair still worsens test BCE to **0.5730**. Fixing visibility does not supply linguistic structure or guarantee easier optimization. There is also still severe frequency-rank crowding: repaired adjacent singleton fidelity averages **0.999938**.

This motivates separating three questions: does the encoding retain the input, does it impose useful similarity, and can this ansatz/optimizer exploit it? The present circuit fails the first question for singleton prefixes; the repair alone does not settle the other two.

### 6. Wider separation, position features, and re-uploading did not win

- **Log-rank expansion and input-ID permutation** substantially increase distinguishability but worsen short-screen loss. Numeric separation alone does not create predictive relationships between words.
- **Explicit slots, Fourier features, and binary features** were all tested. They retain more identity/position structure but gave no consistent short-budget improvement over the original. The explicit ID-slot design includes presence qubits so token ID zero differs from missing padding; tests verify sensitivity to every position.
- **Random mean pooling** reaches 9.52% exact accuracy in confirmation, essentially the 9.47% constant baseline, while producing worse BCE. A single accuracy number would make it look much more useful than its probability quality warrants.
- **A fully learned token table** adds 14,580 encoder parameters, for 14,638 total. Its screening BCE, 0.5971, is essentially the original's 0.5972. This is evidence that this high-dimensional table is not useful under the chosen SPSA budget, not that learned embeddings cannot work. A gradient method suited to many encoder parameters deserves a separate test.
- **Fourier re-uploading** is the weakest screened candidate, 0.6955 BCE and zero exact accuracy. It inserts an additional RY/RZ/CNOT feature block between the two ansatz layers while holding ansatz parameters fixed. It roughly doubles input-gate work and learns little in 20 epochs. This does not demonstrate a barren plateau: no gradient-scaling experiment was done, and the result is specific to this feature map, optimizer, and budget.

## A constraint on the report's fidelity proposal

For fixed encoded states and the same trainable unitary applied to every context,

`|<U(theta) psi(x) | U(theta) psi(y)>|^2 = |<psi(x) | psi(y)>|^2`.

Changing the shared ansatz cannot learn a different pairwise fidelity geometry. The numerical check agrees to **2.3e-15**. Fidelity can still be a useful diagnostic, but fidelity-based training needs a trainable encoder, input-dependent interleaving, a learned comparison construction, or another operation that changes the relevant geometry. Replacing the similarity metric alone is insufficient. The report suggests jointly training the encoder; that part is essential to this approach.

## Confidence and limits

The headline winner is consistent across three optimization seeds. The intervals quoted above use **2,000 paired bootstrap resamples of held-out sentence-text groups**, preserving duplicates and averaging per-example differences across the three fitted seeds. They are conditional on this corpus, split, and fitted runs. They do not include variation from new training splits or from choosing among 26 designs, and are not multiplicity-adjusted claims.

Only the selected seven encodings and the input-free control received full-data confirmation. A short-screen failure should not be generalized to larger corpora, different optimizers, deeper circuits, other amplitude preparations, or hardware. Training-fitted tables also have classical storage and fitting costs: matching 58 ansatz parameters does not make these systems resource-equivalent. No hardware shots, noise study, external language benchmark, or semantic-similarity benchmark was run. The task evaluated is causal next-word prediction; it is not a direct measurement of semantic embedding quality.

The statevector implementation was checked against independent Qiskit circuits for 2, 3, and 10 qubits, both entangling directions, each rotation convention, and interleaved re-uploading. A separate test confirms identical SPSA training updates to the production backend. Tests also cover normalization, train-only feature fitting, split separation, word/padding visibility, and direct conditional-probability loading. The final suite contains **45 passing tests**; lint and the project's type check also pass.

## What I would pursue next

1. **A learned predictive encoder with a gentle quantum residual.** Start from the conditional-bit construction, preserve the identity mapping initially, and add older context through a small trainable projection. Compare every added quantum block with the same classical encoder and an identity ansatz. The current best model is effectively a last-word model.
2. **A matched PPMI calibration ablation.** Confirm fixed PPMI recency, learned gains/biases, and a jointly learned low-rank projection under equal data, parameter, and update budgets. Prefer accurate gradients before increasing the encoder to thousands of independently perturbed angles.
3. **A stronger next-word output objective.** Test a valid-vocabulary categorical decoder or another structured output model, with train-only vocabulary construction. The categorical bigram's 21.51% top-1 and 47.11% top-5 expose a substantial gap that encoding alone may not close. This changes the output model and should be reported separately from an encoding ablation.

Trainable MPS/tensor encoders, subword encodings, pretrained classical embeddings, quantum-kernel retrieval, and the report's BBQC/readout proposals remain **untested** here. They are possible next studies, not entries in the tested ranking. MPS is particularly relevant for scaling simulation; adopting it alone is not evidence of better encoding.

## Reproduce or extend manually

Run these from the **QCSE directory**. No new dependencies are required. The worker setting starts ordinary local experiment processes; each uses one numerical-computation thread. No remote service is used.

Reproduce the full screening matrix in a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m experiments.encoding_study.run \
  --workers 3 --output outputs/encoding-screen-reproduction
```

Reproduce confirmation, including the control:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m experiments.encoding_study.run \
  --methods exponential scaled_ry amplitude_ppmi ppmi_affine bigram_angles \
    exponential_ry random_mean constant_zero \
  --seeds 11 22 33 --epochs 60 --train-limit 0 --val-limit 0 --test \
  --workers 3 --output outputs/encoding-confirmation-reproduction
```

A deliberately longer follow-up using eight ansatz layers and fresh optimization seeds, evaluating **validation only** until a new selection is fixed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m experiments.encoding_study.run \
  --methods exponential scaled_ry ppmi_recency ppmi_affine bigram_angles constant_zero \
  --layers 8 --epochs 100 --seeds 44 55 66 --train-limit 0 --val-limit 0 \
  --workers 3 --output outputs/encoding-deeper \
  > outputs-encoding-deeper.log 2>&1
```

Each completed method/seed saves a JSON history and NPZ weights/predictions. Re-running the **identical command** skips completed jobs. An interrupted individual job restarts; there is no epoch-level resume in this experimental runner. Use a new output directory when changing a configuration. Logging is one line per completed job; there is no need to watch a loop of command outputs.

Experimental NPZ files are study artifacts, **not production `QCSEModel.load` checkpoints**. Fitted features can be reconstructed from the saved data split and deterministic code. Raw NPZ files and logs are kept locally but ignored by Git; JSON results and the report remain shareable. The recorded final runs total about 23.3 worker-minutes, with three workers used for both sweeps; this is summed job time, not wall time or a quantum runtime claim.

To recheck and rebuild the numerical summary for the included study directories:

```bash
uv run pytest -q
uv run ruff check .
uv run pyright
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m experiments.encoding_study.diagnostics
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python -m experiments.encoding_study.summarize
```

## Sources and provenance

- [QCSE paper, local v2 PDF](../../../References/2509.05729v2.pdf): architecture, equations 10–28, and the five original encodings. Its published accuracy ranking is not assumed to transfer to this causal corpus and protocol.
- [Amazigh Halil's report](../../../References/Amazigh-Halil_Rapport_2A_CS_R_10-2-2026.pdf): sections 3.4 and 5–7 motivate trainable encoders, RY/amplitude encoding, re-uploading, and alternative extraction methods.
- [Pérez-Salinas et al., Data re-uploading for a universal quantum classifier](https://arxiv.org/abs/1907.02085): theoretical motivation for interleaving data and trainable operations; not a prediction that this NLP experiment must improve.
- [Schuld, Sweke and Meyer, The effect of data encoding on expressive power](https://arxiv.org/abs/2008.08605): motivation for examining input frequencies. Greater frequency expressivity does not by itself establish trainability or generalization.
- [Levy and Goldberg, Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html): motivation for PPMI factorization as a classical distributional feature source. The directional, weighted variant tested here is specified in the code.

Repository baseline: `87fa9a7cef03522957d442485a9bcee35c749a71`. Exact splits, corpus hashes, package versions, and hyperparameters are recorded in each stage's `config.json`. [diagnostics.json](results/diagnostics.json) preserves the encoding checks and the audit of the existing saved model. [source_manifest.json](results/source_manifest.json) records the final experimental source hashes.
