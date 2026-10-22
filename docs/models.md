# Models and code map

Code paths below are relative to the repository root. The project contains
several experiments; their outputs and metrics are not interchangeable.

| Family | Entry points / implementation | Learned parts and output |
| --- | --- | --- |
| Production QCSE | `qcse` CLI; `context.py`, `circuit.py`, `model.py`, `simulation.py`, `training.py`, `state_cache.py` | Fixed rank-angle context encoding; RX/RZ/CRZ angles trained with SPSA/Adam against bit BCE. Output: Z marginals; threshold decoding may yield an invalid word ID. |
| Semantic prototype | `scripts/semantic_experiment.py`; `src/qcse/semantic.py` | Learned lookup→angles→small circuit→XYZ features→tied word head. Direct/zero/frozen pathway ablations. Cosine retrieval and word-CE scores differ. |
| Quantum attention | Same script; `src/qcse/attention.py` | QCSE or learned classical token input, quantum Q/K/V features and causal real-overlap attention. Optional circuit removal. Not a standard Transformer. |
| Robust benchmark | `scripts/robust_benchmark.py`; `benchmark.py`, `benchmark_models.py`, `benchmark_report.py` | Shared prepared targets/splits and word-level scoring; independent paper QCSE, hybrid, MLP/mean and real scratch BERT adapters. Autograd for these trainable adapters. |
| New controls | `scripts/research_model_controls.py` | Scratch causal GPT decoder and nonlinear classical hybrid-feature replacement. CPU-tested prototypes; not registered in the frozen benchmark factory. |
| Optional pretrained references | `scripts/qwen_baselines.py` | Inference-only causal/embedding probes on a saved semantic run; may download external weights. Different pretraining/tokenization and covered-word populations; not a matched scratch-model result. |

All `src` paths in the table are under `src/qcse/`. See [protocol](protocol.md)
for data, scoring and version boundaries, and [runs](runs.md) for evidence.

## Corpus and context

`phrases.csv` and `cleaned_sentences.csv` are the active inputs. Tatoeba is
excluded because the cleaned file is its curated version. Default cleaning
tokenizes/lowercases and deduplicates exact sentences across sources. Vocabulary
IDs are frequency-ranked with alphabetical ties, built from the complete active
sources before filtering/splitting: 10,864 words, hence 14 bits/qubits for the
paper adapter. This is transductive preprocessing, not a train-only OOV setup.

`make_examples` excludes the target and sentence-crossing windows. Causal
examples use previous words and omit position zero. Native CBOW uses both sides
and includes position zero. The recent L4/L2R2 controls instead share positive
target positions in training and evaluation. L2pad keeps two right slots padded
to isolate adding right information from removing left context.

## Fixed QCSE encoder and simulator

`ContextConfig` selects exponential, diagonal, phase, hash or angular features.
For the default matrix, theta_i=2π·ID_i/V:

```
C[i,j] = exp(-alpha*abs(i-j))*sin(omega*theta_i)*cos(omega*theta_j) + theta_i
```

The row-major matrix is zero-padded into `(encoding_layers, qubits, 2)` angles.
Each pair drives RX then RZ; each encoding layer ends with adjacent CNOTs.
H is applied once initially. Forward/reverse chains and q0-first ordering match
the exported Qiskit circuit. Angular features first pad to a square. No features
are cropped or interpolated.

The batched encoder now preserves the scalar recipe for all five methods,
including its phase/hash indexing. It groups equal-length contexts and bounds
each encoding batch by `simulation_batch_size`. Qiskit remains the independent
reference and circuit exporter. This optimization is separate from the robust
adapter's encoder; it does not retroactively change frozen DGX measurements.

The production simulator stores real/imaginary tensor pairs, combines commuting
diagonal gates per layer and skips the final diagonal block for Z marginals.
SPSA evaluates both perturbed weight vectors together. CPU defaults to float 64;
CUDA/MPS use float 32. `ContextStates` reuses duplicate contexts, packs small
sets on device and streams larger sets through a bounded host cache.

## Robust adapters and readouts

The paper adapter uses H/RX/RZ/CNOT numerical encoding and Z marginals. Its
independent-bit valid-word logits satisfy

```
logit(w) = constant(context) + binary_code(w) · logit(bit_marginals).
```

Word softmax cancels the constant. This is a fixed binary-code linear head on
q log-odds features; it is not the joint Born distribution. BCE and valid-word
CE are distinct training objectives. Existing probability/gradient tests also
cover invalid vocabulary codes. Joint-Born scoring is still a proposed readout.

The hybrid uses a learned word lookup and angle encoder, product-state input,
chosen variational circuit, local XYZ readout and unrestricted decoder/tied
word head. Its identity arm uses exact analytic one-qubit moments, without a
statevector. Local rotations can be absorbed by the decoder; different training
coordinates still affect optimization.

Implemented circuit families are CRZ chain/ring, ZZ chain, XX chain, RY/CNOT,
local and identity. Frozen-circuit arms still train their encoder/head. A final
Z-diagonal block is invisible to any computational-basis probability cost;
XYZ can observe phase effects. See [Hamiltonians](literature.md#hamiltonians-and-ansatz) for the generator mapping.

## Classical references and the GPT gap

MLP uses ordered context embeddings, a GELU feedforward map and tied word head.
BERT is `BertForMaskedLM`, width 32/two layers/two heads in the main comparison,
randomly initialized on the same word IDs. It inserts a MASK in the middle for
L2R2 and at the end for L4. Left-only input does not make its attention causal.

The GPT prototype uses a triangular attention mask, width 32/two layers/two
heads, predicts from the last left state, excludes PAD from predicted words and
does not pass labels through the library's automatic loss shift. It uses no
pretrained weights, tokenizer or external text. BERT/GPT differ in query token,
positions, normalization and dropout, so this is not a pure mask ablation.

| Current/proposed model at V=10,864 | Parameters | Status on common-target study |
| --- | ---: | --- |
| MLP-32 |368,880|Complete |
| BERT-32 |385,394|Complete |
| q8/depth 2 CRZ |186,190|Complete |
| q8 identity |186,144|Complete |
| MLP-16 |187,312|Existing factory; new matched-target fits pending |
| Classical angle features |186,824|Prototype only |
| Causal GPT-32 |373,280|Prototype only |

The nonlinear control shares the hybrid lookup/encoder/decoder/head, replacing
the circuit with 16→16 GELU→24 tanh features. It adds 634 parameters overall, but
its feature module has 680 parameters versus 46 in the circuit. Neither exact
component capacity nor Bloch-vector geometry is matched. Keep those differences
visible when interpreting results.

## Checkpoints

Production `run.npz` stores optimizer/SPSA RNG/history for ordinary continuation;
the one-shot legacy focused early-stopping extension rejects continuation.
Semantic/attention `checkpoint.pt` retains the last chronological weights,
optimizer, sampler and selection state; `selected.pt` is inference-only.
New early-stopping resume uses saved patience/minimum, not fresh CLI defaults.
Historical selected checkpoints without selection state are rejected explicitly.

Robust fits have separate `checkpoint.pt`/`selected.pt`, raw prediction NPZ,
history, runtime, phase and task ledgers. New GPT/control integration must prove
factory, resume and CUDA behavior before it joins that runner.
