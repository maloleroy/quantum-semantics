# Semantic decoder prototype

This is a small experiment inspired by Amazigh Halil's supplied report
`Amazigh-Halil_Rapport_2A_CS_R_10-2-2026.pdf`, sections 3.4 and 4.2–4.4,
especially Figure 3 (physical PDF page 10), and the accompanying roadmap.
It is a chain-circuit prototype, not a reproduction of the complete BBQC method.

## Model

- Token IDs address learned embedding rows; they do not encode numerical meaning.
- Concatenate the ordered, left-padded causal context embeddings. A learned linear
  projection followed by `pi * tanh` produces RY and RZ input angles.
- Apply trainable RX/RZ rotations and controlled RZ gates along the fixed latent
  chain `0 → 1 → ...`. The graph is a chosen prior, not a learned language graph.
- Measure exact X, Y and Z expectations on every qubit. Final diagonal gates are
  retained because their phases affect X/Y measurements.
- A linear decoder maps these features to the embedding dimension. Scores are
  `E @ hidden + output_bias`, using the same embedding matrix as the input lookup.
  Optimize word cross-entropy with Adam through the entire model. Evaluation also
  reports cosine retrieval after normalizing `E` and `hidden`; cosine is not the
  training objective.

Defaults: four qubits, two layers, embedding dimension 16, window 4, Adam LR 0.003,
batch 64, seed 42, and 5,000 replacement draws per epoch. There is no classical
context bypass around the circuit. Full softmax includes every vocabulary word.

The backend represents an exact statevector as real/imaginary Torch arrays.
It uses no shot sampling or bond truncation. Recorded half-chain entropy and
Schmidt rank describe the exact state; they are not an MPS bond-dimension limit.
Apple's MPS device backend and matrix-product states are different concepts.

## Runs and inference

```bash
uv run python scripts/semantic_experiment.py --epochs 10
uv run python scripts/semantic_experiment.py --resume outputs/semantic/<run> --epochs 50 --evaluate-test

uv run python scripts/semantic_experiment.py --datasets phrases --epochs 10
uv run python scripts/semantic_experiment.py --resume outputs/semantic/<phrases-run> --epochs 50 --evaluate-test
```

The default sanity corpus has 96 distinct repetitive sentences. Named real-data
sources use the existing corpus loader and the complete shared vocabulary.
`--datasets phrases cleaned` makes both curated sentence pools eligible;
`--max-sentences N` optionally draws a seeded subset before splitting.

Each new run gets a unique directory. `checkpoint.pt` atomically saves model,
optimizer, vocabulary, examples, grouped splits, history and sampling RNG after
every epoch. Resume restores these settings; `--epochs` sets the new total target,
and `--device` can change the runtime device. Fresh-run model/data flags do not
override saved settings during resume.

`summary.json` records initial settings and data provenance. `history.json`
contains word CE, perplexity, top-1/top-5, gradient norms from the last batch,
and entanglement diagnostics. Training monitoring uses at most 512 fixed examples;
validation uses its complete split. The optional final `test.json` includes a
training-frequency baseline, and `predictions.json` decodes eight test contexts
back to words. Each dataset uses one grouped 64/16/20 split, not cross-validation.

Checkpoints are specific to this prototype. To predict a new context on CPU:

```python
import torch
from qcse.semantic import SemanticModel

saved = torch.load("outputs/semantic/<run>/checkpoint.pt", weights_only=True, map_location="cpu")
model = SemanticModel(**saved["model_config"])
model.load_state_dict(saved["model"])
words = saved["vocabulary"]
lookup = {word: i for i, word in enumerate(words)}
context = [lookup[word] for word in "the red cat".split()]  # Known words only.
window = model.config["window"]
context = context[-window:]
ids = torch.tensor([[len(words)] * (window - len(context)) + context])
with torch.no_grad():
    print([words[i] for i in model(ids).topk(5).indices[0].tolist()])
```

## Measured local results — 2026-09-14

Both runs used the defaults above on CPU, first for 10 epochs and then resumed
to 50 with saved optimizer and RNG state. Each fit therefore used 250,000 draws
in total. Test was scored once after epoch 50. No further tuning was performed.

| Corpus | Epoch | Train CE | Validation CE | Validation top-1 | Validation top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Repetitive, 23 words | 0 | 3.122 | 3.128 | 1.9% | 21.9% |
| Repetitive, 23 words | 10 | 0.646 | 0.756 | 61.9% | 97.1% |
| Repetitive, 23 words | 50 | 0.635 | 0.751 | 63.8% | 98.1% |
| Phrases, 10,864 words | 0 | 9.320 | 9.325 | 0.0% | 0.1% |
| Phrases, 10,864 words | 10 | 3.740 | 4.754 | 18.3% | 39.4% |
| Phrases, 10,864 words | 50 | 1.955 | 6.765 | 19.5% | 42.1% |

The repetitive corpus has 96 sentences and 1,157 model parameters. Its test has
133 token examples: CE 0.781, top-1 **62.4%**, top-5 **93.2%**, versus the
training-frequency baseline's 14.3% and 48.9%.

The real run uses all 1,000 phrases and 185,454 parameters, retaining the full
shared vocabulary. Its test has 1,246 token examples: CE 7.582, top-1 **18.5%**,
top-5 **39.3%**, versus frequency-only 9.5% and 16.9%. These are epoch-50 results,
not results from a validation-selected checkpoint. **Validation CE is best at
epoch 10 and worsens afterward while training CE falls: clear overfitting.**
Top-k accuracy rises slightly even as cross-entropy worsens.

Gradients remained finite through embedding, encoder, ansatz and decoder. Final
mean half-chain entropy was 0.00051 bits (repetitive) and 0.00452 bits (phrases),
with numerical Schmidt rank 2 at the recorded threshold of `1e-6`. This is weak
entanglement and provides no evidence of a useful quantum contribution.

Evidence is in ignored local folders:

- `outputs/semantic-prototype/semantic-causal-20260914T134925Z-kxv04lf5/`
- `outputs/semantic-prototype/semantic-causal-20260914T135100Z-ba943t5l/`

The focused new tests passed (3): exact statevector/XYZ agreement with Qiskit,
finite nonzero gradients through every parameter group, and token-ID permutation
equivariance. Ruff and targeted Pyright passed. CPU training, checkpoint reload,
continuation, full test scoring and word exports completed in both runs.
CUDA and Apple MPS execution of this new differentiable model were not tested.

The follow-up 20-epoch phrase ablations are documented in
[SEMANTIC_ABLATIONS.md](SEMANTIC_ABLATIONS.md), with a validation plot and a
decoder-only control. They do not show a benefit from the trainable ansatz yet.

## Scope

This stage checks learning and word recovery. It does not yet compare a matched
classical model, full-state fidelity retrieval, semantic similarity benchmarks,
learned graph structure or tensor-network truncation. Held-out template prediction
and phrase completion alone do not establish semantic generalization or quantum advantage.
