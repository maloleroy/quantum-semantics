# Next corpus: WikiText-2 Raw

Use **WikiText-2 Raw** for the next Wikipedia experiment. It contains connected
articles rather than isolated example sentences, has established official
train/validation/test partitions, and is small enough to download in full.
The [original paper](https://arxiv.org/abs/1609.07843) introduces WikiText for
longer-context language modeling; the
[publisher's dataset card](https://huggingface.co/datasets/Salesforce/wikitext)
describes its Good/Featured Wikipedia article sources. Choose the raw variant
so that our own training-only vocabulary controls unknown-word replacement.

The pinned files are in `data/wikitext-2-raw-v1/`: 7.73 MB of Parquet and 13.36 MB
of exported UTF-8 text (about 21 MiB total). Bulk files are ignored; the
[source manifest](../data/wikitext-2-raw-v1/manifest.csv) and
[download instructions](../data/wikitext-2-raw-v1/README.md) are tracked.

Our current lowercase, letters/apostrophes-only tokenizer gives:

| Official partition | Article records | Normalized words | Eligible next-word targets |
| --- | ---: | ---: | ---: |
| Train | 629 | 1,691,503 | 1,690,874 |
| Validation | 60 | 176,918 | 176,858 |
| Test | 62 | 198,741 | 198,679 |

Counts include article and section title words. Repeated normalized articles
within one partition retain a common group ID; exact normalized article overlap
between partitions is rejected. Source text preserves case/punctuation/numbers;
the current model inputs do not. Results under this preprocessing and restricted
vocabulary are **not standard published WikiText-2 perplexities**.
The counts above are in [CSV](experiments/wikitext2-corpus.csv).

## Budget measured on this machine

The actual host is a Ryzen 7 7840S with 8 physical cores / 16 threads and about
27 GiB total RAM. The project uses CPU PyTorch. Do not plan against the older
24-core resource ceiling in the research notes.

On 26 October 2026, short probes of the existing dense Hamiltonian gave:

| Register vocabulary including UNK | Qubits | H parameters | Validation UNK targets | Seconds / Adam step | Projected minutes / 30 epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 | 8 | 65,536 | 48.3% | 0.040 | 1.5 |
| 512 | 9 | 262,144 | 40.9% | 0.241 | 8.9 |
| 1,024 | 10 | 1,048,576 | 33.3% | 1.525 | 57.0 |
| 2,048 | 11 | 4,194,304 | 25.8% | 12.448 | 470.2 |

These are **per-fit extrapolations**, not completed 30-epoch runs or convergence
guarantees. Probe: one Torch thread, float64/complex128, context 32, batch 512,
Adam LR .001, NLL, gradient clipping 5, two warm-up steps followed by three timed
steps (median). Forward timing uses batches of 256. Training inputs came from
the prepared 512-word corpus; IDs were reduced for the 256-word timing and reused
for larger registers. Matrix dimensions match each row, but these probes are
not quality comparisons between vocabularies. Coverage uses every validation
target with vocabulary fitted on every training word, excluding the first word
of each validation article as a target.

The projection includes 64 updates per epoch on 32,768 fixed sampled targets,
epoch-zero/epoch-end validation on 4,096 targets, and selected validation/train
evaluation. It excludes preparation, I/O, separate test scoring and sustained-load
thermal effects. Three seeds and additional controls multiply the budget.
See [full timing CSV](experiments/wikitext2-budget.csv); local probe tools and raw
timings are under ignored `output/dataset-selection/`.

A separate end-to-end one-epoch 512-word H fit (seed 11, same subset/window/batch,
NLL selection) completed in 37.56 seconds including checkpointing and selected
train/validation evaluation, while another corpus preparation ran concurrently.
This verifies the new preparation/training path; it is not a competitive quality
result. The checkpoint is local under
`output/dataset-selection/run-20261026T212538Z-ff3e014a/`; test was not scored.

Start with **512 words for screening**, then **1,024 words for a main fit**.
The dense eigendecomposition, not the download size, is the bottleneck. A full
1.69-million-target epoch has about 52 times as many updates as the screening
subset; the few-hours claim does not extend to 30 full-corpus epochs.
Even 2,048 words projects to nearly eight hours per fit with this implementation,
so it falls outside the requested budget.

## Preparation and experiment protocol

```bash
uv run qcse prepare --data data/wikitext-2-raw-v1 \
  --vocabulary 512 --window 32 \
  --train-limit 32768 --validation-limit 4096 --test-limit 8192
```

Use the printed path:

```bash
uv run qcse train --dataset output/dataset-.../dataset.npz \
  --models hamiltonian dephased mlp bigram unigram \
  --epochs 30 --batch-size 512 --seeds 11 --threads 1 \
  --loss nll --selection nll
```

This proposes **validation NLL selection** for language modeling, explicitly
overriding the CLI's sampling-success default. Declare it before comparing fits;
continue reporting expected one-shot success, argmax and NLL separately. It is a
starting recipe, not a tuned LR or a claim that all these fits cost the H time.
Use `--vocabulary 1024` in a fresh preparation for the larger register.

The directory loader keeps official splits, concatenates lines only within each
article, and resets causal context at every new article. Windows can cross
sentences and paragraphs. Title and section words are retained, with `=` markup
removed by tokenization. There is no EOS token and the first word of an article
is not predicted. Vocabulary and count statistics see only the official training
articles, independently of the gradient-example limit. Test remains unscored
until the separate `evaluate` command is invoked after weights/settings lock.

To test whether context matters, compare windows 1, 4, 16 and 32 on identical
partitions and target selections, and retrain controls with older context shuffled
while preserving the most recent word. Hold training exposure and tuning budgets
fixed. The current amplitude encoder heavily discounts distant tokens and uses
periodic position phases; increasing the window alone does not supply a GPT-like
attention mechanism or establish useful long-range memory. Treat the dataset as
an opportunity to measure that limitation.

At 1,024 words, one third of validation targets still map to UNK. A high UNK
prediction rate can inflate aggregate accuracy or sampling success; retain
known-target metrics without renormalizing. Predicting UNK does not recover the
missing word. For broader vocabulary, a structured quantum evolution or a small
latent quantum register with a classical decoder is a more plausible next change
than scaling a dense word Hamiltonian. The hybrid is a different hypothesis and
needs the existing classical-replacement and frozen-core controls. Subword/byte
prediction is another task and would require explicit tokenizer/scoring changes.

## Familiar classical references and alternative corpora

For a stronger context-using classical baseline, start with a **small causal
Transformer trained from scratch** (for example 2 layers, 4 heads, width 128,
context 32/64), using exactly our vocabulary, partitions and prediction targets.
Tune/match its budget explicitly; this configuration is a proposed starting
point, not a measured result. PyTorch's official
[word language model example](https://github.com/pytorch/examples/tree/main/word_language_model)
already demonstrates WikiText-2 LSTM and Transformer models. Adapt its loading
and evaluation to our leakage/selection contract before using it as evidence.
Neither a Transformer nor LSTM is currently implemented in this repository.

[nanoGPT](https://github.com/karpathy/nanoGPT) is a familiar compact GPT reference
with small CPU configurations, but its author now marks it deprecated in favor
of nanochat. Its architecture can be an educational reference; importing its
whole training stack is unnecessary here. A pretrained GPT-2 checkpoint has a
different tokenizer, prior corpus exposure and parameter budget, so it would be
a separate external reference rather than a controlled quantum comparison.

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) is the next
alternative if the goal shifts toward coherent generation with simpler language.
Its [paper](https://arxiv.org/abs/2305.07759) demonstrates small classical language
models on synthetic multi-paragraph stories; it does not establish quantum
feasibility or training within hours on this machine. The full collection has
over two million stories, so select complete stories within a fixed word budget
and preserve story groups rather than downloading/training it indiscriminately.
The published TinyStories-1M/3M models are useful classical references, not quantum
models or fair substitutes for training matched controls from scratch.

WikiText-103 is a later scaling test, with roughly 103 million training tokens;
its much larger corpus does not solve our representation or dense-H bottleneck.
Penn Treebank and Tiny Shakespeare are useful small demonstrations, but offer
less suitable Wikipedia context or a narrow literary domain. Only WikiText-2 Raw
was downloaded for this recommendation.
