# Byte-backed subword prediction: prepared manual screen

30 October 2026. Implemented and tested on small fixtures; full-corpus tokenizer
fitting, preparation, model training and generation have **not been run**.
This addresses the representation barrier exposed by the 41.80% UNK rate in
word-level WikiText validation. It does not establish improved model quality.

## Tokenization and corpus contract

The reference tokenizer follows the frequent-pair merging idea of
[Sennrich, Haddow and Birch (2016)](https://aclanthology.org/P16-1162/), with a
complete UTF-8 byte alphabet instead of their character segmentation. This is
our declared tokenizer, not a reproduction of GPT tokenization or a standard
published WikiText evaluation.

For a 512-outcome register, ID 0 remains UNK, IDs 1–256 represent every byte,
and up to 255 further IDs represent merge rules. Fit those rules on all official
training articles, independently of optimization sample limits. Repeatedly
merge the most frequent adjacent pair, breaking ties lexicographically by token
IDs. Merges stay within whitespace or nonwhitespace chunks; chunks reconstruct
the input exactly. There is no Unicode normalization, lowercasing or removal of
punctuation/numbers. All byte strings obtained from valid UTF-8 input are
representable even if their words/characters were absent from training.

Unused register outcomes, if a tiny training corpus supplies too few merges,
remain explicit unused tokens; they are not silently removed or renormalized.
UNK likewise remains a possible model outcome and counts as an incorrect draw
for byte-backed targets. The input encoder never needs to emit it. Merge rules,
vocabulary and target byte lengths are stored as NPZ arrays, not JSON array dumps.

Keep official WikiText article partitions and the previous normalized-article
overlap guard. Preserve every character and newline within an article, including
title/section markup; omit only separator blanks before the first article.
Reset context at every article boundary. The first token of each article is
not predicted. No EOS is added; generation uses an explicit token limit.
Every article is checked for exact byte round-trip during preparation. Held-out
articles are encoded with the frozen training tokenizer and contribute no merge,
vocabulary-frequency or count-model statistics.

## Scoring units

NLL, one-shot sampling success and argmax now refer to **subword tokens**, not
words. Comparing these numbers directly with the previous word scores would
confound task granularity and preprocessing. Known-target metrics coincide with
all-target metrics when UNK rate is zero; this is coverage, not improved semantic
prediction. Whitespace and punctuation may be frequent/easy targets.

`target_bits_per_byte` is `sum(token NLL) / sum(target byte lengths) / log(2)`
over selected canonical targets. It is not the average of per-token NLL/length
ratios. The first token and unsampled targets are excluded, and context is
limited to the declared token window. This is a sampled conditional-code-length
estimate, not full-corpus perplexity or raw-text likelihood summed over alternate
segmentations. Model selection remains validation token NLL within a single,
fixed tokenizer/dataset. Test is never scored by the study runner.

## Declared six-fit screen

Prepare 32,768 training, 4,096 validation and 8,192 test targets with a 512-token
vocabulary and 128-token context. Each partition uses its own fixed seed-1618
sampling procedure. Training statistics and merges use all training articles.
A 128-token window is not assumed equivalent to 32 words; preparation reports
corpus byte and token totals so compression can be inspected.

Run seed 11, 30 epochs, batch 512, Adam LR .003, clipping 5, NLL loss and validation
NLL selection including epoch zero, float64/complex128, one thread per fit.
There is no LR search in this screen. Quantum recipes have two rank-64 blocks
on nine token qubits plus one auxiliary qubit. Direct token measurement ignores
the auxiliary register and retains all outcomes.

The fixed initial amplitude decay is .9, declared for this new token granularity
without extrapolating earlier word-decay results. Conditional controls pool the
last token, the preceding 15 tokens and older context; their within-pool decays
remain .7/.97. Phase scale is 1. These choices are exploratory, not tuned wins.

Independently train:

- Learned context gates, 128 tokens: 277,632 parameters.
- The same model with only the final 16 tokens retained: identical target/group/
  byte-length arrays and allocated parameter count; older-input parameters are inactive.
- Static auxiliary dynamics: 262,272 parameters.
- Fixed random context gates with trained dynamics: 262,272 trainable parameters
  and the additional 15,360-entry stored control table.
- Frozen quantum dynamics with learned angles: 15,360 trainable parameters.
- The classical amplitude/pooled-angle feature control: 277,315 parameters.

Evaluate unigram and bigram references from all training counts. Their data
exposure exceeds the gradient target subset. The tokenizer is shared across
all models, and its fitting/storage/preprocessing costs are not free information.
Paired article-bootstrap intervals are descriptive conditional on this seed and
reused validation articles; adaptive model selection uncertainty is not included.

## Generation and artifacts

After successful validation audits, generate 64 tokens from each model for three
fixed prompts: `The history of `, `In 2026, the researchers ` and
`Quantum mechanics describes `. Use seeds 20261030, 20261031 and 20261032.
Use unmodified distributions and one categorical/Born draw per token: no
postselection, retries, temperature adjustment or sample selection. These are
qualitative diagnostics, not a language-quality benchmark.

`qcse generate` appends sampled token IDs directly to the rolling context.
Byte fragments are concatenated before UTF-8 rendering. UNK/unused outcomes
remain in the context and appear as explicit markers. Malformed/incomplete UTF-8
is displayed with replacement characters, without rejecting its tokens. The
runner preserves exact generated IDs in NPZ and rendered text in CSV.

Generated tools/artifacts live in `output/byte-bpe-screen/`. Each batch stores
prepared full/short datasets, locked selections, histories, weights, validation
arrays, `validation.csv`, `paired.csv`, `samples.csv` and sample-ID NPZ files.
Source/corpus/runner hashes, a process guard and a file lock protect manual runs.
The optional printed batch path resumes by preserving completed fits and
restarting only missing fits from initialization; optimizer resume is unavailable.
An interrupted preparation is preserved for inspection rather than overwritten.

## Verification and handoff

All 45 tests, strict Pyright and Ruff pass. Tests cover unseen UTF-8/NUL round-trip,
ranked merging against a sequential reference, train-only merge/count invariance,
article boundaries, unchanged held-out samples when the optimization limit changes,
NPZ codec/byte-length validation, byte-normalized metrics, autoregressive byte
assembly, retained non-text outcomes, and count-checkpoint persistence. Existing
word-model and quantum probability/gradient checks also pass. Only small fixture
work was executed; no full-corpus tokenizer fitting or model training was launched.
Runner syntax and CLI/import checks passed, but the full manual pipeline is unrun.

```bash
bash output/byte-bpe-screen/run.sh
```

Allow roughly 10–20 minutes, with uncertainty from first-time tokenizer preparation
and the wider context. This is an estimate based on the earlier six-minute quantum
fits, not a new timing measurement. Six one-thread workers run concurrently.
To preserve completed fits after a failure, rerun with its printed batch path:

```bash
bash output/byte-bpe-screen/run.sh output/byte-bpe-screen/batch-XXXXXX
```

The user's instruction is to hand over training commands and stop. Do not launch
or monitor this screen autonomously. Competitive generation, useful quantum
computation and hardware advantage remain open research goals.
