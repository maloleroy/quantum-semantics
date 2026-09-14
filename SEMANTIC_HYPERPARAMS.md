# 30-epoch causal ablation

This sweep compares the full trainable circuit with the matched no-circuit control. It uses `phrases.csv` and `cleaned_sentences.csv` together, excludes Tatoeba, keeps the vocabulary built from all source rows, and uses sentence-grouped 64/16/20 train/validation/test splits. To keep the local check short, each run used 1,000 curated sentences and 5,000 replacement-sampled training examples per epoch; the cluster sweep can raise `--max-sentences` without changing the code path.

All reported retrieval metrics are cosine similarity over the QCSE vocabulary. The training loss remains tied dot-product cross-entropy internally, but dot-product accuracy is not reported.

The 18 runs vary one setting at a time around `alpha=0.05`, `lr=0.003`, two layers, and batch size 64:

- `alpha`: 0.01, 0.05, 0.2
- learning rate: 0.001, 0.003, 0.01
- layers: 2, 4, 8
- batch size: 32, 64, 128

The highest full-trainable result was `lr=0.01`, `alpha=0.05`, two layers, batch 64 (test cosine top-1 20.2%, top-5 39.6%). The lower-learning-rate alpha sweep reached 19.4% / 36.4% at `alpha=0.01`, `lr=0.003`. More layers did not help: eight layers reached 14.9% top-1 and 33.0% top-5, while the no-circuit control reached 15.2% and 36.0%. The no-circuit control was competitive in every setting and exceeded the trainable circuit for four-layer and `alpha=0.2` runs, so the circuit benefit is not established by this short sweep.

Plots and machine-readable results:

- [test cosine hyperparameter bars](outputs/semantic-hyper-report/test-cosine-hyper-bars.png)
- [no-circuit minus full-trainable delta](outputs/semantic-hyper-report/no-circuit-delta.png)
- [hyperparameter grid JSON](outputs/semantic-hyper-report/hyper-grid.json)

## Inference-only Qwen baselines

`scripts/qwen_baselines.py` adds two optional references without fine-tuning any model:

- `Qwen/Qwen2.5-0.5B-Instruct`: next-token logits, scored on QCSE words that are single tokenizer tokens.
- `Qwen/Qwen3-Embedding-0.6B`: mean-pooled hidden states, cosine-ranked over the full QCSE vocabulary.

Run them against a completed QCSE directory after the cluster or local machine has downloaded the weights:

```bash
uv run python scripts/qwen_baselines.py \
  outputs/semantic-hyper-30/semantic-causal-20260914T145440Z-12a1up1d \
  --device cuda --max-examples 256 \
  --output outputs/qwen-baselines.json
```

The script writes only the baseline JSON and leaves the training checkpoint untouched. Causal results include tokenizer coverage because subword tokenization means that exact word scoring is not defined for every vocabulary item.
