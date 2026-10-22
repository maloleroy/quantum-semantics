"""Run inference-only Qwen causal and embedding baselines on a saved QCSE run.

The script never fine-tunes or writes to the QCSE checkpoint. Model weights are
downloaded by Transformers into its normal cache when they are not present.
"""

import argparse
import json
from pathlib import Path

import torch


def load_run(run):
    checkpoint = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
    vocabulary = json.loads((run / "vocabulary.json").read_text(encoding="utf-8"))
    return checkpoint, vocabulary


def causal_baseline(run, model_id, device, max_examples):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint, vocabulary = load_run(run)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    contexts, targets = checkpoint["contexts"], checkpoint["targets"]
    test_ids = checkpoint["test_ids"][:max_examples]
    token_to_words = {}
    for index, word in enumerate(vocabulary):
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            token_to_words.setdefault(ids[0], []).append(index)
    correct1 = correct5 = covered = 0
    with torch.inference_mode():
        for example_id in test_ids:
            context = contexts[example_id]
            words = [vocabulary[index] for index in context.tolist() if index < len(vocabulary)]
            encoded = tokenizer(" ".join(words), return_tensors="pt").to(device)
            logits = model(**encoded).logits[0, -1]
            candidate_tokens = list(token_to_words)
            scores = logits[candidate_tokens]
            top = torch.topk(scores, min(5, len(candidate_tokens))).indices.tolist()
            ranked_tokens = [candidate_tokens[index] for index in top]
            target = int(targets[example_id])
            target_tokens = tokenizer.encode(" " + vocabulary[target], add_special_tokens=False)
            if len(target_tokens) != 1 or target_tokens[0] not in token_to_words:
                continue
            covered += 1
            predictions = [word for token in ranked_tokens for word in token_to_words[token]]
            correct1 += int(target in predictions[:1])
            correct5 += int(target in predictions[:5])
    return {
        "model": model_id,
        "examples": len(test_ids),
        "covered_examples": covered,
        "coverage": covered / max(1, len(test_ids)),
        "cosine_top1": correct1 / max(1, covered),
        "cosine_top5": correct5 / max(1, covered),
        "scoring": "next-token logits; words must be single tokenizer tokens",
    }


def embedding_baseline(run, model_id, device, max_examples, batch_size):
    from transformers import AutoModel, AutoTokenizer

    checkpoint, vocabulary = load_run(run)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    contexts, targets = checkpoint["contexts"], checkpoint["targets"]
    test_ids = checkpoint["test_ids"][:max_examples]

    def encode(texts):
        batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return torch.nn.functional.normalize(pooled, dim=-1)

    vocab_vectors = []
    with torch.inference_mode():
        for start in range(0, len(vocabulary), batch_size):
            vocab_vectors.append(encode(vocabulary[start : start + batch_size]).cpu())
        vocab_vectors = torch.cat(vocab_vectors)
        correct1 = correct5 = 0
        for start in range(0, len(test_ids), batch_size):
            ids = test_ids[start : start + batch_size]
            texts = [
                " ".join(
                    vocabulary[token]
                    for token in contexts[example_id].tolist()
                    if token < len(vocabulary)
                )
                for example_id in ids
            ]
            scores = encode(texts).cpu() @ vocab_vectors.T
            top = scores.topk(min(5, len(vocabulary)), dim=-1).indices
            target = targets[ids]
            correct1 += int((top[:, 0] == target).sum())
            correct5 += int((top == target[:, None]).any(1).sum())
    return {
        "model": model_id,
        "examples": len(test_ids),
        "cosine_top1": correct1 / max(1, len(test_ids)),
        "cosine_top5": correct5 / max(1, len(test_ids)),
        "scoring": "mean pooled last hidden state; cosine over the full QCSE vocabulary",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="completed QCSE run directory")
    parser.add_argument("--causal-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--max-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-causal", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {"run": str(args.run), "inference_only": True}
    if not args.skip_causal:
        results["causal"] = causal_baseline(
            args.run, args.causal_model, args.device, args.max_examples
        )
    if not args.skip_embedding:
        results["embedding"] = embedding_baseline(
            args.run, args.embedding_model, args.device, args.max_examples, args.batch_size
        )
    output = args.output or args.run / "qwen-baselines.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
