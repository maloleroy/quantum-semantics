"""Small CPU/device forward-and-gradient parity check for every benchmark adapter."""

import argparse
import copy
import json
import os

import torch

from qcse.benchmark import configurations
from qcse.benchmark_models import create_benchmark_model, training_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), required=True)
    parser.add_argument("--trainability", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    contexts = torch.tensor([[16, 16, 1, 2], [3, 4, 5, 6]])
    targets = torch.tensor([3, 7])
    results = []
    for arm in configurations(args.trainability):
        reference = create_benchmark_model(16, 4, "causal", arm).eval()
        actual = copy.deepcopy(reference).to(args.device).eval()
        left, right = reference(contexts), actual(contexts.to(args.device))
        torch.testing.assert_close(left, right.cpu(), atol=3e-5, rtol=3e-4)
        trainable = [p for p in reference.parameters() if p.requires_grad]
        if trainable:
            training_loss(reference, contexts, targets).backward()
            training_loss(actual, contexts.to(args.device), targets.to(args.device)).backward()
            for (_, a), (_, b) in zip(
                reference.named_parameters(), actual.named_parameters(), strict=True
            ):
                if a.grad is not None:
                    assert b.grad is not None
                    torch.testing.assert_close(a.grad, b.grad.cpu(), atol=3e-5, rtol=3e-4)
                    if not torch.isfinite(b.grad).all():
                        raise ValueError(f"Non-finite {arm['name']} gradient")
        results.append(
            dict(arm=arm["name"], max_error=float((left - right.cpu()).detach().abs().max()))
        )
    print(json.dumps(dict(device=args.device, checked=len(results), results=results), indent=2))


if __name__ == "__main__":
    main()
