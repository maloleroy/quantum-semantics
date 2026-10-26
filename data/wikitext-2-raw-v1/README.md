# WikiText-2 Raw, pinned source snapshot

Downloaded 26 October 2026 from the publisher's
[Salesforce/wikitext dataset](https://huggingface.co/datasets/Salesforce/wikitext),
configuration `wikitext-2-raw-v1`, revision
`6231e49f19a707241d6b84d9cff60a3a86b85a85`.

The three Parquet files are unchanged source downloads, verified against the
SHA-256 values in the pinned Git LFS pointers. `wiki.train.raw`, `wiki.valid.raw`
and `wiki.test.raw` export their `text` rows in source order, adding a terminating
newline only when absent. Blank rows, article titles, section titles, punctuation,
case and numbers are preserved. These are text exports, not byte-identical copies
of the original historical ZIP archive.

The local download occupies approximately 21 MiB, including both Parquet and text.
The bulk files are ignored by Git; this README and [manifest.csv](manifest.csv)
are tracked. Every downloaded/exported file has a checksum and source URL in the
manifest. No Git LFS installation or extra project runtime dependency is required.
The unmodified source card is saved locally as `upstream-README.md`.

Attribution: Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher,
*Pointer Sentinel Mixture Models* (2016), and the original Wikipedia contributors.
See the [paper](https://arxiv.org/abs/1609.07843) and the
[pinned source card](https://huggingface.co/datasets/Salesforce/wikitext/blob/6231e49f19a707241d6b84d9cff60a3a86b85a85/README.md)
for source and license notices. The card's metadata lists CC BY-SA 3.0 / GFDL;
its licensing prose links CC BY-SA 4.0; both are retained verbatim upstream.

To download/export the exact same snapshot again, run from the repository root:

```bash
uv run --with pyarrow==26.0.0 python - <<'PY'
import csv
import hashlib
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq

root = Path("data/wikitext-2-raw-v1")
with (root / "manifest.csv").open(encoding="utf-8", newline="") as source:
    manifest = list(csv.DictReader(source))
for row in manifest:
    path = root / row["file"]
    if path.suffix == ".raw":
        split = {"train": "train", "valid": "validation", "test": "test"}[path.stem.split(".")[1]]
        texts = pq.read_table(root / f"{split}.parquet", columns=["text"])["text"].to_pylist()
        content = "".join(text if text.endswith("\n") else text + "\n" for text in texts).encode("utf-8")
    else:
        with urlopen(row["source_url"], timeout=60) as response:
            content = response.read()
    if len(content) != int(row["bytes"]) or hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}")
    path.write_bytes(content)
    print(path)
PY
```

Preparation needs only the three `.raw` files. Pass this **directory**, rather
than one file, to `qcse prepare`; this preserves the official article partitions.
See [the dataset recommendation and budget](../../docs/datasets.md).
