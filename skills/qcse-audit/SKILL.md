---
name: qcse-audit
description: Audit this QCSE implementation against its local paper and distinguish circuit fidelity from predictive performance.
---
Read ../../reports/01-paper-audit.md and ../../PROGRESS.md before extending the audit.
Use the supplied paper4.pdf for equations, not the ideas document as a specification.
Check little-endian marginals with independent small dense matrices. In RX-then-RZ
encoding on |+>, the first RX values are global phase; final diagonal gates cannot
change Z marginals. Check these identities before attributing gate sensitivity to bugs.
The half-bits-match accuracy is not word prediction. Always include BCE, exact word
accuracy, train-only bit priors and the fair-bit null. Distinguish input ID remapping
from relabeling target bits. Record corpus, vocabulary, split and optimizer differences.
