# Interim focused sweep — short report

The 14 completed non-causal fits show two clear conclusions:

1. Classical encoding is essential in the encoding arm: about 19–20% test top-1 versus 2.8% without encoding. A trainable ansatz adds only a small, uncertain improvement over a zero ansatz.
2. Quantum attention is highly dependent on the encoder. It improves QCSE by about 26 top-1 points, but classical encoding by only about 3 points. Classical encoding remains best overall, while QCSE + quantum attention recovers much of the gap.

These are interim directional results from two repetitions. Encoding used 10,000 test examples; attention used a 133-example synthetic test set. The causal arm and the new 1,024-sample protocol are still running and are not represented here.
