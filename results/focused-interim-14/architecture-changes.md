# Architecture changes: encoding and attention

These figures describe the two ablations without discussing their results.

## Figure 1 — Common model path

```mermaid
flowchart LR
    C[Context words] --> E[Encoding]
    E --> S[State for each position]
    S --> A[Causal attention]
    A --> D[Decoder]
    D --> O[Next-word scores]
```

Encoding decides how words become states. Attention decides how the states at different positions interact.

## Figure 2 — Exact encoding configuration

```mermaid
flowchart LR
    I[Token IDs] --> E[Learned embedding lookup]
    E --> F[Flatten ordered context window]
    F --> L[Linear encoder<br/>to 2m angles]
    L --> P[Quantum state preparation<br/>RY + diagonal phase]
    P --> A[Ansatz<br/>zero/frozen or trainable]
    A --> X[Measure X/Y/Z features]
    X --> D[Linear decoder<br/>to embedding dimension]
    D --> O[Similarity/logits<br/>against output embedding table]
```

The encoding comparison changes the representation pathway:

- `classical + trainable ansatz`: all stages above are active and trainable;
- `classical + zero ansatz`: the ansatz is the identity and is frozen;
- `no encoding`: embeddings are averaged and the quantum pathway is bypassed.

There is no explicit one-hot tensor and no context matrix in this encoding model. An embedding lookup is mathematically equivalent to multiplying a one-hot vector by an embedding matrix, but the implementation uses token IDs directly.

## Figure 3 — Where the context matrix appears

```mermaid
flowchart LR
    I[Token IDs] --> T[Token angles]
    T --> C[Exponential QCSE context matrix]
    C --> Q[Quantum state per position]
    Q --> U[Optional trainable Q/K/V circuits]
    U --> A[Causal overlap attention]
    A --> O[Decoder and vocabulary logits]
```

This context-matrix path belongs to the QCSE attention encoder and the separate causal QCSE model, not to the classical-encoding ablation above.

## Figure 4 — Attention ablation

```mermaid
flowchart LR
    subgraph INPUT[States at context positions]
        X[Encoded states]
    end

    X --> QKV
    subgraph QKV[Optional quantum Q/K/V transformation]
        Q[Query circuit]
        K[Key circuit]
        V[Value circuit]
    end

    Q --> SCORE[Real query-key overlaps]
    K --> SCORE
    SCORE --> MASK[Causal lower-triangular mask]
    MASK --> MIX[Weighted value mixing]
    V --> MIX
    MIX --> LAST[Final-position state]
    LAST --> OUT[Prediction]
```

The attention comparison is a 2×2 design:

```mermaid
flowchart TB
    C[Classical encoding] --> C0[Raw encoded states]
    C --> C1[Trainable quantum Q/K/V]
    Q[QCSE encoding] --> Q0[Raw QCSE states]
    Q --> Q1[Trainable quantum Q/K/V]

    C0 --> CA[Causal overlap attention]
    C1 --> CA
    Q0 --> QA[Causal overlap attention]
    Q1 --> QA
```

“No quantum attention” still performs causal attention. It only removes the trainable Q/K/V transformations, so query, key, and value are the encoded states themselves.

## Figure 5 — What changes and what stays fixed

```mermaid
flowchart LR
    P[Same causal next-word task] --> S[Same sentence-level split]
    S --> M[One architectural choice changed]
    M --> R[Validation-selected epoch]
    R --> T[One held-out test evaluation]

    M --> E[Encoding: classical vs no encoding]
    M --> A[Attention: Q/K/V circuits off vs on]
```

The encoding and attention arms are separate experiments. The attention arm uses a small repetitive sanity corpus, while the encoding arm uses the full corpus; their percentages should therefore not be compared directly.
