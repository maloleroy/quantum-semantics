# Quantum text prediction: what we have learned

As of 15 November 2026. Detailed inventories: [successful steps](experiments/successful.md) ·
[negative and inconclusive experiments](experiments/unsuccessful.md).

## Data, prediction and optimization

We began with Tatoeba sentences and **word tokens**: one ID per vocabulary word, with unfamiliar
words mapped to UNK. Today's task uses **WikiText-2 Raw** Wikipedia articles and training-only
byte-pair encoding (BPE). A word may span several tokens; case, whitespace, punctuation and numbers
are preserved. The vocabulary has $V=512$ IDs: 256 bytes, 255 learned byte combinations and UNK.
Every source word is representable without UNK; word-level and byte-token scores are different units.

Write an article as token IDs $x_1,\ldots,x_T$. For $t\ge2$, predict $y_t=x_t$ from
$C_t=(x_{\max(1,t-W)},\ldots,x_{t-1})$, with **$W=128$ preceding tokens**. Context stops at article
boundaries. Training, validation and test articles are separate; tokenizer and count statistics use
training articles only. Each of 30 logical epochs draws 524,288 positions from a 2,097,152-position
pool: **30,720 Adam updates**, batch 512, constant learning rate .003.

Classical parameters $\theta$ are optimized through the simulated quantum evolution to minimize
**negative log-likelihood**, using natural logarithms:

$$
\mathcal L_{\mathrm{train}}(\theta)
=-\frac{1}{N_{\mathrm{train}}}\sum_t\log p_\theta(y_t\mid C_t).
$$

We select the lowest-NLL checkpoint on 4,096 fixed validation targets. Separately,
$S=N^{-1}\sum_t p_\theta(y_t\mid C_t)$ is expected **one-draw success**, whereas argmax accuracy
counts $\arg\max_y p_\theta(y\mid C_t)=y_t$. We also inspect non-space targets and unfiltered
autoregressive generation, drawing $x_t\sim p_\theta(\cdot\mid C_t)$ and feeding it back as context.
The 8,192-target WikiText test remains **unscored**. Repeated training seeds check robustness on
reused validation, not independent generalization.

## From the basic Hamiltonian to the current model

The basic Tatoeba model used **eight qubits, $V=256$, $W=4$**, fixed context amplitudes
$|\psi_C\rangle=\sum_j a_j(C)|j\rangle$, and one learned Hermitian matrix $H=H^\dagger$ with
$256^2=65,536$ real coordinates. Evolution $U=e^{-iH}$ mixes amplitudes; the Born rule gives
$p(y\mid C)=|\langle y|U|\psi_C\rangle|^2$. Training helped, and preserving interference beat a
separately trained dephased control.

Today's replicated **prior_content** model uses **nine token qubits plus one auxiliary qubit**:
state dimension $D=2^{10}=1024$, **three Hamiltonian stages**, rank $r=128$, and **1,073,900
trainable real coordinates**. Its input is a superposition of a smoothed training-bigram prior:

$$
\begin{aligned}
|\psi_0(C)\rangle&=\sum_{y=0}^{511}\sqrt{q(y\mid x_{t-1})}\,|y\rangle|0\rangle,\\
q(y\mid x)&=\frac{n_{xy}+10\hat p(y)}{\sum_z n_{xz}+10}.
\end{aligned}
$$

where $n_{xy}$ counts training transitions and $\hat p$ is the training unigram distribution.
Learned position weights and a classical 16-dimensional query/key selector pool the last token,
the preceding 15 tokens and older context. They produce spectral shifts $s_\theta(C)$ and local
phase angles $\alpha_{\ell k}(C)$. Each stage has

$$
H_\ell(C)=Q_\ell\operatorname{diag}\!\bigl(e_\ell+s_\theta(C)\bigr)Q_\ell^\dagger,
\qquad Q_\ell\in\mathbb C^{1024\times128},\quad Q_\ell^\dagger Q_\ell=I_{128}.
$$

The learned columns of $Q_\ell$ define 128 active directions; $e_\ell,s_\theta(C)\in\mathbb R^{128}$
set their energies, with evolution time absorbed into them.
Between evolutions, local rotations $R_\ell(C)=\bigotimes_{k=1}^{10}R_Z(\alpha_{\ell k}(C))$
modify relative phases, where $R_Z(\alpha)=e^{-i\alpha Z/2}$ and $Z$ is the Pauli-Z operator.
The complete prediction is

$$
\begin{aligned}
U_\theta(C)&=e^{-iH_3(C)}R_2(C)e^{-iH_2(C)}R_1(C)e^{-iH_1(C)},\\
p_\theta(y\mid C)&=\sum_{a=0}^{1}
\left|\langle y,a|U_\theta(C)|\psi_0(C)\rangle\right|^2.
\end{aligned}
$$

Measuring the token register ignores the auxiliary outcome; no classical vocabulary decoder or
output softmax is used. Count preparation and context pooling are nevertheless classical inputs.
Count amplitudes replaced the earlier learned product-state preparation. **“Depth three” means
three evolution blocks and two local-rotation banks; native hardware gate depth has not been
established.** General state preparation and learned Hamiltonian compilation remain unresolved costs.

| Current evidence | Hamiltonian stages / rank | NLL ↓ | One-draw success ↑ | Argmax ↑ |
| --- | --- | ---: | ---: | ---: |
| Replicated quantum, three-seed mean | 3 / 128 | 2.4655 | 28.52% | 44.10% |
| Equally exposed strong classical MLP, three-seed mean | — | 2.2228 | 39.45% | 50.35% |
| Lowest observed quantum NLL, one seed | 4 / 128 | 2.4091 | 30.02% | 46.09% |
| Its matched classical MLP, one seed | — | 2.1662 | 42.40% | 51.81% |

The replicated model gains a small NLL benefit from 128 versus 16 tokens, although sampling and
argmax slightly worsen. Longer training and more capacity help at one seed; four stages do not
clearly beat comparable extra width on NLL. **Generation remains incoherent.** No advantage over
classical controls, entanglement-specific benefit or practical quantum speedup has been demonstrated.
[Replicated results](experiments/coverage-seeds.md) · [later capacity candidates](experiments/depth-four.md).

## What the model actually generates

Below are all three saved continuations from the replicated three-stage/rank-128 model's
**training-seed-22 checkpoint**. Each contains 64 generated BPE tokens, sampled directly from the
Born distribution. The prompts are supplied text; only the text inside the blocks is generated.
These uncorrected samples contain recognizable fragments, but do not form coherent sentences.
Source: [archived samples](experiments/coverage-seeds-samples.csv), rows labelled `quantum_s22`.

**Prompt:** `The history of ` · sampling seed 20261030

```text
its first game ream , butvarBan7 was an untrHMNmanedthe

 Pecr8 Germania , New Ingaglas an delfirst there is
```

**Prompt:** `In 2026, the researchers ` · sampling seed 20261031

```text
modere aally Redyogical States suballyivult memorth with Fueld Comheore cr

 un wechuere was a feat from 190
```

**Prompt:** `Quantum mechanics describes ` · sampling seed 20261032

```text
on her Chan 1921 the starod eleages and sawulture of the Suned , fricon , but theatch Withach a funion ,
```
