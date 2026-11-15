# Successful steps toward the current model

Companion to the [one-page explainer](../research-summary.md), covering evidence available on 15
November 2026. “Successful” means a useful improvement within our research path, not a quantum
advantage. Pilots, expanded studies and seed replications of the same idea are grouped together. NLL
is lower-is-better; S is ideal expected one-draw success. Scores refer to validation unless
explicitly marked test. Changing tokenization or datasets breaks direct numerical comparability.

The established reference is the three-stage `prior_content` recipe with expanded training coverage.
Some earlier successful components were subsequently replaced. Later positive candidates are
separated below because they have not earned the same replication status.

## 1. Train a Hamiltonian directly, without a learned classical encoder/head

- **Change:** fixed recency/position amplitudes encode four preceding words; a fully trainable
  Hamiltonian and direct Born measurement predict the next word.
- **Result:** on Tatoeba, test NLL improves **6.5861 → 3.5974**, S **1.91% → 6.76%**, argmax **2.57%
  → 24.21%**, averaged over three confirmation seeds.
- **Control:** separately trained dephasing gives NLL **3.6683**, S **5.71%**. Retaining
  interference helps within this architecture; the matched classical MLP is much better (**2.6205 /
  24.77% S**).
- **Retained:** trainable quantum dynamics and direct token measurement. This establishes learning
  and a restricted coherence contribution, not a general classical or entanglement advantage.
- **Evidence:** [findings §§1–2](findings.md#1-hamiltonian-only-next-word-prediction).

## 2. Controlled checks that Hamiltonian learning can use coherence

- **Test:** learn 16-symbol modular composition and classify handwritten digits using fixed
  amplitudes, without learned classical input/output networks.
- **Result:** symbol argmax reaches **100%**, versus **8.33%** after retrained dephasing; digits
  reach **91.96%**, versus **61.88%** dephased.
- **Interpretation:** these validate the learning mechanism underlying the language work. The digit
  likelihood also beats a derived bound for the specified completely dephased readout family.
- **Boundary:** classical addition solves the symbol task, and a similar-size digit MLP reaches
  **95.43%**. These are supporting mechanism experiments, not language-model additions or evidence
  of overall advantage.
- **Evidence:** [controlled experiments](findings.md#2-controlled-hamiltonian-learning).

## 3. Low-rank Hamiltonian evolution

- **Change:** learn an active orthonormal subspace and its energies, instead of exponentiating a
  fully dense Hamiltonian. Evolution remains exact for that restricted family.
- **Result:** two rank-128 blocks on the 512-word task reach **3.8744 NLL / 11.58% S** in **213
  seconds** with 32,768 training targets.
- **Boundary:** the earlier dense recipe scored 4.0110 with fewer examples and epochs; that
  difference does not isolate a rank effect. This was chiefly a useful simulation/recipe change.
- **Retained:** low-rank spectral evolution is the computational basis of the later models. It does
  not demonstrate cheap compilation on quantum hardware.
- **Evidence:** [low-rank study](lowrank.md).

## 4. Add an unmeasured auxiliary qubit

- **Change:** evolve the token register jointly with one auxiliary qubit; measure the token register
  while keeping all auxiliary outcomes.
- **Result:** word-level NLL improves **3.8744 → 3.6379**, S **11.58% → 14.01%**, versus the earlier
  nearly parameter-matched word-only recipe.
- **Boundary:** dephasing is better on NLL (**3.5954**) and known-word S; the coherent model's
  higher aggregate S is driven by unknown-word targets. Entanglement is present, but its causal
  benefit was not isolated.
- **Retained:** the extra qubit and direct marginal token measurement. This is a within-quantum
  recipe improvement, not an established entanglement advantage.
- **Evidence:** [auxiliary-register study](ancilla.md).

## 5. Learn context-dependent phase gates

- **Change:** small token tables turn recent/older context into local phase rotations between
  Hamiltonian blocks, rather than using static evolution alone.
- **Result:** after the pilot, equal-learning-rate word-level controls give **3.6360 → 3.5518 NLL**,
  **14.09% → 14.36% S**; fixed random gates give 3.6230 NLL.
- **Control:** frozen quantum dynamics regress to **4.7747 NLL**. The angle tables alone do not
  reproduce the jointly trained result at this budget.
- **Retained:** learned local phase interleaving. The short-context refit and classical control
  still do better; useful older context was not established here.
- **Evidence:** [pilot and expanded controls](conditional.md).

## 6. Replace the restricted word vocabulary with lossless byte BPE

- **Change:** learn byte-pair merges on training articles, keeping every UTF-8 byte, case,
  whitespace, numbers and punctuation in a 512-outcome register.
- **Result:** **zero unknown targets** and exact source-byte reconstruction, versus **41.80% unknown
  targets** in the earlier word-level WikiText validation sample.
- **Retained:** this tokenizer, raw-text task and autoregressive direct-draw generation. It removes
  a coverage barrier to generating arbitrary words.
- **Boundary:** this is a representation success, not a measured NLL improvement across
  tokenizations. The first quantum model on it predicted poorly; that negative result is listed
  separately.
- **Evidence:** [byte-BPE preparation and first screen](byte-bpe.md).

## 7. Learn local product-state preparation

- **Change:** replace fixed context amplitudes with learned local rotations on all ten qubits before
  the Hamiltonians; keep context gates and direct measurement.
- **Result:** on the same byte-token task, NLL improves **4.2773 → 3.1975**, S **5.68% → 21.79%**.
- **Control:** frozen Hamiltonians give **4.3635 NLL**. The small-data matched MLP gives **3.2306
  NLL**, but higher S (**26.48%**); its NLL deficit disappears with more training coverage.
- **Disposition:** this was a major intermediate reference and remains the preparation used by
  `adaptive`. The current `prior_content` quantum model later replaces it with count-amplitude
  preparation.
- **Evidence:** [product-state study](product.md).

## 8. Increase unique training targets: 32,768 → 131,072

- **Change:** expand the nested gradient-training subset; compare against repeating the old subset
  for the same **7,680 optimizer updates**.
- **Result:** product NLL is **2.9673**, versus **3.1793** on the repeated small set; S **23.17%
  versus 21.91%**.
- **Retained:** broader training exposure rather than extra repetitions of a tiny subset. Frozen
  dynamics remain much worse (**4.2567 NLL**).
- **Boundary:** validation-checkpoint opportunities differ between the 30- and 120-epoch arms. The
  equally exposed MLP improves more, to **2.7966 / 30.84% S**; longer context still gives no NLL
  benefit.
- **Evidence:** [nested exposure comparison](product-exposure.md).

## 9. Increase Hamiltonian rank to 128

- **Change:** test product ranks 32/64/128, with corresponding classical parameter controls.
- **Result:** NLL improves **3.0327 / 2.9673 / 2.9281** as rank increases.
- **Retained:** rank 128 as the subsequent working choice. This is a likelihood improvement rather
  than a sampling breakthrough: rank-128 S is **23.20%**, versus rank-64 **23.17%**.
- **Boundary:** the matched rank-128 MLP reaches **2.8303 NLL / 31.04% S**; text remains fragmented.
  Increasing rank alone does not close the gap.
- **Evidence:** [product capacity screen](product-capacity.md).

## 10. Increase unique training targets: 131,072 → 524,288

- **Change:** compare 524,288 unique positions with four copies of the old set, matching **30,720
  updates** and validation-checkpoint opportunities.
- **Result:** product NLL improves **2.9077 → 2.7870**, S **23.79% → 24.42%**.
- **Retained:** the larger training subset for subsequent architecture experiments, with tokenizer,
  counts and held-out targets unchanged.
- **Boundary:** the equally exposed MLP reaches **2.4941 / 36.66% S**, improving more. More data
  helps the quantum recipe without resolving its predictive deficit or fragmented generation.
- **Evidence:** [coverage comparison](product-coverage.md).

## 11. Let context change Hamiltonian energies

- **Change:** pooled token features shift each Hamiltonian's energies within its learned basis,
  rather than using only context-independent energies.
- **Result:** adaptive quantum reaches **2.7007 NLL / 25.77% S**, versus nearly parameter-matched
  static capacity **2.7558 / 24.62%**.
- **Retained:** context-dependent spectral shifts, shared across evolution stages, alongside local
  phase gates. There is no classical vocabulary head.
- **Boundary:** fixed bases still allow spectral shifts to train, so that control is not a frozen
  Hamiltonian. The matched MLP remains better (**2.4393 / 38.44% S**).
- **Evidence:** [adaptive-spectrum study](adaptive-spectrum.md).

## 12. Start from a training-count next-token distribution

- **Change:** prepare square-root amplitudes of a smoothed bigram distribution; learned quantum
  evolution refines it using context. Initialization exactly reproduces the count baseline.
- **Result:** three-seed mean NLL improves **2.70146 adaptive → 2.66776 prior**, passing the
  new-seed replication rule.
- **Tradeoff:** mean S falls **25.74% → 25.51%**, while argmax improves **40.29% → 40.76%**. At seed
  11, unigram-only input and fixed bases are worse (**2.7178 / 3.0198 NLL**).
- **Retained:** count-amplitude preparation, replacing product-state preparation in the quantum
  reference. The count tables and preparation cost are real classical inputs, not free quantum
  learning.
- **Evidence:** [initial screen](prior-screen.md) · [seed replication](prior-seeds.md).

## 13. Learn a different position preference for each control feature

- **Change:** replace fixed recency decay with learned normalized lag weights for spectral shifts
  and phase gates. Different features can emphasize different preceding positions.
- **Result:** three-seed mean NLL improves **2.66776 → 2.60325**, beating the matched-capacity
  fixed-pooling control (**2.66729**) at each new seed and on average.
- **Retained:** feature-specific position weights. S rises **25.51% → 26.57%**; the mean NLL
  difference from prior is **−.06451**, conditional article interval **[−.07600, −.05207]**.
- **Boundary:** the 16-token refit is better (**2.59486 NLL**). This improves local prediction
  without establishing a longer-context benefit.
- **Evidence:** [position screen](prior-position.md) · [replication](position-seeds.md).

## 14. Make context selection depend on token content

- **Change:** a small classical query/key selector uses the ordered last four tokens to reweight
  preceding tokens, alongside the learned position preferences.
- **Result:** three-seed mean NLL improves **2.60325 → 2.57588**, beating matched-capacity
  position-only control (**2.60124**) at each new seed and on average.
- **Retained:** the selector in `prior_content`; S rises **26.57% → 27.29%**. Mean NLL improvement
  is **−.02737 [−.03741, −.01762]**.
- **Boundary:** this is classical preprocessing, not quantum attention. The independently refitted
  short model still wins (**2.56249 NLL**), and the strong MLP remains ahead.
- **Evidence:** [content screen](content-screen.md) · [replication](content-seeds.md).

## 15. Add a third context-conditioned Hamiltonian stage

- **Change:** compare three rank-128 stages with two stages and a nearly parameter-matched
  two-stage/rank-176 width control; count the extra phase-controller parameters too.
- **Result:** three-seed mean NLL is **2.53554**, versus **2.57588** original content and
  **2.56465** matched width. The replication rule passes.
- **Retained:** three stages. S reaches **28.02%**; depth-minus-width NLL is **−.02910 [−.03828,
  −.01968]**.
- **Boundary:** this supports the complete depth/interleaving recipe, not an isolated quantum-depth
  effect. Short context still wins at this training coverage (**2.51807 NLL**).
- **Evidence:** [depth screen](depth-screen.md) · [replication](depth-seeds.md).

## 16. Broaden gradient coverage to a 2,097,152-position pool

- **Change:** draw 524,288 examples per epoch from a larger nested pool, preserving 30 epochs,
  **30,720 updates**, and the validation-selection budget.
- **Result:** three-seed mean full-model NLL improves **2.53554 → 2.46546**; difference **−.07009
  [−.08197, −.05750]**. Both new seeds improve overall and non-space NLL.
- **Retained:** this is the **current established quantum recipe**. S/argmax reach **28.52% /
  44.10%**.
- **Context result:** full 128-token context now beats a retrained 16-token model (**2.47748**) on
  NLL: **−.01202 [−.02025, −.00475]**. Sampling/argmax remain slightly worse for full context.
- **Boundary:** validation is reused; the equally exposed strong MLP reaches **2.22277 / 39.45% S**,
  and quantum text remains incoherent.
- **Evidence:** [coverage screen](depth-coverage.md) · [replication and complete
  metrics](coverage-seeds.md).

## Later positive candidates: not yet part of the replicated reference

### 17. Train for up to 90 logical epochs

- **Change:** fresh three-stage fits with **92,160 updates**, compared with the expanded-pool
  30-epoch references.
- **Result:** seed-11 full quantum NLL improves **2.46253 → 2.43218**, selecting epoch 68; the
  paired improvement interval excludes zero. S/argmax improve to **29.20% / 45.14%**.
- **Status:** useful optimization-budget gain, **single seed only**. The 30-epoch three-seed recipe
  remains the established reference.
- **Boundary:** full versus short NLL is now inconclusive, and the MLP reaches **2.17231 NLL**.
  Extra optimization does not solve the classical gap or generation quality.
- **Evidence:** [longer-budget screen](budget-screen.md).

### 18. Increase overall quantum capacity

- **Change:** at the 90-epoch budget, test four stages/rank 128 and three stages/rank 164 against
  the smaller three-stage/rank-128 reference.
- **Result:** NLL improves **2.43218 → 2.40905 / 2.41051**, respectively; both gains against the
  smaller model have article intervals below zero and improve non-space NLL.
- **Status:** positive **single-seed capacity candidates**, not replicated replacements. Four-stage
  S/argmax are **30.02% / 46.09%**.
- **Boundary:** four stages do not clearly beat matched width on NLL; that hypothesis fails in the
  [negative inventory](unsuccessful.md#26-a-fourth-stage-specifically-rather-than-more-capacity).
  Matched MLP NLL is **2.16622**; there is no four-stage short-context refit.
- **Evidence:** [fourth-stage/width screen](depth-four.md).

## Supporting engineering results

CPU/CUDA probabilities, gradients and checkpoint reloads were checked. A full-A100 benchmark gave
**4.14× quantum / 3.30× MLP simulator fit speedups** over the faster measured CPU configurations;
later resource probes showed the default 10 GB MIG slice was sufficient. These made the research
practical, but are neither predictive gains nor quantum hardware speedups. See [DGX
measurements](../dgx.md) and [content resource probe](content-controller.md).
