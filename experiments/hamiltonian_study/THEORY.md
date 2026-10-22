# What the Hamiltonian encoder does, and what it cannot do

## 1. The symmetry goal is correct

Let vocabulary labels be renamed by a permutation matrix P. Let input states,
corpus Hamiltonian and context potential transform together:

$$\psi'_C=P\psi_C,\quad H'_G=PH_GP^T,\quad H'_C=PH_CP^T.$$

Every matrix exponential then conjugates in the same way, so for any depth L,

$$U'_\theta(C)\psi'_C=P U_\theta(C)\psi_C.$$

Born probabilities are permuted, leaving predictions for named words unchanged.
This property holds for shared scalar beta/gamma parameters. A generic local-qubit
ansatz appended afterward does **not** automatically preserve it; its generators
must also transform. The original bit-marginal output is likewise not invariant
under arbitrary vocabulary permutations. This study therefore tests a replacement
encoder/readout system as well as two original-encoding controls.

This symmetry does not mean that exchanging words inside a sentence should leave
the result unchanged. "Dog bites man" and "man bites dog" should differ. Vocabulary
renaming and sentence reordering are separate operations.

Graph construction must itself respect symmetry. Keeping exactly k neighbors by
an arbitrary index-based tie break can violate the property. We keep all ties at
the kth-weight threshold and symmetrize by the union of retained edges. The maximum
degree can therefore exceed k, especially for frequent-word hubs.

## 2. The draft's normalization needs repeated-token care

The draft adds sqrt(alpha_r) for every occurrence of word i. Define

$$a_i=\sum_{r:w_r=i}\sqrt{\alpha_r},\qquad
 Z=\sum_i |a_i|^2.$$

Z is **not generally** sum_r alpha_r: repeated occurrences interfere coherently.
For [a,a,b] with equal weights, measurement probabilities are (4/5,1/5).
An alternative that preserves weighted empirical frequencies is

$$c_i=\frac{\sum_{r:w_r=i}\alpha_r}{\sum_r\alpha_r},\qquad
 |\psi_C\rangle=\sum_i\sqrt{c_i}|i\rangle.$$

The literal `bag` control uses coherent sums; `walk` uses the latter rule with
alpha_r = 0.7^(age from the newest word). Padding has no weight. Both are normalized.
These are different inductive biases, not equivalent notations.

## 3. Shared unitary evolution cannot learn a fidelity geometry

For any input-independent unitary U,

$$|\langle U\psi_C,U\psi_D\rangle|^2=
 |\langle\psi_C,\psi_D\rangle|^2.$$

In particular, basis words remain mutually orthogonal after a fixed graph walk.
No amount of depth changes this fact if all layers use only H_G:

$$\prod_\ell e^{-i\beta_\ell H_G}=e^{-i(\sum_\ell\beta_\ell)H_G}.$$

A walk can change the distribution visible in a chosen measurement basis, but
cannot by itself make distinct orthogonal word states closer in fidelity. Learning
semantic geometry requires context-dependent operations, nonunitary operations,
a restricted measurement representation, or a different comparison construction.
Alternating H_C and H_G can change cross-context fidelity because the unitary now
depends on the input. Measuring a different distribution is still not evidence of
semantic similarity: that needs an external similarity/disambiguation evaluation.

## 4. Order is not recovered merely by allocating a position register

With a joint state sum_r sqrt(alpha_r)|r>|w_r> and evolution I_pos tensor U_G,
ignoring position at readout gives

$$p(j|C)=\frac{1}{\sum_r\alpha_r}
 \sum_r\alpha_r |\langle j|U_G|w_r\rangle|^2.$$

The orthogonal position tags remove interference between occurrences. Uniform
weights yield a bag of words again; a reversal is invisible. To use genuine order,
use position-conditioned potentials/mixing, an order-sensitive measurement, or
explicit recency weights. The `ordered` experiment cycles chronological token
potentials through the layers. Its state register still uses V amplitudes, avoiding
the cost of an unused position register.

## 5. Undirected association differs from next-word prediction

For a real symmetric fixed H, U(t)^T=U(t), hence transition probabilities obey
p(j|i)=p(i|j). The matrix |U_ji|^2 is doubly stochastic for any fixed unitary.
Language transitions generally have neither property: p("the"|"of") need not
equal p("of"|"the"). A symmetric co-occurrence graph captures association, whereas
a causal language model needs directional and frequency-sensitive conditionals.
Context-dependent potentials and nonuniform initialization can escape some of
these restrictions; the fixed-walk impossibility does not apply to every variant.

To test direction while keeping a Hermitian generator, take a row-stochastic
training bigram table T and form B_ij=sqrt(T_ij), then

$$H_{\rm dir}=\frac{1}{\|B\|_2}
 \begin{pmatrix}0&B\\B^T&0\end{pmatrix}.$$

The two blocks denote source and target roles. Starting at source i, a short walk
puts target amplitude -it B_ij/||B||_2 + O(t^3) on j. Consequently,

$$p(j\mid\text{target role},i)=T_{ij}+O(t^2),\qquad
 p(\text{target role}\mid i)=t^2/\|B\|_2^2+O(t^4).$$

This is a coherent implementation of a supplied classical prior in the short-time
limit. It is not evidence that quantum evolution discovered the prior. Conditioning
on the target role must be disclosed: repeated state preparation costs on average
1/p_success attempts. We report the mean of the per-example inverse success, which
need not equal the inverse mean success.

## 6. Adjacency versus normalized Laplacian is often redundant

For the precise choice L=I-A,

$$e^{-itL}=e^{-it}e^{+itA}.$$

The first factor is a global phase; the other reverses evolution time. With real
input amplitudes and a single real-symmetric walk, both choices give the same Born
probabilities. With arbitrary signed beta parameters, replacing A with I-A can
also be absorbed by beta sign changes layer by layer. It is not an independent
architecture improvement. Conventions for isolated-vertex Laplacians may differ;
this identity refers specifically to I-A as written in the proposal.

## 7. Phases must be converted into measurable interference

A final diagonal gate has no effect on computational-basis probabilities. Each
implemented layer first applies exp(-i gamma_l H_C), then exp(-i beta_l H_G), so
mixing follows the phase. For a singleton initial context, the first context phase
is just a global phase. Without graph mixing, **all** context phases remain
invisible to Born measurement. The graph-removal control is therefore an important
mechanism check, but outperforming it does not establish superiority over classical
graph propagation.

Unsmoothed Born outputs can assign zero probability to a target outside the reachable
support. All main graph systems use a learned mixture with a strictly positive
training unigram prior. This is a declared departure from the literal objective:

$$p_\theta=(1-\eta)|\phi_C|^2+\eta u,\quad\eta=\sigma(a).$$

The mixture is classical postprocessing, or operationally a randomized choice of
which distribution to sample. It is not the same as mixing amplitudes coherently.
Raw Born likelihood and graph-removal effects are diagnosed separately.

## 8. Unary implementation and the factor of two

For an undirected matrix with off-diagonal entries H_ij, the correct unary hopping
operator is

$$\widehat H_G=\frac12\sum_{i<j}H_{ij}(X_iX_j+Y_iY_j).$$

On the single-excitation subspace its hopping matrix is exactly H_G. Using the
same one-half coefficient while summing both (i,j) and (j,i) doubles the hopping
strength. An unrestricted double sum would need coefficient one-quarter instead.
A diagonal context potential is sum_i c_i (I-Z_i)/2 in the same subspace. Independent
Qiskit Pauli-matrix checks verify the restriction and its time evolution.

V unary qubits and ceil(log2 V) binary qubits encode the same V-dimensional
one-particle dynamics. Binary state-space compactness does not imply cheap circuits:
a generic Hermitian matrix can need O(V^2) Pauli coefficients. A generic dense input
state also has an O(V) specification/preparation burden. Sparse Hamiltonian simulation
requires suitable access to entries/neighbors; classical corpus storage is not free.
Our sparse exponential-multiply reference and dense eigensolver agree numerically.
For the bounded vocabulary, dense spectral evolution is faster for repeated gradient
training and introduces no Trotter approximation. Neither simulator is quantum hardware.
