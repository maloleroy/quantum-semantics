# Scientific poster: revision 3

Revised 17 November 2026. Title restored to “Can ten qubits finish your sentence?”
The two equal-width columns and Hamiltonian focus remain.

## Author's latest direction

- Present the move from Quantum Context-Sensitive Embedding (QCSE) to a learned
  Hamiltonian as the scientific narrative.
- Use the existing results and explicitly state the unmatched-task limitation.
  The author confirmed this after we checked whether a controlled comparison existed.
- Enlarge text and headings, spending less area on blank space.
- Explain every symbol and operator immediately below the Hamiltonian equation.
- Use ordinary LaTeX mathematical lettering. Prose remains sans serif.
- Define acronyms at first use or write the term in full.

The poster does not assert that the current experiments establish a controlled
Hamiltonian-over-embedding win. Its QCSE and Hamiltonian results differ in task,
vocabulary, context, training recipe and evaluation split. The limitation is
visible above and below the two separate case studies. No common-axis comparison,
fold improvement or significance claim is used.

## Deliverables

- poster.pdf in this directory is the submission A0 portrait PDF.
- It matches output/poster/quantum-semantics-poster-a0.pdf at the repository root.
- A4 proof, PNG preview and editable source ZIP are also in output/poster/.
- Older PDFs: Git commits d8cb5b4 and 2857d51, and output/poster/revisions/.
- presentation-notes.md contains the revised three-minute script and questions.

## Layout and typography

Body and footer columns are each 375.5 mm wide, at x = 30 and x = 435.5 mm,
with a 30 mm gutter. Only the required institutional/logo/title strip spans the
whole A0 portrait sheet (841 x 1189 mm).

IBM Plex Sans supplies all prose. Latin Modern Math supplies mathematical
lettering. Nominal sizes:

- Title 96 pt; semibold subtitle 56 pt; author names 36 pt.
- Section headings 44 pt, increased from 36 pt.
- Main body 34 pt, increased from 28 pt.
- Captions, chart labels and equation definitions 29 pt, increased from 24 pt.
- Displayed Hamiltonian 54 pt; circuit evolution labels 36 pt.
- Result percentages 66 pt; final takeaway 46 pt; references and credits 22 pt.

The equation has a complete key for C, stage index l, H, Q, e, s, trainable
parameters theta, diag, and the conjugate-transpose dagger. Circuit notation is
explained separately in its caption. There are no undefined prose acronyms:
QCSE is expanded on introduction, negative log-likelihood and multilayer
perceptron are written in full, and “standard deviation” replaces SD.
The institution L2S and the workshop's artificial-intelligence wording are expanded.

The five model/metric audits and generation excerpts from earlier versions do
not compete with the main narrative. The genuine training curve, tokenizer,
Hamiltonian equation, circuit and two case studies carry the page.

## Scientific traceability

| Poster item | Evidence and meaning |
| --- | --- |
| QCSE definition and circuit | Report Methods and Varmantchaonala et al. (2026), arXiv:2509.05729v2. Context is encoded and a trainable circuit predicts word-code bits. |
| 1.07% | Report tab:qcse-other: full-corpus configuration sweep, test exact-word accuracy 1.068%, rounded. Vocabulary 10,864 words, 14 qubits. This is not the 57-target miniature metric audit. |
| 44.10% | coverage-seeds-seed-summary.csv: Hamiltonian mean most-likely-token validation accuracy 44.09993489583333%, seeds 11/22/33, 4,096 targets, 512 subwords, 10 qubits. This is not expected accuracy of one measurement. |
| Comparison limitation | Word versus subword outcomes, different vocabularies/contexts/recipes and test versus reused validation. These descriptive case studies do not establish a controlled model ranking. |
| Hamiltonian equation | Report Appendix “Final model”: H_l(C) = Q_l diag(e_l + s_theta(C)) Q_l†. State dimension 1024, rank 128, three evolutions. Q has orthonormal complex columns and energies are real. |
| Model size | 1,073,900 trainable real parameters, displayed as 1.07 million. Nine token qubits plus one auxiliary qubit. |
| Tokenizer example | Actual archived training-only byte-pair tokenizer: “language” gives IDs 109 / 260 / 502 / 466, pieces l / an / gu / age. Recorded in tokenization-example.csv. Vocabulary is 512 IDs including byte coverage, merges and an unknown bucket. |
| Loss history | Full-context Hamiltonian fits at seeds 11, 22, 33, epochs 0 through 30. 30,720 updates. training-sources.csv gives original history paths and SHA256s. |
| Epoch alignment | History row e stores online training loss for epoch e+1. The plot uses epoch e+1. Online training averages changing weights and sampled rows; validation evaluates a checkpoint on fixed targets. |
| Loss band | Mean plus/minus one sample standard deviation across the three training seeds. It is not an article confidence interval. All seed points and means are retained as CSV. |
| Loss endpoints | Count-prepared initial model 3.059255 nats, trained model 2.465456, displayed as 3.06 and 2.47. Zero energies reproduce the bigram prior. |
| Classical context | Same WikiText task: a classical multilayer perceptron reaches mean validation loss 2.222766, displayed as 2.22. Shared data/update exposure, differing capacities and schedules. This qualification remains on the poster. |
| Readout | Direct measurement of the token register, marginalizing over the auxiliary outcome. No classical vocabulary head. Preparation and context pooling use classical computation. |
| Limits | Noiseless simulation, reused WikiText validation, unscored held-out WikiText test and incoherent generation. Native-gate compilation and state-preparation costs remain open. |

source-data.csv contains the scalar values displayed. training-seeds.csv,
training-mean.csv and training-sources.csv preserve the epoch-level evidence.
No training, model changes or new held-out scoring occurred for this revision.

## Requirements from the email

The email overrides the earlier Markdown plan. The poster retains the supplied
CentraleSupélec / Université Paris-Saclay, EUGLOH and EU co-funding logos, plus
L2S credit, at the top. It includes purpose, methods, results, conclusions,
authors/address and the confirmed declaration “None”.

Title and subtitle physical lettering exceed the required 15 mm and 10 mm.
The submission remains one A0 portrait sheet, due 18 November 2026 at 12:00
on Edunao. Workshop date: 20 November 2026. No external submission was made.
Missing files referenced only by the older Markdown plan are ignored.

## Rebuild and checks

Compile from this directory with LuaLaTeX:

~~~bash
mkdir -p ../../output/poster/build
latexmk -lualatex -interaction=nonstopmode -halt-on-error \
  -outdir=../../output/poster/build poster.tex
~~~

A standalone source ZIP compiles with the same command without the outdir option.
TeX Live packages: fontspec, unicode-math, IBM Plex, Latin Modern Math, TikZ,
PGFPlots and tabularx. No hard-coded font paths.

Generated verification/production tools remain in output/poster/tools/.
From the repository root:

~~~bash
uv run python -m output.poster.tools.prepare_hamiltonian_figures
uv run --with pymupdf --with pypdf python output/poster/tools/finish_poster.py
~~~

Checks cover archived history hashes, all mean/band values, epoch alignment,
case-study scalar values, complete equation notation, acronym definitions,
two-column geometry, prose fonts, print margins, page size, lettering and logo
resolution. Page renders are inspected at both full-page and detail scales.
The A4 proof preserves vector content with at least 5 mm margins.
Actual print color and distance legibility have not been physically tested.
