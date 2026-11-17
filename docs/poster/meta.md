# Scientific poster: revision 5

Revised 17 November 2026. Title restored to “Can ten qubits finish your sentence?”
The two equal-width columns and Hamiltonian focus remain.

This revision replaces a sequence of paper topics with a four-stage argument.
Read across the top row, then across the shaded results row. The two equal
columns, centered equation and mathematical operator lettering are retained.

## The story and its reading order

1. **We first tested quantum word embeddings.** Define the prediction problem,
   explain context visibility, and show the earlier exact-word recovery result.
   The weak recovery motivates a question about learning dynamics.
2. **We train the Hamiltonian itself.** Answer that question with one integrated
   design: subword input, learned generator, and direct token measurement.
   Tokenization, equation/key and circuit are steps within this design.
3. **Training lowers prediction loss.** Test the design using the actual epoch
   history, then report most-probable-subword validation accuracy.
4. **A classical network still predicts better.** Put learning in context with
   the common-task control, then answer the title with the saved incoherent sample.

The top row develops the question and design. A single pale results band groups
the two tests below it. Numbered claims, aligned row starts, gutter arrows and
the downward transition make the reading order visible. The headings alone
communicate the argument. Connecting sentences explain why one stage leads to
the next; the diagrams no longer introduce independent paper sections.

The earlier side-by-side QCSE/Hamiltonian case figure is removed. The 1.07% word
score is the starting observation in stage 1; the 44.10% subword score belongs
to the new experiment in stage 3. The task-change caveat stays next to the
starting observation. This avoids presenting those scores as a controlled gain.

## Author's latest direction

- Present the move from Quantum Context-Sensitive Embedding (QCSE) to a learned
  Hamiltonian as the scientific narrative.
- Use the existing results and explicitly state the unmatched-task limitation.
  The author confirmed this after we checked whether a controlled comparison existed.
- Enlarge text and headings, spending less area on blank space.
- Explain every symbol and operator immediately below the Hamiltonian equation.
- Use ordinary LaTeX mathematical lettering. Prose remains sans serif.
- Define acronyms at first use or write the term in full.
- Show the difference between word embeddings, bidirectional masked prediction
  and causal next-token prediction. Expand all three model names in the diagram.
- Add a graphical comparison with a classical model on the Hamiltonian's task.
- Center the equation and render diag with operatorname in the mathematical font.
- Make the wording and layout tell a story rather than concatenate report sections.

The poster does not assert that the current experiments establish a controlled
Hamiltonian-over-embedding win. Its QCSE and Hamiltonian results differ in task,
vocabulary, context, training recipe and evaluation split. The limitation is
visible in the transition to the new experiment. No common-axis comparison,
fold improvement or significance claim is used.

## Deliverables

- poster.pdf in this directory is the submission A0 portrait PDF.
- It matches output/poster/quantum-semantics-poster-a0.pdf at the repository root.
- A4 proof, PNG preview and editable source ZIP are also in output/poster/.
- Older PDFs: Git commits d8cb5b4, 2857d51, d11013f and 82ff719,
  and output/poster/revisions/.
- presentation-notes.md contains the revised three-minute script and questions.

## Layout and typography

Body and footer columns are each 375.5 mm wide, at x = 30 and x = 435.5 mm,
with a 30 mm gutter. Both rows use those same column widths. Only the required
institutional/logo/title strip spans the whole A0 portrait sheet (841 x 1189 mm).
The results band is a background treatment, not a change in content width.

IBM Plex Sans supplies all prose. Latin Modern Math supplies mathematical
lettering. Nominal sizes:

- Title 96 pt; semibold subtitle 56 pt; author names 36 pt.
- Section headings 44 pt, increased from 36 pt.
- Main body 34 pt, increased from 28 pt.
- Captions, chart labels and equation definitions 29 pt, increased from 24 pt.
- Displayed Hamiltonian 54 pt; circuit evolution labels 36 pt.
- Starting result 66 pt; final answer 38 pt; references and credits 22 pt.

The equation has a complete key for C, stage index l, H, Q, e, s, trainable
parameters theta, diag, and the conjugate-transpose dagger. Circuit notation is
explained separately in its caption. There are no undefined prose acronyms:
QCSE is expanded on introduction, negative log-likelihood and multilayer
perceptron are written in full, and “standard deviation” replaces SD.
The institution L2S and the workshop's artificial-intelligence wording are expanded.
BERT expands to Bidirectional Encoder Representations from Transformers; GPT
to Generative Pre-trained Transformer. Both are defined in the new figure.

The equation sits in an explicitly centered makebox in the design column. Operatorname
remains the correct semantic LaTeX operator command, while
setoperatorfont with symup forces Latin Modern Math for its letters rather
than inheriting the surrounding sans-serif text font. PDF inspection checks
the displayed equation's center and the actual font used for diag.

The new diagram treats representation and context direction as different axes.
QCSE maps context to a quantum state and word-code bits. BERT's missing token
receives both left and right context. GPT's target receives preceding tokens only.
All three use representations; the Hamiltonian experiment uses the causal
next-token task. These diagrams explain objectives/readout and do not claim
that QCSE, BERT and GPT are mutually exclusive representation types.

Tokenization and circuit figures are shorter vertically while retaining their
font sizes. The classical comparison supports the fourth claim with a common-axis
loss chart. The original word-level observation and new subword result retain
their explicit task-change caveat.

## Scientific traceability

| Poster item | Evidence and meaning |
| --- | --- |
| QCSE definition and circuit | Report Methods and Varmantchaonala et al. (2026), arXiv:2509.05729v2. Context is encoded and a trainable circuit predicts word-code bits. |
| Bidirectional prediction | Devlin et al. (2019), BERT abstract and masked-language-model description: representations condition on left and right context. Primary source: https://aclanthology.org/N19-1423/ . |
| Causal next-token prediction | Radford et al. (2018), Improving Language Understanding by Generative Pre-Training, Section 3.1, Equation 1: target probability conditions on preceding tokens. Primary source: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf . |
| 1.07% starting observation | Report tab:qcse-other: full-corpus configuration sweep, test exact-word accuracy 1.068%, rounded. Vocabulary 10,864 words, 14 qubits. This is not the 57-target miniature metric audit. |
| 44.10% new-experiment result | coverage-seeds-seed-summary.csv: Hamiltonian mean most-likely-token validation accuracy 44.09993489583333%, seeds 11/22/33, 4,096 targets, 512 subwords, 10 qubits. This is not expected accuracy of one measurement. |
| Comparison limitation | Word versus subword outcomes, different vocabularies/contexts/recipes and test versus reused validation. These descriptive case studies do not establish a controlled model ranking. |
| Hamiltonian equation | Report Appendix “Final model”: H_l(C) = Q_l diag(e_l + s_theta(C)) Q_l†. State dimension 1024, rank 128, three evolutions. Q has orthonormal complex columns and energies are real. |
| Model size | 1,073,900 trainable real parameters, displayed as 1.07 million. Nine token qubits plus one auxiliary qubit. |
| Tokenizer example | Actual archived training-only byte-pair tokenizer: “language” gives IDs 109 / 260 / 502 / 466, pieces l / an / gu / age. Recorded in tokenization-example.csv. Vocabulary is 512 IDs including byte coverage, merges and an unknown bucket. |
| Loss history | Full-context Hamiltonian fits at seeds 11, 22, 33, epochs 0 through 30. 30,720 updates. training-sources.csv gives original history paths and SHA256s. |
| Epoch alignment | History row e stores online training loss for epoch e+1. The plot uses epoch e+1. Online training averages changing weights and sampled rows; validation evaluates a checkpoint on fixed targets. |
| Loss band | Mean plus/minus one sample standard deviation across the three training seeds. It is not an article confidence interval. All seed points and means are retained as CSV. |
| Loss endpoints | Count-prepared initial model 3.059255 nats, trained model 2.465456, displayed as 3.06 and 2.47. Zero energies reproduce the bigram prior. |
| Classical control chart | Same WikiText task and 4,096 validation targets: Hamiltonian mean loss 2.465456 versus classical multilayer perceptron 2.222766, shown as 2.47 and 2.22. Both are three-seed means. Shared data/update exposure, differing capacities and schedules. Bars start at zero. This comparison is within a common task, unlike the QCSE case studies. |
| Readout | Direct measurement of the token register, marginalizing over the auxiliary outcome. No classical vocabulary head. Preparation and context pooling use classical computation. |
| Limits | Noiseless simulation, reused WikiText validation, unscored held-out WikiText test and incoherent generation. Native-gate compilation and state-preparation costs remain open. |
| Sample excerpt | Uncorrected prefix of the training-seed-22 continuation for prompt “The history of ”: “its first game ream , butvarBan7”. Source: report appendix “Context length and generation” and coverage-seeds-samples.csv. It illustrates the remaining fluency problem and answers the title without a general language-ability or hardware claim. |

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
case-study and classical-control values, complete equation notation, operator font,
equation centering, acronym definitions,
two-column geometry, prose fonts, print margins, page size, lettering and logo
resolution. Page renders are inspected at both full-page and detail scales.
The A4 proof preserves vector content with at least 5 mm margins.
Actual print color and distance legibility have not been physically tested.
