# Hamiltonian scientific poster: revision 2

Revised 17 November 2026 in response to the author's feedback.

## Deliverables and build

The submission PDF is poster.pdf, identical to
output/poster/quantum-semantics-poster-a0.pdf at the repository root.
The A4 proof, PNG preview and editable source ZIP are in output/poster/.
The previous version remains in Git commit d8cb5b4 and
output/poster/revisions/v1-2026-11-16/poster.pdf.

From this directory:

~~~bash
mkdir -p ../../output/poster/build
latexmk -lualatex -interaction=nonstopmode -halt-on-error \
  -outdir=../../output/poster/build poster.tex
~~~

For a standalone source ZIP, run the same command without the outdir argument.
Dependencies: LuaLaTeX, fontspec, unicode-math, IBM Plex Sans, Fira Math,
TikZ and PGFPlots. All font families ship with TeX Live. No hard-coded font paths.

## Design and priority

The body and footer use exactly two equal 375.5 mm columns at x = 30 and
x = 435.5 mm, separated by a 30 mm gutter. Only the required title and logo
strip span the full sheet. Column 1 explains the model; column 2 reports its
learning and prediction. There are no changing column widths or spanning results.

The scientific emphasis is the trainable Hamiltonian, its token representation,
context conditioning and measurement readout. The QCSE metric audit and hybrid
circuit benchmark are removed from this poster. They remain in the report.

Every prose element uses IBM Plex Sans. Mathematical letters and symbols use
the sans-serif Fira Math family. The title is bold and subtitle semibold.
Quantum quantities use teal, classical controls or conditioning use burgundy,
and reference quantities use a neutral navy tint. Shapes and direct labels
supplement color. Charts, diagrams and equation remain editable vector source.

The large learning plot and coherence bars show trainability and a restricted
interference benefit. A second chart retains the input-prior and MLP comparisons.
The large final sentences and uncorrected generation sample replace the former
small Interpretation/Conclusion text panels.

## Requirements retained from the email

- One A0 portrait sheet, 841 x 1189 mm, exported as PDF.
- Original CentraleSupélec / Université Paris-Saclay, EUGLOH and EU co-funding
  logos at the top, plus L2S credit.
- Title, authors and institutional address across the upper edge.
- Purpose, methods, results and conclusions.
- Physical title/subtitle lettering above the required 15 mm / 10 mm.
- Declaration of interests “None”, confirmed by the user.
- Submission deadline: 18 November 2026 at 12:00 on Edunao.
- Workshop date: 20 November 2026. No upload or external submission has been made.

The email overrides the earlier Markdown plan. The author's latest request
overrides the plan's serif body font, QCSE/hybrid emphasis and mixed-width layout.
Missing files mentioned only by that plan are ignored.

## Data and scientific traceability

| Visual or claim | Source and handling |
| --- | --- |
| Tokenization | The archived training-only byte BPE tokenizer from the seed-22 coverage run encodes “language” as IDs 109, 260, 502, 466, decoding to l / an / gu / age. tokenization-example.csv records the exact example. |
| Vocabulary/register | V = 512 token IDs, nine token qubits and one auxiliary qubit, dimension 1024. Vocabulary contains 256 bytes, 255 merges and UNK. Whitespace, punctuation and case remain intact. Report Appendix “Final model” and research-summary.md. |
| Hamiltonian equation | Report Appendix “Final model”: H_l(C) = Q_l diag(e_l + s_theta(C)) Q_l†. Q has 1024 x 128 orthonormal columns. Context shifts are classical pooling outputs. |
| Architecture | Three rank-128 evolutions alternate with two banks of local phase rotations. The token register is measured; the auxiliary outcome is ignored. Count-based preparation and context pooling remain classical ingredients. The diagram is at register level, not a compiled native-gate circuit. |
| Learning plot | Full-context coverage reference, seeds 11, 22, 33. 31 fixed-validation evaluations, epochs 0 through 30, and 30 online training-epoch averages per seed. All source histories and SHA256s are listed in training-sources.csv. |
| Training loss alignment | History row e stores next_epoch_online_loss for epoch e+1. The plot places it at e+1, not e. It is an online average over changing weights and sampled training rows, not a fixed full-training-set evaluation. |
| Aggregation | training-seeds.csv retains every plotted seed value. training-mean.csv stores the mean, sample SD, and mean ± SD validation band. The band shows training-seed variability, not a confidence interval over articles. |
| Endpoint benchmark | Count prior 3.059255, Hamiltonian 2.465456, MLP 2.222766 validation NLL. Replicated final results on 4,096 targets. Quantum and MLP share inputs and update exposure; capacities and schedules differ. |
| One-draw success | 28.521166% is mean correct-token probability in one Born draw. It differs from 44.099935% argmax accuracy, which is not displayed. Source: coverage-seeds-seed-summary.csv. |
| Coherence bars | Report tab:tatoeba: coherent 6.76% vs separately trained dephased 5.71% expected one-draw success, three seeds and 8,192 test targets. This is the earlier 256-word task, not WikiText. It does not isolate entanglement or beat the classical MLP. |
| Sample | Exact prefix of the report's training-seed-22 continuation for “The history of ”, uncorrected. Prompt text is separate from generated text. |
| Limits | Noiseless simulation, reused WikiText validation, unscored WikiText test. State-preparation and hardware-compilation costs remain unresolved. No hardware speedup is claimed. |

The loss plot is a genuine epoch history of one model recipe. It replaces the
previous design-milestone curve. No training, test scoring or model changes were
performed to make this revision.

The initial model reproduces the bigram prior when energies are zero. Thus the
decrease from 3.06 to 2.47 shows improvement over preparation alone. The stronger
MLP still leads. The positive contribution is trainable direct Hamiltonian
prediction and a measured input-coherence benefit within these architectures.

## Verification and auxiliary tools

Generated tools are kept in output/poster/tools/. From the repository root:

~~~bash
uv run python -m output.poster.tools.prepare_hamiltonian_figures
uv run --with pymupdf --with pypdf python output/poster/tools/finish_poster.py
~~~

The preparation tool reads archived data only. It does not train or score.
The finalization tool checks data provenance, two-column geometry, all-sans fonts,
page dimensions, lettering, print margins, and logo resolution. It exports the
vector A4 proof and packages the editable sources, including all plotted CSVs.
The A4 proof retains vector artwork and a minimum 5 mm margin.

Final page renders are inspected at whole-poster and detail scales. These checks
do not establish actual printed color or reading distance. Verification output:
output/poster/build/verification.txt.
