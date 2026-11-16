# Scientific poster: production notes

Prepared 16 November 2026. The final poster is in English and is a single A0
portrait sheet (841 x 1189 mm).

## Deliverables

- Submission PDF: poster.pdf in this directory, identical to
  output/poster/quantum-semantics-poster-a0.pdf at the repository root.
- A4 proof: output/poster/quantum-semantics-poster-a4-proof.pdf. This is a
  scaled proof with a 5 mm minimum margin, not the submission file.
- Preview: output/poster/quantum-semantics-poster-preview.png.
- Editable source bundle: output/poster/quantum-semantics-poster-source.zip.
- Oral presentation and questions: presentation-notes.md.
- All figure values and their report locations: source-data.csv.

The original V0 remains in versions/v0-2026-11-16/. Other earlier template
experiments are unchanged.

## Instruction precedence

The email and its attachments override the Markdown plan. Accordingly:

- A0 portrait, one sheet, with purpose, methods, results and conclusions.
- Supplied CentraleSupélec / Université Paris-Saclay, EUGLOH and EU co-funding
  logos appear unmodified on a white strip at the top. L2S is also credited.
- The title, authors and institutional address span the upper part of the sheet.
- Title is bold, subtitle uses the same sans-serif family in semibold.
  Font outline measurements check the 15 mm / 10 mm lettering requirements.
  This takes precedence over the plan's smaller title range.
- Declaration of interests is “None”, explicitly confirmed by the user.
- Submission deadline: 18 November 2026 at 12:00 on Edunao, following the email.
  The attachment's earlier generic printing dates do not override that deadline.
- Workshop presentation: 20 November 2026. The email's isolated “September”
  is inconsistent with its subject, later reminder and workshop attachment.
- No QR code, since no public report or repository URL was supplied.
- Missing files referenced only by the Markdown plan can be ignored, as the user instructed.

No upload, email or other external submission has been made.

## Design

The content and five-section narrative follow the plan, with a revised results
layout: two broad panels above a full-width Hamiltonian comparison. This avoids
crowding three narrow graphs. The largest continuous area contains original
results. A light background, dark title band and flat section structure preserve
the scientific poster format.

IBM Plex Sans supplies headings and chart labels. TeX Gyre Termes supplies body
text and mathematics. Quantum models use dark teal (#187E8A), classical MLP/prior
controls use burgundy (#600030), and identity/count references use navy or a
neutral tint. Marker shape and direct labels supplement color. Graphics and
tables remain editable vector LaTeX/TikZ/PGFPlots source.

Nominal sizes: title 96 pt, subtitle 56 pt, section headings 36 pt, result
headings 34 pt, body 28 pt, captions/labels 24 pt, references/credits 20 pt.
The title and subtitle satisfy the email's physical lettering requirement.
The title outline and key results remain visible at reduced scale. Actual
reading distance and printed color reproduction require a physical print.

## Scientific traceability

The report PDF and its matching main.tex are the primary scientific sources.
The research summary provides context, but does not override the report.

| Region | Source | Editorial decisions |
| --- | --- | --- |
| Meaning as operators | Report Background, Figure 1; Galofaro et al. and Dubois & Toffano | Redraw the semiotic square with a temperature example. Explicitly distinguish prediction from testing semantic relations. |
| Purpose and methods | Report Methods and configuration tables | Keep the three families and controls. QCSE is 10 qubits in the audit, 14 in the full corpus. Hamiltonians span an earlier 8-qubit study and the final 10-qubit model. |
| Panel A | Report Figure 3(a), QCSE audit, 57 held-out targets | Show half-bit and exact-word scores. The constant prior uses training statistics. Do not generalize this miniature audit to the full corpus. |
| Panel B | Table “Final comparison of circuits with a classical output layer” (tab:robust) | Show CRZ for L4, ZZ for L2R2, identity and MLP. Values are two-seed validation means. Identity removes the circuit and preserves the classical head. |
| Panel C | Figure “Retained design steps” (fig:progress) | Show five selected design milestones, not epochs. Last three plotted milestones are three-seed means. Earlier recipes are exploratory. Compare vertical model pairs, not slopes as controlled causal effects. |
| NLL callout | Endpoint 2.46546 vs 2.22277 | Difference 0.24269 nats rounds to 0.24. These are validation results on 4,096 targets. |
| Generation | Report appendix “Context length and generation”; archived seed-22 continuation | Copy the initial substring exactly, including the unusual punctuation and spelling. Prompt text is separate from generated text. |
| Interpretation | tab:tatoeba | 6.76% vs 5.71% is expected success of one Born draw, on a separate word-level test. The dephased model is separately trained, not an inference-time ablation. |
| Conclusions and limits | Discussion and conclusion | Restrict the conclusion to the tested recipes. Mention reused WikiText validation and the unscored test. Do not claim an entanglement benefit or hardware speedup. |

MLP means multilayer perceptron. NLL means negative log-likelihood, using natural
logarithms. Top-1 accuracy and expected one-draw success are different quantities.
The three result panels concern different tasks and must not be pooled.

One correction to the old draft: Hamiltonian checkpoints use validation NLL,
whereas the final hybrid benchmark reports the last epoch. Another: a 50% random
half-bit line would be wrong for the 10-bit audit (the correct independent-bit
probability is 62.3046875% because ties count). The final chart omits that line.

## Rebuild

Use LuaLaTeX with TeX Live packages for fontspec, unicode-math, IBM Plex, TeX Gyre,
TikZ and PGFPlots. The source has no hard-coded system font paths and does not
require the old Gemini theme.

From this directory, a standalone source bundle can be compiled with:

~~~bash
latexmk -lualatex -interaction=nonstopmode -halt-on-error poster.tex
~~~

To keep build products in output/ when working in the repository:

~~~bash
mkdir -p ../../output/poster/build
latexmk -lualatex -interaction=nonstopmode -halt-on-error \
  -outdir=../../output/poster/build poster.tex
~~~

The repository's finalization script is generated in
output/poster/tools/finish_poster.py. It verifies figure values against the
report, checks geometry/type/logo resolution, exports the A4 proof, and
packages the editable sources. From the repository root:

~~~bash
uv run --with pymupdf --with pypdf python output/poster/tools/finish_poster.py
~~~

Verification results are in output/poster/build/verification.txt. Page renders
are reviewed visually at both whole-poster and detail scales. Neither software
checks nor the A4 proof establish real-world legibility at six metres.

Final checks: 23 plotted values match their declared report figures/tables;
the NLL gap is 0.24269 nats before rounding. Measured title/subtitle x-heights
are 17.71 / 10.27 mm. The four logos have effective resolution of at least
150 dpi. There are no overfull boxes or missing glyphs. The PDF contains
454 extracted words including references, chart labels and credits.
