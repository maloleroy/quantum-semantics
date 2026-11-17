# Poster plan mirror

Status: **V0 prepared, 16 November 2026** for the 17 November preparation session. The one-page PDF is A0 portrait. Approximate word tally: **396 words**, counting poster text, chart labels, and short references; lettering embedded in raster logos is excluded.

## Decisions

- Layout A from the plan: three introductory cells; section 1 at left and methods across the remaining width; full-width tinted results band; interpretation and take-home row; references and credits footer.
- Gemini theme recolored for CentraleSupélec / L2S. Raleway headings, Times-style body, navy `#1F3050`, teal `#2A9DB0`, burgundy `#600030`, pale gray `#F3F4F7`.
- Title: “Can ten qubits finish your sentence?” Subtitle: “Not better than a small classical network: three quantum language models tested against fair controls.”
- Authors and supervisor follow report v1.2: Ethan Bandasack, Malo Leroy, Thierry Zhang; Zeno Toffano.
- No QR code: no public destination was specified. The footer includes the LLM-use disclosure.
- English copy for an AI and OneHealth audience. The main point is methodological: check the metric and the classical control before claiming a gain.

## Block mirror

| Poster block | Role | Report source | Plan requirement |
| --- | --- | --- | --- |
| Three introductory cells | State the motivation, evidence gap, and comparison. | `main.tex`, abstract and Introduction | C4, C6 |
| 1. Why quantum semantics? | Explain the semiotic square and Pauli relations with a redrawn diagram. | `main.tex`, Background and Fig. 1; Galofaro et al. (2018), Dubois and Toffano (2026), Coecke et al. (2010) | C6, C8, C10 |
| 2. Three models, one rule | Diagram QCSE, hybrid, and Hamiltonian approaches; list sizes and controls; state matched inputs, targets, budget, and validation checkpoint selection. | `main.tex`, Methods; Varmantchaonala et al. (2026), Bengio et al. (2003) | C6, C8 |
| 3. Results | Compare the 57-target QCSE audit; hybrid top-1 against identity and MLP; Hamiltonian validation NLL milestones and Tatoeba coherence control. | `main.tex`, Results, summary table, Fig. 3, Appendix B | C7, C8 |
| 4. What it means | Summarize classical contributions, exact-simulation and seed limits, unscored test split, and next relation task. | `main.tex`, Discussion and conclusion | C6, C7 |
| 5. Take-home message | Give three complete conclusions without claiming a universal result. | `main.tex`, Results and Discussion | C9 |
| References and credits | List the five works cited; identify the workshop and LLM assistance. | `references.bib`; `main.tex`, acknowledgements | C10, C11 |

## Data notes

- QCSE audit: 57 test targets; 10 qubits in the miniature audit and 14 in the separate full-scale setting. QCSE scores 56.1% half-bit and 0.0% exact-word; the constant prior scores 96.5% and 22.8%.
- Hybrid validation top-1: L4 circuit / identity / MLP = 33.32 / 33.36 / 35.66%; L2R2 = 50.38 / 50.03 / 54.89%. Across five folds and three seeds, the circuit gained 0.23 points over identity at 2.7 times the compute; fold effects have both signs.
- WikiText-2 validation NLL, Hamiltonian / MLP: first BPE 4.2773 / 3.4424; product state 3.1975 / 3.2306; count prior 2.66776 / 2.41207; 64× pool 2.46546 / 2.22277. The final gap is 0.24 nats. Model and data recipes change between milestones.
- Tatoeba expected one-draw success: coherent Hamiltonian 6.76%; dephased control 5.71%. The sample continuation is copied from Appendix B with prompt “The history of”: “its first game ream, butvarBan7…”

## Thirty-second pitch

“Quantum language models promise to learn meaning with quantum circuits. We asked a simple question: if a classical network gets exactly the same data, does the quantum part still help? We rebuilt one published model and designed two of our own. Each time the answer was no: a constant guess beat the published metric, removing the circuit changed nothing, and a small classical network stayed ahead. The lesson holds for any AI claim: check the metric and the control.”

## Questions to prepare for

- **Why can the constant prior score 96.5%?** The published half-bit criterion rewards label statistics; the exact-word audit scored 0.0% for QCSE and 22.8% for the prior.
- **What makes the controls fair?** They use the same inputs, targets, and update budget, with checkpoints selected on validation data.
- **Does this show quantum models cannot help?** No. These exact-simulation experiments found no advantage for the tested models; the planned semiotic-square relation task remains open.

## Build and review

The plan requested `latexmk -lualatex poster.tex`. LuaLaTeX could not start because its font cache had no writable location in this environment. The copied Gemini theme points directly to the installed TeX Live Raleway and Lato font files, and the poster was built with XeLaTeX; the body remains Times-style. No poster drafts were consulted.

The PDF compiles to one A0 portrait page (841 × 1189 mm). The rendered page was reviewed for chart legibility, spacing, contrast, and clipping. The source has no overfull boxes. Title is 72 pt, block titles 40 pt, body 28 pt, chart annotations at least 24 pt, and references / credits 20 pt. The raster logos are sized above the plan’s 150 dpi minimum. A physical print check at six metres has not been done.

For the V0 meeting, `versions/v0-2026-11-16/` preserves the exact source and PDF. After feedback on 17 November, prepare a V1 checkpoint and A4 print proof; the final PDF is due 18 November at noon.
