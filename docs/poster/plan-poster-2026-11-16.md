# Plan : poster A0 « Quantum language models, fairly tested »

## Contexte

Public : mélange IA et santé, pas spécialiste du quantique. Le message qui parle à ce public est méthodologique : vérifier la métrique et le contrôle classique avant de croire un gain.

Des informations et ressources complémentaires sont situées dans 

## Grille tirée des consignes

Source : `How to make a poster.pdf` (pages PDF entre parenthèses).

| # | Consigne |
| --- | --- |
| C1 | Format A0, PDF ; orientation non précisée (p. 9 : « à demander ») |
| C2 | Fond clair et uniforme (p. 8, 11) |
| C3 | Palette de 3 ou 4 couleurs, reprise dans les figures (p. 11, 16) |
| C4 | Titre en haut, sans empattement, 50 à 80 pt, lisible à 6 m, message principal, intrigant voire provocateur, sous-titre possible (p. 5, 13) |
| C5 | Corps avec empattement, 24 à 30 pt (p. 14) |
| C6 | Boîtes ou colonnes ; contenu type : titre, introduction, résultats, méthodes, discussion, take-home, références, crédits (p. 12) |
| C7 | Résultats propres les plus visibles ; « less is more » ; pas de survente (p. 15) |
| C8 | Figures refaites pour le poster : annotation sur le graphe (pas de légende), traits épais, libellés intuitifs ; 150 dpi au moins (p. 16, 17) |
| C9 | Take-home en phrases complètes et claires (p. 20) |
| C10 | Références courtes, seulement celles citées, format (Auteur et al., année) plus explicite (p. 21) |
| C11 | Crédits : institution, coauteurs, encadrant ; QR codes possibles (p. 22) |
| C12 | Alignement précis (grilles), marges d'impression, peu de polices et de puces (p. 10) |
| C13 | Outil : « LaTeX (beamer) : bien s'il y a beaucoup d'équations » (p. 9) |

Deux indices sur le goût de l'enseignant : son propre exemple (p. 25) est un poster à bandeau de titre foncé, fond clair, trois colonnes de **boîtes numérotées 1 à 5** finissant par « Take-home message » ; et la vidéo « a radical way for making posters » (p. 30, QR décodé) est le #betterposter de Mike Morrison.

## Classement des gabarits

Mesures faites ici : fichiers `.sty` lus en entier ; Gemini et Scribe compilés en A0 dans le scratchpad (TeX Live 2025 locale), tailles relevées dans le PDF avec PyMuPDF. Inria : source et aperçu lus, non compilé.

| Rang | Gabarit | Conforme d'origine | Écarts | Effort |
| --- | --- | --- | --- | --- |
| 1 | **Gemini** (`anishathalye/gemini`, MIT, 1 282 étoiles, CI, LuaLaTeX) | A0 natif (118,9 × 84,1) ; fond blanc ; 4 couleurs dans un fichier de thème ; titre Raleway gras **61,7 pt** ; corps **24,8 pt** ; colonnes ; `alertblock` pour le take-home ; `\logoleft`/`\logoright` ; pied de page | corps sans empattement (Lato Light) ; pas de QR ; références numérotées | faible : environ 10 lignes de préambule, aucun patch du `.sty` |
| 2 | **Scribe** (`FedericoBruzzone/scribe`, Apache/MIT, 35 étoiles, XeLaTeX) | corps Libertine avec empattement **24,8 pt** ; boîtes à barre de titre ; encadrés (info, tip, warn) ; QR natif ; deux logos ; fond quasi blanc | titre **avec** empattement, 51,4 pt, famille codée en dur dans l'en-tête ; conçu pour 170 × 110 cm (filets et marges en cm absolus) ; 5 à 6 teintes ; thème v0.1.0 du 18/09/2026 | moyen : patch du `.sty`, palette à réduire |
| 3 | **Inria** (`Inria-Chile/inria-latex-poster`, MIT pour le code, XeLaTeX) | A0 portrait natif ; titre sans empattement (60 pt calculés) ; bandeau d'accroche en trois phrases ; blocs « Finding 1..4 » ; bloc final ; QR et citation ; bibliographie APA auteur-année | **marque République Française / Inria dans l'en-tête et charte Inria** (nous sommes CentraleSupélec / L2S) ; 13 couleurs ; fond en dégradé ; corps sans empattement à 34,8 pt ; `fontawesome7` absent en local, police Inria Sans non trouvée, `minted`, `biber` | élevé : à réécrire, il ne resterait que les boîtes |

Autres gabarits regardés :

| Gabarit | Verdict |
| --- | --- |
| **tcbposter** (bibliothèque `poster` de tcolorbox, installée) | pas un thème mais une **grille** : boîtes nommées placées par cases (`column`, `row`, `span`, `rowspan`). pdfLaTeX, donc mêmes figures et même chaîne que le rapport ; polices Times et Helvetica par `newtx` + `helvet`, soit exactement les exemples des slides. Tout le style est à écrire. |
| tikzposter (CTAN, installé) | A0 et option 25 pt, mais thèmes à fond coloré, environnement `figure` incompatible, tout en sans empattement |
| betterposter-latex (`rafaelbailo`, GPL-3, 2019) | concept de la vidéo citée p. 30 ; fond plein coloré par défaut, structure IMRaD reléguée dans des barres étroites |
| Poster Université Paris-Saclay (Overleaf, non officiel) | beamerposter A0/A1 portrait, charte « prune » ; non inspecté |
| Jacobs, a0poster, baposter (latextemplates.com, cités p. 31) | anciens ; `baposter` absent en local |

## Shortlist

1. **Gemini, recoloré CentraleSupélec / L2S, corps en Times** (recommandé). Seul des trois dont titre et tailles tombent déjà dans les fourchettes ; le plus éprouvé ; compile ici sans rien installer. Les blocs sans cadre tolèrent des hauteurs inégales, ce qui compte avec un V0 à rendre demain.
2. **tcbposter**, si l'on veut une grille stricte avec des boîtes cadrées qui s'étendent sur plusieurs colonnes. Conformité maximale, mais une à deux heures de plus.
3. **Scribe**, en repli, si l'on préfère ses boîtes à barre de titre toutes faites.

Du gabarit Inria, on ne garde que deux idées : le bandeau d'accroche en trois phrases et les résultats numérotés.

## Message

Titre (sans empattement, 70 à 80 pt), trois propositions :

1. **Can ten qubits finish your sentence?** (recommandé : intrigant, exact, le modèle final a dix qubits)
2. Quantum language models, fairly tested: the classical baseline still wins
3. Take the circuit out and nothing changes

Sous-titre : *Not better than a small classical network: three quantum language models tested against fair controls.*

Budget de texte : 350 à 400 mots sur tout le poster, environ 40 % de la surface en figures.

## Mises en page

### A. Portrait, grille de 3 colonnes, bande de résultats pleine largeur (retenue)

Portrait confirmé par Ethan le 16/11. Il tient sur n'importe quel panneau ; l'A0 paysage demande 1,19 m de large. Dans Gemini : `width=84.1,height=118.9`, marges `\sepwidth` de 3 % de la largeur, rangées empilées de `columns` (cellules d'un tiers ou de deux tiers).

```
+----------------------------------------------------------------+
| [CS]     CAN TEN QUBITS FINISH YOUR SENTENCE?         [L2S]    | 13 %
|   Not better than a small classical network: three quantum...   |
|   E. Bandasack, M. Leroy, T. Zhang - supervisor Z. Toffano      |
+--------------------+--------------------+----------------------+
| Meaning as         | Thin evidence      | Our test             |  7 %
| operators (1 ph.)  | (1 phrase)         | (1 phrase)           |
+--------------------+-------------------------------------------+
| 1 WHY QUANTUM      | 2 THREE MODELS, ONE RULE                  | 21 %
|   SEMANTICS?       | [circuit QCSE] [chaine hamiltonienne]     |
| [carre semiotique] | tableau 3 lignes : modele / controle      |
| 2 puces            | 1 ligne de protocole                      |
+--------------------+-------------------------------------------+
| 3 RESULTS (bande legerement teintee, pleine largeur)           | 35 %
| +------------------+------------------+----------------------+ |
| | (1) A constant   | (2) Remove the   | (3) It learns, but   | |
| | guess beats the  | circuit: nothing | the MLP stays ahead  | |
| | published metric | changes          |                      | |
| | [barres]         | [points]         | [courbes]            | |
| | 1 ligne          | 1 ligne          | 1 ligne + exemple    | |
| +------------------+------------------+----------------------+ |
+--------------------+-------------------------------------------+
| 4 WHAT IT MEANS    | 5 TAKE-HOME (bloc mis en avant)           | 16 %
| 3 puces + suite    | 3 phrases completes, 34 pt                |
+--------------------+-------------------------------------------+
| References (5)     | Credits, workshop  | [QR]                 |  6 %
+----------------------------------------------------------------+
```

Remplissage, zone par zone (texte du poster en anglais, chiffres du rapport v1.2) :

| Zone | Rôle (consigne) | Visuel | Phrases d'exemple | Mots |
| --- | --- | --- | --- | --- |
| Accroche | raconter l'histoire en trois temps (p. 23) | aucun | « Quantum semantics writes relations between meanings as operators. » / « Quantum language models claim to learn them, but on tiny corpora and with permissive metrics. » / « We gave a classical network the same data and compared. » | 35 |
| 1 Why quantum semantics? | contexte, motivation (C6) | carré sémiotique, opérateurs en couleur sur les arêtes (figure 1 du rapport, agrandie) | « Contraries, contradictions and inclusions become Pauli operators. » « A cosine similarity confuses antonyms; this vocabulary separates them. » | 40 |
| 2 Three models, one rule | méthodes, courtes (p. 18) | circuit à 3 qubits et chaîne `prior -> e^{-iH} -> Born` (figure 2 simplifiée) ; **tableau** ci-dessous | règle en gras : « Every classical control gets the same inputs, targets and training budget. » ligne fine : « Exact simulation, checkpoints chosen on validation data only. » | 60 |
| 3 (1) | résultat le plus visible, phrase-titre à la place de la légende (p. 17) | deux paires de barres : critère publié 96.5 % (constant guess) contre 56.1 % (QCSE), mot exact 22.8 % contre 0.0 % ; trait pointillé « chance = 50 % » ; flèche « a constant guess wins » | « The published score counts a word as right when half of its bits match. » | 30 |
| 3 (2) | idem | points alignés, top-1 : circuit 33.32, identité 33.36, MLP 35.66 (gauche seul) ; 50.38, 50.03, 54.89 (bidirectionnel) ; accolade « +0.23 pt at 2.7x the cost, 5 folds x 3 seeds » | « An identity circuit does as well as a trained one. The MLP leads everywhere. » | 30 |
| 3 (3) | idem | courbes NLL au fil des 11 étapes (figure 3 simplifiée) : quantique 4.28 vers 2.47, MLP 2.22, comptes de bigrammes 3.06 ; écart hachuré « 0.24 nats » ; petit insert « coherent 6.76 % vs dephased 5.71 % » | « Coherence helps against a dephased copy, not against a classical network. » ; encart d'exemple : le texte généré cité au rapport (annexe B, invite « The his... »), repris mot pour mot | 45 |
| 4 What it means | interprétation, limites, suite (p. 19) | aucun | « Most gains came from classical ingredients: learned embeddings, count priors, context pooling. » « Limits: noiseless simulation, two or three seeds, test split left unscored. » « Next: test the semiotic square directly, on a relation task. » | 55 |
| 5 Take-home | phrases complètes (C9) | fond teinté, filet de couleur | « No quantum advantage: every quantum or hybrid model was matched or beaten by a classical model given the same information. » « Check the metric first: a constant guess scored 96.5 % on the published criterion. » « Always run the identity control: it shows what the circuit really adds. » | 60 |
| Pied | C10, C11 | QR, logos déjà en tête | (Galofaro et al., 2018) (Dubois and Toffano, 2026) (Coecke et al., 2010) (Varmantchaonala et al., 2026) (Bengio et al., 2003) ; « AI and OneHealth workshop, 20 November 2026 » | 40 |

Tableau du bloc 2 (remplace un paragraphe de méthodes) :

| Model | Qubits | Readout | Fair control |
| --- | --- | --- | --- |
| QCSE, reimplemented | 7 to 14 | one bit per qubit | constant guess |
| Hybrid circuit | 8 | classical head | identity circuit, MLP, BERT |
| Hamiltonian | 10 | Born rule, no classical head | dephased copy, MLP |

(Le nombre de qubits de la ligne QCSE est à relire dans le rapport avant d'être écrit.)

### B. Paysage, trois colonnes (repli, non construite)

Disposition native de Gemini et de l'exemple de l'enseignant. Elle ne sert que si la séance du 17/11 impose le paysage.

```
+-------------------------------------------------------------------------+
| [CS]         CAN TEN QUBITS FINISH YOUR SENTENCE?               [L2S]   |
+---------------------+-------------------------------+-------------------+
| accroche (3 phr.)   | 3 RESULTS                     | 4 WHAT IT MEANS   |
| 1 WHY QUANTUM       | (1) [barres]  | phrase-titre  | 3 puces           |
|   SEMANTICS?        | (2) [points]  | phrase-titre  +-------------------+
| [carre semiotique]  | (3) [courbes] | phrase-titre  | 5 TAKE-HOME       |
| 2 THREE MODELS      |               | + exemple     | 3 phrases         |
| [schemas] [tableau] |                               | References, [QR]  |
+---------------------+-------------------------------+-------------------+
        27 %                       46 %                       27 %
```

Mêmes blocs et mêmes phrases que A ; les trois résultats s'empilent au centre, figure à gauche et phrase à droite.

### C. Variante « phrase centrale » (idée #betterposter, plus risquée)

Panneau central de 46 % portant le take-home en 110 pt (« Give a classical network the same data, and the quantum circuit stops helping. »), le tableau de score et le QR ; contexte et méthodes à gauche, les trois figures en petit à droite. Accroche forte à 6 m, mais structure IMRaD moins lisible et grande surface teintée (C2, C6). À garder comme idée pour la séance du 17/11, pas pour le V0.

## Palette et typographie

- Quatre couleurs, relevées sur les logos : bleu nuit `#1F3050` (L2S) pour le bandeau et les titres de blocs ; bleu-vert `#2A9DB0` (L2S, assombri) ; bordeaux `#600030` (CentraleSupélec) ; gris clair `#F3F4F7` pour les fonds teintés. Texte noir.
- Code unique dans toutes les figures : **quantique = bleu-vert, classique = bordeaux, hasard ou référence = gris**, doublé par la forme des marqueurs.
- Deux polices : titres en sans empattement (Raleway du gabarit, ou TeX Gyre Heros si l'on veut l'allure Arial) ; corps en TeX Gyre Termes (Times), mathématiques assorties.
- `beamerposter` avec `scale=1.13` : titre 70 pt, titres de blocs 40 pt, corps 28 pt, annotations de figures 24 pt au moins, références et crédits 20 pt au moins. Traits des figures de 3 pt au moins.
- Logos : `CS.png` (2362 px) et `L2S_logo.png` (1485 px) restent au-dessus de 150 dpi jusqu'à 40 cm et 25 cm de large.

## Fichiers créés

```
  poster.tex                   # contenu, sans commentaire méta
  beamerthemegemini.sty        # copie non modifiée (MIT), avec LICENSE-gemini.md
  beamercolorthemecsl2s.sty    # palette ci-dessus
  fig/square.tex, models.tex, metric.tex, identity.tex, progress.tex
  references.bib               # 5 entrées reprises de qlm-short-report/references.bib
  logos/CS.png, L2S_logo.png   # copies de qlm-report/logos/
  meta.md                      # miroir : rôle, source et consigne de chaque bloc ; pitch ; prochaines étapes
  versions/v0-2026-11-16/      # checkpoint avant toute modification
  poster.pdf
```

Réutilisé : code TikZ, quantikz et pgfplots de `report/main.tex`, chiffres des tableaux. Rien n'est modifié.

## Étapes

1. Dossier, logos, thème Gemini et sa licence, thème de couleurs.
2. Squelette `poster.tex` de la mise en page A, avec le texte définitif et des cadres vides à la place des figures ; compiler et contrôler la géométrie.
3. Redessiner les cinq figures au style poster (C8).
4. Assembler, vérifier (ci-dessous), corriger.
5. Écrire `meta.md` : pour chaque bloc, rôle, paragraphe ou tableau source du rapport, consigne satisfaite ; pitch de 30 s ; questions probables ; prochaines étapes. Checkpoint `v0`.
6. Après la séance du 17/11 : intégrer les retours, checkpoint `v1`, PDF final avant le 18/11 à 12h00, version A4 à imprimer (p. 23).

Pitch de 30 s proposé (80 mots) : « Quantum language models promise to learn meaning with quantum circuits. We asked a simple question: if a classical network gets exactly the same data, does the quantum part still help? We rebuilt one published model and designed two of our own. Each time the answer was no: a constant guess beat the published metric, removing the circuit changed nothing, and a small classical network stayed ahead. The lesson holds for any AI claim: check the metric and the control. »

## Vérification

- `latexmk -lualatex poster.tex` sans erreur ; aucun `Overfull \vbox` (beamer coupe en silence ce qui dépasse le bas de page).
- Script PyMuPDF dans le scratchpad : format 841 × 1189 mm ; titre sans empattement entre 60 et 80 pt ; corps avec empattement entre 24 et 30 pt ; aucun texte sous 20 pt ; tout texte à 10 mm au moins du bord ; résolution effective des logos au moins 150 dpi ; nombre de mots au plus 400.
- Chaque nombre pertinent du poster retrouvé dans `../report/main.tex` (script PASS/FAIL), puis relecture ligne à ligne.
- Rendu en PNG à l'échelle A4 et relecture visuelle : hiérarchie, alignements, chevauchements dans les figures, couleurs.
- Forme : aucun tiret cadratin, quatre couleurs, deux polices, chaque référence listée est citée.
- Dans le compte rendu : ce qui n'a pas pu être vérifié (rendu imprimé, lisibilité réelle à 6 m).

## À trancher par Ethan

Réglé : orientation portrait, mise en page A. Sans réponse sur les points suivants, le V0 prend le choix indiqué.

1. Gabarit : Gemini recoloré (défaut), tcbposter ou Scribe.
2. Titre parmi les trois (défaut : « Can ten qubits finish your sentence? »).
3. Cible du QR code (PDF du rapport ou dépôt) : il faut une adresse publique (défaut : pas de QR dans le V0).
4. Mention d'usage de LLM en pied de poster, comme dans le rapport (défaut : oui, une ligne).
5. Auteurs : même liste et même ordre que le rapport (défaut : oui).
