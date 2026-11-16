# Miroir méta de `main.tex`

Ce fichier suit `main.tex` paragraphe par paragraphe. Pour chaque paragraphe, tableau et figure : **rôle** (pourquoi il est là) et **sources** (d'où viennent les affirmations et les chiffres). Rien de ce fichier ne figure dans le rapport.

Chemins relatifs à `Filière recherche/`. Conventions :
- `main:` = `git show origin/main:<chemin>` dans `quantum-semantics/` ;
- `ordering:` = `git show research/qcse-ordering:<chemin>` ;
- `results/`, `docs/` = fichiers de la branche courante `codex/qcse-bert-robust-benchmark`.

## Gabarit et contraintes

- Rapport en anglais, exigé par le sujet (`Administration/Sujet21_Zeno_Toffano.pdf`, « Livrables attendus »).
- Structure IMRaD et règles de forme du cours d'A. Chaillet (`OneDrive_1_16-11-2026/Slides  - How to write a paper.pdf`, p. 13, 24, 29, 31, 34, 41, 45) : titre clair, résumé de moins de 250 mots sans référence ni formule (236 mots), résultats négatifs mentionnés, discussion qui interprète au lieu de récapituler.
- Budget demandé : corps de 3 à 4 pages ; figures, tableaux, bibliographie et annexes hors décompte. Mesure par `pdflatex "\def\bodyonlyflag{}\input{main}"` (flottants, annexes et bibliographie masqués) : le corps finit aux trois quarts de la page 4, titre et résumé compris.
- Murs de texte évités : tout ce qui est énumérable est en tableau ou en liste ; aucun paragraphe de prose au-delà d'environ 110 mots, hors résumé.
- Logos copiés de `qlm-report/logos/`. Thème et contenu de `qlm-report/` non repris.
- Aucun tiret cadratin ; aucun commentaire LaTeX méta dans `main.tex`.

## En-tête

- **Titre** : annonce le parcours (sémantique quantique, puis prédicteurs entraînables). « Next-Word » plutôt que « Next-Token » : la plupart des études portent sur des mots ; la dernière porte sur des sous-mots, précisé dans le texte.
- **Auteurs et encadrant** : repris de `Papers/synthesis/Literature_Review_21_Quantum_Semantics.tex`. À confirmer.

## Résumé

Rôle : version réduite du rapport (contexte, question, méthodes, trois résultats, conclusion). Chaque résultat cité est détaillé en section 4. Aucun chiffre, pour respecter la consigne « no formulas ».

## 1 Introduction

| § | Rôle | Sources |
|---|---|---|
| 1 | Contexte : relations de sens implicites dans les modèles statistiques, explicites en sémantique quantique et dans DisCoCat ; rappel de l'objectif du sujet (cadre hybride). | Sujet 21 ; `Papers/onboarding/paper1.md`, `paper2.md`, `paper3.md` |
| 2 | Verrou : modèles quantiques entraînables à preuves empiriques minces. | `Papers/onboarding/paper4.md` (V-A, V-B) ; `quantum-semantics/docs/literature.md` |
| 3 | Question unique, puis le parcours réel en trois phrases (QCSE, hybrides, hamiltoniens) pour expliquer l'enchaînement ; contributions en liste ; plan. | Historique : `Conversations/README.md`, `quantum-semantics/backlog.md`, `main:docs/research-summary.md` |

## 2 Background

| § | Rôle | Sources |
|---|---|---|
| Quantum-like semantics, § 1 | Modèle HAL de Galofaro-Toffano-Doan, corrélations ±1, lien au carré de Greimas. | `Papers/.work/paper1.txt` p. 8-9 |
| Quantum-like semantics, § 2 | Dictionnaire des opérateurs de Dubois-Toffano (figure 1), eigenlogic, limite (pas de règle texte vers état, pas de données annotées). | `Papers/french/paper2.md` 3-P4 à 3-P13 ; limites dans `Papers/onboarding/paper1.md` §6.4 et `paper2.md` §3 |
| Compositional semantics | Deux phrases sur DisCoCat, exigées par le sujet, sans en faire un axe : non expérimenté. | `Papers/onboarding/paper3.md` |
| Trainable quantum LMs | Une phrase de cadrage, puis renvoi au tableau 1. | voir tableau 1 |

- **Figure 1** : carré sémiotique et opérateurs de Pauli. Exemples et flèche d'inclusion S1 vers ¬S2 tirés de `Papers/french/paper2.md` 3-P4, 3-P5, 3-P9.
- **Tableau 1** (travaux liés) : QCSE, 85,8 % contre 70,0 % pour CBOW-80 (`Notes/Recherche_bibliographique_et_synthese.md` §5.5, `Papers/paper4.pdf`) ; Liao-Ferrie théorique (`Papers/synthesis/state_of_art_quantum_gpt.tex`) ; Quixer battu par sa référence Transformer (`quantum-semantics/docs/literature.md`) ; HyQuT, remplacements ciblés stables et larges instables (`state_of_art_quantum_gpt.tex`) ; QISA (même source).

## 3 Methods

| § | Rôle | Sources |
|---|---|---|
| QCSE reimplementation | Ce qui a été codé, et pourquoi on peut s'y fier (simulation exacte, parité Qiskit à 1e-15). | `ordering:reports/01-paper-audit.md` (écart max 1,22e-15) ; `main:docs/experiments/findings.md` §5 (3e-16) |
| Hybrid benchmark | Modèles hybrides, six réglages de circuit, références MLP et BERT de zéro, tâches L4 et L2R2. | `docs/models.md`, `docs/protocol.md` |
| Hamiltonian predictors | Lecture de Born sans couche classique de sortie, contrôle déphasé, modèle final (10 qubits, 3 étages, rang 128, 1 073 900 paramètres). | `main:docs/research-summary.md` ; `main:docs/research.md` |
| Data and protocol | Partitions par groupes, vocabulaire BPE appris sur l'entraînement, équité des contrôles, matériel ; métriques renvoyées au tableau 2. | `docs/protocol.md` ; `main:docs/research-summary.md` ; `main:docs/dgx.md` |

- **Figure 2** : (a) circuit QCSE schématique sur 3 qubits (le vrai en a ⌈log2 V⌉) ; (b) chaîne du prédicteur final.
- **Tableau 2** (métriques) : définitions et étude où chacune sert. Évite un paragraphe de définitions.

## 4 Results

| § | Rôle | Chiffres et sources |
|---|---|---|
| The QCSE metric… (3 puces) | Premier résultat : métrique, paramètres inactifs, pleine échelle. | Hasard 1/2 : `Papers/onboarding/paper4.md` V-A. 96,5 / 56,1 / 0,0 / 22,8 : `ordering:reports/01-paper-audit.md`. Étiquettes mélangées : `ordering:reports/07-…md`. « 19 sur 58 » : audit (portes diagonales finales de l'ansatz). 1,07 % et 44 configurations : `results/cluster-sweep/short-report.md` |
| In hybrid models…, § 1 | Deuxième résultat : circuit ≈ identité, ZZ instable, CRZ et ZZ même classe de fonctions. | `results/robust-causal-controls/README.md` ; `docs/literature.md` |
| In hybrid models…, § 2 | Campagnes exploratoires DGX (demande d'Ethan : les citer si elles apportent quelque chose). Elles apportent la CV multi-plis et multi-graines et un test tenu à part qui confirment le résultat. | « plus de 500 » = 90 + 150 entraînables + 258 + 30 = 528 ajustements (sondes de gradient exclues). 0,23 point et 2,7× : `results/robust-timed-cv/README.md` (+0,233, 2,69×). Test : `results/robust-followup/README.md` (33,432 contre 33,474 ; MLP 35,537) |
| In hybrid models…, § 3 | Écarte l'hypothèse du plateau stérile sans surinterpréter. | `results/gradient-replicated-10324/README.md` |
| Hamiltonian learning works | Troisième résultat, côté positif : apprentissage et cohérence. | `main:docs/experiments/findings.md` §1-2 |
| But it does not catch up… | Côté négatif : écart au MLP, génération incohérente, idées rejetées (annexe). | 4,28 vers 2,47 : `main:docs/experiments/successful.md`. Écart 0,24 nat (0,242690), 10,929 et 6,250 points : `main:docs/experiments/coverage-seeds.md` |

- **Tableau 3** (synthèse) : un bloc par étude ; colonnes « Acc. 1 / Acc. 2 » redéfinies par bloc. WikiText : `main:docs/experiments/coverage-seeds.md` et `coverage-seeds-validation.csv`.
- **Figure 3** : rend visible le constat central (le MLP progresse au moins aussi vite). Points de `main:docs/experiments/successful.md` §6 à 18 ; MLP des étapes 7 à 10 = MLP fort à trois graines (2,41207). Étape 2 : le MLP (3,2306) est derrière le quantique (3,1975), seul cas, gardé tel quel. « 4×, 16×, 64× » = pools de 131 072, 524 288 et 2 097 152 positions rapportés aux 32 768 initiales.

## 5 Discussion and conclusion

| § | Rôle |
|---|---|
| 1 | Interprétation : gains surtout classiques ; seul effet quantique isolé = cohérence face à un contrôle restreint ; leçon de méthode de l'audit QCSE. Sources : `main:docs/research.md`, `successful.md`. |
| 2 | Limites (exigées par le cours) : simulation, coûts non chiffrés, validation réutilisée, test WikiText non évalué, peu de graines, capacités inégales. Sources : `docs/protocol.md`, `main:docs/research.md`. |
| 3 | Retour au sujet : le carré sémiotique n'a pas été testé ; expérience proposée tirée de `Notes/Recherche_bibliographique_et_synthese.md` §7. |
| 4 | Conclusion en deux phrases, qui répond à la question de l'introduction. |

## Remerciements

Encadrant ; matériel (« DGX A100 server » : le propriétaire du serveur n'est écrit nulle part). Phrase de déclaration d'usage des LLM : obligation rappelée dans `Notes/Recherche_bibliographique_et_synthese.md` §2, à confirmer.

## Annexes

| Partie | Contenu | Sources |
|---|---|---|
| A.1 | Équations QCSE (matrice, nombre de couches, marginales), portes inactives, hasard du critère, marginales contre Born. | `Papers/onboarding/paper4.md` III-B à V-A ; `ordering:reports/01-paper-audit.md` ; `docs/models.md` |
| A.2 | Identité CRZ/ZZ, borne de rang de Schmidt. | `docs/literature.md` |
| A.3 | Modèles hamiltoniens, $S$ contre argmax. | `main:docs/research.md`, `main:docs/research-summary.md` |
| B.1, tableaux 4-5 | Audit QCSE complet ; pilote d'ordre, étiquettes mélangées, Qwen, balayage. | `ordering:reports/00-summary.md`, `01-…md`, `07-…md` ; `results/cluster-sweep/short-report.md` |
| B.2, tableau 6 | Prototypes à encodeur appris et attention quantique. | `main:docs/experiments/unsuccessful.md` §3-5 |
| B.2, tableau 7 | Campagnes exploratoires DGX. | `results/robust-timed-cv/`, `robust-screen-main/`, `robust-followup/`, `gradient-replicated-10324/`, `robust-causal-controls/` (README) ; `docs/runs.md` |
| B.2, tableau 8 et effets appariés | Comparaison hybride finale complète. | `results/robust-causal-controls/README.md` |
| B.3, tableaux 9-10 | Tatoeba et tâches jouets ; idées rejetées. | `main:docs/experiments/findings.md` §1-2 ; `unsuccessful.md` §10, 17, 20-27 |
| B.3, dernier § | Contexte long et échantillon généré. | `main:docs/experiments/coverage-seeds.md` ; `main:docs/research-summary.md` (graine 22) |
| C, tableaux 11-12 | Données puis entraînement de chaque étude. | `docs/protocol.md`, `results/*/README.md`, `findings.md` §1, `coverage-seeds.md`, `01-paper-audit.md` |

## Contrôles effectués (v1.2)

- Compilation `latexmk -pdf` : 0 erreur, 0 référence ou citation indéfinie.
- Rendu de toutes les pages relu visuellement.
- 294 nombres distincts de `main.tex` confrontés par script au corpus des sources : tous retrouvés, sauf artefacts (séparateurs de milliers, notation « .012 », largeurs de mise en page, « 85,8 % » écrit à la française) et « more than 500 », borne arrondie de 528.
- Paragraphes de prose de 90 mots ou plus relevés par script : il reste le résumé, l'introduction § 1 (91), le § 3 avec sa liste, « Hybrid benchmark » (106), « Hamiltonian predictors » (92), A.1 et A.3 (équations affichées). Aucun ne dépasse 110 mots de prose continue.
- Bibliographie : auteurs et titres relus sur les PDF de `Papers/` ; Quixer et Cerezo et al. vérifiés en ligne ; McClean, Schuld, Bengio, Devlin et Merity cités de mémoire. `schuld2021` est dans le `.bib` mais non cité.

## Historique des versions

- **v1 (16 novembre 2026)** : `versions/v1-2026-11-16/`, premier jet.
- **v1.1 (16 novembre 2026)** : `versions/v1.1-2026-11-16/`. Deux corrections de fond : origine des « 19 sur 58 » (portes diagonales finales de l'ansatz, pas la première $R_X$) ; plateau stérile, « does not explain » remplacé par « no evidence… not probed ».
- **v1.2 (16 novembre 2026, version courante)** :
  - travaux liés, métriques et campagnes DGX passés en tableaux (1, 2, 7) ;
  - configurations éclatées en deux tableaux à colonnes (11, 12) ;
  - autres résultats QCSE et prototypes en tableaux (5, 6) ;
  - contributions, audit QCSE et effets appariés en listes ;
  - paragraphes longs scindés (sémantique quantique, hamiltoniens, campagnes) ;
  - phrase sur les campagnes exploratoires DGX ;
  - ton : tournures toutes faites retirées (« remarkably well », « The common gap is », « To give the circuit a fair chance »), parcours réel raconté, phrases de longueur variée ;
  - précision : l'écart de 0,05 point au test tenu à part ne vaut qu'en gauche seule.

## Prochaines étapes

1. **Relecture par Ethan** du ton et de la phrase sur les campagnes DGX.
2. **Coauteurs** : confirmer le rôle de Malo Leroy et Thierry Zhang.
3. **Déclaration LLM** : garder, préciser ou retirer la phrase des remerciements.
4. **Corpus du balayage cluster** (« phrases, cleaned sentences ») : déduit du vocabulaire de 10 864 mots, à relire dans ses journaux.
5. **Références citées de mémoire** : revérifier McClean, Bengio, Devlin, Merity si le rapport est diffusé.
6. **Mise en page** :
   - le tableau 12 arrive seul en page 13 ;
   - le tableau 9 a des étiquettes multiligne légèrement décalées (tâches jouets) ;
   - à reprendre si une version imprimée est prévue.
7. À chaque modification :
   - copier l'état dans `versions/vN-date/` ;
   - recompiler et mesurer le corps ;
   - relancer le contrôle des nombres ;
   - mettre à jour ce miroir.
