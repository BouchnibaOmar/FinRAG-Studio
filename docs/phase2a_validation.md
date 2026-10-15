# Validation historique de la Phase 2A

Mesures du 13 octobre 2026, Windows x64 et CPython 3.12.14. BM25 et l’évaluation
dev étaient alors les nouvelles fonctions ; le README décrit l’état actuel.

## Corpus

| Rapport | Pages physiques | Pages textuelles | Pages blanches |
|---|---:|---:|---:|
| BNP | 936 | 934 | 2 |
| Crédit Agricole S.A. | 1 082 | 1 081 | 1 |
| Société Générale | 716 | 714 | 2 |
| Total | 2 734 | 2 729 | 5 |

Les extractions natives conservent toutes les positions, sans certifier la
structure des tableaux ni l’absence de zones scannées. Les PDF originaux ont
été copiés avec contrôle d’empreinte, sans importer les anciennes extractions.

| Configuration | Maximum / chevauchement, caractères | Passages |
|---|---:|---:|
| A | 1 000 / 100 | 13 178 |
| B | 2 000 / 200 | 7 141 |
| C | 3 000 / 300 | 5 108 |

Un défaut JSONL a été corrigé : `str.splitlines()` séparait les caractères
Unicode de paragraphes à l’intérieur des chaînes. Le lecteur délimite désormais
les enregistrements sur LF et conserve le texte.

## Résultats dev

Sur 28 questions répondables ; les deux non répondables sont séparées.

| Configuration | Hit@1 | Hit@5 | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|
| A | 1/28 | 8/28 | 9/28 | 0,130159 |
| B | 1/28 | 9/28 | 10/28 | 0,144388 |
| C | 3/28 | 8/28 | 9/28 | 0,174107 |

La politique fixée avant l’expérience privilégie Hit@5, puis MRR@10, Hit@1,
moins de passages et moins de chevauchement. Elle sélectionne B sans consulter
le test. Agrégats : `experiments/phase2a/reports/dev_results.json`.

Pour B : 19 échecs à @5, dont 16 sans le bon document et trois avec le document
mais pas la page annotée. Ces constats ne prouvent pas une cause linguistique
ou tabulaire. La Phase 2B utilise un contrôle apparié de 8 325 passages, distinct
de ces 7 141 fenêtres.
