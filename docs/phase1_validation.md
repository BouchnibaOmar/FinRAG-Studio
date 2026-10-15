# Validation historique de la Phase 1

Compte rendu du 13 octobre 2026, Windows x64 et CPython 3.12.14. Il décrit cette
phase au moment de son exécution, pas les fonctionnalités actuelles.

La Phase 1 a livré contrats typés, extraction par page physique, découpage en
caractères, import privé des annotations, validation et CLI.

| Contrôle historique | Résultat enregistré |
|---|---|
| Suite finale | 62 réussites, aucune erreur ni exclusion, 11,93 s |
| Ruff | Lint et format conformes, 25 fichiers Python |
| Dépendances | `pip check` sans conflit |
| Cas CLI synthétique | 3 pages physiques, 2 textuelles, 1 blanche, 4 passages |
| Import privé | Sept sources contrôlées par SHA-256, copies identiques |
| Validation de l’import | Extraits présents dans les pages fournies |

Il ne s’agit pas d’une nouvelle exécution sur les données privées. La
[revue actuelle](publication_review.md) donne les contrôles les plus récents.
L’import conserve snapshot, statuts et correction de formulation BNP documentée.
Les données et empreintes détaillées restent dans le manifeste privé. Voir
[la provenance](provenance.md).

Les fenêtres comptent des caractères Unicode, pas des tokens. OCR, interprétation
des tableaux et certification d’authenticité ne faisaient pas partie de cette
phase. La publication temporaire réduit les sorties partielles sans garantir
une transaction après coupure.
