# Recherche dense et hybride fixe de la Phase 2B

Le protocole est dans `protocol.json`. Modèle unique :
`intfloat/multilingual-e5-small`, MIT, révision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`. Vecteurs 384 dimensions, préfixes
`query: ` et `passage: ` dans les deux langues, moyenne masquée et normalisation
float32 L2. Inférence locale hors ligne après préparation des actifs.

Chaque passage B est compté avec préfixe et tokens spéciaux. Les dépassements
de 512 tokens produisent des enfants déterministes avec offsets, texte et page
conservés. Aucune troncature. Les 8 325 enfants servent au contrôle BM25 comme
à dense et hybride ; B original reste distinct.

FAISS IndexFlatIP score les vecteurs normalisés. Toutes les lignes sont scorées
avant départage stable par ID, y compris au seuil 50. L’hybride somme
`1 / (60 + rang)` à poids égaux pour au plus 50 candidats uniques par composant.
Ni annotations, ni document attendu, ni preuves n’entrent dans ces composants.

## Préparation et reprise

Les dépendances sont fixées dans l’extra `retrieval`. Une installation CUDA est
spécifique au matériel et ne doit pas remplacer automatiquement l’environnement
fonctionnel. Les mesures historiques, limites et commandes sont dans
[la validation](../../docs/phase2b_validation.md).

Les checkpoints enregistrent empreintes, lignes, dimensions et normalisation.
Une incohérence ou un batch non publié requiert inspection ; aucun écrasement
silencieux. Si l’estimation dépasse cinq minutes, l’exécution est manuelle.
Les 30 requêtes dev sont classées ; les deux non répondables sont exclues des
métriques de pertinence. Aucun test n’est évalué par ce protocole.

## Isolation Windows

Torch et FAISS chargent des runtimes OpenMP incompatibles dans les versions
inspectées. Le parent conserve Torch ; un worker frais exécute FAISS sans Torch.
Les échanges sont des tableaux NPY float32 contrôlés et JSON empreinté, sans
pickle. Aucun contournement OpenMP, remplacement de DLL ou changement global de
PATH. Voir [la validation OpenMP](../../docs/phase2b_openmp_validation.md).

Les sorties privées sous `artifacts/private/phase2b/` contiennent texte, vecteurs,
checkpoints et diagnostics. `reports/dev_results.json` contient seulement les
agrégats et IDs. La présence littérale d’un extrait n’est pas une précision de
réponse générée. Les résultats terminés restent BM25 9/28, E5 1/28 et hybride
3/28 en Hit@5, conservés sans nouvelle exécution.
