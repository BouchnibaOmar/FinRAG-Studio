# Politique des données

Les rapports bancaires, extractions, preuves, questions et annotations sont des
données privées de recherche locale. Aucun droit de redistribution des documents
d’émetteurs n’est établi. La licence MIT du code ne couvre ni ces documents,
ni les modèles, ni les dépendances tierces.

| Emplacement | Contenu | Publication |
|---|---|---|
| `data/private/` | Rapports, extractions, annotations, références, modèles | Exclu |
| `artifacts/private/` | Prédictions, diagnostics, caches, réponses, consommations | Exclu |
| `models/`, `model_cache/`, `indexes/` | Poids et index locaux | Exclus |
| `.artifact-budget/` | Budgets personnels sans rapport avec FinRAG | Exclu |
| `data/public_examples/` | Exemples fictifs de contrats | Public après revue |
| `experiments/*/reports/` | Agrégats et diagnostics d’IDs/pages/rangs | Public après revue |
| `artifacts/prototype_baseline_reference.json` | Référence historique agrégée | Public ; pas une reproduction |

`.gitignore` protège aussi PDF, extractions JSONL, prédictions, vecteurs, poids
usuels, secrets Streamlit, clés privées et journaux. Il ne détecte pas tout texte
confidentiel et n’a aucun effet rétroactif sur un fichier déjà suivi. Ne pas
publier une archive complète du répertoire local.

## Provenance et interprétation

Un SHA-256 confirme l’accord des octets avec une référence, pas l’authenticité
ni la validité financière. Les pages physiques conservent les blancs et se
distinguent des numéros imprimés. Période, unité, devise et périmètre restent
à vérifier. Une revue assistée ne promeut pas une annotation candidate.

L’import conserve les snapshots autorisés et leurs empreintes. Les événements
personnels et identités des réviseurs ne sont pas importés. Les droits sur la
création des actifs hérités restent à clarifier avant redistribution.

## Développement, test et services externes

Dev et test sont distincts. Les expériences présentées évaluent dev seulement ;
le prototype avait déjà utilisé le test. Documents et certains faits sont
partagés : ce test ne doit pas être décrit comme inédit. La validation structurelle
historique est distincte de l’évaluation.

Recherche et lecteur restent locaux. Une génération explicitement demandée
transmet question et cinq passages à OpenRouter. Le catalogue peut être actualisé
par GET public sans texte documentaire. L’interface ne lance ni traduction ni
BGE automatique. Les diagnostics techniques excluent secrets et headers sensibles ;
les réponses brutes nécessaires à l’analyse restent privées. Les conditions du
fournisseur doivent être acceptées avant envoi.
