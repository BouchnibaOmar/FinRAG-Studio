# Audit historique de DocsGPT-Custom

Synthèse de l’inspection locale en lecture seule du 13 octobre 2026. Aucun module
du prototype n’a été importé ou exécuté pour cet audit ; ses constats ne décrivent
pas les fonctionnalités actuelles de FinRAG Studio.

Le prototype associait application générique DocsGPT, adaptations financières
et workflow distinct d’évaluation. Son historique établit une dérivation
d’arc53/DocsGPT sous MIT, avec copyright arc53 de 2023. Cela n’établit pas les
droits sur les rapports ni sur tous les actifs d’évaluation non suivis.

Son architecture utilisait React/TypeScript, Flask, MongoDB, Redis et Celery.
Elle n’a pas été copiée dans FinRAG. Les actifs retenus pour l’import contrôlé
sont provenance du corpus, annotations, correspondances de pages, configuration
figée et résultats historiques.

Les empreintes des rapports, jeux de données, passages, configuration, manifeste
et modèles inspectés correspondaient aux références locales. Cet accord ne
prouve ni authenticité officielle, ni approbation humaine, ni reproductibilité.

La référence MPNet est conservée séparément dans
`artifacts/prototype_baseline_reference.json`. Extraction et segmentation
diffèrent de FinRAG : aucune équivalence expérimentale ni précision de génération
n’est déduite. Aucun nouveau score test n’est produit ici.

L’audit a trouvé 87 méthodes de test statiques mais uniquement un compte rendu
antérieur de 80 réussites. Compter des méthodes ne prouve pas leur exécution ;
87 tests réussis n’ont pas été établis.

Les pages physiques doivent survivre au découpage. Un aperçu tronqué n’est pas
la prédiction complète utilisée pour évaluer. Une taille en caractères n’est pas
une taille en tokens. Statut d’annotation et droits de redistribution restent
distincts de l’intégrité technique.

Les snapshots privés approuvés ont été importés en Phase 1 ; les événements
personnels de revue ne l’ont pas été. Voir [provenance](provenance.md) et
[politique des données](data_policy.md). Les chemins personnels et transcriptions
d’opérations sont absents de cette synthèse ; les documents antérieurs sont
conservés localement.
