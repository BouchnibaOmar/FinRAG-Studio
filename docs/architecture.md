# Architecture de FinRAG Studio

Le projet conserve le lien entre un passage financier et sa page physique.
Recherche, évaluation, génération et présentation ont des responsabilités
séparées. La configuration est explicite et sérialisable ; aucun conteneur de
services ni fabrique générique de fournisseurs n’est nécessaire.

## Modules

| Module | Responsabilité |
|---|---|
| `domain/` | Contrats Pydantic stricts : source, page, passage, contexte financier et prédiction |
| `ingestion/` | Empreintes, extraction physique, lecture JSON/JSONL et publication contrôlée |
| `chunking/` | Fenêtres par page et enfants compatibles avec les tokens E5 |
| `corpus/` | Import privé et caches vérifiables |
| `retrieval/` | BM25, routage, E5, FAISS et reclassement de candidats |
| `evaluation/` | Chargement dev, références scellées, métriques et diagnostics |
| `generation/` | Contexte limité, requête OpenRouter, validation et cache privé |
| `ui/` | Trois espaces Streamlit, consentement et lecteur PDF autorisé |

## Décisions et compromis

| Problème | Solution retenue | Alternative écartée | Compromis |
|---|---|---|---|
| Une page blanche ne doit pas décaler les citations | Une entrée par page physique, depuis 1 | Numéroter seulement les pages textuelles | Fichiers plus longs, références stables |
| Un nom ne prouve pas l’identité du rapport | SHA-256 des octets originaux et nom sans chemin | Identité par nom ou date | Une modification d’octets change l’identité |
| Le texte financier doit rester traçable | Passages limités à une page, positions et empreintes | Concaténation entre pages | Certaines phrases peuvent être coupées |
| Les caractères ne sont pas des tokens | Comptage réel E5 avec préfixe et tokens spéciaux ; enfants déterministes | Troncature silencieuse | Plus de passages ; contrôle BM25 apparié |
| Des chiffres similaires désignent des faits différents | Période, devise, unité, périmètre et opérandes explicites | Notes non structurées | Certains champs requièrent encore une revue |
| Une interruption peut endommager les sorties | Validation temporaire, publication contrôlée, refus d’écrasement | Réécriture directe | Pas de transaction garantie contre une coupure |
| JSONL peut contenir des séparateurs Unicode dans les chaînes | Enregistrements séparés sur LF seulement | `str.splitlines()` | Convention stricte, texte conservé |
| Les expériences doivent garder leur identité | Empreintes des configurations, versions, entrées et sorties | Réutilisation par nom | Cache incohérent refusé explicitement |
| Le routage ne doit pas lire l’évaluation | Texte et catalogue, variantes de noms explicites | Document attendu ou règle par question | Sans nom reconnu, recherche sur tout le corpus |
| Nom commercial et périmètre juridique diffèrent | Famille d’entreprise distincte de l’entité du rapport | Assimiler Crédit Agricole S.A. au Groupe Crédit Agricole | Vérification du périmètre toujours nécessaire |
| Une année peut être comparative | Aucun filtrage automatique d’exercice | Déduire une année documentaire unique | Corpus parfois plus large |
| Une traduction peut modifier la question | Égalité exacte du texte original et de la langue cible | Cache approximatif ou API implicite | Une reformulation peut invalider le cache |
| BM25 dépend du corpus sélectionné | Statistiques sur les passages des documents routés | Comparer des scores d’index différents sans le signaler | Scores non universels, non probabilistes |
| BM25 et cosinus ont des échelles différentes | RRF fixe, constante 60, poids égaux, départage par ID | Somme de scores bruts | Pas de calibration apprise |
| Torch et FAISS chargent deux OpenMP incompatibles sous Windows | Processus distincts, NPY float32 et JSON empreinté | Ignorer le conflit ou remplacer des DLL | Démarrage supplémentaire ; `allow_pickle=False` |
| Un reclasseur ne doit pas changer son pool | Candidats scellés sans rangs/scores BM25 ni annotations ; permutation complète | Réparer une liste partielle | 49 IDs sur 50 restent un échec |
| BGE utilise un autre tokenizer | Compter les paires complètes avant chargement des poids | Supposer la compatibilité E5 | Précontrôle supplémentaire, aucune troncature |
| Un oracle utilise une information privilégiée | Étiquette diagnostique explicite, document annoté | Présenter le score comme du routage réel | Pas un score déployable |
| Une page peut ne pas porter toutes les preuves | Métriques de pages, extraits et couverture complète séparées | Un seul Hit@5 pour les comparaisons | Plusieurs mesures nécessaires |
| La génération doit rester indépendante des références | Question originale et cinq passages projetés sur les champs publics | Réponses et preuves annotées en contexte | Refus possible malgré une réponse ailleurs |
| Les citations recopiées peuvent perdre les espaces | Contrat v2 : unités consécutives de 600 caractères maximum, positions exactes et IDs | Normalisation approximative ou réparation | Une frontière peut couper une ligne ; toutes les unités sont disponibles |
| Le modèle peut inventer une page | Citations résolues depuis les métadonnées locales | Pages proposées par le modèle | Traçabilité contrôlée, justification financière non certifiée |
| Une comparaison exige deux valeurs | Gardes sur opérandes cités et valeurs littérales | Compléter depuis des connaissances générales | Détection lexicale et contrôles limités |
| Les reruns ne doivent pas répéter les appels | Clic, identité complète, réservation persistante et aucun retry | Appel pendant le rendu | Échecs conservés pour examen, facturation inconnue possible |
| Un contrat nouveau ne doit pas réinterpréter l’historique | Caches v1/v2 distincts ; prompt et paramètres dans l’identité | Promouvoir une ancienne réponse rejetée | Deux contrats maintenus ; l’interface utilise v2 |
| Les tarifs ne suffisent pas à compter les tokens d’entrée | Prix datés, réservation de sortie partielle et consentement | Ratio caractères/tokens ou prix absent égal à zéro | Coût total et taille du contexte non certifiés |
| Les capacités varient selon le fournisseur | Modèle exact, catalogue daté, paramètres requis, aucun fallback | Substitution silencieuse | Refus fournisseur encore possible |
| Les exemples peuvent sembler calculés instantanément | Espaces séparés et mention précalculée | BGE CPU à chaque interaction | Recherche libre BM25, exemples distincts |
| Les détails masquent la réponse | Paramètres techniques fermés, consentement visible, réponse avant sources | Retirer les contrôles ou la provenance | Détails accessibles après ouverture |
| Un lecteur peut exposer des fichiers arbitraires | Catalogue autorisé, confinement, empreinte et bornes physiques | Chemin libre ou lecteur externe | Mémoire PDF ; fragment de page dépendant du navigateur |

## Modules à savoir expliquer

- `domain/documents.py`, `ingestion/hashing.py`, `chunking/page_chunker.py` :
  identité des octets, page physique et positions exactes.
- `retrieval/bm25.py`, `company_router.py`, `routed_bm25.py` : tokenisation,
  statistiques documentaires et absence d’annotations dans le routage.
- `chunking/model_chunker.py`, `retrieval/e5.py`, `embedding_cache.py`,
  `faiss_process.py` : limites de tokens, pooling, normalisation et lignes du cache.
- `evaluation/document_oracle_metrics.py` : page retrouvée, extrait littéral et
  couverture complète de plusieurs preuves sont des mesures différentes.
- `generation/citation_units.py`, `cache.py`, `openrouter.py` : contrat v2,
  identité des requêtes, tentatives conservées et frontière réseau.
- `ui/resources.py`, `app.py`, `generation_panel.py` : classements sauvegardés,
  sources autorisées, consentement et état des reruns.

Les tests synthétiques associés montrent les invariants et leurs cas d’échec.
Les résultats historiques restent séparés des validations actuelles.
