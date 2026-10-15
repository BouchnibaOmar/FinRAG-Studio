# Reclassement des candidats BM25 sauvegardés

Cette expérience lit uniquement `cached_translation` dans
`artifacts/private/bm25_company_routing_dev/run_1/predictions_top50.jsonl`.
Ce JSONL contient aussi `original`, qui n'est ni utilisée ni mélangée au pool.
La préparation vérifie 28 questions dev répondables, au plus 50 candidats par
question et la couverture complète des extraits : 11/28 à @5 et 20/28 à @50.
Elle ne lance aucune recherche.

## Frontière de données

Les requêtes du reclasseur contiennent la question effectivement recherchée et
les candidats : identifiant opaque, texte exact, identité publique du rapport,
entreprise du catalogue, entité du rapport, langue et page physique source.
Cette page est une métadonnée documentaire, pas une page annotée pertinente.
La projection ne contient ni scores, ni rangs, ni identifiant de question,
réponse attendue, preuve, statut ou métrique. Les métadonnées des rapports
proviennent du catalogue public, pas des annotations.

Les IDs opaques sont les empreintes de l'identité document/chunk. Leur tri
lexicographique fixe l'ordre d'entrée indépendamment du rang BM25 et des
références. Cet ordre est enregistré. Aucun passage n'est modifié, tronqué ou
retiré pour adapter la requête au modèle. L'évaluateur conserve séparément la
correspondance vers les candidats historiques.

Le prompt opérationnel demande une permutation complète et privilégie question,
entreprise, périmètre, période, unités, devise et preuves complémentaires dans
les comparaisons. Il traite les passages comme des données non fiables et
demande d'ignorer leurs instructions. Cette consigne ne garantit pas à elle
seule l'immunité du modèle aux injections ; aucune commande documentaire n'est
exécutée et seuls les IDs autorisés peuvent devenir un résultat valide.

La sortie est `{"ordered_candidate_ids": [...]}`. Tous les IDs doivent être
présents une seule fois, sans autre champ. Réponse vide, JSON malformé, texte
formaté, champ dupliqué, ID inconnu, doublon, omission, refus, outil ou fin autre
que `stop` produit un échec. Aucun résultat n'est réparé ou complété automatiquement.

## Adaptateur et historique

Le reclasseur reprend les conventions de l'adaptateur de traduction et son
contrôle d'identifiant exact, mais conserve une frontière HTTP distincte : son
contrat de sortie est une permutation JSON, pas une question traduite.
La clé est lue uniquement dans `OPENROUTER_API_KEY`, jamais enregistrée.
Le modèle vient de `--model`, `OPENROUTER_RERANK_MODEL`, `OPENROUTER_MODEL` ou de
la configuration explicitement enregistrée, dans cet ordre. Aucun modèle n'est
choisi automatiquement. Aucun fallback de fournisseur ou retry n'est autorisé.

Les [sorties structurées OpenRouter](https://openrouter.ai/docs/guides/features/structured-outputs)
et [paramètres de routage](https://openrouter.ai/docs/guides/routing/provider-selection)
sont utilisés avec `require_parameters=true` et `allow_fallbacks=false`.
Le timeout est de 60 secondes ; la réponse est limitée à 1 MiB et un dépassement
est refusé. Température 0 est demandée par défaut, sans garantie de reproductibilité.
`--omit-temperature` permet de la retirer explicitement si nécessaire, après
vérification du modèle, en créant une autre identité de requête.

Une réservation et le payload précèdent chaque tentative. Un verrou empêche
les dépenses concurrentes. Les corps HTTP sont conservés en base64 dans le
cache privé, avec statut HTTP et latence ; aucun header d'authentification ni
contenu d'erreur réseau n'est enregistré. Les records valides conservent modèle
demandé/retourné, prompt, paramètres, ordre d'entrée, usage et coût disponible.
Le coût absent reste null. Les échecs restent dans le journal et dans le budget,
sans replay automatique. Une reprise réutilise les réussites. Les fichiers
historiques et les fichiers de cache existants ne sont pas écrasés.

## Dry-run et pilote manuel

Le dry-run construit les tailles des 28 requêtes sans connexion HTTP.
L'estimation réserve un token par octet UTF-8 du payload sérialisé et 512 tokens
de marge de framing. Elle inclut textes, métadonnées et schéma JSON, et est
conservative ; elle n'est ni le comptage natif du modèle ni un plafond monétaire
garanti. Le budget de sortie est de 2 048 tokens par requête. L'usage disponible
est conservé, et tout dépassement signalé bloque les nouvelles tentatives.

Le pilote contient les trois plus petites empreintes des entrées sans références.
Le choix précède tout appel et ne consulte ni score ni métrique. Les limites
par défaut sont 3 appels et 600 000 tokens réservés, échecs compris.
Le pilote vérifie format, exécution, latence, usage et coût fourni ; il ne permet
aucune annonce de gain global.

Avant le pilote, choisir un identifiant exact localement et vérifier le modèle
avec un nouveau snapshot officiel de moins de 24 heures : ID, contexte, limite
de sortie, paramètres structurés et tarifs. La commande `pricing` existante ne
génère pas de texte mais effectue une lecture du catalogue public ; elle n'a
pas été exécutée pendant cette préparation. Utiliser un nouveau nom de fichier
pour chaque snapshot et dry-run tarifé.

```powershell
if (-not $env:OPENROUTER_RERANK_MODEL) { throw 'Configurer un identifiant exact dans OPENROUTER_RERANK_MODEL.' }
$rerankModel = $env:OPENROUTER_RERANK_MODEL
& .\.venv\Scripts\python.exe -m finrag.evaluation.bm25_translation_experiment pricing --model $rerankModel --output artifacts/private/bm25_candidate_reranking_dev/model_snapshot.json
if ($LASTEXITCODE -ne 0) { throw 'Verification du modele impossible.' }
& .\.venv\Scripts\python.exe -m finrag.evaluation.candidate_reranking_experiment dry-run --experiment artifacts/private/bm25_candidate_reranking_dev --model $rerankModel --model-snapshot artifacts/private/bm25_candidate_reranking_dev/model_snapshot.json --output artifacts/private/bm25_candidate_reranking_dev/dry_run_priced.json --max-calls 3 --max-total-tokens 600000
if ($LASTEXITCODE -ne 0) { throw 'Dry-run impossible.' }
```

La sortie du dry-run indique le coût conditionnel et la compatibilité par question.
Le pilote refuse un contexte inconnu ou insuffisant et ne tronque pas les entrées :

```powershell
& .\.venv\Scripts\python.exe -m finrag.evaluation.candidate_reranking_experiment pilot --experiment artifacts/private/bm25_candidate_reranking_dev --model $rerankModel --model-snapshot artifacts/private/bm25_candidate_reranking_dev/model_snapshot.json --max-calls 3 --max-total-tokens 600000
```

Cette commande effectue des appels payants : elle est préparée, pas exécutée ici.
Un modèle compatible et une clé configurée localement sont nécessaires ; aucune
clé ne doit être collée dans une conversation.

## Évaluation des caches

L'évaluateur charge exclusivement dev via le garde existant, vérifie v1 contre
les empreintes du baseline et ne relance aucune recherche. Il réordonne les
hits existants et calcule Hit@1, Hit@5 et couverture complète des extraits @5.
Les gains/régressions sont par question. La couverture du pool @50 doit rester
identique : une permutation ne peut pas réparer une preuve absente des candidats.

Les erreurs API, erreurs de validation, caches manquants après succès apparent,
tentatives incertaines et questions non tentées sont distingués. Les échecs et
non-tentatives restent au dénominateur 28, comme misses d'exécution. Un échec
d'API n'est pas interprété comme une dégradation de la pertinence du modèle.
Le rapport indique que l'expérience est incomplète si toutes les questions
n'ont pas un ordre valide. Les scores d'un pilote ne sont pas un gain global.

```powershell
& .\.venv\Scripts\python.exe -m finrag.evaluation.candidate_reranking_experiment evaluate --experiment artifacts/private/bm25_candidate_reranking_dev --model $rerankModel --output artifacts/private/bm25_candidate_reranking_dev/evaluation_pilot_1
```

Cette commande est séparée, hors ligne et protégée contre l'écrasement. Son accès
aux fichiers dev privés devra être disponible. Sorties : `evaluation.json` et
`rapport.md`. Ni les pages alternatives proposées ni le split test ne sont utilisés.

## Profils explicites par modèle

`--reasoning-effort none|high` ajoute un paramètre de requête, sans modifier les
messages. Le snapshot doit annoncer l'effort choisi ; `none` est refusé si le
modèle impose le raisonnement. Ne pas confondre omission de ce paramètre avec
désactivation : certains modèles, dont Mistral Large 4, annoncent high par défaut.
`--max-completion-tokens` fixe explicitement la sortie commune au JSON et au
raisonnement éventuellement facturable. Une petite sortie JSON doit conserver
assez de budget visible ; les sorties tronquées restent des échecs.

`--cache-directory` permet au pilote et à l'évaluation de partager un nouveau
cache privé dédié à un modèle ou à une configuration. Les journaux d'un autre
cache restent intacts, avec leurs consommations historiques. Le payload, dont
modèle et effort, reste inclus dans l'identité de requête : les configurations
ne sont pas interchangeables. Les limites d'appels et tokens s'appliquent au
journal du cache explicitement sélectionné. Les anciens chemins et requêtes
sans réglage de raisonnement restent compatibles.
