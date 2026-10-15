# Recherche BM25 par entreprise

`retrieval/company_router.py` reçoit le texte de la question et un catalogue
public. Il ne lit aucun fichier et n'importe pas les modèles d'évaluation.
`retrieval/routed_bm25.py` reçoit les passages, ce catalogue, la configuration
BM25 et, éventuellement, des traductions de questions sans annotations.
L'identifiant attendu et les preuves restent dans l'évaluateur.

Le catalogue `experiments/company_routing/corpus_catalog.json` reprend les
identités, fichiers et entreprises des `REPORTS` de `corpus/import_corpus.py`.
Les langues proviennent des noms de fichiers du corpus (`_en.pdf`, `_fr.pdf`),
pas du dataset. La recherche vérifie les identités du catalogue contre les
métadonnées des passages. Aucun contenu financier protégé n'est dans le catalogue.

## Règles visibles

La normalisation applique NFKC, casefold, NFKD, suppression des marques d'accent,
puis séparation de la ponctuation en mots alphanumériques. La correspondance
porte sur des séquences de mots entiers. Les formes reconnues sont BNP Paribas,
B.N.P. Paribas, Crédit Agricole et Société Générale, avec leurs variantes de
casse, accents et séparateurs. CA, SG, BNP seul et les variantes non documentées
ne sont pas reconnus. L'absence de nom reconnu conduit au corpus complet.

Une entreprise sélectionne tous ses rapports disponibles. Plusieurs entreprises
sélectionnent leur union. Un nom reconnu sans rapport disponible produit une
erreur explicite, pas un repli silencieux. Les années de la question ne filtrent
pas les documents. Le statut, les entreprises, les termes normalisés déclencheurs
et les identifiants des documents sont conservés.

Crédit Agricole est une famille de routage ; `reporting_entity` conserve
Crédit Agricole S.A. dans ce corpus. Sélectionner ce rapport pour une question
sur le Groupe ne prouve pas que ses chiffres ont le bon périmètre. Aucune
résolution financière ou juridique automatique n'est revendiquée.

## Statistiques et traductions

Le moteur conserve les 8 325 passages Phase 2B, le tokenizer
`unicode-financial-v1`, k1=1.2 et b=0.75. Chaque index BM25 calcule IDF et longueur
moyenne sur tous les passages de la route. Une route unique utilise donc les
statistiques documentaires de l'oracle ; une union utilise ses propres
statistiques, plutôt qu'une fusion de scores issus de plusieurs index.

Le texte original décide toujours la route. Le cache est consulté ensuite par
égalité exacte du texte original et par langue cible. Il ne fournit au moteur
ni question_id, ni document attendu, ni périmètre annoté. Une traduction absente
ou incompatible laisse la question originale, avec un statut explicite. Si les
documents sélectionnés ont plusieurs langues, la recherche conserve l'original
sur leur union, sans mélanger des scores de requêtes différentes.

L'évaluation réutilise le translation-set corrigé de `evaluation_run_2`, dont
certaines entrées sont des corrections linguistiques locales assistées. Le
routage est autonome ; cette expérience ne valide pas une traduction autonome
pour de nouvelles questions. Elle ne déclenche jamais une API.

## Évaluation dev

```powershell
& .\.venv\Scripts\python.exe -m finrag.evaluation.company_routing_experiment --catalogue experiments/company_routing/corpus_catalog.json --translation-set artifacts/private/bm25_query_translation_dev/translator/cache/sets/00bf6cd63861496ea3bae30f1858e7d7.json --output artifacts/private/bm25_company_routing_dev/run_1
```

Le dossier de sortie doit être nouveau. L'évaluateur vérifie les empreintes des
entrées, charge uniquement dev, exclut les deux questions non répondables et
compare les 28 questions aux oracles original et traduit existants. Les sources
figées et les records de traduction sont revérifiés avant publication.

Pour une route qui correspond exactement au document attendu et à la requête
oracle, le top 50 sauvegardé et toutes les métriques, y compris les rangs de
référence calculés au-delà de 50, doivent être identiques. Une différence bloque
la publication. L'identité de tout le classement au-delà des résultats conservés
par l'oracle n'est pas revendiquée.

Sorties privées : `predictions_top50.jsonl`, `comparison.json`, `rapport.md`.
Les métriques v1 séparent la présence d'une page et la couverture complète des
extraits. Les rangs sont calculés sur tous les candidats de score positif ; les
listes publiées sont limitées à 50. Les comparaisons par question portent sur A
contre l'oracle original, B contre l'oracle traduit et B contre A, sans promotion
de statut ou modification de référence.

## Validation

Les tests synthétiques couvrent noms, accents, casse, limites de mots, faux
positifs, abréviations exclues, unions, absence d'entreprise, années, périmètres,
traductions absentes/incompatibles et refus d'écrasement. Un scénario complet de
28 questions synthétiques contient volontairement une référence de document
incompatible avec le nom explicite : le moteur suit le texte, jamais cette
référence. Ces résultats synthétiques ne sont pas des scores dev.
