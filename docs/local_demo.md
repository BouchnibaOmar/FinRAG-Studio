# Démonstration locale

Lancer depuis le terminal où `OPENROUTER_API_KEY` est configurée, sans afficher sa valeur :

```powershell
# Depuis le dossier FinRAG-Studio
& .\.venv\Scripts\python.exe -m streamlit run .\src\finrag\ui\app.py `
  --server.address 127.0.0.1 --server.port 8501 `
  --server.headless true --server.fileWatcherType none `
  --browser.gatherUsageStats false
```

Ouvrir <http://127.0.0.1:8501>. Arrêter l’ancien serveur avec Ctrl+C avant de
redémarrer. Aucun fichier `.env` n’est chargé. Sans clé, la recherche et les
exemples restent disponibles ; seules les réponses déjà en cache sont réutilisables.

## Parcours

- **Poser une question** : recherche locale par mots-clés (BM25), après clic
  sur Rechercher. La requête réellement utilisée, les entreprises et les documents
  sont consultables dans les détails. Aucune exécution de BGE dans ce mode.
- **Explorer les exemples** : les classements BM25 et BGE des 28 exemples dev
  sont lus depuis les fichiers existants. « Reclassement par pertinence » désigne
  ici BGE déjà exécuté. Ce parcours ne mesure pas une recherche nouvelle.
- **Comprendre le fonctionnement** : explications et limites, sans chargement
  du corpus ni requête réseau.

La réponse apparaît au-dessus des sources. Les aperçus sont identifiés comme
tels ; le texte original complet, l’identifiant et la provenance restent dans
les sections dépliables. Le bouton de consultation n’ouvre que les PDF du catalogue
local autorisé. Une page désigne toujours sa position physique dans le PDF.

Cinq sources sont affichées initialement. **Explorer davantage de passages**
permet d’en afficher plus sans changer les cinq passages envoyés pour générer.
**Détails de la recherche**, fermé initialement, contient la requête réellement
recherchée, les entreprises et la provenance du classement. Un exemple reste
explicitement signalé comme précalculé.

## Modèle et envoi

La barre latérale permet de rechercher un nom ou un identifiant exact.
`OPENROUTER_GENERATION_MODEL` sert uniquement de sélection initiale ; sa valeur
absente du catalogue ne provoque aucun remplacement automatique.

La navigation, le modèle et le budget estimé restent visibles. **Paramètres
avancés**, fermé initialement, regroupe limite de sortie, raisonnement,
capacités JSON, limites du modèle, identifiant complet, tarifs et date du
catalogue. Les choix de modèle, budget et classement sont conservés lors des
interactions visuelles et du passage par l’aide.

**Actualiser le catalogue** effectue seulement un GET public vers
`https://openrouter.ai/api/v1/models`, sans clé, question ou passage. Un snapshot
horodaté est ajouté dans `artifacts/private/local_ui/model_catalogues/`.
Au chargement et lors des reruns, seul le snapshot local est lu. Les informations
manquantes sont indiquées « non disponible ». Les prix `prompt` et `completion`,
en USD par token, sont multipliés par un million pour l’affichage ; les frais
par requête absents restent inconnus. Les tarifs du catalogue peuvent varier
selon le fournisseur et la tarification appliquée.

La génération exige une entrée et une sortie textuelles, les paramètres déclarés
`response_format`, `structured_outputs` et `max_tokens`, une limite de contexte
connue et une sortie compatible avec la limite déclarée. Ces déclarations ne
garantissent pas l’exécution ni l’acceptation de chaque contrainte JSON par un
fournisseur. `require_parameters=true` et `allow_fallbacks=false` restent actifs.
Le modèle retourné doit correspondre exactement au modèle demandé.

L’effort de raisonnement n’est proposé que parmi les `supported_efforts`
déclarés. Une liste nulle signifie les efforts normalisés exposés par OpenRouter ;
un champ absent ne permet pas de sélectionner un effort. `mandatory=true`
exclut `none`. Pour Mistral Large 4, `none` est présélectionné seulement si le
catalogue le déclare. Pour les autres modèles, le paramètre est initialement omis :
les valeurs par défaut du fournisseur s’appliquent et peuvent inclure du raisonnement.

La sortie est initialement limitée à 1 024 tokens ; sa limite est modifiable
jusqu’à 32 768 dans cette interface, sous réserve du modèle. Avant chaque envoi,
les cinq passages effectivement transmis et la question originale sont accessibles
dans les détails. La profondeur affichée des résultats ne change pas ce contexte.

## Coût

Le budget est un **budget estimé par appel**, pas un plafond de facturation.
La partie de sortie réservée utilise le tarif courant et la limite choisie.
Elle exclut les tokens d’entrée et les éventuels frais supplémentaires.
Si cette partie seule dépasse le budget, le bouton reste bloqué.

Aucun compteur de tokens compatible avec tous les modèles n’est installé ni
téléchargé. L’application n’utilise aucun ratio caractères/tokens. Le coût total
et l’occupation exacte de la fenêtre de contexte ne peuvent donc pas être
vérifiés avant l’appel. Une acceptation explicite de cette incertitude est exigée.
Le budget ne peut alors pas servir de contrôle du coût total. Le calcul pur
prend en charge un comptage compatible lorsqu’il est fourni, avec blocage si
l’estimation dépasse le budget ; l’interface ne fabrique pas ce comptage.

Le modèle, le nombre de passages et le coût connu ou inconnu sont affichés
près de **Générer une réponse**. Quand le coût total est inconnu, la case
**J’accepte le coût variable de cet appel OpenRouter.** est initialement
décochée, juste au-dessus du bouton. Le motif du blocage reste visible.
**Détails de l’envoi et du coût**, fermé initialement, conserve les prix,
la réservation partielle pour la sortie, les volumes et les paramètres.

Après un clic, un seul appel est possible pour une nouvelle empreinte. Aucun
retry automatique. Un échec conserve son marqueur et ne sera pas rejoué par un
clic supplémentaire. Les tokens, le coût déclaré, le modèle, le fournisseur et
la durée sont affichés lorsqu’ils sont disponibles ; une valeur absente ne vaut
jamais zéro. Pour une réponse en cache, la durée est celle de l’appel initial.

## Citations et historique

Le contrat de citation v2 est conservé : unités de 600 caractères Unicode au
maximum, positions exactes et références par IDs. Le texte original, y compris
espaces insécables et retours à la ligne, reste intact. Les citations sont
reconstruites côté application. Un identifiant valide ne garantit pas la
justification financière ; vérifier période, unité, devise et périmètre.

La version de prompt `financial-answer-citation-units-v2-concise-1` demande une
réponse directe et courte sans faits supplémentaires. Elle crée une nouvelle
identité dans le cache v2, comme tout changement du modèle, des paramètres ou du
contexte. Aucune ancienne réponse n’est raccourcie, réparée ou promue.

La clé reste uniquement en environnement. Aucun envoi de PDF, chemin local ou
annotation ; aucun appel de traduction ni reclassement à l’ouverture ou lors
d’un rerun. La génération transmet des textes à OpenRouter seulement après clic
et acceptation du coût incertain. Les droits d’accès aux fichiers restent inchangés.

Documentation officielle vérifiée pour cette intégration :
[catalogue et tarifs](https://openrouter.ai/docs/guides/overview/models),
[API du catalogue](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties),
[sorties structurées](https://openrouter.ai/docs/guides/features/structured-outputs),
[capacités de raisonnement](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).
