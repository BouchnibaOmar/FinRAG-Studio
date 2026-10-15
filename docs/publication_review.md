# Revue avant publication

Revue du 15 octobre 2026. Aucun staging, commit, push, changement de remote ou
création de dépôt. Aucun CV préparé ou modifié. La publication reste manuelle.

## Changements

- README réécrit en français : problème financier, fonctions disponibles,
  diagramme, installation, lancement, modèles, coûts, résultats et limites.
- Guides publics et synthèses historiques en français, avec suppression des
  chemins personnels et des transcriptions devenues inutiles. Les versions
  antérieures sont archivées en privé, sans perte de l’historique local.
- Aides et messages d’enveloppe de la CLI principale en français, identifiants
  inchangés. Le détail original des exceptions est conservé.
- Deux tests d’oracle corrigés : ils créent leur projet temporaire et ses racines
  privées plutôt que de dépendre du dossier temporaire global du dépôt. Un test
  supplémentaire confirme le rejet d’une sortie hors des racines privées.
- `.gitignore` complété pour le dossier de budgets personnels, secrets Streamlit,
  clés privées, vecteurs, poids usuels et journaux. Aucun fichier privé supprimé.

## Revue du code

La lecture a porté sur contrats, sérialisation, BM25, routage, chargement des
exemples, contrôle PDF, génération, citations, cache et métriques. Les frontières
sont explicites : domaine, disque, réseau, évaluation et présentation. Les noms
du code sont conservés en anglais ; aucune abstraction ou réorganisation n’a
été ajoutée. Aucun défaut de classement ou de génération n’a été démontré par
cette revue, donc aucun correctif du moteur n’a été introduit.

Les validations répétées aux frontières servent des entrées différentes : cache,
requête fournisseur et métadonnées locales. Elles n’ont pas été fusionnées
artificiellement. Les commentaires utiles expliquent provenance, atomicité et
isolation ; aucune réécriture cosmétique du code n’a été faite.

Les attributions réelles sont conservées dans [la provenance](provenance.md),
la licence MIT reste inchangée et les licences tierces restent distinctes.

## Contrôles exécutés

| Contrôle | Résultat |
|---|---|
| Suite complète sur données synthétiques | **383 réussites, 35,78 s** |
| Ruff `check src tests` | Conforme |
| Ruff `format --check src tests` | 111 fichiers conformes |
| `pip check` | Aucun conflit déclaré |
| Création d’un environnement local de contrôle | Conforme, sans installation de paquets |
| Installation `.[ui,dev]` | Résolution en dry-run, hors ligne, dépendances déjà présentes ; aucune réinstallation |
| Aide CLI et options Streamlit | Vérifiées ; aide Streamlit avec `-X utf8` pour la sortie capturée Windows |
| Démarrage local Streamlit | Santé HTTP vérifiée en quelques secondes, sans clé ; serveur arrêté |
| Exclusions de publication | 15 chemins représentatifs exclus, six chemins publics conservés |
| Scan des fichiers publics candidats | Aucun motif de secret, chemin personnel ou bannière artificielle détecté dans les catégories contrôlées |

Les commandes réutilisables sont dans le [README](../README.md). Une installation
complète sur machine vierge, les téléchargements de dépendances et une facturation
OpenRouter réelle n’ont pas été testés pendant cette revue. Le serveur de contrôle
n’avait aucun navigateur connecté : ce contrôle vérifie le lancement, pas une
nouvelle inspection visuelle. Les états UI restent couverts par les tests hors
ligne et la vérification synthétique de l’étape précédente.

Les essais initiaux sont conservés dans le rapport privé : une erreur Windows de
renommage lors du premier lancement sous OneDrive ; puis deux tests dépendant de
la racine privée lors du passage au dossier temporaire externe. Les tests ont
été corrigés sans modifier les permissions ou les contrôles de production.
Le lanceur `py -3.12` ne trouvait pas de Python : le README demande désormais
un exécutable installé explicitement, sans chemin propre à une machine.

## Points restant à régler avant publication

- Le scan porte sur les fichiers de travail non ignorés, pas sur tous les objets
  d’un historique Git ni sur une archive complète. Un fichier déjà suivi n’est
  pas protégé rétroactivement par `.gitignore`. Une revue finale du contenu
  effectivement choisi pour publication reste nécessaire.
- Ne pas publier données privées, PDF, annotations, questions, modèles, index,
  caches, réponses brutes, journaux ou budgets personnels. Le scan de motifs ne
  prouve pas l’absence de tout secret ou de tout texte confidentiel arbitraire.
- Les droits de redistribution des actifs financiers et d’évaluation hérités
  ne sont pas établis. Conserver ces actifs privés ; ne pas présenter les
  annotations candidates comme entièrement vérifiées.
- Les fichiers privés requis par l’interface ne sont pas fournis dans une copie
  publique. Les tests synthétiques fonctionnent séparément ; une démonstration
  financière externe exige une préparation de données autorisées.
- Les prix du catalogue vieillissent et le coût total peut rester inconnu.
  Aucun prix actuel ni plafond de facturation n’est promis.
- Les dépendances directes sont fixées, mais aucun verrou complet de toutes les
  roues et dépendances transitives pour chaque plateforme n’est livré.

Les moteurs, prompts, unités de citation, références et résultats historiques
n’ont pas été modifiés. Aucun appel API, téléchargement de modèle, inférence
longue ou accès au split test réel n’a eu lieu pendant cette revue.
