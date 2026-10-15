# Diagnostic historique de la recherche dense

Scores dev conservés, Hit@5 : BM25 apparié 9/28, E5 1/28, hybride 3/28. Aucun
défaut de production n’a été démontré par les contrôles suivants.

- Préfixes, moyenne masquée, mode évaluation et normalisation vérifiés ; écart
  maximal avec un pooling indépendant : 1,49e-8.
- Douze passages et six questions déterministes ; leurs batchs d’origine ont
  produit 21 vecteurs de passage et 22 de requête frais, identiques aux caches.
  Ce contrôle ne réencode pas tout le corpus.
- Checkpoints, métadonnées, échange FAISS et 8 325 lignes d’index identiques.
- Douze passages recherchés par leur propre vecteur : propre ID au rang 1 dans
  12/12 cas, égalités de contenu autorisées.
- Relecture des top 50 d’échange et top 10 dense des 30 requêtes : mêmes scores et IDs.
- Six paires fictives FR/EN : passage prévu premier dans 5/6 cas, deuxième dans
  le dernier. Ce petit contrôle n’est pas un benchmark général.

Les 32 extraits v1 étaient présents sur les pages annotées et dans au moins un
enfant complet. Aucune perte ou fragmentation d’extrait n’a été trouvée.

Parmi 27 échecs à @5, le meilleur passage de preuve a un rang minimal de 8,
médian de 289 et maximal de 4 031 ; 24/27 sont au-delà de 50. Le premier résultat
vient d’un autre document dans 11/27 cas. Les similarités ne sont pas des
probabilités et ces constats n’établissent pas une cause linguistique ou tabulaire.

Les références restent candidates. Pour 12 questions, au moins un extrait se
répète sur d’autres pages ; cette présence ne valide pas une référence alternative.
Pour `finrag_v1_casa_11`, une preuve partielle ailleurs ne remplace pas les deux
opérandes nécessaires. Aucun échec n’a été transformé en succès.

Agrégats et IDs : `experiments/phase2b/reports/dense_diagnostics.json`. Textes et
comparaisons : `artifacts/private/phase2b/dense_audit/`. Les 4 210 fichiers protégés
du diagnostic historique sont restés inchangés.
