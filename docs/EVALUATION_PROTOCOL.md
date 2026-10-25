# Protocole d'évaluation de la version POC

## Données

Conserver train, val et test par match ; auditer les doublons et l'historique des
poids importés. Val sert aux réglages et à la sélection. Le test final est réservé
à une recette gelée. Aucun nouveau match test n'a été ajouté lors de la livraison
technique du 18 octobre 2026.

Les scores joueurs actuels portent sur 401 images des premières secondes de deux
clips val. Publier ce support temporel et les comptes par match. Les séquences
densément voisines ne sont pas des exemples indépendants. Ne pas mélanger un
échantillon systématique avec des blocs denses dans un score global sans exposer
leur pondération.

## Détection

Comparer la même sélection exportée, les mêmes coordonnées source, matériel,
versions et précision FP32. Publier AP@50 et AP@50:95 par rôle, précision/rappel/F1
à score 0,25 et IoU 0,50, ainsi que les supports. Plusieurs chaînes emploient le
même détecteur arbitres E-BARD ; leurs scores arbitres ne sont pas des expériences
indépendantes. Voir `training/players/EVALUATION.md`.

Mesurer batch 1, cinq warmups et trois répétitions, avec synchronisation GPU.
Séparer adaptateur et décodage ; garder les temps bruts. Les mesures du forward
ne représentent pas la latence de la chaîne vidéo. Les allocations PyTorch ne
représentent pas toute la VRAM.

## Ballon

Publier seuil 0,5 et tolérance 15 pixels source pour le protocole courant.
Évaluer chaque frame une fois et conserver les erreurs importantes dans la
distribution des paires localisées. Une localisation trop éloignée produit un FP
et un FN. Rapporter séparément couverture et précision : les frames privées de
contexte ne sont pas de fausses réussites.

La politique `causal_last` ne lit pas de frame future. Fournir les coupes revues
par clip et conserver les redémarrages. Les exclusions d'images de l'export
limitent le contexte disponible. Les positions inconnues ne constituent pas des
absences certaines. Un benchmark spécifique d'absence/occlusion reste nécessaire.

## Tracking, identité, géométrie et distance

Utiliser un gold revu lié au hash vidéo et des identités humaines persistantes,
avec boîtes exhaustives sur les frames sélectionnées. Les segments gold peuvent
être indépendants des segments prédits ; les erreurs de coupe se mesurent lorsque
`scene_cut` a été explicitement revu. Des repères terrain `holdout: true` permettent
de contrôler une homographie avec des points qui n'ont pas servi à son ajustement.

Publier IDF1, changements d'ID et fragmentation, précision des identités acceptées,
couverture et abstentions sur cas connus. Distinguer les identités invérifiables
et les acceptations sur fausses détections. Préserver les chaînes `0` et `00`.

Publier erreur médiane/P95 de position au sol et couverture de projection. La
distance se compare sur les mêmes intervalles consécutifs valides, sans traverser
un changement de track ou de segment prédit. Publier couverture et erreurs par
trajectoire, y compris la référence nulle pour laquelle l'erreur relative reste
indisponible. Ne pas assimiler une référence manuelle à une mesure physique exacte.

## Décision

`configs/quality_gates.yaml` est une règle technique provisoire sur val : support
minimum de 100 images, baisse d'AP globale au plus 0,01, baisse de rappel globale
au plus 0,02, latence adaptateur P95 au plus 1,25 fois la baseline. Elle ne qualifie
pas une précision métier de distance ou d'identité. Avant une release finale,
figer les seuils métier et la baseline retenue sans exploiter les scores test.

Conserver commit et empreintes du code réellement exécuté, dépendances,
checkpoints, exports, sélection, configurations et protocole. La présence d'un
fichier de gel atteste la recette liée aux contenus ; elle ne prouve pas
l'absence de consultation antérieure du test.
