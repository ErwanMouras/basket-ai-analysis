# Registre des preuves pour la démonstration

Ce document sert à choisir des formulations défendables. Les anciens rapports
restent des instantanés de leurs dates ; le comportement actuel est celui du
viewer après la PR #8. Le manifeste généré lors des captures donne les valeurs
et hashes effectivement utilisés dans les médias.

| Affirmation utilisable | Preuve | Périmètre à conserver |
| --- | --- | --- |
| « J’ai construit un pipeline et un viewer pour explorer des vidéos de basket. » | [Pipeline](../../pipeline/README.md), [viewer](../../viewer/README.md), code et artefacts du run | Prototype d’analyse, sans promesse de performance sur tout match |
| « Le tir à 16,80 s, frame 504, ouvre une relecture avant le départ. » | `/api/shots` du run local | Proposition probable ; destination de relecture : 16,20 s |
| « Les candidats sont distingués des probables. » | `viewer/shots.js`, `viewer/team-shots.js`, `viewer/shot_analytics.py` et tests | Deux statuts heuristiques, pas des probabilités calibrées |
| « Les cartes sont issues des observations de l’extrait. » | [Guide du viewer](../../viewer/README.md), `/api/heatmaps`, `shot_maps.scope` | Présence filtrée par situation / quart-temps ; tirs sur l’extrait entier |
| « Les compteurs de tirs joueurs conservent seulement les probables. » | `summarize_shots`, tests de `viewer.test_shot_analytics` | Les candidats sont inclus dans la carte d’équipe, pas dans ces compteurs |
| « Ce profil a traité 1 842 frames en 318,49 s. » | `runs/analysis/hou-lal-sam-yolo26l-ball/run.json`, `timings.processing_s` | 320,94 s total, dont 2,40 s chargement ; environ 5,78 images/s de traitement. Run de développement sur RTX 5080 |
| « Le candidat ballon atteint un F1 de localisation de 0,9451 dans cette validation. » | [Évaluation ballon](../../training/ball/EVALUATION.md) | 2 627 frames communes de deux matchs de validation, tolérance 15 px source. Sélection de checkpoint sur validation |
| « Données versionnées, expériences suivies, promotion contrôlée et rollback. » | [guide de release](../EVALUATION_AND_RELEASE.md) | Exercice de release exécuté pour les modèles de personnes ; ne pas généraliser un registre complet à tous les adaptateurs |
| « La version de la PR #8 passe les suites de régression et le navigateur. » | [PR #8](https://github.com/ErwanMouras/basket-ai-analysis/pull/8), workflow POC validation | En local : 500 tests Python recensés, 28 ignorés, 472 réussis ; 30 tests JavaScript, six scénarios Chromium. Pas une certification tous navigateurs |

## Valeurs attendues dans les captures réelles

Run : `hou-lal-sam-yolo26l-ball`.
Source SHA-256 : `e841e7d9c584884d8db0dc7202795f590560f3de159c5a104e01ebadeadde8b9`.
Observations SHA-256 de la revue : `53462a1d409f2825b7c1ff31db00f8ae9c39be43c6853813c01439813ad17881`.
Le script recalcule ces hashes avant d’enregistrer le manifeste.

Le détecteur propose cinq probables et quatre candidats sur cet extrait. Parmi
les événements attribués et localisables, LAL a trois probables et un candidat,
HOU un probable et un candidat. Trois événements n’ont pas d’équipe attribuée.
Ces nombres sont des sorties du détecteur sur l’extrait, pas des statistiques NBA.
Le texte promotionnel n’a pas besoin de les utiliser ; les cartes les rendent
inspectables.

## Formulations à ne pas employer

| Formulation trompeuse | Formulation exacte |
| --- | --- |
| « 94,5 % de précision sur le basket » | « F1 de localisation ballon 0,9451 dans le protocole de validation publié » |
| « L’analyse tourne à 30 FPS » | « La lecture utilise des résultats calculés en amont ; le profil montré traite environ 5,78 images/s » |
| « Il détecte tous les tirs / mesure le taux de réussite » | « Il propose des tentatives probables et à vérifier ; marqué/raté reste inconnu » |
| « Distance parcourue sur le match » | « Distance observée sur les intervalles valides de l’extrait » |
| « Tout le système est entraîné de zéro » | « Poids existants, adaptations, entraînements et intégration sont documentés » |
| « Le sample public prouve les performances » | « Le sample synthétique permet d’essayer le viewer » |
| « La calibration est correcte sur 84 % des frames » | « La calibration est disponible sur cette fraction du run ; sa disponibilité ne mesure pas son erreur » |

## Ce qu’un recruteur doit pouvoir approfondir

Pour chaque brique, être prêt à expliquer son entrée, sa sortie, son mode
d’échec et le test associé. Priorités : synchronisation vidéo/masque, comparabilité
des évaluations ballon, abstention sur l’identité, validité de la projection et
distinction entre disponibilité, couverture et exactitude.

La prochaine preuve forte serait une évaluation figée sur de nouveaux matchs,
avec annotations de suivi, identité et géométrie. C’est une suite du projet,
pas une propriété déjà validée de la démo.
