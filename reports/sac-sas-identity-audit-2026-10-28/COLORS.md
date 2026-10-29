# Correction des couleurs — 29 octobre 2026

Source : `runs/analysis/sac-sas-2026-10-28-identites-v2`.
Nouvelle sortie : `runs/analysis/sac-sas-2026-10-29-couleurs-v3`.
Suivi : http://127.0.0.1:8766/.
Journal : `runs/sac-sas-2026-10-29-couleurs-v3.log`.

## Périmètre

Cette passe réutilise les détections, rôles, trajectoires, poses, segmentations,
lectures PARSeq, ballon, projections et calibrations du run v2. Elle ne lance
aucun des six modèles, ne retire aucune détection et conserve les scores.
Les couleurs, identités et statistiques qui en dépendent sont recalculées.
Les calculs de tenseurs restent sous garde CUDA stricte, sans repli CPU.

## Couleurs adaptées au match

Les quantiles de luminosité des pixels neutres de deux zones segmentées du torse
sont réutilisés lorsqu’ils existent. La vidéo est décodée avec NVDEC pour
compléter les seuls quantiles manquants, sans inférence des modèles :

- 90 743 mesures de torse déjà disponibles réutilisées ;
- 118 649 observations réexaminées pour leurs mesures manquantes ;
- 38 662 rejets de qualité déjà connus conservés ;
- 198 416 observations avec deux zones exploitables au total ;
- préparation sur les 59 827 images en 253 secondes, avec zéro calcul de tenseur
  CPU ni repli. Audit dans
  `runs/recovery/sac-sas-2026-10-29-couleurs/features/features-audit.json`.

Les médianes et dispersions robustes des deux équipes sont estimées sur CUDA
à partir de numéros OCR uniques dans les effectifs, à deux chiffres, avec au
moins trois lectures fortes (≥ 0,95), au moins 90 % d’accord et une corroboration
par un cadrage large fondé sur la boîte. Les anciens votes d’équipe ne servent
pas d’étiquettes de calibration.

Les premiers essais utilisant aussi les numéros à un chiffre ne passaient pas
le contrôle d’accord. L’inspection a confirmé des joueurs en maillot sombre
dont l’OCR prétendait un numéro unique des Spurs. Un « 11 » ou « 34 » partiel
peut produire une hypothèse à un chiffre. Des cadrages de pose peuvent aussi
confondre des numéros à deux chiffres. Les ancres sans cadrage large corroborant
sont donc exclues de la calibration.

La calibration finale utilise 1 526 exemples issus de 132 groupes de
trajectoires, dont 330 exemples réservés à la validation. Une trajectoire entière
appartient au même côté de la séparation entraînement/validation. Sur les
330 exemples de contrôle, 317 sont acceptés et 316 concordent avec les étiquettes
OCR indépendantes. Cela mesure l’accord avec des ancres OCR, pas une précision
mesurée sur des annotations humaines du match.

La prédiction exige l’accord des deux zones, une séparation suffisante entre
classes et une distance bornée aux prototypes appris. Un consensus d’au moins
trois images échantillonnées peut prolonger la couleur sur les images de la
même trajectoire continue. Les interruptions et changements de couleur stables
séparent les épisodes. Les anciens votes sont réévalués : ils peuvent être
confirmés, remplacés ou effacés, plutôt que conservés par défaut.

## Contrôles avant lancement complet

176 tests pipeline passent :
`runs/review/sac-sas-2026-10-28/color-all-tests-validated.log`.
L’essai de bout en bout est
`runs/analysis/sac-sas-color-integration-100-v4` :

- 100 images, 755 détections de joueurs conservées, sans suppression ;
- 203 observations nommées avant, 216 après ;
- 258 observations avec couleur avant, 579 après ;
- 330 ajouts, un remplacement et neuf suppressions de votes de couleur ;
- horodatages, identités uniques, empreintes et statistiques validés ;
- empreinte de la géométrie et des détections strictement identique à la source.

Le contrôle d’intégrité du run complet compare la même empreinte pour les
59 827 images. Elle couvre notamment boîtes, scores, rôles, identifiants de
trajectoire, poses, masques, positions, ballon, terrain et métadonnées d’image.
Les identités et les couleurs sont les champs autorisés à changer.

Sur les 1 248 observations de joueurs conservées du passage de cinq secondes
déjà revu, la couleur d’équipe correcte est disponible sur 546 observations
avant, contre 1 221 après. Aucune couleur opposée à la référence revue n’est
observée dans cette comparaison ; 27 observations restent sans couleur.
Ce contrôle local utilise la référence existante, sans modifier les votes ni
les identités. Il ne constitue pas une mesure de précision sur le match complet.
Rapport : `runs/review/sac-sas-2026-10-28/color-reviewed-interval-check.json`.

Sur le match complet, les votes recalculés apportent une couleur d’équipe à
450 003 observations, contre 154 290 avant : 303 591 ajouts, 887 remplacements
et 7 878 anciens votes effacés. Parmi ces 450 003 observations, 185 303 ont
une classification acceptée sur l’image échantillonnée et 265 067 bénéficient
du consensus temporel. Toutes les 544 188 détections de joueurs sont conservées.
Ce taux décrit la disponibilité des équipes, pas celle des noms.

Le nombre d’épisodes ayant une majorité OCR mais aucune équipe établie passe
de 974 à 74. Le nombre total d’épisodes peut changer puisque les nouvelles
couleurs redéfinissent les frontières des séquences. Comparaison dans
`runs/review/sac-sas-2026-10-28/color-episode-comparison.json`.

## Résultat complet validé

Le run `sac-sas-2026-10-29-couleurs-v3` est terminé. Les 59 827 images, leurs
horodatages et les identifiants uniques sont vérifiés. L’empreinte de toute la
géométrie est identique à celle de la source :
`61eb700be113dcb725392722022a25f188b21af507a95eebdd23e794f1d340d4`.
Les 287 202 lectures OCR sont strictement identiques hors métadonnées de
filiation des runs ; voir `runs/review/sac-sas-2026-10-28/color-ocr-reuse-check.json`.
Les douze artefacts finaux ont été revérifiés par leur empreinte.

| Disponibilité sur les mêmes 544 188 observations | Avant | Après |
| --- | ---: | ---: |
| Équipe, mesures et consensus temporel | 154 290 (28,35 %) | 450 003 (82,69 %) |
| Nom | 203 351 (37,37 %) | 246 984 (45,39 %) |

Le gain de noms est de 43 633 observations, soit 8,02 points, avec un nombre
de détections inchangé. Ces taux mesurent la disponibilité, pas la justesse
des noms sur l’ensemble du match. Les conflits restants sont supprimés de
l’affichage des identités, et non publiés comme doublons d’un même joueur.

Le recalcul après préparation des mesures dure 905 secondes, dont 323 secondes
de statistiques. Avec les 253 secondes de préparation, le traitement représente
environ 19 min 18 s. Aucun modèle visuel ni OCR n’est relancé ; aucun calcul de
tenseur CPU ou repli CPU n’est enregistré.

Sur l’extrait de cinq secondes déjà revu, les 645 noms corrects disponibles
restent identiques avant/après, sans nom contraire à la référence dans les
observations comparées. La couleur y progresse fortement mais l’OCR demeure
limitant pour les noms manquants. Cette comparaison locale n’évalue pas la
justesse du reste du match. Vérifications complètes dans
`runs/review/sac-sas-2026-10-28/color-final-check.json`.

Le viewer a chargé le nouveau run et sa dernière image a été vérifiée :
http://127.0.0.1:8765/?run=sac-sas-2026-10-29-couleurs-v3.

## Reproduction

Utiliser des répertoires de sortie neufs. La commande unique complète uniquement
les mesures de couleur absentes, puis rejoue les étapes dépendantes :

```bash
.venv-players/bin/python -m pipeline.cuda.refine \
  --source runs/analysis/sac-sas-2026-10-28-identites-v2 \
  --output runs/analysis/nouvelle-correction-couleurs-sac-sas \
  --colors-only
```

Pour réutiliser les mesures déjà préparées pendant cette intervention :

```bash
.venv-players/bin/python -m pipeline.cuda.refine \
  --source runs/analysis/sac-sas-2026-10-28-identites-v2 \
  --output runs/analysis/nouvelle-correction-couleurs-sac-sas \
  --colors-only \
  --color-features runs/recovery/sac-sas-2026-10-29-couleurs/features
```

Les empreintes du run source, de ses observations, de ses lectures OCR et des
mesures de couleur sont contrôlées. Les contraintes de composition officielle
restent limitées à l’intervalle dont la synchronisation a déjà été vérifiée.
Le nouveau `color-audit.json` et le rapport `identity-quality.json` donnent les
mesures propres au résultat complet après succès ; les valeurs de l’essai de
100 images ne préjugent pas du gain global.
