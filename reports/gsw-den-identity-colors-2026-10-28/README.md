# Couleurs et identités GSW–DEN — 28 octobre 2026

Run : `runs/analysis/gsw-den-2026-10-26-full`, 57 861 frames, 25 fps.
Les preuves détaillées, crops, captures et sauvegardes restent dans
`runs/audit-gsw-den-2026-10-28/` (fichiers locaux ignorés par Git).

## Résultat appliqué au viewer

| Mesure | Avant | Après |
| --- | ---: | ---: |
| Observations de joueurs nommées | 122 362 | 133 850 |
| Disponibilité d'une identité nommée | 22,21 % | 24,30 % |
| Tracks comportant au moins une observation nommée | 783 | 882 |
| Observations avec équipe seule | 102 812 | 98 314 |
| Observations sans équipe ni nom | 325 655 | 318 665 |
| Fragments de tracks | 6 433 | 6 433 |
| Observations sans ID de tracking | 52 691 | 52 691 |

Le gain net est de **11 488 observations nommées** (+9,39 % par rapport au
nombre initial), soit +2,09 points de disponibilité. Il comprend 12 194
observations nouvellement nommées et 706 observations redevenues anonymes
lorsque les nouvelles preuves créent un doublon d'identité simultané.
Le garde-fou existant conserve ces conflits anonymes. Aucun nom existant
n'est remplacé par un autre et aucune track auparavant nommée ne perd toutes
ses observations nommées. **99 tracks supplémentaires** ont un nom observable.
Les tracks sont des fragments, pas des joueurs uniques.

Ces chiffres mesurent la disponibilité, **pas l'exactitude des identités**.
La confiance OCR est un score du modèle, pas une probabilité de justesse validée
sur ce match. Les cas récupérés ont été échantillonnés visuellement ; cela ne
constitue pas une annotation exhaustive du match.

## Couleurs des bbox et segmentations

Les deux couleurs du roster étaient `#1D428A` et `#0E2240`, deux bleus foncés.
Lorsque la distance RGB des deux couleurs est inférieure à 120, les overlays
utilisent la paire stable **GSW cyan `#45C7F5` / Denver rose `#F279C8`**.
Le serveur recalcule cette paire pour les anciens runs ; le client prend cette
palette pour les boîtes et les masques, y compris pour les joueurs dont le
numéro est temporairement supprimé mais dont l'équipe est établie.
Une identité non établie conserve la couleur neutre.

Les champs `color` et `uniform_color` du roster, ainsi que les preuves de
couleur utilisées par l'association, restent intacts.

## Causes des identités manquantes

Le relevé initial montre plusieurs limites qui se cumulent :

- **5 097 / 6 433 tracks** n'ont pas de majorité numérique suffisante.
- **3 646 tracks** n'ont aucune couleur de torse classifiable ; les maillots
  multicolores, la peau, les angles et les petits crops limitent ce classifieur.
- **2 311 tracks** durent moins de quinze images ; les 320 changements de plan
  et les interruptions du tracker fragmentent fortement les trajectoires.
- Sur 128 195 lectures finales v4, seules 6 615 sont acceptées initialement :
  105 341 sont rejetées pour faible confiance et 16 239 pour numéro hors effectif.
- Les 52 691 détections sans ID de track ne peuvent pas être nommées par une
  majorité temporelle de track.

Une cause corrigeable a été reproduite à **30 s, frame 750, segment 4,
track 95** : la bbox d'un autre joueur recouvre entièrement le rectangle du
numéro de Jamal Murray. Son masque n'en occupe aucun pixel. L'ancien filtre
rejette donc le crop alors que PARSeq lit bien « 27 » sur GPU.

## Corrections de l'association

### Mesurer l'occultation avec les masques

`training/jersey/crops.py` mesure la surface du crop OCR occupée par les masques
des autres joueurs lorsque leurs segmentations sont disponibles et que le
masque cible couvre au moins 50 % du crop. La bbox reste le repli pour chaque
voisin dont le masque est indisponible, ou si le masque cible est insuffisant.
Le seuil d'occultation reste 40 %. Des masques visibles disjoints ne font pas
ignorer une vraie occultation : la surface du **crop** occupée par l'autre
joueur est mesurée, même si les deux masques ne se chevauchent pas.
Les exigences de netteté et de contraste restent appliquées aux pixels source.

### Numéro unique avec preuves de couleur rares

La majorité habituelle exige au moins cinq votes de couleur. Un cas strict
supplémentaire accepte un numéro unique dans l'effectif éligible si :

- sa majorité numérique atteint 90 % ;
- au moins trois frames distinctes donnent ce numéro avec confiance ≥ 0,95,
  sur un intervalle d'au moins quatre frames ;
- au moins un vote de couleur correspond à son équipe, sans aucun vote adverse ;
- aucune lecture à confiance ≥ 0,95 ne donne un numéro concurrent.

Ces associations portent `resolution_method: unique_number_corroborated`.
Un numéro partagé, des couleurs contradictoires ou l'absence totale de preuve
de couleur bloquent ce mécanisme. Les collisions et les restrictions des
liaisons spatiales restent actives.

### Récupération ciblée sur GPU

`pipeline/identity_recovery.py` réutilise les détections et masques enregistrés.
Les 783 tracks déjà nommées sont exclues de la nouvelle inférence. Chaque track
non nommée d'au moins quinze frames reçoit douze échantillons répartis sur sa
durée, plus les frames à ±2, ±4, ±8 et ±16 des lectures numériques fiables.
Seuls les crops nouvellement débloqués par le filtre de masques sont relus.
Une seule lecture est conservée par track et frame, même avec trois variantes
du crop. Une ancienne lecture v4 n'est remplacée que par une lecture plus
confiante de la même frame ; les autres preuves historiques sont conservées.

La passe finale utilise **PARSeq FP32 sur `cuda:0`, RTX 5080**, sans repli CPU :
**16 657 crops**, 7 231 frames de track relues et 1 081 lectures répondant au
seuil de vote. Sa durée est de 128,13 s après préparation des échantillons.
Le décodage vidéo, les opérations sur les pixels et le rejeu JSON/statistiques
restent des traitements CPU ; toute l'inférence OCR de cette récupération
s'exécute sur CUDA. `audit.json` enregistre `device: cuda:0` et
`fallback_reads: {}`. Le SHA-256 des poids PARSeq reste
`bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`.

Le numéro 27 de Murray est maintenant associé sur la track 4:95, notamment à
30 s. Les informations de récupération sont enregistrées dans
`run.json.identity_resolution.recovery`.

## Application et contrôles

Le rejeu a été calculé dans `staged-run/`, puis comparé frame par frame avec le
run original. Les bbox, poses, masques, IDs, segments et nombres de détections
sont identiques. Les observations, votes, distances, statistiques, palette,
rapport de santé et hashes des artefacts ont été mis à jour ensemble.
Le contrôle de santé valide les 57 861 frames et leurs contrats.
La vidéo source, sa copie de lecture, le scoreboard et le roster sont conservés.
La sauvegarde précédente est dans `backup-before-apply/`.

Vérifications automatisées :

- 72 tests pipeline ;
- 34 tests jersey, dont un test optionnel ignoré ;
- 19 tests serveur ;
- 31 tests JavaScript ;
- six suites Playwright : navigation, heatmaps, scoreboard, tirs, cartes de tirs
  et heatmaps d'équipe ;
- contrôle du vrai run à 30 s : couleurs, association de Murray, métadonnées
  CUDA, modes bbox + segmentation et segmentation seule, absence d'erreur JS.

La majorité des observations reste anonyme. Lever cette limite demande des
preuves supplémentaires sur les maillots peu lisibles et la continuité des
tracks ; ce correctif conserve les refus lorsque les preuves restent ambiguës.

## Reproduire la récupération

Depuis la racine, dans un nouveau dossier de sortie :

```bash
.venv-players/bin/python -m pipeline.identity_recovery \
  runs/analysis/gsw-den-2026-10-26-full runs/nouvelle-recovery-gsw-den
```

La commande écrit uniquement les preuves dans ce nouveau dossier. Elle échoue
si CUDA est indisponible. Le run déjà corrigé a davantage de tracks nommées :
un nouveau passage les exclura, donc il ne reproduira pas exactement les
compteurs de la passe initiale. Pour les retrouver, utiliser le run sauvegardé
et sa configuration d'origine.
