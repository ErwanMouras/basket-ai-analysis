# Calibration NBA et positions sur le terrain

Le module ajoute au pipeline joueurs une homographie par image, une projection
des joueurs en mètres, les lignes NBA superposées à la vidéo et une vidéo du
terrain vu du dessus. Le détecteur de terrain préentraîné est installé localement ;
l'inférence ne télécharge pas de modèle et ne réentraîne pas les détecteurs.

## Lancer

Utiliser l'environnement joueurs existant, CPU ou CUDA. Aucune nouvelle
dépendance n'est nécessaire. Le modèle de terrain YOLOv8x représente environ
418 Mo sur disque ; son inférence est plus coûteuse que YOLO26 Nano joueurs.

```bash
make setup-court
make calibrate-court \
  VIDEO=datas/train/HOU_LAL/G1/video.mp4 \
  CHECKPOINT=/chemin/vers/detecteur-joueurs.pt \
  OUTPUT=runs/court/mon-essai \
  COURT_DEVICE=cuda:0 \
  MAX_FRAMES=300
```

Le dossier de sortie doit être nouveau. `COURT_DEVICE` choisit uniquement le
device du détecteur de terrain ; le détecteur joueurs garde le device de la
recette. Copier [predict_video.yaml](../players/configs/predict_video.yaml), y
mettre `device: cuda:0` et passer `CONFIG=/chemin/recette.yaml` pour les deux sur
GPU. Le défaut est CPU. `PLAYERS_PYTHON=.venv-players-cpu/bin/python` sélectionne
l'environnement CPU. `MAX_FRAMES` omis traite le clip entier.

Une recette GPU utilisable avec les poids locaux est fournie :

```bash
make predict-players CONFIG=training/court/configs/nba_gpu.yaml OUTPUT=runs/court/nba-demo
```

Elle traite 300 images HOU/LAL avec BoT-SORT. `models/players/yolo26n.pt` est un
détecteur de personnes généraliste pour cet essai ; remplacer ce checkpoint par
le détecteur joueurs entraîné/validé pour réduire arbitres et public.

Installation à partir du même checkpoint déjà téléchargé :

```bash
make setup-court COURT_SOURCE=/chemin/court_keypoint_detector.pt
```

Le SHA-256 attendu est
`f6263105e5c2338fafcfd5a6fefd7d1d441e87364635e918dfdbb849f2df1377`.
Le schéma des 18 repères est lié à ces poids : un modèle YOLO pose humain ou
un modèle terrain avec un autre ordre de points est refusé.

L'option fonctionne aussi avec le pipeline pose/maillots :

```bash
make predict-jersey VIDEO=... CHECKPOINT=... OUTPUT=... COURT=1 COURT_DEVICE=cuda:0
# Ou directement :
.venv-players/bin/python -m training.players.predict \
  --config /chemin/recette.yaml --court --court-device cuda:0
```

Les effectifs `jersey.json` et poids de pose/OCR restent requis pour leurs
options respectives. La calibration seule n'exige ni OCR ni pose.

## Géométrie

Les coordonnées viennent du [règlement officiel NBA 2025–26, page 8](https://cdn.nba.com/manage/2026/01/Official-2025-26-NBA-Playing-Rules.pdf)
et de la [règle 1](https://official.nba.com/rule-no-1-court-dimensions-equipment/).
La conversion est exactement `1 ft = 0.3048 m`.

| Mesure | NBA | Mètres |
| --- | --- | --- |
| Longueur intérieure | 94 ft | 28.6512 |
| Largeur intérieure | 50 ft | 15.24 |
| Largeur extérieure de la raquette | 16 ft | 4.8768 |
| Ligne de fond → bord extérieur de la ligne de lancer franc | 19 ft | 5.7912 |
| Ligne de fond → projection au sol du centre de l'arceau | 5 ft 3 in | 1.6002 |
| Rayon extérieur de l'arc à trois points | 23 ft 9 in | 7.239 |
| Distance latérale des trois points en corner au centre du panier | 22 ft | 6.7056 |
| Rayon de la zone restrictive | 4 ft | 1.2192 |
| Largeur des lignes | 2 in | 0.0508 |

Le modèle de terrain conserve ces **bords de référence réglementaires**, et non
une approximation FIBA de 28 × 15 m. Les tracés de contrôle sont des lignes
fines sur ces références ; ils ne simulent pas l'épaisseur complète de peinture.
Les annotations apprises ne distinguent pas systématiquement les deux bords
d'une bande de 5.08 cm : la précision des prédictions n'est donc pas celle des
constantes réglementaires.

Origine : coin lointain gauche ; x vers la droite du terrain dans la vue de
diffusion, y vers la ligne latérale proche. Les labels gauche/droite parfois
inversés du modèle sont normalisés géométriquement. Cette orientation vaut
**par segment caméra** : elle ne désigne pas le panier d'une équipe et ne garantit
pas une orientation physique identique après passage à une caméra opposée.
Le centre est `(14.3256, 7.62)`. Le panier élevé n'est pas utilisé comme un point
du parquet ; tous les points de l'homographie sont au sol.

## Estimation et caméra mobile

1. Détection des 18 repères sémantiques, par défaut toutes les cinq images.
2. Filtrage des faibles scores, points hors image et labels au même pixel.
3. RANSAC **terrain → image**, avec seuil en pixels, puis inversion pour obtenir
   **image → mètres**. Minimum cinq inliers, contrôle de surface couverte,
   reprojection, singularité, horizon et orientation.
4. Entre deux détections : suivi Lucas–Kanade aller/retour sur la texture du
   parquet, après masquage des personnes et incrustations configurées. Une
   homographie inter-images propage la calibration et absorbe pan/zoom.
5. Un recalage accepté et compatible avec le suivi corrige progressivement la
   calibration propagée : les projections des repères sont mélangées dans
   l'image courante, puis une homographie est ajustée. Le mouvement de caméra
   est déjà compensé. La constante `court.smoothing_seconds` vaut 0,15 s ;
   zéro restaure le recalage direct. Le lissage conserve les contrôles
   d'horizon, d'orientation, de singularité et le seuil de reprojection ;
   s'il les dépasse, le recalage direct est gardé. Il n'utilise pas d'images
   futures et ne mélange pas les matrices coefficient par coefficient.
6. La propagation expire après 0.5 seconde sans recalage absolu. Une estimation
   incompatible avec le mouvement mesuré est rejetée. Si ni les repères ni le
   mouvement ne suffisent, la calibration vaut `null` immédiatement.
7. Détection des coupes par variation d'image/histogramme avec contrôle de
   structure pour éviter de confondre les flashs photo avec des changements de
   caméra. Les coupes vident aussi le tracker et, via le segment, la mémoire OCR.
   `tracking.reset_frames` reste disponible pour les coupes connues.

La détection des coupes est heuristique. Les fondus, replays, vues inhabituelles
et longues transitions peuvent nécessiter des limites explicites. Aucune
calibration n'est interpolée à travers une coupe. Les réglages et diagnostics
effectifs sont exportés dans chaque run.

## Positions et sorties

| Fichier | Contenu |
| --- | --- |
| `annotated.mp4` | Vidéo source avec détections et lignes NBA projetées |
| `tactical.mp4` | Vue de dessus à l'échelle, positions et IDs, même cadence |
| `calibrations.jsonl` | Une ligne par frame : matrice, statut, segment, repères, inliers, reprojection, mouvement et coupes |
| `court_positions.jsonl` | Observations des joueurs : ID, point image choisi, position en mètres ou `null`, motif de rejet |
| `court.json` | Mesures, convention de coordonnées, SHA-256 du modèle et du code, configuration |
| `result.json` | Comptages `fit` / `propagated` / `unavailable` permettant de calculer la couverture, temps et empreintes des sorties |

`predictions.jsonl`, `tracks.jsonl`, les poses et l'OCR gardent leurs contrats.
L'option est désactivée dans les anciennes recettes. Un arrêt conserve les
fichiers `.partial.*` ; aucune reprise en milieu de vidéo n'est proposée.

Le point projeté par défaut est le **bas-centre de la boîte**, une approximation
du contact au sol. `court.footpoint: ankles` utilise la moyenne des deux chevilles
valides lorsque la pose existe, avec repli sur la boîte. Les chevilles sont
également au-dessus du sol : cela ne résout pas automatiquement les sauts.
Le point et la méthode utilisés sont toujours exportés. Le centre du torse et
la balle en l'air ne sont pas projetés comme des positions au sol.

Les observations trop faibles, pieds coupés, projections à l'infini, positions
hors terrain (marge configurable de 0.5 m), ou à plus de 4 m du support des
repères sont refusées. Les coordonnées ne sont jamais rabattues artificiellement
sur le bord du terrain. Aucun joueur hors champ n'est inventé.

Une faible erreur de reprojection indique une cohérence avec **les repères
prédits**, pas une précision absolue mesurée sur le vrai terrain. Les poids
préentraînés peuvent avoir un biais commun sur plusieurs points. Les positions
ne sont pas encore un benchmark de précision ni une base validée pour calculer
des vitesses instantanées.

## Sources et validation

Poids : [basketball_analysis, révision 197eef8](https://github.com/abdullahtarek/basketball_analysis/tree/197eef8ad7dc0b4daeeaa2086275aeaca8fbb8eb),
référencés dans son README. L'implémentation géométrique et temporelle ici est
indépendante ; les dimensions 28 × 15 m et la validation par rapports de longueurs
image de ce dépôt ne sont pas reprises (les rapports ne sont pas invariants
par homographie).

```bash
make test-court
make test-players PLAYERS_PYTHON=.venv-players/bin/python PYTHON=.venv-players/bin/python
make test-jersey PYTHON=.venv-players/bin/python
```

Les tests utilisent des projections analytiques et des vidéos synthétiques :
dimensions NBA, mètres/pixels, points aberrants, dégénérescence, labels inversés,
pan/zoom, expiration, coupe, flash, masques, intégration vidéo et interruption.
Les observations sur les vidéos locales sont consignées dans
[VALIDATION.md](VALIDATION.md).

## Revue des améliorations du 26 octobre 2026

Le lissage et l’affichage dans le viewer ont été examinés sur 5 100 images locales.
Les diagnostics `smoothing_applied`, `smoothing_alpha`,
`stabilized_reprojection_rmse_px`, `correction_median_px` et
`output_correction_median_px` complètent le rapport des recalages concernés.
L'erreur brute `reprojection_rmse_px` garde sa signification d'origine.
