# Validation du POC — 27 septembre 2026

La chaîne complète a été exécutée sur **5 100 images réelles**, soit 170 secondes
de trois vidéos NBA locales, avec le modèle préentraîné de terrain, YOLO26 Nano
personnes et BoT-SORT. Aucun entraînement n'a été effectué. Les deux vidéos du
dossier `val` ont servi à cette inspection qualitative, pas à une mesure contre
des annotations de calibration.

## Fonctionnement mesuré

RTX 5080, PyTorch 2.11.0 CUDA 13.0, FP32, entrée des détecteurs 640, OpenCV
4.11.0, Ultralytics 8.4.39, quatre threads CPU. Traitement séquentiel des clips.
Pose et OCR désactivés pour ces trois essais. Les temps incluent chargement,
décodage, détection, suivi, calibration, dessin, encodage des deux vidéos et JSON.

| Extrait | Images / durée | Recalages directs | Images propagées | Indisponibles | Couverture | Débit complet |
| --- | --- | --- | --- | --- | --- | --- |
| HOU/LAL G1 | 1 800 / 60 s | 290 | 1 212 | 298 | 83,44 % | 19,15 img/s |
| POR/SAS G5 | 1 500 / 50 s | 266 | 1 110 | 124 | 91,73 % | 18,93 img/s |
| PHX/OKC G4 | 1 800 / 60 s | 287 | 1 135 | 378 | 79,00 % | 19,35 img/s |

La couverture vaut `(fit + propagated) / frames`. **Elle ne mesure pas la
précision géométrique.** Ces débits proviennent d'un essai par clip, sans
échauffement distinct ; ce n'est pas un benchmark contrôlé de performances.
Ce profil n'atteint pas le temps réel 30 FPS sur ces mesures.

Les observations projetées sont respectivement 8 413, 7 805 et 8 284. Le détecteur
personnes est généraliste : il inclut des arbitres ou du public. Ce comptage
n'est ni le nombre de joueurs uniques ni une mesure de qualité du tracking.
Les candidats de faible confiance restent exportés dans le contrat joueurs mais
sont rejetés par la projection. Les positions éloignées du support des repères
sont également laissées à `null`, même lorsque l'image possède une homographie.

## Inspection et limites observées

Des images réparties dans chaque extrait ont été inspectées avec les lignes
projetées. Le système retrouve les deux moitiés de terrain, suit des panoramiques
et des changements de cadrage et refuse des vues centrales trop pauvres en
repères. Il reste des décalages visibles entre les prédictions et les lignes,
notamment autour des lancers francs et sur les parties éloignées des repères.
Les cinq points acceptés peuvent partager un biais : une erreur de reprojection
inférieure à un pixel sur ces points ne prouve pas une précision métrique
équivalente sur le parquet. Aucune erreur moyenne en mètres n'est revendiquée.

La première version du détecteur de coupes prenait un flash photo HOU/LAL pour
deux coupes successives aux frames 29 et 30. Le contrôle de corrélation de
structure corrige ce cas ; un test de changement d'éclairage couvre ce
comportement. Les essais finaux signalent 5, 0 et 2 coupes sur les trois clips.
Ces coupes n'ont pas toutes été annotées manuellement : ni rappel ni précision
de détection de coupes n'est mesuré, et `tracking.reset_frames` reste nécessaire
pour imposer les limites connues d'un montage.

Un essai HOU/LAL sans limite a échoué à la vérification de fin de source : le
conteneur annonce **1 843 images**, tandis qu'OpenCV et `ffprobe -count_frames`
en décodent **1 842**. Les sorties de cet essai sont conservées en `.partial.*`
dans `runs/court/validation/hou-lal-full/`. La vérification existante du pipeline
n'a pas été relâchée. L'essai final de 1 800 images ci-dessus est terminé avec
tous ses artefacts vérifiés ; la source n'a pas été modifiée.

## Artefacts disponibles localement

- `runs/court/validation/hou-lal-1800/` : vidéo annotée, vue tactique, positions,
  calibrations, provenance et `preview.jpg` à la frame 100.
- `runs/court/validation/por-sas-1500/` : mêmes sorties, plus `contact.jpg`.
- `runs/court/validation/phx-okc-1800/` : mêmes sorties, plus `contact.jpg`.
- `runs/court/validation/summary.json` : mesures, motifs de rejet, coupes et bilan
  des tests.

Les trois séries JSONL ont exactement le nombre d'images demandé. Les indices
de frame et segments sont identiques dans calibrations, positions et tracks.
Les deux vidéos de chaque run s'ouvrent et déclarent ce même nombre d'images.
Les SHA-256 de tous les artefacts enregistrés dans les trois `result.json` ont
été revérifiés après écriture.

Les runs documentent le chemin initial local du checkpoint joueurs
`/home/erwan/NBA/models/yolo26n.pt`. Une copie identique est disponible dans
`models/players/yolo26n.pt` pour la recette fournie. Aucun import de code du
projet NBA n'est ajouté. Les poids terrain sont dans
`models/court/court_keypoints.pt`, avec téléchargement et SHA-256 reproductibles.

## Tests automatisés

| Suite | Réussis | Optionnels désactivés |
| --- | --- | --- |
| Calibration NBA, environnement CPU joueurs | 21 | 0 |
| Joueurs, environnement joueurs existant | 123 | 10 |
| Maillots, environnement joueurs existant | 27 | 1 |

Les tests de calibration couvrent les dimensions et l'ordre des repères, la
projection analytique en mètres, RANSAC, labels inversés, doublons, points
colinéaires, horizon, pan/zoom, expiration, coupe, flash, masques, coordonnées
hors terrain, extrapolation, chevilles/repli, IDs après coupe, intégration vidéo,
option désactivée et interruption. Ils ne nécessitent aucun téléchargement.
La suite est aussi ajoutée à la tâche CPU de la CI ; la CI distante n'a pas été
exécutée dans cette session.

`git diff --check` et la compilation Python passent. Les annotations des matchs,
les jeux DVC et le modèle PARSeq n'ont pas été modifiés.
