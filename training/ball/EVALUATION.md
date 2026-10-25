# Évaluer le ballon sans entraînement

## Comparer YOLO et les différentes versions de TrackNet

Le comparateur `training.ball.comparison` couvre les entraînements complets
retrouvés dans MLflow : YOLO26 S/M/L, TrackNet V3, V5 et V5+TOTNet, ainsi que
l'ancien V5 de deux époques comme référence historique. Il évalue les fichiers
`best.pt` et `best_f1.pt` disponibles ; `last.pt` est un état de reprise, pas un
candidat ajouté après lecture des scores. Les essais limités à quelques batches
et les entraînements interrompus sont exclus du classement.

Avant toute prédiction, le gel vérifie les exports, l'absence de matchs communs
entre train et validation, les annotations et les octets JPEG identiques entre
formats. Les poids locaux doivent correspondre aux artefacts du run MLflow.
Le protocole contient les SHA-256 des checkpoints, configurations et évaluateur,
les identifiants des exports, la liste des images communes et les règles de calcul.

La comparaison porte sur **une position par image**, ramenée en pixels de la
vidéo source. YOLO fournit le centre de la boîte classe 0 la plus confiante ;
TrackNet fournit le centre du plus grand contour de la heatmap seuillée.
Une erreur supérieure à 15 pixels compte à la fois comme faux positif et faux
négatif. Les erreurs médiane/P95 incluent les positions incorrectes, et ne sont
pas limitées aux détections réussies. Les tolérances 5/10/15/25 px complètent
le résultat principal sans changer les prédictions.

Les seuils sont fixés avant évaluation : 0,25 pour YOLO, 0,5 pour les heatmaps.
Ces scores ne sont pas des probabilités calibrées comparables entre familles.
Le classement caractérise donc des **modèles avec leur prétraitement et leur
décodeur**, aux seuils annoncés. Les initialisations préentraînées, augmentations
et coûts d'entraînement diffèrent aussi : 30 époques ne constituent pas un budget
de calcul égal et ne permettent pas d'isoler l'effet de l'architecture seule.

L'inférence est causale : image courante seule pour YOLO, dernière tête des trois
images pour V5, dernière tête des huit images pour V3. Aucun vote de têtes
précédentes ni aucune moyenne utilisant le futur. Les trous d'export, frontières
de clips et coupes renseignées interrompent le contexte. Le classement principal
utilise l'intersection des images couvertes ; un second résultat indique la
couverture propre à chaque modèle et pénalise les images non couvertes comme
des faux négatifs. La première mesure compare la qualité sur un support identique ;
la seconde expose le coût du contexte temporel.

**Limites du jeu actuel :** pas de test indépendant, pas de négatifs revus où la
balle est certainement absente, pas de liste de coupes revue. Les inconnues restent
exclues et ne deviennent jamais des négatifs. Les fenêtres ne représentent donc
pas encore tout le contexte disponible dans une vidéo brute. Les images voisines
sont corrélées et le choix entre `best` et `best_f1` utilise la validation ; les
résultats sont une sélection de développement, pas une estimation indépendante.

Pour préparer une nouvelle comparaison dans l'environnement CUDA installé :

```bash
.venv-players/bin/python -m training.ball.comparison freeze \
  --output runs/ball/comparison-new/protocol.json
# Versionner et relire ce protocole avant de lancer les prédictions.
.venv-players/bin/python -m training.ball.comparison run \
  --protocol runs/ball/comparison-new/protocol.json \
  --output runs/ball/comparison-new/evaluations
.venv-players/bin/python -m scripts.ball_comparison_report \
  --results runs/ball/comparison-new/evaluations/results.json \
  --protocol runs/ball/comparison-new/protocol.json \
  --output runs/ball/comparison-new/report
```

Les exports, poids et références épinglées doivent être présents localement.
Chaque checkpoint tourne dans un processus distinct pour isoler les imports des
implémentations amont. Chaque évaluation crée un run `ball-model-comparison`
dans MLflow avec protocole, environnement, métriques et prédictions par image.
Le gel refuse un changement du code de calcul ; modifier celui-ci nécessite une
nouvelle comparaison. Ce workflow refuse le split test et sert à la sélection.

Le chronométrage utilise le même GPU, FP32, batch 1, cinq appels de chauffe et
trois répétitions sur 30 fenêtres aux mêmes images cibles. CUDA est synchronisé
avant et après l'appel. La mesure inclut préparation des images RGB déjà décodées,
transfert, réseau et décodage de sortie ; elle exclut disque, chargement des poids,
restauration des coordonnées source et autres étapes du pipeline. Les fenêtres
temporelles sont recalculées en entier ; aucune optimisation de streaming n'est
supposée. La médiane/P95 décrit ces 90 appels, pas le débit de la chaîne complète.

## Évaluateur V5 et test indépendant

L'évaluateur charge les checkpoints locaux V5 ou V5+TOTNet et un export SDK
vérifié. Il accepte un export val ou un export test indépendant : les IDs des
données d'entraînement restent dans la provenance, sans imposer que l'export
d'évaluation soit celui d'entraînement.

```bash
.venv-players/bin/python -m training.ball.evaluation \
  --config configs/evaluate_ball_causal.yaml
```

Deux politiques sont disponibles :

- `offline_mean` : comportement historique, moyenne des heatmaps des fenêtres
  qui se recouvrent ; une prédiction peut utiliser des frames futures.
- `causal_last` : dernière tête de chaque triplet consécutif, sans fusion et
  sans contexte futur. Les deux premières frames de chaque bloc restent sans
  prédiction et sont comptées dans la couverture.

`cut_frames` associe des clip IDs aux indices source des coupes connues. Une
fenêtre traversant une coupe est exclue ; le contexte reprend au début du plan.
Le benchmark porte sur les frames présentes dans l'export. Lorsque des images
sans annotation ont été exclues de cet export, leur contexte manque aussi :
ne pas présenter ce protocole comme une équivalence complète à l'analyse vidéo.
Les coupes ne sont pas détectées automatiquement par cet évaluateur.

Les métriques sont calculées une fois par frame, en pixels source si
`tolerance_space: source`. Publier seuil, tolérance, support, couverture et erreurs
médiane/P95, en distinguant les groupes visibles, occultés localisables et inconnus.
Une position inconnue ne prouve pas une absence certaine ; ne pas la convertir
en négatif sans convention et revue adaptées.

## Test final gelé

Préparer une recette avec `split: test`, `frozen_recipe` pointant vers un nouveau
fichier, et un export test explicitement créé. Les lecteurs d'entraînement
continuent à refuser le split test par défaut.

```bash
.venv-players/bin/python -m training.ball.evaluation \
  --config /recettes/test.yaml --freeze-recipe /recettes/test.frozen.json
# Examiner et versionner ce gel avant de lire les scores.
.venv-players/bin/python -m training.ball.evaluation --config /recettes/test.yaml
```

Le gel lie checkpoint, données, sélection, code et politique temporelle. La
commande de gel ne charge pas le modèle et n'exécute aucune inférence. Les poids
smoke ou entraînés avec des plafonds de batches sont refusés pour le test final.
Le checkpoint et l'export sont revérifiés avant finalisation d'une évaluation.

Le rapport `protocol.json`, les prédictions JSONL, les métriques et le CSV de revue
sont journalisés dans MLflow. Les vidéos de revue sont optionnelles.
