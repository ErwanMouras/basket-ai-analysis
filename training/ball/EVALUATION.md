# Évaluer le ballon sans entraînement

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
