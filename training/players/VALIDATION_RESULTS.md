# Bilan de livraison — phase 7

Essais locaux exécutés le 25 septembre 2026 ; bilan établi le 26 septembre.
Python 3.12.3, Linux x86-64, PyTorch 2.11.0+cpu / torchvision 0.26.0+cpu,
environnement neuf installé depuis `requirements-cpu.lock`. `uv pip check`
réussit sur les 137 distributions installées. Les adaptateurs exécutent les
entraînements courts en FP32, résolution 128, batch 2, deux threads CPU.

## Tests exécutés

| Vérification | Résultat observé |
| --- | --- |
| Suite joueurs avec MLflow local activé | 104 tests : **99 réussis, 5 ignorés**, 23,73 s |
| Suite balle dans l'environnement CPU dédié | 69 tests : **53 réussis, 16 ignorés**, 2,76 s |
| Parcours réel YOLO26 Nano sur fixtures | **1 réussi**, 24,30 s |
| Parcours réel RF-DETR Nano sur fixtures | **1 réussi**, 45,37 s |
| Comparaison CLI commune, mode par défaut | Réussie : les deux runs smoke sont exclus |
| Comparaison CLI avec `--include-smoke` | Réussie : deux lignes, un seul groupe données/protocole |
| `make test-ball` dans l'environnement historique `.venv` | 69 tests : **56 réussis, 13 ignorés**, 5,34 s |
| Syntaxe des fichiers Python modifiés et parsing du YAML CI | Réussis localement |
| `git diff --check` | Réussi |

La commande complète CPU a duré **103,44 s**, création des poids incluse.
Son rapport détaillé est conservé localement dans
`runs/players/phase7-validation/cpu-delivery/summary.json`, avec les identifiants
de tous les tests passés/ignorés, leurs raisons et les logs. Les deux parcours
utilisent de vrais frameworks, optimisateurs, évaluateurs COCO et un registre
MLflow SQLite ; ils ne remplacent pas les modèles par des mocks.

Chaque famille crée sept images synthétiques (4 train, 2 val, 1 test), dont deux
négatifs. Le parcours exporte et utilise seulement les six images train/val,
avec quatre boîtes au total. Le test synthétique reste exclu de la sélection.
La vidéo de démonstration comporte quatre frames. Les poids initiaux sont
aléatoires, générés localement ; aucun téléchargement de poids n'est nécessaire.

Les parcours couvrent export, entraînement interrompu après une epoch, reprise
stricte jusqu'à deux epochs, évaluation, comparaison, inférence vidéo, inscription
et rechargement natif/pyfunc d'un candidat. Ils vérifient notamment :

- refus d'une reprise sur checkpoint corrompu ; restauration des octets de la
  fixture avant la vraie reprise ;
- refus d'un changement de seed sur un checkpoint réel, sans création d'un run
  MLflow supplémentaire ;
- réutilisation complète à la relance, sans nouvelle tentative ;
- recalcul de la seule comparaison après corruption de son CSV ;
- absence de nouvelle version lors d'un second enregistrement identique,
  refus de promotion smoke et conservation de la provenance du meilleur epoch ;
- absence de modification des sources et de création d'annotations depuis la vidéo.

Les contrats unitaires complètent ces essais : corruption des métadonnées/poids
du registre refusée avant chargement, interruption du parent avec arrêt du
worker, reçus incomplets, checkpoint incompatible ou incomplet, publication
atomique, transformations de coordonnées, TP/FP/FN, doublons, images vides,
égalité des métriques entre adaptateurs pour des prédictions identiques.

## Tests ignorés et non exécutés

Les cinq tests ignorés dans la découverte joueurs sont : l'interface Tk,
l'inférence de poids externes préentraînés, leur évaluation autonome,
l'intégration de reprise comparée à un entraînement continu et le parcours réel.
**Ce dernier est ensuite exécuté séparément et réussit pour les deux familles.**
Les quatre autres restent ignorés dans cette livraison ; leurs validations
historiques ne sont pas comptées comme des essais de phase 7.

Les seize tests balle ignorés dans le nouvel environnement comprennent trois
tests Tk (module absent), six tests numériques PyTorch optionnels, six smokes
d'entraînement optionnels et une intégration de reprise. Dans `.venv`, Tk est
disponible et les trois tests GUI réussissent ; les treize intégrations optionnelles
restent désactivées. Aucun environnement ni code d'exécution balle n'a été changé.
Seul le setup de ses tests GUI reconnaît désormais explicitement Tk absent.

Les smokes GPU YOLO26/RF-DETR sont disponibles via une activation explicite,
mais **n'ont pas été exécutés en phase 7**. Aucun succès GitHub Actions hébergé
n'est revendiqué : le workflow a été écrit et son YAML contrôlé localement.
MLflow distant, authentification, restauration DVC distante, déploiement et test
final sur matchs ne sont pas exécutés. Aucun transfert DVC n'a été lancé.

## Volume réel et limites

Le comptage en lecture seule de `datas/` a trouvé **0 fichier
`*.playersann.json`, 0 frame joueurs vérifiée et 0 boîte joueurs vérifiée**.
Rapport local : `runs/players/phase7-validation/annotations.json`. Ce constat ne
compte ni les fixtures sous `runs/`, ni les prédictions NBA, ni les annotations
balle/terrain/maillots. Il ne préjuge pas de datasets externes non présents ici.

Cette livraison valide donc le fonctionnement technique, **pas la qualité d'un
détecteur sur des matchs**. Aucun AP synthétique ne constitue une baseline
sportive. Deux images val, un préchauffage et une répétition ne permettent pas
un benchmark robuste de latence/P95. Les rapports conservent le protocole et les
mesures, sans les présenter comme une recommandation de modèle.

Deux problèmes ont été rencontrés puis corrigés pendant ces essais : l'absence
de Tk était une erreur de setup au lieu d'un skip explicite, et le test de
registre supposait que `best` venait toujours du dernier run. Un meilleur epoch
peut rester celui du parent interrompu, notamment avec des poids aléatoires.
Le registre embarquait également toujours le lock CUDA ; son packaging choisit
désormais le lock du runtime, vérifié par l'intégration CPU.

Les avertissements RF-DETR sur l'initialisation sans préentraînement, le backbone,
le nombre de classes et l'inférence non optimisée ont été observés et conservés
dans les logs. Ils ne sont pas masqués. Le garde réseau ne remplace pas un
pare-feu système ; il interdit les connexions Python des tests et de leurs workers.

## Écarts au plan et reproduction

Le périmètre logiciel de phase 7 est livré. Restent à réaliser pour une recette
sur matchs : annotation humaine, première baseline, sélection sur val puis
évaluation finale figée. Le workflow doit encore s'exécuter dans GitHub ; les
smokes GPU et les services distants demeurent optionnels et non validés ici.
Les limites des phases précédentes restent documentées : groupes d'occultation
par image, évaluation FP32, RSS échantillonné, vidéo sans audio et alias distants
sans opération atomique distribuée.

Après l'installation décrite dans [VALIDATION.md](VALIDATION.md), depuis la racine :

```bash
# Choisir un dossier nouveau à chaque validation complète.
make validate-players PLAYERS_PYTHON=.venv-players-cpu/bin/python \
  VALIDATION_OUTPUT=runs/players/validation-reproduction
make test-ball
.venv-players-cpu/bin/python -m training.players.tests.annotation_inventory \
  --source datas --output runs/players/annotations-inventory.json
# Facultatif, explicitement activé, dans l'environnement CUDA :
make smoke-players-gpu PLAYERS_PYTHON=.venv-players/bin/python \
  VALIDATION_OUTPUT=runs/players/validation-gpu
```

Les commandes de restauration utilisent directement DVC : les cibles historiques
`make data-*` citées dans `docs/DATA.md` sont absentes du Makefile actuel. Cette
différence est signalée et les commandes effectives figurent dans le guide de
validation. Les exemples ajoutés ne dépendent d'aucun chemin personnel ni secret.
