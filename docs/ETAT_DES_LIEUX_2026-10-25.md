# État des lieux — objectif portfolio MLOps junior

Audit du 25 octobre 2026. Objectif : rendre Vyneros convaincant sur un CV et
facile à évaluer à travers une démo réelle de 30 secondes et quelques exemples
ciblés. Ce document distingue les fonctionnalités présentes, les mesures
effectivement disponibles et le travail recommandé.

## Verdict

Le socle technique est déjà assez riche pour ce positionnement : traitement
vidéo batch, entraînements, suivi MLflow, exports traçables, registre,
promotion/refus/rollback, conteneurs, rapports d'exécution et tests. La priorité
est de terminer une version cohérente et démontrable, puis de rendre ses preuves
visibles dans GitHub. Ajouter de nouveaux modèles ou de nouvelles briques
d'infrastructure apporterait moins que cette finalisation.

Les principales faiblesses actuelles sont la démo publique synthétique, les
modèles récents encore absents de la recette d'inférence par défaut, la validation
limitée de la chaîne complète et la quantité de travail encore hors commits.

## Vérifications réalisées

- Lecture du code pipeline, viewer, évaluateurs et outils MLOps, des
  configurations, du workflow CI et des rapports de livraison.
- Consultation ciblée de l'historique Codex du projet, notamment les échanges
  d'octobre sur l'audit et les entraînements, recoupés avec SQLite MLflow et
  les artefacts locaux.
- Relance de `make check-poc CHECK_OUTPUT=runs/audit-2026-10-25` :
  **329 tests réussis, 28 ignorés, aucun échec**, et syntaxe JavaScript valide.
  Ce contrôle désactive les entraînements réels optionnels.
- Nouvel inventaire des huit vidéos, avec comptage des images décodées et
  lecture des statuts d'annotation : `runs/audit-data-2026-10-25/`.
  Il ne constitue pas une nouvelle revue visuelle des annotations.
- Contrôle Git distant : `origin/main` pointe sur `0463c19`, comme le HEAD local.
- Contrôle DVC local : `dvc status datas.dvc` signale `datas` modifié.
- Inspection des captures existantes du viewer réel et synthétique. Aucune
  nouvelle validation interactive en navigateur ni nouvelle inférence GPU
  complète n'a été exécutée pendant cet audit.

## Ce qui existe déjà

| Domaine | État et preuve | Limite actuelle |
| --- | --- | --- |
| Pipeline vidéo | Détections joueurs/arbitres, tracking par rôle, ballon, pose, calibration, OCR, identité et distance observée | Qualité bout en bout encore peu mesurée sur vérité terrain indépendante |
| Viewer | Vidéo et calques synchronisés, terrain projeté, effectifs et fiche joueur | La fiche expose la distance sans la durée mesurée ni sa couverture ; titre « temps réel » ambigu pour un traitement batch |
| Expériences | Six entraînements ballon complets récents, paramètres et métriques dans MLflow | Le README et `configs/analysis.yaml` pointent encore vers l'ancien V5 de deux époques |
| Évaluation personnes | Comparaisons locales, dont E-BARD à AP joueur 0,7318 sur le protocole du 18 octobre | 401 images provenant des premières secondes de deux clips val, fortement corrélées |
| Cycle modèle | Acceptation, refus et rollback exécutés dans une base MLflow isolée | Cette preuve concerne les détecteurs de personnes ; la sélection des nouveaux modèles ballon reste à relier à la démo |
| Reproductibilité | Locks, bundles avec empreintes, Docker CPU, inférence hors réseau et exemple sans modèles | L'essai CPU enregistré est très court ; le pointeur DVC ne décrit plus les données actuelles |
| Vérification | 329 tests locaux réussis aujourd'hui | Les intégrations optionnelles ignorées et la syntaxe JS ne remplacent pas les essais réels et navigateur |
| Publication | Sources, documentation et workflow prêts localement | README, `mlops/`, Docker, CI, exemples et nombreux guides sont encore non suivis par Git |

Les mesures historiques proches de 20 images/s concernent un clip et une RTX
5080. Elles ne prouvent pas du temps réel à 30 FPS. La lecture fluide de sorties
précalculées dans le viewer est un parcours distinct.

## Les entraînements récents changent le diagnostic

Les six runs ci-dessous sont terminés, sur 30 époques en 512 × 288. Les scores
sont les maxima enregistrés pendant l'entraînement, pas une nouvelle évaluation
commune des checkpoints sauvegardés.

| Modèle ballon | Meilleur F1 de validation interne | Run MLflow |
| --- | ---: | --- |
| TrackNet V5 | 0,8943 | `c6bcdc638b0c45c9882d71e03b2ad533` |
| TrackNet V5 + TOTNet | 0,9181 | `8813c2970cb241bf9147f3e95524baad` |
| TrackNet V3 | 0,8620 | `874b776b96d34fcea14e8a0cfb749680` |
| YOLO26S | 0,8627 | `814c488a9bb54f76897ee3a44923d69e` |
| YOLO26M | 0,8725 | `613beb3177c84b32894f2a3991d9c03f` |
| YOLO26L | 0,8930 | `0873327e0d7f402a95ef68fb02d7e626` |

Ces F1 ne permettent pas de désigner directement un vainqueur : TrackNet mesure
une localisation de centres en pixels d'entrée, sur des fenêtres temporelles ;
YOLO utilise une métrique de détection de boîtes. Le nombre d'images de contexte
diffère aussi. Il faut comparer les prédictions dans les coordonnées source,
sur une sélection commune, avec tolérance, couverture et politique temporelle
explicites. Ne pas comparer directement ces valeurs au F1 causal 0,7539 du README.

L'évaluateur ballon actuel et le pipeline intégré acceptent V5/V5+TOTNet.
V3 et YOLO ballon demandent des adaptateurs pour intégrer cette comparaison et
la chaîne complète. Le chemin le plus court vers une démo actualisée est donc
d'évaluer les checkpoints V5 et V5+TOTNet récents avec `causal_last`, puis
d'intégrer celui qui convient le mieux en qualité et coût de calcul.

Vérifier le critère de sauvegarde : `best.pt`, `best_f1.pt` et `last.pt` peuvent
correspondre à des époques différentes. Le meilleur score historique ne suffit
pas à certifier les performances du fichier retenu.

## Données disponibles aujourd'hui

| Split | Clips | Images joueurs vérifiées | Images avec position ballon | Position ballon inconnue | Images sans annotation ballon |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 6 | 225 | 9 913 | 1 558 | 1 263 |
| Validation | 2 | 401 | 2 929 | 378 | 199 |
| Test | 0 | 0 | 0 | 0 | 0 |

Les 225 images vérifiées de train viennent de HOU/LAL (113) et MIN/DEN (112).
Les quatre autres clips train ont encore uniquement des propositions de boîtes.
Une préannotation n'est pas une annotation humaine validée.

Les annotations ballon distinguent position renseignée et position inconnue,
sans statut indépendant de revue. Les 1 462 images sans annotation ne sont pas
automatiquement des exemples où le ballon est absent. Conserver la politique
`unknown_position: exclude` déjà choisie ; ne pas fabriquer de négatifs à partir
de ces cas.

Le répertoire contient un ancien gold OCR HOU/LAL marqué `draft`, lié à un autre
format historique. Ce fichier ne fournit pas un benchmark revu de trajectoires
compatible avec l'évaluateur `mlops.gold` actuel.

Les chiffres du rapport du 18 octobre sont historiques : la validation ballon
compte maintenant une position de plus. Chaque score publié doit conserver la
référence exacte de l'export qui a servi à son calcul.

## Ordre de travail recommandé

### 1. Fermer la sélection des modèles

Évaluer l'ancien V5, le V5 complet et V5+TOTNet complet sur la même validation
causale. Publier précision, rappel, F1 de localisation, couverture, erreurs
médiane/P95 et coût d'inférence. Examiner les erreurs importantes et les coupes.
Sélectionner un checkpoint et consigner la décision, son empreinte et la recette.

Étendre ensuite à YOLO/V3 si la comparaison répond à une question utile. Ces
adaptateurs supplémentaires ne doivent pas bloquer une première release basée
sur V5. Aucun nouvel entraînement n'est nécessaire pour cette première étape.

Critère de fin : un tableau comparable, un modèle sélectionné et une
configuration d'inférence qui charge exactement ce modèle.

### 2. Construire la démo réelle de 30 secondes

Choisir un extrait de jeu lisible, avec un cadrage exploitable et des mouvements
intéressants. Les extraits existants sont des candidats à inspecter, pas encore
une sélection visuelle validée. HOU/LAL peut servir de démonstration de
développement, avec cette provenance affichée ; il ne devient pas un test
indépendant parce qu'il est utilisé dans une vidéo de présentation.

Recalculer un run dédié avec la recette finale et le roster explicite lorsque
des noms doivent être montrés. Le run `color-groups` utilisé pour la capture
réelle actuelle ne contient aucune identité confirmée dans ses statistiques.
Le MP4 existant de `release/private-demo/` dure environ 4,12 secondes.

Produire une vidéo principale avec détections et terrain projeté, puis une
capture du viewer montrant une fiche joueur effectivement renseignée. Limiter
les calques visibles pour garder l'action lisible ; la pose et les repères
techniques peuvent apparaître dans des exemples séparés.

Ajouter à la fiche joueur durée observée/mesurée et couverture à côté de la
distance. Renommer « Terrain en temps réel » pour rendre le mode batch clair.
Les valeurs montrées doivent venir du run, avec leurs incertitudes et exclusions.

Prévoir deux ou trois exemples de 5 à 10 secondes :

- ballon rapide, petite taille ou occultation, pour expliquer une réussite et
  une limite du modèle retenu ;
- changement de caméra ou calibration indisponible, pour montrer la rupture
  correcte des mesures ;
- identification lisible d'un joueur, ou abstention face à un numéro ambigu,
  avec un cas réellement vérifié.

Critère de fin : 30 secondes de vidéo réelle relues, des sorties portables et
deux exemples expliqués. Le média public doit être choisi en tenant compte des
conditions de diffusion déjà identifiées dans la documentation du projet.
L'exemple synthétique reste utile pour essayer le viewer sans données privées.

### 3. Ajouter une validation ciblée indépendante

Réserver un ou deux nouveaux matchs absents du développement et des réglages.
Annoter un petit ensemble systématique pour joueurs/ballon et quelques
séquences courtes denses pour le tracking. Déclarer ce volume comme un premier
benchmark limité, sans prétendre couvrir tous les matchs NBA.

Pour les trajectoires : identités humaines persistantes, coupes revues et
repères terrain indépendants. Mesurer d'abord IDF1, changements d'ID,
précision/couverture d'identité, erreur de position et disponibilité de la
projection. Garder les distances comme estimations observées tant que leur
précision physique ne dispose pas d'une référence suffisante.

Le gold peut rester court. Terminer toutes les annotations et entraîner tous
les modèles disponibles ne sont pas des conditions nécessaires à une première
version portfolio honnête. En revanche, consulter le test pour régler les
modèles lui retire son rôle de test final.

Critère de fin : un rapport reproductible avec nombres de clips/images,
protocole, résultats, erreurs typiques et limites.

### 4. Terminer la release Git et la reproduction

Relire les modifications suivies et non suivies, puis préparer des commits
cohérents : données/évaluation, cycle MLOps, déploiement, viewer/documentation.
Le dépôt distant ne contient actuellement pas le README local ni les nouveaux
dossiers MLOps et déploiement. Vérifier les fichiers retenus avant publication.

Actualiser le pointeur DVC après gel des annotations retenues et vérifier la
disponibilité de cette version pour les utilisateurs autorisés. Le simple
fait d'avoir un fichier `datas.dvc` ne garantit pas qu'il décrit les données
des expériences actuelles.

Rejouer les vérifications depuis les sources à publier et lancer la CI GitHub
après publication. Tester le parcours viewer depuis un clone neuf, et le
parcours d'inférence avec les modèles installés explicitement. Arrêter aussi
le choix de licence source mentionné comme restant dans le README.

Critère de fin : commit de release identifiable, CI distante verte et exemple
viewer utilisable avec la commande annoncée.

### 5. Réécrire le README pour un premier lecteur

Conserver l'anglais et organiser l'entrée en matière ainsi :

1. Une phrase sur le problème et le résultat produit.
2. La démo réelle de 30 secondes, avec provenance et mode batch.
3. Une commande pour essayer le viewer et ses sorties précalculées.
4. Une architecture compacte reliant données, expériences, évaluation,
   promotion, inférence et rapports.
5. Un petit tableau des résultats et des choix : qualité, couverture, latence,
   matériel, volume du benchmark et limites.
6. Les preuves MLOps : version de données, expérience, décision de promotion,
   exemple de refus/rollback, CI et image reproductible.
7. Deux exemples ciblés et les liens vers les guides détaillés.

Le README actuel contient beaucoup de matière utile, mais sa longue suite de
commandes, de réserves et de procédures dilue le résultat visible. Déplacer les
procédures détaillées dans les guides existants, tout en conservant près des
scores les limites indispensables à leur interprétation.

## Version minimale suffisante pour candidater

Une première version convaincante peut se limiter à un détecteur de personnes,
un modèle ballon sélectionné, la projection terrain et le viewer. Elle montre
une vraie démo de 30 secondes, deux cas expliqués, un benchmark limité mais
reproductible, un cycle de promotion/refus/rollback et une CI publiée.

L'argument CV repose sur l'ensemble de la chaîne et les décisions justifiées.
L'identification exhaustive, la distance exacte sur un match, un service
d'upload public, Kubernetes et une nouvelle matrice d'entraînements ne sont pas
des prérequis à cette version.

Formulation possible, à utiliser après publication des preuves correspondantes :

> Développement de Vyneros, pipeline batch d'analyse vidéo de basketball :
> suivi d'expériences MLflow, versionnement des données DVC, évaluation de
> modèles, promotion et rollback contrôlés, inférence Docker et CI GitHub
> Actions ; démonstration sur clips réels avec visualisation des prédictions.

Ajouter ensuite une mesure représentative, associée au matériel et au protocole
publiés. Préparer pour les entretiens une explication de la sélection du modèle,
du découpage des données, d'un échec observé et du mécanisme de retour arrière.

## Références locales

- `runs/audit-2026-10-25/summary.json` : vérification locale actuelle.
- `runs/audit-data-2026-10-25/inventory.json` : inventaire actuel.
- `mlflow.db` et `runs/ball/full-*/training/` : entraînements récents.
- `configs/analysis.yaml`, `pipeline/adapters.py` et
  `training/ball/evaluation/inference.py` : intégration et limites des adaptateurs.
- `docs/EVALUATION_PROTOCOL.md` et `mlops/gold.py` : contrat de validation.

Les fichiers sous `runs/`, la base MLflow et les médias privés sont locaux et
ignorés par Git ; leurs chemins sont des références d'audit, pas des liens de
reproduction accessibles à un visiteur du dépôt.
