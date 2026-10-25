# Numéros de maillot — audit NBA et plan d’intégration dans Vyneros

Date : 26 septembre 2026. Statut initial : **plan proposé** ; mise à jour d’implémentation ci-dessous.
Périmètre audité : `/home/erwan/NBA`, son historique Git et ses artefacts locaux,
puis les contrats, l’inférence et les utilitaires de `/home/erwan/Vyneros`.

**Mise à jour du 27 septembre — baseline vidéo livrée :** tracking et pose sont
maintenant raccordés à `training/jersey/` : crops torse/poitrine, PARSeq générique,
sélection causale par fenêtres, confirmation temporelle, budgets CPU/GPU,
exports et vidéo annotée. Voir [le guide opérationnel](../training/jersey/README.md).
`setup-jersey`, `predict-jersey` et `test-jersey` existent. Les lots d’annotation,
d’entraînement, de benchmark gold et de registre décrits plus bas restent un
plan ; les anciens résultats NBA ne sont pas des résultats de cette baseline.

**Mise à jour du 26 septembre — prérequis tracking livré :** Vyneros fournit
désormais des `track_id` via BoT-SORT/ByteTrack dans l’inférence vidéo joueurs.
Voir [le guide de tracking](../training/players/TRACKING.md). Le constat initial
ci-dessous sur l’absence de tracking est donc historique ; la pose et les numéros ont depuis été livrés dans la baseline ci-dessus ;
les équipes restent à implémenter. Au lot 5, réutiliser cet adaptateur de suivi
plutôt que reporter le tracker NBA.

## 1. Décision recommandée

Créer **`training/jersey/`**, au même niveau que `training/players/` et
`training/ball/`, par **portage sélectif du code NBA et réimplémentation de son
intégration**. La reconnaissance doit fonctionner d’abord sur crops ou observations
importées, puis sur les détections vidéo Vyneros via des adaptateurs explicites.

Le code NBA contient une base substantielle récupérable. Copier simplement son
ancien dossier `training/jersey/` serait insuffisant : ce dossier ne contenait
essentiellement que l’évaluation ; l’inférence était dans `pipeline/` et dépendait
du tracking, de la pose et éventuellement des équipes/effectifs.

| Option | Avantage | Limite | Conclusion |
| --- | --- | --- | --- |
| Restaurer et copier tout le pipeline NBA | Retrouver rapidement son environnement historique | Couplages nombreux, mémoire non bornée, contrats incompatibles, poids spécialisés absents | À écarter pour l’intégration ; une extraction isolée peut servir de référence |
| Porter les fonctions utiles et refaire les frontières | Conserver crops, OCR, fusion et cas de tests, avec les garanties du dépôt actuel | Travail d’adaptation et de validation nécessaire | **Option recommandée** |
| Tout réécrire et entraîner un nouveau modèle immédiatement | Contrôle complet | Refaire inutilement des briques existantes avant d’avoir un benchmark cible | À réserver aux composants déficients démontrés par l’évaluation |

Le premier objectif est **lire un numéro avec abstention explicite**. L’attribution
d’un nom demande en plus une équipe et un effectif fiables. Une identité stable
sur tout le match demande encore une gestion des tracklets et des changements de
plan : ce sont des livrables distincts.

## 2. Ce qui existe réellement dans NBA

### 2.1 Historique et récupération

La documentation NBA n’est pas entièrement synchronisée : `README.md` et
`AGENTS.md` mentionnent encore la reconnaissance alors que `PROGRESS.md` signale
sa suppression. L’arbre Git et les fichiers ont été contrôlés directement.

| Élément | Source vérifiée | État |
| --- | --- | --- |
| Runtime maillots | `77be5484ea10bcfca138a5e3fe979e8c313c1b01:pipeline/src/pipeline/jersey_recognition.py` | **2 090 lignes**, récupérables dans Git ; parent du commit de suppression `3a4f331` |
| Tests runtime | Même révision, `pipeline/tests/test_jersey_recognition.py` | 613 lignes, 11 fonctions de test ; crops, sélection, fusion, roster, conflits, chargement OCR |
| Configuration historique | Même révision, `pipeline/src/pipeline/config.py`, classe `JerseyIdConfig` | Mélange de paramètres utilisés et hérités d’anciennes implémentations |
| Évaluateur et tests | `e19d117976ffba71491c87b8f1b6fead692c6573:training/jersey/` | Évaluateur de 337 lignes ; dossier supprimé ensuite par `3256dc0` |
| Premier portage historique | Commit `3ec985646d628a464ef75273a89d1ca922a54a05` | Contient notamment `src/nba/pipeline/jersey_recognition.py` et le gold NBA |
| PARSeq externe | `/home/erwan/NBA/third_party/parseq/` | Présent ; révision locale `1902db043c029a7e03a3818c616c06600af574be` |
| Poids PARSeq génériques | `/home/erwan/NBA/models/parseq/parseq-bb5792a6.pt` | Présents ; chargement et empreinte à vérifier lors du portage |
| Poids JNR SoccerNet | Ancien `models/jnr/parseq_soccer_jnr.ckpt` | Absents du dossier de modèles inspecté ; Git ne restaure pas ces poids ignorés |
| Données de diagnostic | `NBA/runs/jersey_crops/`, `jersey_case_debug/`, rapports JSON et run figé | Encore présents ; provenance à vérifier avant réutilisation |

Révisions des dépôts à l’audit : NBA
`b3a151edba8a10b6975a318f7861137f584e881b`, Vyneros
`fdca8effae55b963af409bd3d0dd010777e6f746`. Les deux arbres comportent des
modifications locales ; cet audit n’a ni restauré ni modifié le dépôt NBA.

Extraction consultative possible, sans checkout ni rétablissement du pipeline :

```bash
git -C /home/erwan/NBA show \
  77be5484ea10bcfca138a5e3fe979e8c313c1b01:pipeline/src/pipeline/jersey_recognition.py
git -C /home/erwan/NBA show \
  e19d117976ffba71491c87b8f1b6fead692c6573:training/jersey/scripts/evaluate_jersey_gold.py
```

Le journal mentionne aussi des essais antérieurs `src/nba/jersey_number/`,
classificateurs de chiffres/lisibilité et entraînements E-BARD. Les recherches
dans les chemins Git historiques concernés n’ont pas retrouvé ces sources.
Le journal du 6 mai décrit leur suppression. **Ne pas planifier leur transfert
comme s’il existait un trainer JNR réutilisable et validé.**

### 2.2 Chaîne implémentée

```text
Vidéo + observations {frame, track_id, bbox, pose COCO17}
                     + équipes/effectifs facultatifs
  → régions pose_torso / center_chest et autres crops relatifs
  → qualité : netteté, contraste, taille, pose, orientation, chevauchements
  → sélection de keyframes par tracklet
  → OCR PARSeq générique ou JNR spécialisé, par lots
  → confirmation sur une autre région autour d’une lecture forte
  → votes pondérés par confiance OCR × qualité du crop
  → filtrage par effectif, résolution des conflits simultanés
  → numéro par tracklet, diagnostics, JSON/CSV et affectations par frame
```

Ce n’est pas un détecteur entraîné à localiser une boîte autour des chiffres.
La région est obtenue par géométrie du joueur/torse, puis un modèle de
reconnaissance de texte lit le numéro. Un futur détecteur `number_box` serait
donc une **nouvelle brique**, à justifier par les erreurs de localisation.

Les points COCO17 utilisés pour le torse sont les épaules et hanches
`5, 6, 11, 12`. La confirmation multi-région ne renforce un candidat que si elle
relit le même numéro que son crop d’ancrage.

Les backends réellement construits sont PARSeq, JNR, un adaptateur Python et une
commande externe. Plusieurs noms acceptés par la configuration historique
(`easyocr`, `paddleocr`, etc.) ne correspondent pas à des lecteurs construits
dans cette version : ne pas reprendre cette liste comme support effectif.

### 2.3 Résultats et données réutilisables

Les rapports suivants ont été lus, y compris le comptage de leurs lignes :

| Rapport dans `NBA/runs/` | Résultat observé | Portée |
| --- | --- | --- |
| `jersey_gold_eval_pose_keyframes_best.json` | 15/20, soit 75 % | PARSeq générique, ancienne version du gold |
| `jersey_gold_eval_jnr_best.json` | 19/20, soit 95 % ; 12/13 à identifier, 7/7 unknown | JNR, ancienne version du gold |
| `jersey_gold_eval.json` | 17/20, soit 85 % ; 14/17 à identifier, 3/3 unknown | Gold révisé ; 3 faux négatifs, 3 tracklets exclues |

Le gold révisé reclasse notamment les tracks 10, 25, 30 et 31 comme identifiables.
Les 95 % et 85 % ne mesurent donc pas le même protocole. Tous ces résultats
portent sur **HOU/LAL, 300 frames, 20 tracklets évaluées** ; aucune généralisation
aux matchs complets ou aux vidéos FFBB n’est démontrée.

Attention aux associations d’artefacts : le `NBA/runs/jersey_stats.json` courant
concerne **Hyeres**, avec 10 tracks identifiées sur 16 et aucun filtrage roster
effectivement appliqué. Il ne doit pas être associé au rapport HOU/LAL courant.
Le dossier `runs/frozen/hou_lal_g1_300f_jersey_gold_v1/` contient pour sa part une
référence **EasyOCR antérieure**, pas la sortie JNR finale. Les chemins seuls ne
suffisent pas à reconstituer une expérience.

Vyneros possède déjà :

- `datas/train/HOU_LAL/G1/gold/jersey_id_hou_lal_g1_300f_jersey_gold_v1.yaml` :
  23 tracklets, gold révisé, mais encore marqué `draft` et référant à des chemins NBA ;
- la vidéo HOU/LAL et son `match.yaml` ;
- des caches de détection NBA, qui fournissent des boîtes mais ne remplacent pas
  un export validé de tracking et de pose.

Les identifiants de track du gold sont liés à son run historique. Un nouveau
tracker peut attribuer d’autres IDs : **aucune comparaison par simple égalité
des IDs entre deux runs**. Il faudra conserver les observations de référence,
ou construire un appariement explicite validé par image/temps/boîtes.

## 3. Ce qu’il faut corriger pendant le portage

| Point observé dans le code | Conséquence | Traitement proposé |
| --- | --- | --- |
| `collect_jersey_crops` accumule les images de tous les crops avant sélection | Mémoire croissante avec la vidéo, y compris pour les crops rejetés | Flux borné, sélection par fenêtres et journalisation disque |
| `_normalize_number` concatène les chiffres, tronque à deux caractères et retire les zéros initiaux | `0` et `00` confondus ; texte invalide pouvant devenir un numéro plausible | Chaînes conservées exactement, alphabet ASCII et longueur validés sans correction implicite |
| `jnr_max_digits` configurable jusqu’à 3, normalisation limitée à 2 | Configuration et résultat incohérents | Une seule règle de longueur définie par le contrat |
| Sélection avec écart temporel puis remplissage sans cet écart | Deux votes peuvent provenir de crops quasi identiques | Compter séparément crops et observations temporelles indépendantes |
| Confiance fusionnée = moyenne d’un score OCR × qualité heuristique | Ce score n’est pas une probabilité calibrée | Conserver les scores bruts, calibrer sur validation, mesurer précision/couverture |
| Choix du candidat suivant après conflit | Une identité alternative peut être promue sans preuve suffisante | Abstention par défaut ; alternative seulement si elle passe ses propres critères |
| Équipe reprise du premier crop et roster susceptible de compléter l’équipe | Erreur amont pouvant contaminer l’identité | Provenance distincte équipe/numéro, mode sans roster, pas d’identité complète si ambiguë |
| Repli CUDA vers CPU implicite dans le lecteur historique | Mesures de vitesse et attentes matérielles trompeuses | Device réellement utilisé enregistré ; repli configurable et signalé |
| Import PARSeq via `sys.path`, chargement historique `weights_only=False` | Dépendance à l’environnement local et au format du checkpoint | Environnement dédié, révision figée, artefacts de provenance connue et chargement explicite |
| Évaluateur excluant via équipe absente et mot « contamin » dans les notes | Exclusions fragiles, équipe inconnue confondue avec exemple invalide | Champs d’exclusion structurés et périmètre d’évaluation explicite |
| Rapports sans liaison forte gold/run/vidéo/poids | Mélange possible des résultats de clips différents | Identifiants et SHA-256 vérifiés avant évaluation |

Le filtrage numérique forcé de JNR (`EOS + chiffres`, softmax restreint) donne
une confiance conditionnelle à cet alphabet ; il ne démontre pas qu’un numéro
est réellement visible. Il faut conserver de vrais exemples illisibles/négatifs.
Les seuils historiques `0.95`, deux votes, douze crops et confirmation sur une
frame sont des paramètres de reproduction NBA, **pas des valeurs validées ici**.

## 4. Frontières avec le dépôt actuel

`training.players.inference.predict_image` renvoie des boîtes XYXY en coordonnées
source, une classe et une confiance. `training/players/predict.py` produit un
JSONL `players_video_predictions` avec SHA-256 vidéo, indice et temps nominal.
Il ne produit ni track ID, ni pose, ni équipe.

Les contrats `training/players/contracts.py` refusent les champs inconnus.
L’`object_id` des annotations est unique **dans une frame** ; il ne représente
pas une identité temporelle. Ajouter directement `jersey_number` dans ces
documents casserait le contrat et ne résoudrait pas le tracking.

Proposition : conserver ces sorties et produire des sidecars maillots séparés.
Le module importe les utilitaires génériques existants :

- `training/common/files.py` : écritures atomiques ;
- `training/common/provenance.py` : empreintes et état du code ;
- `training/common/sources.py` : identité match, sources et séparation des splits ;
- `training/common/tracking.py` : utilitaires **MLflow**, pas suivi de joueurs.

Ne pas réutiliser directement le registre, l’évaluateur COCO ou l’orchestrateur
joueurs : ils sont spécialisés. Reprendre leurs garanties, puis extraire un
utilitaire commun uniquement lorsqu’un besoin partagé est démontré.

## 5. Compartiment cible et contrats

Arborescence proposée, à créer progressivement selon les lots :

```text
training/jersey/
  README.md, Makefile, requirements.in, requirements-*.lock
  contracts.py, config.py
  crops.py, quality.py, selection.py
  inference.py, fusion.py, roster.py, predict.py
  backends/{base,parseq,jnr}.py
  adapters/{nba,players,tracking,pose}.py
  annotator/             # revue des crops et des tracklets
  export/                # manifestes crops, labels et splits
  evaluation/            # numéro, abstention, identité, rapports
  learning/              # seulement au lot entraînement
  orchestration/         # seulement après stabilisation des commandes
  configs/, tests/
exports/jersey/           # données dérivées régénérables
runs/jersey/              # expériences, prédictions, diagnostics, poids locaux
.external/parseq/         # dépendance épinglée, hors Git
.venv-jersey*/            # environnements isolés
```

Le cœur crops/OCR/fusion doit être importable sans les dépendances du détecteur,
du tracker ni de la pose. Les adaptateurs optionnels fournissent ces entrées.
Si tracking/pose deviennent des briques partagées, les déplacer ensuite dans
des compartiments dédiés ; leur présence dans `jersey/adapters` ne doit pas
en faire des dépendances implicites du cœur.

### Contrats versionnés à définir avant le code

| Artefact | Contenu minimal |
| --- | --- |
| Observations | Vidéo et SHA-256, dimensions/FPS, `frame_index`, temps, `observation_id`, boîte source, score détecteur ; `tracklet_id`, pose COCO17 et équipe optionnels avec provenance |
| Manifestes de crops | ID, observation parente, boîte joueur et boîte crop, région, transformations, image/empreinte, scores qualité, motif de rejet, match/salle/split |
| Annotations maillots | Numéro chaîne ou `null`, lisibilité, visibilité partielle, région éventuellement annotée, statut de revue, exclusion structurée, provenance humaine |
| Lectures OCR | Crop ID, texte brut, candidat numérique exact ou `null`, scores bruts, backend, checkpoint, configuration et raison d’abstention |
| Résultats par tracklet | ID qualifié par source/run/segment, numéro, équipe éventuellement inconnue, candidats, preuves crop/frame, nombre de votes indépendants, statut et motifs |
| Rapport d’évaluation | Empreintes gold/prédictions/protocole, numérateurs/dénominateurs, exclusions, groupes, métriques, latences et environnement |

Un `tracklet_id=null` est autorisé pour lire un crop isolé ; il interdit la fusion
temporelle. Un changement de plan ou une réutilisation d’ID doit créer un nouveau
segment. Une identité complète est au minimum `(match_id, team_id, number)` ;
deux équipes peuvent avoir le même numéro.

Conserver `"0"` et `"00"` comme valeurs différentes. `null` n’est jamais le numéro
zéro. Une lecture illisible, une observation non revue et une erreur de modèle
sont trois états distincts. Les préannotations n’entrent dans l’entraînement
qu’après validation humaine explicite.

## 6. Plan d’exécution par lots

### Lot 0 — figer les références et préparer les dépendances

- Écrire `MIGRATION.md` : révisions source, chemins, SHA-256 des fichiers portés,
  attribution et correspondance ancien/nouveau code.
- Inventorier les crops, vidéos, poids, effectifs et rapports disponibles ;
  vérifier chaque lien vidéo/run/gold. Les artefacts orphelins restent des
  diagnostics, sans prétention à reproduire une mesure.
- Extraire uniquement les fichiers nécessaires dans un espace de travail isolé.
  Ne pas cherry-pick les refontes globales NBA.
- Tester la compatibilité de PARSeq local avec un environnement Python 3.12
  dédié ; produire des locks CPU/GPU vérifiés. Les versions installées chez
  `players` ne constituent pas une preuve de compatibilité PARSeq.
- Pour JNR, retrouver une copie fiable ou récupérer explicitement le checkpoint
  officiel, enregistrer origine/empreinte/licence. Son absence ne bloque pas
  les contrats, les crops ni les tests avec lecteur factice.

**Sortie / validation :** inventaire reproductible, environnement importable,
une lecture réelle sur crop si un checkpoint compatible est disponible ; chaque
ressource manquante et chaque expérience non reproductible sont identifiées.

### Lot 1 — contrats et noyau autonome

- Créer schémas, validateurs, fixtures et configurations minimales.
- Porter les structures crop/lecture et le protocole de lecteur ; séparer
  configuration et logique métier du module monolithique.
- Porter géométrie torse, qualité et clipping ; tests des coordonnées, bords,
  crops vides, poses insuffisantes et images BGR/RGB.
- Réimplémenter normalisation stricte et statuts d’abstention ; supprimer les
  options historiques sans implémentation.
- Ajouter un lecteur factice pour tester sans poids, réseau ni GPU.

**Sortie / validation :** import et tests du cœur sans NBA, PARSeq, BoxMOT ou
ONNXRuntime installés ; distinction `0`/`00`, rejets des entrées ambiguës,
provenance et coordonnées validées.

### Lot 2 — données, import et annotation

- Importer le gold déjà présent dans `datas/` avec correspondance explicite des
  chemins et des IDs du run historique. Le conserver comme fixture de régression
  en split train ; ne pas en faire un nouveau test indépendant.
- Fournir un importeur de crops NBA et, lorsqu’ils existent, d’observations
  tracking/pose. Sans observations historiques, ne pas inventer de correspondance.
- Ajouter la revue par planche de crops et contexte vidéo : numéro/lisibilité,
  correction, validation, exclusion motivée, sauvegarde atomique et verrou.
- Prévoir un sidecar `video.mp4.jerseyann.json` versionné, distinct du sidecar
  joueurs ; rattacher les annotations aux sources et observations immuables.
- Constituer un premier gold cible sur plusieurs matchs/salles : ordre de
  grandeur **150–250 tracklets revues**, incluant illisibles, profils, petits
  joueurs, chevauchements, arbitres/faux positifs et numéros proches. Ce volume
  lance le benchmark ; il ne suffit pas à certifier une fiabilité de production.
- Respecter `datas/train|val|test`, regrouper tous les crops d’un match dans son
  split et garder une séparation par salle si elle est annoncée. Aucune
  redistribution silencieuse des données actuelles.
- Exporter images + labels/manifestes ; LMDB seulement si nécessaire au trainer
  PARSeq. Conserver l’export canonique indépendant de ce format.

**Sortie / validation :** annotations revues séparées des propositions, exports
déterministes et vérifiables, absence de fuite par match/contenu. Les sources
restent sous le DVC existant ; exports/caches régénérables hors `datas.dvc`.

### Lot 3 — OCR et diagnostic par crop

- Porter les lecteurs PARSeq et JNR séparément ; modèle/tokenizer/transformations
  attachés au checkpoint, longueur et tokens numériques vérifiés.
- Charger localement, sans téléchargement implicite pendant l’inférence ;
  batching configurable, CPU puis smoke GPU, device effectif enregistré.
- Conserver texte brut et sorties invalides ; vérifier la correspondance exacte
  entre nombre de crops et nombre de résultats. Une dépendance absente en mode
  OCR requis provoque une erreur explicite.
- Construire planches d’erreurs et évaluation exact-match par crop. Évaluer la
  lisibilité séparément : une vérité connue au niveau tracklet n’est pas un
  label visible valable sur toutes ses frames.
- Si les erreurs viennent surtout du cadrage, comparer une région `number_box`
  géométrique et des boîtes manuelles oracle avant d’entraîner un localisateur.

**Sortie / validation :** inférence réelle sur crops annotés, rechargement des
poids reproductible, rapport par lisibilité/taille/longueur du numéro. PARSeq
générique est une baseline technique, sa qualité n’est pas présumée suffisante.

### Lot 4 — sélection temporelle, fusion et conflits

- Porter les scores de sélection et la confirmation multi-région sous tests de
  caractérisation, puis corriger les votes corrélés et les règles de conflit.
- Traiter les observations en flux. Première passe : scores et candidats dans
  un index disque, sélection par fenêtres avec plafond de candidats ; seconde
  passe séquentielle : extraction des seules frames retenues, OCR par lots bornés.
- Conserver dans chaque fenêtre les alternatives nécessaires à la confirmation
  multi-région ; documenter que cette sélection bornée peut différer du tri global
  historique. Comparer les lectures retenues sur les fixtures.
- Ne conserver en RAM ni tous les crops, ni toutes les observations de tout le
  match. Écrire diagnostics et votes au fil de l’eau ; finaliser par tracklet ou
  partition avec état disque. Borner aussi les états des tracks terminées.
- Séparer preuves temporelles indépendantes et corroborations du même instant.
  Documenter pondération, marge entre candidats et causes de rejet.
- Modes sans roster, avec équipe connue et avec roster validé. Un roster
  incomplet ou une équipe incertaine ne doit pas forcer une identité.
- Conserver une sortie par tracklet ; pas de fusion des identités long terme à
  ce stade. Signaler contamination, changement d’équipe et contradictions fortes.

**Sortie / validation :** tests de numéro tardivement visible, faux positif
confiant, `0`/`00`, deux équipes même numéro, conflit simultané, track fragmentée,
équipe absente et ID réutilisé ; mémoire mesurée sur séquence prolongée.

### Lot 5 — raccordement à la vidéo Vyneros

Deux niveaux successifs, avec capacités annoncées explicitement :

1. **Mode détections seules** : lire `players_video_predictions`, vérifier SHA-256
   vidéo, frames/dimensions et protocole, générer crops relatifs au joueur puis
   lectures par observation. Fonctionnel sans tracking ; aucune identité stable.
2. **Mode tracklets et pose** : adaptateur de suivi alimenté par les boîtes
   existantes, puis pose top-down COCO17 et fusion du lot 4. Les observations
   enrichies deviennent un artefact réutilisable et indépendant de l’OCR.

Pour le second niveau, extraire les interfaces pertinentes de
`NBA/pipeline/src/pipeline/runner.py` (`_create_player_tracker`, `_track_stats`)
et `pipeline/src/pipeline/inference/pose_rtmpose.py`. Écrire les adaptateurs
autour des contrats locaux : les imports `pipeline.config/types/paths` et le
runner NBA ne sont pas portables tels quels. BoT-SORT/RTMPose constituent les
candidats de continuité, à vérifier sur les versions réellement épinglées.

L’équipe peut d’abord venir d’une annotation fiable ou rester inconnue. La
classification automatique d’équipe est une dépendance optionnelle distincte,
pas une fonctionnalité incluse discrètement dans le lecteur de numéros.

Produire JSONL lectures, résultats tracklets, progression et vidéo annotée
optionnelle dans `runs/jersey/`. Reprendre les conventions d’écriture partielle
et finale des joueurs, les erreurs de décodage et les limites de timestamps
nominaux. Les phases doivent être réutilisables par empreintes ; une reprise
du tracker exige son état exact ou le retraitement du segment, pas seulement
le dernier indice de frame.

**Sortie / validation :** une vidéo brute → détections joueurs → observations
suivies/pose → numéros, sans accès à NBA ; JSONL cohérents, interruption propre,
ressources bornées et non-régression balle/joueurs.

### Lot 6 — benchmark commun et seuils

- Adapter l’évaluateur NBA au contrat strict ; recalculer tous les compteurs
  depuis les lignes, valider les IDs et refuser les mélanges de sources/runs.
- Séparer lecture du numéro, identité équipe+numéro et performance de toute la
  chaîne. Une erreur de tracking exclue de l’OCR pur reste comptée/visible dans
  le bilan de bout en bout.
- Rapporter exact-match crop, rappel des tracklets identifiables, précision des
  identifications acceptées, couverture, fausses identifications sur illisibles,
  abstentions et mauvaise identité. Toujours afficher les supports.
- Définir la couverture comme acceptées/éligibles, la précision comme
  correctes/acceptées, et le rappel identifiable comme correctes/identifiables.
  Afficher séparément la part du dataset exclue de l’éligibilité.
- Mesurer par match/salle, taille, face/dos/profil, visibilité, chiffres simples
  ou doubles ; annoncer les groupes trop petits et l’incertitude statistique.
- Comparer bbox seule/pose, PARSeq/JNR, avec/sans roster, avec/sans confirmation,
  à gold et observations identiques. Séparer budgets d’OCR et de calcul amont.
- Mesurer latence OCR médiane/P95 après préchauffage, temps de décodage, tracking,
  pose, sélection, fusion et débit complet, RAM/VRAM et matériel exact.
- Calibrer sur val, figer recette puis évaluer test une seule fois pour le
  candidat final. Aucun classement automatique sur le smoke ou le gold historique.

**Sortie / validation :** rapport JSON/CSV et planches de revue, expériences
comparables et choix documenté. Cibles initiales proposées pour une V1 utile :
précision des identifications ≥ 95 % et couverture des tracklets identifiables
≥ 75 %, avec supports et intervalles ; à confirmer sur le cas d’usage réel.
« Zéro erreur observée » sur quelques clips ne signifie pas risque nul.

### Lot 7 — entraînement adapté, si le benchmark le justifie

- Implémenter un trainer PARSeq dédié : données propres revues, initialisation
  déclarée, vocabulaire numérique, budget/seed/augmentations enregistrés.
- Comparer initialisation autorisée, données réelles et synthétiques ; ne jamais
  rendre une autre inscription par-dessus un numéro réel encore visible. Prévoir
  fonds sans numéro ou masquage contrôlé, polices et images de provenance connue.
- Tester flou, faible résolution, perspective et occultations plausibles ;
  préserver les zéros initiaux et éviter les transformations inversant les chiffres.
- Ajouter un modèle de lisibilité seulement si les faux positifs le justifient.
  Ajouter un détecteur de numéro seulement si l’oracle de cadrage montre un gain
  suffisant. Chacun demande annotations, entraînement et métriques propres.
- MLflow `jersey-training` / `jersey-evaluation` : config, manifeste, code,
  environnement, pertes, métriques, poids initiaux/best/last et liens de reprise.
- Distinguer fine-tuning et reprise stricte avec optimiseur/scheduler/RNG/états
  requis. Un simple rechargement des poids n’est pas une reprise stricte.

**Sortie / validation :** entraînement court réel, rechargement et interruption/
reprise vérifiés ; entraînement long seulement après validation des données.
Conserver la baseline si l’amélioration sur val n’est pas démontrée.

### Lot 8 — orchestration, registre et identité persistante

- Orchestrer les commandes stabilisées par processus séparés et séquentiels,
  avec reçus, hashes, reprise et invalidation des seules étapes dépendantes.
- Packaging `jersey-reader` avec code/poids/config/tokenizer et provenance ;
  enregistrement et promotion explicites, smoke exclu de la promotion.
- Conserver séparément le lecteur crop et le profil complet tracking/pose/fusion.
  Leurs métriques et environnements ne sont pas interchangeables.
- Seulement après validation des numéros : relier les tracklets à une entité
  joueur par match/équipe/numéro, sans conflit temporel et avec preuves suffisantes.
  Les replays/changements de plan sont segmentés ; pas de raccordement aveugle.
- Évaluer les mauvaises fusions/séparations sur un gold d’identité dédié et
  conserver l’abstention quand l’équipe ou le numéro reste ambigu.

**Sortie / validation :** exécution reproductible, candidat rechargé sans NBA,
promotion traçable ; identité persistante activée uniquement si son évaluation
propre passe les critères figés.

## 7. Ordre de livraison et commandes prévues

Chemin critique : **0 → 1 → 2 → 3 → 4 → 5 → 6**. L’annotation cible du lot 2
peut se poursuivre pendant le développement du lecteur. Le lot 7 dépend des
erreurs observées ; le lot 8 vient après stabilisation des interfaces.

| Livraison | Contenu démontrable | Dépendance principale |
| --- | --- | --- |
| A — compartiment autonome | Import, revue, export et lecture de crops | Sources/poids compatibles et annotations revues |
| B — reconnaissance temporelle | Numéro ou abstention par tracklet importé | Observations temporelles liées à la vidéo |
| C — vidéo Vyneros | Détecteur local + tracking/pose + numéros | Adaptateurs tracking/pose validés |
| D — adaptation et packaging | Modèle choisi/entraîné, protocole figé, registre | Gold cible suffisamment diversifié |
| E — identité sur le match | Raccordement contrôlé des tracklets | Équipes fiables et gold d’identité |

Interface Make proposée lors de l’audit ; `test-jersey` et `predict-jersey`
existent désormais (mode vidéo). Les autres commandes ci-dessous restent prévues :

```text
make test-jersey
make import-jersey CONFIG=...
make annotate-jersey CONFIG=...
make export-jersey CONFIG=...
make predict-jersey CONFIG=...       # mode crops / observations / vidéo
make evaluate-jersey CONFIG=...
make train-jersey CONFIG=...         # lot 7
make pipeline-jersey CONFIG=...      # lot 8
make register-jersey CONFIG=...      # lot 8, action explicite
```

Mettre à jour le Makefile racine, `.gitignore` pour `.venv-jersey*`, les locks de
sidecar et les checkpoints spécifiques, puis `docs/DATA.md` et les guides
d’utilisation au moment où les commandes sont effectivement livrées.

Pas d’estimation ferme avant le lot 0 : disponibilité du checkpoint JNR,
compatibilité PARSeq et coût de l’annotation sont les principales incertitudes.
Le portage du noyau est borné ; la qualité sur les vidéos cibles demande une
mesure et éventuellement un entraînement, pas seulement une copie de fichiers.

## 8. Dépendances externes et limites de cet audit

Le dépôt officiel JNR fournit la chaîne et les liens de poids, et annonce une
licence **CC BY-NC 3.0**. Conserver ce candidat comme référence expérimentale et
tracer cette contrainte avant tout packaging produit. Réentraîner à partir de
ces poids ne doit pas être supposé supprimer la restriction. Source consultée
le 26 septembre 2026 : [JNR officiel](https://github.com/mkoshkina/jersey-number-pipeline).

Le code PARSeq amont est sous **Apache-2.0** ; la provenance des données et les
conditions des poids choisis doivent être enregistrées séparément. Source :
[licence PARSeq](https://github.com/baudm/parseq/blob/main/LICENSE). Cette distinction
permet de préparer une voie d’entraînement adaptée sans rendre le module dépendant
du checkpoint SoccerNet.

L’audit a inspecté les sources, historique, configurations, tests existants,
rapports JSON et gold local. Il n’a pas chargé les modèles, lancé d’entraînement,
rejoué les tests NBA, vérifié visuellement tous les crops, ni validé les
empreintes de tous les artefacts historiques. Les scores ci-dessus sont des
**résultats conservés**, pas des mesures refaites pendant cette session.

La livraison initiale de cet audit ajoutait uniquement ce plan. Aucun code de reconnaissance, poids,
dataset, environnement, export DVC ou pipeline existant n’avait alors été modifié.
