# Inventaire et programme d'annotation, vidéo par vidéo

**Mise à jour après préannotation :** les 851 images de la phase 1 ont été
préannotées dans les six vidéos train avec E-BARD : 7 090 boîtes joueur et
1 850 boîtes arbitre proposées. Aucune image n'a été validée automatiquement.
Les deux sidecars val sont inchangés. Voir le
[rapport d'exécution](../preannotation-train-2026-10-17.json).
L'inventaire et les états « absents » du CSV ci-dessous décrivent le point de
départ avant cette opération ; le travail restant est la correction et la revue.

Inventaire local du 17 octobre 2026. Huit vidéos examinées, avec comptage complet
des frames par FFmpeg, vérification des dernières frames accessibles par le lecteur
OpenCV du projet et lecture des sidecars. Les empreintes des deux vidéos val
correspondent à leurs annotations joueurs. Aucun média ni annotation source n'a
été modifié. Les contenus n'ont pas fait l'objet d'une nouvelle revue visuelle :
les comptages ne certifient pas la justesse humaine des annotations.

## Fichiers pour suivre le travail

- [inventory.csv](inventory.csv) : une ligne par vidéo.
- [inventory.json](inventory.json) : détails, empreintes, plages manquantes et comparaison des fichiers terrain.
- [players_annotation_queue.csv](players_annotation_queue.csv) : chaque image joueur à traiter, avec phase, frame et temps.
- [ball_review_ranges.csv](ball_review_ranges.csv) : plages ballon manquantes ou avec position inconnue.
- [gold_sequences.csv](gold_sequences.csv) : quatre séquences candidates pour le suivi, les identités et la géométrie.

Les indices commencent à zéro. Les plages écrites `a–b` dans ce document sont
inclusives. `PLAYER_STOP` est exclusif. Le temps indiqué est `frame / 30`, depuis
le début du fichier, pas le chronomètre du match.

## Inventaire actuel

Toutes les vidéos sont en 1920 × 1080, à 30 images/s. Les durées ci-dessous sont
celles du conteneur ; les colonnes de frames utilisent le nombre réellement décodé.

| Split / vidéo | Durée | Frames décodées | Ballon avec coordonnées | Position inconnue | Frames sans annotation ballon | Images joueurs vérifiées | Terrain brouillon / registration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train HOU/LAL G1 | 61,43 s | 1 842 | 1 482 | 215 | 145 | 0 | 223 / 50 |
| train MIN/DEN G3 | 60,53 s | 1 815 | 1 413 | 334 | 68 | 0 | 21 / 20 |
| train ORL/DET G4 | 60,50 s | 1 814 | 1 443 | 189 | 182 | 0 | 32 / 32 |
| train ORL/DET G5 | 81,60 s | 2 448 | 1 938 | 240 | 270 | 0 | 19 / 19 |
| train PHI/BOS G4 | 90,67 s | 2 720 | 1 825 | 297 | 598 | 0 | 30 / 30 |
| train TOR/CLE G4 | 69,87 s | 2 095 | 1 812 | 283 | 0 | 0 | 30 / 30 |
| val PHX/OKC G4 | 62,13 s | 1 864 | 1 507 | 214 | 143 | 201 | 30 / 30 |
| val POR/SAS G5 | 54,77 s | 1 642 | 1 421 | 165 | 56 | 200 | 37 / 33 |

Train : **6 vidéos, 7 min 04,6 s, 12 734 frames**, 11 471 annotations ballon dont
9 913 positions, aucun sidecar joueurs. Val : **2 vidéos, 1 min 56,9 s, 3 506 frames**,
3 307 annotations ballon dont 2 928 positions, 401 images joueurs vérifiées,
3 100 proposées et 5 en cours. Aucun test local.

Le ballon possède seulement **deux positions marquées occultées**, une dans chacun
des deux clips ORL/DET. Les 1 937 positions inconnues ne sont pas des absences
certaines et ne constituent pas un benchmark de localisation sous occlusion.

Les 401 images joueurs vérifiées correspondent à **3 759 boîtes joueur et 969
boîtes arbitre**. Pour 4 727 des 4 728 boîtes, le champ d'occultation est inconnu.
Les états `verified` sont des états déclarés, pas une nouvelle certification de qualité.

## Ordre de travail et volumes précis

| Phase | Action | Images joueurs supplémentaires |
| --- | --- | ---: |
| 1 | Train : une frame sur 15 dans les six clips | 851 |
| 2 | Val : une frame sur 30 après le bloc initial, plus 5 images en cours | 109 |
| 3 | Compléter les boîtes sur les quatre séquences gold candidates | 575 |

Après phases 1 et 2 : **851 images train et 510 images val vérifiées**, sous réserve
de revue effective. Après phase 3 complète, avec ces sélections inchangées :
**991 images train et 945 images val**. L'identité persistante, le numéro et le
point au sol demandent des annotations supplémentaires distinctes des boîtes.

Ordre conseillé pour la phase 1 : HOU/LAL, MIN/DEN, TOR/CLE, PHI/BOS, ORL/DET G4,
puis ORL/DET G5. Les trois premiers fournissent 384 images et permettent un
premier contrôle de la chaîne avant de terminer les 467 suivantes. Les deux
ORL/DET restent en train ; ne pas en déplacer un vers val pour fabriquer une
diversité de splits après coup.

## Règles communes avant de commencer

1. Pour les personnes, annoter tous les joueurs participant au jeu et arbitres
   visibles ; exclure public, banc et entraîneurs. Utiliser les classes de l'outil,
   `0=player` et `1=referee`. Corriger les boîtes et renseigner les occultations et
   troncatures quand elles sont jugeables, puis valider toute l'image avec `V`.
2. Une préannotation peut accélérer le travail, mais reste `proposed`. Les poids
   génériques personne ne séparent pas automatiquement arbitres, public et joueurs.
3. Pour le ballon, visionner les plages manquantes. Ajouter une coordonnée seulement
   si elle est justifiable. Sinon conserver le caractère inconnu et documenter la
   raison. Un replay, une coupe ou une absence certaine doit être explicitement
   distingué ; le contrat actuel ne représente pas tous ces états séparément.
4. Revoir aussi 50 positions ballon existantes par clip, réparties sur toute sa
   durée, et les transitions vers/depuis les plages inconnues. Cette revue pilote
   détermine si une correction plus large est nécessaire.
5. `video.mp4.courtann.json` est déclaré brouillon ; `court_keyframes.json` est un
   artefact de registration. Leur présence ne prouve pas une précision métrique.
   Ne pas écraser l'un avec l'autre sur la seule base du nombre de repères.
6. Les sidecars ballon/terrain portent d'anciens chemins. Ne pas déplacer les vidéos
   sur cette seule base. L'association actuelle doit être documentée par les hashes,
   dimensions et indices ; les chemins historiques ne prouvent pas le split passé.

Les commandes train ci-dessous préannotent joueurs et arbitres avec E-BARD avant
d’ouvrir l’éditeur. Les propositions existantes et corrections sont conservées.
La préannotation utilise l’environnement ML et l’éditeur utilise `.venv` avec Tk.
Les commandes ci-dessous utilisent les environnements déjà présents. `PLAYER_STOP`
évite de sélectionner une frame terminale annoncée mais non décodable dans certains
clips. Pour ouvrir le ballon : `make annotate-ball VIDEO=<chemin-video>`.

**Limite de l'éditeur joueurs actuel :** il conserve toutes les frames déjà présentes
dans le sidecar, en plus de la sélection `START/STOP/STEP`. Sur val, les paramètres
ne masquent donc pas les autres propositions. Suivre les indices du CSV et la frame
affichée via le curseur. Pour rendre cette revue plus pratique, une amélioration
utile serait un filtre strict de sélection et une saisie directe de frame ; ce
programme ne les a pas implémentés. Ne pas supprimer les propositions existantes.

## 1. Train — HOU/LAL G1

Chemin : `datas/train/HOU_LAL/G1/video.mp4`.

**État.** 1 842 frames décodées contre 1 843 annoncées. Ballon : 1 697 annotations,
1 482 positions et 215 inconnues. Pas d'annotations joueurs. `match.yaml` et
`jersey.json` sont présents. Un gold maillot historique existe, avec statut draft,
anciens chemins et anciennes références de tracks : il n'est pas directement
utilisable pour évaluer les sorties actuelles.

**À faire, dans l'ordre :**

1. Annoter **123 images**, frames `0, 15, 30, …, 1830` :

   ```bash
   make annotate-players MEDIA=datas/train/HOU_LAL/G1/video.mp4 PLAYER_STEP=15 PLAYER_STOP=1842 \
     CHECKPOINT=models/players/ebard_yolov8n.pt PLAYER_CLASS=2 REFEREE_CLASS=3 PLAYER_DEVICE=cuda:0
   ```

2. Examiner les quatre trous ballon : **400–414, 416–437, 983–1016, 1243–1316**.
   Ils représentent 145 frames. Ne pas chercher à annoter la frame 1842, illisible.
3. Réconcilier le terrain : 223 keyframes dans le brouillon contre 50 dans la
   registration. Les 50 communes ont des listes de repères différentes, sans
   différence de coordonnées sur les repères de même nom. Déterminer quelles
   listes ont été revues et pour quel usage, puis conserver leur provenance.
4. Vérifier que le roster et les couleurs correspondent au clip. Le roster est
   une référence de noms, pas une preuve d'identité pour chaque track.
5. Pour le développement, proposer le gold **450–599**, soit **15–20 s**. Vérifier
   visuellement son exploitabilité et segmenter s'il contient une coupe. Compléter
   les boîtes des 150 frames, puis annoter identités et géométrie jugeables.
   Cela ajoute **140 images** aux 123 systématiques, soit **263 images train**
   pour ce clip si ce gold est retenu. Ne pas annoncer ce résultat comme test indépendant.

**Terminé lorsque :** les 123 images systématiques sont vérifiées, les quatre trous
sont qualifiés, le rôle des deux fichiers terrain est documenté et le gold est
aligné sur la vidéo actuelle, si cette phase est entreprise.

## 2. Train — MIN/DEN G3

Chemin : `datas/train/MIN_DEN/G3/video.mp4`.

**État.** 1 815 frames décodées contre 1 816 annoncées. Ballon : 1 747 annotations,
1 413 positions et 334 inconnues. Aucun sidecar joueurs. `match.yaml` est présent,
mais aucun `jersey.json` n'est présent.

1. Annoter **121 images**, frames `0, 15, …, 1800` :

   ```bash
   make annotate-players MEDIA=datas/train/MIN_DEN/G3/video.mp4 PLAYER_STEP=15 PLAYER_STOP=1815 \
     CHECKPOINT=models/players/ebard_yolov8n.pt PLAYER_CLASS=2 REFEREE_CLASS=3 PLAYER_DEVICE=cuda:0
   ```

2. Qualifier les trous ballon **644–675, 947, 956, 1473–1506** : 68 frames.
   La frame 1815 est annoncée mais illisible et n'entre pas dans cette liste.
3. Examiner les 334 positions inconnues par plages, sans les transformer en négatifs.
4. Réconcilier les fichiers terrain : 21 contre 20 keyframes. La frame **1684**
   n'existe que dans le brouillon. Sur la frame **1353**, le repère
   **`left_three_bottom_baseline`** a des coordonnées différentes : vérifier
   visuellement ce point. Cinq frames communes ont aussi des listes de repères différentes.
5. Si l'OCR nominatif doit être démontré sur ce clip, préparer un `jersey.json`
   conforme au contrat courant à partir des données match vérifiées. Ne pas
   renommer simplement `match.yaml` : les formats et champs attendus diffèrent.

**Terminé lorsque :** 121 images vérifiées, trous qualifiés et divergence terrain
1353 résolue ou conservée comme ambiguïté documentée.

## 3. Train — TOR/CLE G4

Chemin : `datas/train/TOR_CLE/G4/video.mp4`.

**État.** 2 095 frames décodées contre 2 096 annoncées. Les **2 095 frames décodables
ont une annotation ballon** : 1 812 coordonnées et 283 inconnues. Aucun sidecar
joueurs, roster ou fichier match. Les deux fichiers terrain ont les mêmes 30 keyframes.

1. Annoter **140 images**, frames `0, 15, …, 2085` :

   ```bash
   make annotate-players MEDIA=datas/train/TOR_CLE/G4/video.mp4 PLAYER_STEP=15 PLAYER_STOP=2095 \
     CHECKPOINT=models/players/ebard_yolov8n.pt PLAYER_CLASS=2 REFEREE_CLASS=3 PLAYER_DEVICE=cuda:0
   ```

2. Ne pas refaire l'annotation ballon intégrale : contrôler 50 positions étalées
   et revoir les 283 inconnues suivant le CSV.
3. Ne pas annoter la frame 2095, illisible. L'absence à cet indice n'est pas un
   trou dans les images réellement accessibles.
4. Contrôler un échantillon des 30 keyframes terrain. Si leur provenance n'atteste
   pas une revue humaine, conserver le statut brouillon.
5. Préparer match/roster seulement si ce clip rejoint la démonstration d'identité ;
   cela n'est pas un préalable à l'entraînement des boîtes joueurs ou du ballon.

**Terminé lorsque :** 140 images joueurs vérifiées et audit ballon effectué sans
recréer les annotations déjà présentes.

## 4. Train — PHI/BOS G4

Chemin : `datas/train/PHI_BOS/G4/video.mp4`.

**État.** 2 720 frames décodées, 2 122 annotations ballon dont 1 825 positions et
297 inconnues. **598 frames n'ont pas d'annotation**, plus gros volume manquant
du train. Aucun sidecar joueurs, fichier match ou roster. Les 30 keyframes terrain
sont identiques entre les deux fichiers.

1. Annoter **182 images**, frames `0, 15, …, 2715` :

   ```bash
   make annotate-players MEDIA=datas/train/PHI_BOS/G4/video.mp4 PLAYER_STEP=15 PLAYER_STOP=2720 \
     CHECKPOINT=models/players/ebard_yolov8n.pt PLAYER_CLASS=2 REFEREE_CLASS=3 PLAYER_DEVICE=cuda:0
   ```

2. Visionner en priorité **273–449** (9,10–15,00 s) et **711–1121**
   (23,70–37,40 s). Ces deux blocs représentent 588 des 598 frames manquantes.
   Déterminer s'il s'agit de jeu exploitable, replay, interruption ou ballon
   impossible à localiser. Ajouter les positions seulement lorsqu'elles sont justifiables.
3. Examiner les trous restants **633, 640, 1268–1273, 1515, 2336**.
4. Documenter le chemin historique contenant `val/PHI_BOS` dans les sidecars.
   Le split actuel est train. Reconstituer son historique si des métriques plus
   anciennes sont reprises ; ne pas présenter ce clip comme un test vierge.
5. Revoir un échantillon terrain et les 297 positions ballon inconnues.

**Terminé lorsque :** 182 images vérifiées, tous les trous classés et historique
de split explicité dans la provenance des expériences concernées.

## 5. Train — ORL/DET G4

Chemin : `datas/train/ORL_DET/G4/video.mp4`.

**État.** 1 814 frames décodées contre 1 815 annoncées. Ballon : 1 632 annotations,
1 443 positions dont une marquée occultée, 189 inconnues, 182 frames sans
annotation. Aucun sidecar joueurs, match ou roster. Les 32 keyframes terrain sont
identiques entre fichiers.

1. Annoter **121 images**, frames `0, 15, …, 1800` :

   ```bash
   make annotate-players MEDIA=datas/train/ORL_DET/G4/video.mp4 PLAYER_STEP=15 PLAYER_STOP=1814 \
     CHECKPOINT=models/players/ebard_yolov8n.pt PLAYER_CLASS=2 REFEREE_CLASS=3 PLAYER_DEVICE=cuda:0
   ```

2. Examiner d'abord le trou **1019–1191** (33,97–39,73 s), soit 173 frames.
3. Qualifier les autres trous **496, 610–612, 661–662, 1695–1697**.
   La frame 1814 est illisible et exclue du décompte de travail.
4. Vérifier l'annotation occultée avec position à la frame **0**. Une seule image
   ne permet pas de conclure sur les performances sous occlusion.
5. Conserver G4 et G5 dans le train. Leur proximité de confrontation peut réduire
   la diversité effective : publier les résultats par clip et par match.

**Terminé lorsque :** 121 images vérifiées et 182 frames manquantes qualifiées.

## 6. Train — ORL/DET G5

Chemin : `datas/train/ORL_DET/G5/video.mp4`.

**État.** Les **2 448 frames sont décodables**. Ballon : 2 178 annotations,
1 938 positions dont une occultée, 240 inconnues et 270 frames non annotées.
Aucun sidecar joueurs, match ou roster. Les deux fichiers terrain possèdent les
mêmes 19 keyframes.

1. Annoter **164 images**, frames `0, 15, …, 2445` :

   ```bash
   make annotate-players MEDIA=datas/train/ORL_DET/G5/video.mp4 PLAYER_STEP=15 PLAYER_STOP=2448 \
     CHECKPOINT=models/players/ebard_yolov8n.pt PLAYER_CLASS=2 REFEREE_CLASS=3 PLAYER_DEVICE=cuda:0
   ```

2. Examiner les trois blocs ballon principaux : **397–502** (13,23–16,77 s),
   **1144–1199** (38,13–40,00 s), **1361–1464** (45,37–48,83 s).
3. Examiner aussi **1724, 2444–2445, 2447**. Ici, **2447 est réellement décodable** :
   ne pas appliquer automatiquement la règle de frame terminale illisible des autres clips.
4. Revoir les 240 positions inconnues et identifier dans le JSON la position
   marquée occultée pour contrôler sa convention.
5. Sur le terrain, revoir les changements de cadrage et les moments éloignés
   d'une keyframe ; 19 repères temporels ne garantissent pas la couverture du clip.

**Terminé lorsque :** 164 images vérifiées et les 270 frames manquantes classées.

## 7. Validation — POR/SAS G5

Chemin : `datas/val/POR_SAS/G5/video.mp4`.

**État.** 1 642 frames décodées contre 1 643 annoncées. Ballon : 1 586 annotations,
1 421 positions et 165 inconnues. Joueurs : **frames 0–199 vérifiées**,
**200–204 en cours**, **205–1641 proposées**. Les 200 images vérifiées représentent
1 966 boîtes joueur et 571 arbitres et couvrent seulement les 6,67 premières secondes.
Aucun fichier match ou roster local.

1. Finir la revue des cinq images **200–204** : corriger notamment les arbitres,
   car ces images en cours ne contiennent actuellement que des boîtes joueur.
2. Valider les **48 images** réparties `210, 240, …, 1620`. Revoir toutes les boîtes,
   y compris les arbitres absents des propositions. Cela porte le clip à **253
   images vérifiées**, soit 53 de plus qu'aujourd'hui.

   ```bash
   make annotate-players MEDIA=datas/val/POR_SAS/G5/video.mp4 PLAYER_START=200 PLAYER_STEP=1 PLAYER_STOP=205
   make annotate-players MEDIA=datas/val/POR_SAS/G5/video.mp4 PLAYER_START=210 PLAYER_STEP=30 PLAYER_STOP=1642
   ```

   Ces commandes ouvrent l'éditeur ; sa limite de filtrage décrite plus haut
   impose de suivre les indices du CSV. Elles ne valident aucune image automatiquement.
3. Examiner le trou ballon **1586–1641**, soit 56 frames, de 52,87 à 54,73 s.
   Ne pas annoter la frame 1642, illisible.
4. Réconcilier les **37 keyframes brouillon contre 33 registration**. Les frames
   **10, 20, 80, 110** ne sont que dans le brouillon. Les listes de repères diffèrent
   sur les 33 frames communes, sans changement des coordonnées des repères partagés.
5. Pour le gold, proposer **750–899**, soit **25–30 s**. Après revue visuelle,
   compléter les 150 images, les IDs persistants et les cas où le maillot est lisible.
   Avec les cinq images de cette plage déjà prévues en phase 2, cela ajoute **145
   images** supplémentaires et porte le total à **398** si ce bloc est retenu.
6. Préparer un roster vérifié pour mesurer l'identité nominative. Sans roster,
   limiter la référence à la continuité des personnes et aux numéros jugeables.

**Terminé lorsque :** 253 images distribuées sont vérifiées ; en phase gold, la
séquence dense et ses identités sont revues. Le clip reste val et sert aux réglages.

## 8. Validation — PHX/OKC G4

Chemin : `datas/val/PHX_OKC/G4/video.mp4`.

**État.** 1 864 frames décodées. Ballon : 1 721 annotations, 1 507 positions,
214 inconnues et 143 frames non annotées. Joueurs : **0–200 vérifiées**,
**201–1863 proposées**. Les 201 images vérifiées représentent 1 793 boîtes joueur
et 398 arbitres et couvrent seulement les 6,70 premières secondes. Pas de roster
ou fichier match. Les fichiers terrain ont les mêmes 30 keyframes.

1. Revoir les **56 images** `210, 240, …, 1860`, puis les valider. Cela porte ce clip
   à **257 images vérifiées** avec une meilleure répartition dans le temps.

   ```bash
   make annotate-players MEDIA=datas/val/PHX_OKC/G4/video.mp4 PLAYER_START=210 PLAYER_STEP=30 PLAYER_STOP=1864
   ```

   Suivre le CSV : le filtre actuel conserve les autres propositions du sidecar.
2. Examiner le trou ballon **1410–1552**, soit 143 frames, de **47,00 à 51,77 s**.
   Ne pas le remplir par interpolation sans inspection vidéo.
3. Revoir les 214 positions inconnues et 50 positions existantes étalées.
4. Documenter le chemin historique contenant `train/PHX_OKC` dans les sidecars.
   Le split courant est val ; ce chemin seul ne prouve ni ne réfute une fuite de
   données. Vérifier les données des entraînements dont on reprend les scores.
5. Proposer deux golds : **900–1049** (30–35 s) et **1200–1349** (40–45 s).
   Vérifier que chacun est exploitable ; si une coupe apparaît, segmenter les IDs.
   Compléter les boîtes, identités et références géométriques jugeables.
   Cela ajoute **290 images** au-delà de la phase 2 et porte le total à **547**.
6. Préparer un roster si l'identité nominative fait partie du benchmark.
   Les données test devront venir d'autres matchs : ce clip a déjà servi à valider
   et inspecter les modèles du projet.

**Terminé lorsque :** 257 images distribuées sont vérifiées ; les deux golds
denses sont documentés en phase 3, si les scènes conviennent.

## Annotation supplémentaire pour les quatre golds

Les plages proposées sont des points de départ opérationnels, pas des scènes
certifiées adaptées : aucune nouvelle inspection visuelle n'a sélectionné ces
moments. Revoir chaque extrait avant d'engager sa labellisation dense. S'il est
inexploitable, changer la plage dans le plan et régénérer sa liste de frames.

Pour chaque frame, conserver : hash vidéo, frame/timestamp, segment caméra,
identifiant humain de personne, boîte, rôle, équipe et numéro seulement lorsqu'ils
sont certains, lisibilité et motifs d'incertitude. Ne pas utiliser l'object_id
local des boîtes ou l'ID prédit du tracker comme identité humaine persistante.
Le schéma boîtes actuel ne contient pas cette référence de tracking : créer un
sidecar gold séparé et versionné.

Pour la géométrie, commencer par 10 frames par séquence, distribuées dans le temps,
avec des repères terrain indépendants et points de contact des pieds jugeables.
Réserver des points de contrôle hors ajustement de l'homographie. Une référence
manuelle reste approximative ; sans trajectoire dense et référence suffisamment
fiable, publier une erreur de position, pas une précision de distance réelle.

L'image systématique, l'image dense et l'identité n'apportent pas le même type de
support statistique. Publier séparément scores sur échantillon systématique et
séquences denses : ne pas laisser les trois séquences val surpondérer un score
global au détriment de la diversité. Des frames voisines ne sont pas des milliers
d'exemples indépendants. Les futurs matchs test restent nécessaires.

## Contrôler le travail après chaque vidéo

1. Fermer les éditeurs pour terminer les écritures.
2. Régénérer l'inventaire dans un **nouveau dossier**, par exemple :

   ```bash
   .venv-players/bin/python -m scripts.inventory_data \
     --source datas --output reports/data-inventory-after-first-video
   ```

3. Contrôler les frames prévues dans le CSV et les nouveaux états `verified`.
4. Après les premières annotations train, exécuter `make export-players` et
   `make verify-players-export`, puis vérifier les supports par split. Les nouveaux
   exports auront de nouveaux IDs : ne pas réutiliser les anciennes métriques.
5. Si le ballon a changé, exécuter `make export-ball` avant de nouveaux runs.
6. Enregistrer la version de données avec `.venv/bin/dvc status`, puis
   `.venv/bin/dvc add datas` quand l'état voulu est prêt. L'envoi remote et le
   commit du pointeur sont des opérations distinctes.

La première séance recommandée est **HOU/LAL : 30 images systématiques**, frames
0 à 435 par pas de 15. Mesurer le temps réel de correction, vérifier la présence
des arbitres et l'export, puis terminer les 93 images restantes. Ne pas lancer
une longue campagne joueurs avant d'avoir vérifié cette chaîne sur les données
annotées et figé un premier export train/val.
