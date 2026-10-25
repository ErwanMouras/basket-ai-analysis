# Reconnaissance temporelle des numéros de maillot

Le compartiment `training/jersey/` ajoute la lecture PARSeq aux boîtes suivies et
aux squelettes existants. Il sélectionne un crop du torse, lit le texte et
confirme un numéro après plusieurs observations espacées. La sortie reste
`null` quand les preuves sont insuffisantes. `0`, `00` et `01` restent des
chaînes distinctes. Aucun entraînement n'est nécessaire pour cette baseline.

## Effectifs obligatoires : `jersey.json`

Lorsque l'OCR est activé, placer un fichier **`jersey.json` dans le même dossier
que la vidéo**. Le chargement est automatique, une seule fois avant les modèles.
Un fichier absent ou invalide arrête le run avant toute inférence ; aucun mode
sans restriction n'est utilisé implicitement. Ce filtre ne sollicite pas le GPU.

Format minimal (les numéros restent des chaînes) :

```json
{
  "schema_version": 1,
  "teams": [
    {
      "team_id": "equipe_a",
      "name": "Équipe A",
      "color": "#45C7F5",
      "players": [{"number": "23", "name": "Joueur A"}]
    },
    {
      "team_id": "equipe_b",
      "name": "Équipe B",
      "players": [{"number": "00", "name": "Joueur B"}]
    }
  ]
}
```

Les deux équipes sont requises, avec des identifiants distincts, des noms non
vides et 1 à 64 joueurs chacune. Un numéro doit être unique **dans son équipe**,
mais peut exister dans les deux équipes. `0` et `00` sont différents. Les valeurs
numériques JSON (`23` au lieu de `"23"`), doublons de clés, doublons de numéros
intra-équipe et champs inconnus sont refusés.

`color` est facultatif pour chaque équipe et accepte une couleur hexadécimale
`#RRGGBB`. Le pipeline lui attribue une couleur d'affichage stable si elle est
absente. Une équipe non identifiée reste neutre ; les groupes de maillots
anonymes utilisent leur propre palette et ne sont pas assimilés à une équipe
de l'effectif.

`uniform_color` est aussi facultatif : il décrit la couleur dominante du maillot
porté dans ce match, distincte de la couleur utilisée pour dessiner les boîtes.
Le décodeur OCR final n'émet que `0`–`9` et accepte un ou deux chiffres. Il
examine trois bandes du torse, dont une centrée sur le numéro, toutes les deux
images, en conservant une seule lecture par image. Il s'arrête dès que trois lectures fiables donnent une
majorité d'au moins 75 %, puis reprend si la track change durablement de
couleur ou après une interruption. La confiance reste la probabilité du
vocabulaire complet du modèle :
forcer un chiffre sur du texte ne crée donc pas un vote fiable. Un numéro exige
au moins trois lectures fiables et une majorité stricte des lectures
numériques fiables sur l'ensemble de la track.

La couleur est estimée séparément sur la zone épaules–hanches de **chaque**
image visible. L'équipe demande au moins cinq couleurs classifiables et une
majorité stricte. Quand les deux équipes portent le même numéro, la couleur
majoritaire doit aussi correspondre aux images où le numéro a été lu. Une
portion de couleur contradictoire reste anonyme. Les identités sont appliquées
rétroactivement aux images compatibles ; les distances de plusieurs tracks du
même joueur sont additionnées sans relier leurs positions après une rupture.

Un joueur peut porter `eligible: false` pour être conservé dans l'effectif sans
autoriser la reconnaissance de son numéro. Le défaut est `true`. `status` est
facultatif et descriptif (`played`, `dnp`, `dnd`, `inactive`) ; seul `eligible`
pilote le filtre. Les métadonnées `match`, `sources` et `notes` sont facultatives.

Une lecture OCR absente de l'union des numéros éligibles est rejetée **avant le
vote temporel**, avec `rejection_reason: not_in_roster`. Elle n'est pas remplacée
par un numéro autorisé proche. Le texte et `raw_number` restent dans les
diagnostics, tandis que `number` vaut `null`. Le compteur `roster_rejections`
permet de mesurer ces rejets. Les budgets et seuils de confirmation restent actifs.

Après confirmation, les sorties `jersey` ajoutent `player_name`, `team_id`,
`team_name`, `identity_status` et `candidates`. Si le numéro n'appartient qu'à un
joueur éligible, son nom est résolu (`unique_number`). Si plusieurs équipes le
partagent, les noms restent dans `candidates` et les champs d'identité uniques
valent `null` (`ambiguous`). Sans numéro confirmé : `unresolved`. Cela prépare
l'affichage des noms ; la vidéo affiche encore uniquement l'ID et le numéro.
Aucune équipe n'est choisie arbitrairement pour lever une ambiguïté.

Le même crop du torse sert aussi à distinguer deux **groupes de couleur** dans
une vidéo. Un crop trop chevauché ou sans couleur dominante est ignoré. Il faut
trois observations cohérentes et espacées d'au moins 0,35 s sur une track avant
d'émettre `team_group: team_1` ou `team_2`. Ces étiquettes désignent seulement
les deux couleurs observées ; leur ordre dépend de la première confirmation du
run. Sans correspondance de couleur fournie par l'effectif, elles ne désignent
pas Houston ou Los Angeles et ne lèvent pas l'ambiguïté des numéros partagés.
Le visualiseur affiche `G1`/`G2` sur les joueurs et « Groupe couleur 1/2 » dans
les statistiques. Un groupe absent reste `null`.
Si une track présente ensuite des preuves cohérentes de la couleur opposée, le
groupe est retiré et `team_group_conflict` est signalé ; il n'est pas réattribué
à l'autre couleur sur cette même track.

Le run conserve une copie de l'effectif dans `roster.json` et le SHA-256
du fichier source dans `jersey.json` de provenance et les sidecars. L'effectif
est figé pour le run : modifier le fichier pendant le traitement ne modifie pas
ses décisions. Le `jersey.json` source n'est jamais écrasé par le run.

### Exemple Lakers–Houston, match 1 des playoffs 2026

Le fichier opérationnel est `datas/train/HOU_LAL/G1/jersey.json`. Une copie
versionnable est fournie dans
[examples/hou_lal_g1_2026/jersey.json](examples/hou_lal_g1_2026/jersey.json),
car `datas/` est suivi séparément par DVC et ignoré par Git.

L'effectif du **18 avril 2026** provient de la
[feuille officielle NBA](https://statsdmz.nba.com/pdfs/20260418/20260418_HOULAL.pdf).
Il contient 15 joueurs par équipe, y compris les quatre inactifs mentionnés
séparément. Les numéros des inactifs sont documentés dans les sources du JSON.
25 joueurs sont éligibles : les joueurs entrés et les remplaçants DNP. Kevin
Durant (DND), Steven Adams, Fred VanVleet, Luka Dončić et Austin Reaves (inactifs)
sont conservés avec `eligible: false`. Les numéros `10` (Jabari Smith Jr./Luke
Kennard) et `28` (Alperen Sengun/Rui Hachimura) restent notamment ambigus sans
reconnaissance d'équipe.

## Utilisation

Les environnements joueurs restent séparés de la balle. Synchroniser le lock
CPU ou CUDA décrit dans [le guide joueurs](../players/VALIDATION.md) ; l'ajout
est `timm==0.9.16`, compatible avec le modèle PARSeq de référence.

```bash
# Profil CPU :
uv pip sync --python .venv-players-cpu/bin/python --torch-backend cpu \
  training/players/requirements-cpu.lock
make setup-jersey PLAYERS_PYTHON=.venv-players-cpu/bin/python
make setup-players-pose PLAYERS_PYTHON=.venv-players-cpu/bin/python
make predict-jersey PLAYERS_PYTHON=.venv-players-cpu/bin/python \
  VIDEO=datas/train/HOU_LAL/G1/video.mp4 CHECKPOINT=/poids/yolo26n.pt \
  OUTPUT=runs/jersey/match MAX_FRAMES=300
```

`setup-jersey` télécharge explicitement le checkpoint officiel dans
`models/jersey/parseq.pt`, vérifie son SHA-256 avant publication atomique et
refuse d'écraser un modèle différent. Pour un import local, passer
`JERSEY_SOURCE=/chemin/parseq-bb5792a6.pt`. Le modèle est déjà installé sur le
poste de développement. L'inférence ne télécharge ni code ni poids.

`predict-jersey` active tracking, pose et OCR dans le parcours vidéo joueurs.
YOLO et RF-DETR fournissent le même contrat de détections. Utiliser `CONFIG=...`
pour choisir le détecteur, les poids et son device. Le dossier de sortie doit
être nouveau. Les options `JERSEY_WEIGHTS=...` et `JERSEY_DEVICE=cpu|cuda:0`
sont disponibles. Les options CLI sont `--jersey`, `--no-jersey`,
`--jersey-weights` et `--jersey-device`.

Pour ajouter seulement l'OCR à une recette existante :
`make predict-players CONFIG=... JERSEY=1`. Le tracking doit être activé ; la
pose est facultative, avec une région centrale de poitrine en son absence.
Les recettes anciennes et `JERSEY=0` conservent le parcours précédent.

## Budget de calcul et sélection temporelle

L'OCR utilise **le CPU par défaut, avec deux threads**, même si le détecteur et
la pose sont sur GPU. Les threads Torch précédents sont restaurés après lecture.
Le modèle est chargé une fois, au premier lot utile. Aucun travail OCR n'est
lancé pour une frame vide, une boîte sans ID confirmé ou un crop rejeté.

| Réglage `jersey` | Défaut | Effet |
| --- | ---: | --- |
| `sample_interval_seconds` | 0.1 | Espacement des évaluations de crops par track |
| `selection_window_seconds` | 0.3 | Fenêtre pendant laquelle conserver le meilleur crop |
| `read_interval_seconds` | 0.75 | Espacement minimal entre lectures d'une track incertaine |
| `confirmed_interval_seconds` | 3.0 | Recontrôle moins fréquent après confirmation |
| `batch_size` | 4 | Plafond strict des crops par appel, jamais supérieur à 4 |
| `crops_per_video_second` | 8 | Recharge du budget global en temps vidéo, réserve initiale d'un lot |
| `max_duty_cycle` | 0.15 | Temps OCR puis pause proportionnelle, mesurés en temps réel |
| `min_batch_interval_seconds` | 0.1 | Intervalle minimal entre débuts de lots |
| `gpu_reserve_mb` | 1024 | Mémoire GPU libre à préserver, plus marge de travail OCR |
| `max_tracks` | 64 | Nombre maximal d'états temporels conservés |
| `max_evidence` | 8 | Nombre maximal de votes conservés par état |
| `state_ttl_seconds` | 10 | Expiration des tracks absentes |
| `evidence_ttl_seconds` | 12 | Expiration des preuves, même si la track reste visible |

Les durées de sélection, de lecture et d'expiration sont en **temps vidéo**.
Le budget par seconde utilise un réservoir plafonné à un lot : après une pause,
il n'y a pas de rattrapage massif. La priorité va aux tracks jamais lues, puis
aux lectures les plus anciennes. Un numéro confirmé est recontrôlé ; une
abstention n'est pas définitive et pourra être levée lorsque le maillot devient
visible. À saturation, un état absent peut être finalisé pour libérer une place ;
les observations exclues portent `capacity_limit`.

Le budget de temps réel impose après un lot de durée `d` une pause d'au moins
`d × (1 / max_duty_cycle − 1)` avant le prochain lot. Il inclut le chargement à
froid. La boucle vidéo continue pendant cette pause : elle ne dort pas et
n'accumule pas une file de travaux. C'est une limite moyenne de travail OCR,
**pas un plafond instantané d'utilisation du GPU** ; un lot reste indivisible.
Les sorties peuvent varier entre deux machines, puisque le budget réel change
les crops effectivement lus. Les lectures choisies sont toutes journalisées.

Le mode CUDA est volontaire : `JERSEY_DEVICE=cuda:0` avec le lock CUDA 13.
Les inférences détecteur, pose et OCR sont séquentielles, sans processus OCR
concurrent. Avant chaque lot, la mémoire libre est contrôlée : réserve utilisateur
plus 768 Mio avant chargement, puis 512 Mio de marge de travail. Une pression
mémoire reporte l'OCR et produit `resource_deferred`. Un OOM désactive l'OCR CUDA
pour le reste du run ; les autres étapes continuent si leur propre budget le
permet. Aucun repli CPU implicite et aucun `empty_cache()` par frame.
Ces contrôles ne réservent pas la mémoire vis-à-vis d'autres processus.

Seuls des crops RGB **128×32** sont gardés en attente, un par état (768 Kio au
plafond par défaut, hors objets Python/modèle/frame courante). Les votes sont
bornés et les tracks terminées sont écrites au fil de l'eau. Les crops complets
et la vidéo entière ne sont jamais accumulés en mémoire.

## Cadrage et confirmation

Les épaules et hanches COCO17 donnent une région centrale du torse lorsqu'elles
sont valides. Les géométries incohérentes et profils trop étroits sont rejetés.
Sinon, la région poitrine est relative à la boîte. Taille minimale, netteté,
contraste et chevauchement avec d'autres personnes filtrent les crops sur CPU.
Le meilleur score de qualité de la fenêtre est conservé ; la qualité ne prouve
pas qu'un numéro soit visible.

PARSeq lit l'alphabet complet, puis seules les chaînes ASCII d'un ou deux
chiffres terminées par EOS sont acceptées. Aucune concaténation de chiffres,
correction de lettres ou troncature n'est appliquée. La confiance est le produit
des probabilités des caractères et de la fin de séquence ; elle n'est pas calibrée
comme une probabilité que le numéro attribué au joueur soit correct. Le numéro doit en plus être autorisé
par `jersey.json` avant de contribuer à la fusion.

Par défaut, un vote exige `min_ocr_confidence: 0.9`. Il est pondéré par confiance
OCR × qualité du crop. La confirmation demande au moins deux votes espacés,
`min_agreement: 0.75` du poids total et les deux dernières lectures acceptées
concordantes. Une contradiction acceptée retire immédiatement le numéro affiché.
Des lectures illisibles n'ajoutent pas de vote ; les preuves précédentes expirent
normalement. Deux régions d'une même frame ne peuvent pas compter comme deux
preuves : une seule est lue par observation.

L'identité reste `(tracking_run_id, segment_id, track_id)` ; aucune équipe n’est reconnue visuellement ; un nom peut seulement être résolu
par un numéro unique dans l’effectif. Aucune unicité globale n’est imposée. Deux joueurs peuvent avoir le même
numéro. Les remises à zéro explicites du tracker purgent les preuves. Les coupes
connues doivent être déclarées dans `tracking.reset_frames` : leur détection
automatique et la correction des changements erronés d'ID ne sont pas incluses.

## Artefacts

| Fichier | Contenu |
| --- | --- |
| `jerseys.jsonl` | Une ligne par frame : mêmes boîtes, scores et IDs, plus `jersey` |
| `jersey_reads.jsonl` | Lectures brutes, crop source, temps/frame choisis, qualité, confiance et motif de rejet/report |
| `jersey_tracks.jsonl` | État final de chaque track ou segment d'état expiré, avec les votes conservés |
| `jersey.json` | Configuration, SHA-256 des poids, du code et de l’effectif, liens vers vidéo/détecteur/suivi/pose |
| `roster.json` | Copie de l’effectif utilisé pour le run |
| `annotated.mp4` | Label `#track_id jersey numéro` uniquement après confirmation |
| `result.json` | Compteurs OCR, temps d'inférence, pics d'états/crops et empreintes des sorties |

Le champ `jersey` contient `number` (chaîne ou `null`), `status`, `votes` et
`agreement` (part du poids du candidat dominant, pas une probabilité calibrée).
Les états possibles sont `confirmed`, `uncertain`, `unknown`, `untracked`,
`below_detection_confidence` et `capacity_limit`. Les crops lus après attente
conservent leur `frame_index` d'origine ; `processed_frame` indique la frame à
laquelle leur résultat est devenu disponible. Il n'y a pas de réécriture des
frames passées après confirmation.

Les sorties tracking/pose/détection gardent leurs contrats. L'export maillot
n'y injecte pas de numéros. Les sidecars maillot partagent les identifiants du run
et le SHA-256 vidéo. Après interruption, les JSONL restent `.partial.jsonl` et
`progress.json` signale l'échec ou l'arrêt. Recommencer dans un nouveau dossier ;
il n'y a pas de reprise intermédiaire du tracker/OCR.

## Sources et vérification

Le lecteur utilise un sous-ensemble d'inférence du
[PARSeq officiel](https://github.com/baudm/parseq), révision
`1902db043c029a7e03a3818c616c06600af574be`, avec licence et notices conservées dans
[vendor/parseq](vendor/parseq/UPSTREAM.md). Adaptations : retrait de dépendances
d'entraînement et du tokenizer lourd ; chargement strict de tous les paramètres,
`weights_only=True`, inférence FP32 sans gradients, décodage limité à trois
caractères plus EOS afin de rejeter les textes trop longs sans lire 25 positions.
L'architecture et les poids viennent de la baseline générique utilisée dans NBA,
sans dépendance de code ou chemin NBA à l'exécution. Le checkpoint JNR spécialisé
n'était pas disponible : les anciens scores JNR ne s'appliquent pas ici.

Poids : [release PARSeq](https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt).
SHA-256 : `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00`.
Les détails de cette implémentation diffèrent de la fusion historique NBA :
aucune équivalence de qualité n'est annoncée. Le gold NBA utilise d'autres IDs
et ne permet pas une comparaison directe avec les nouvelles tracks.

```bash
make test-jersey PYTHON=.venv-players-cpu/bin/python
JERSEY_MODEL_TESTS=1 make test-jersey PYTHON=.venv-players-cpu/bin/python
JERSEY_MODEL_TESTS=1 JERSEY_DEVICE=cuda:0 \
  .venv-players/bin/python -m unittest training.jersey.tests.test_integration.RealReaderTests -v
```

Les contrats n'ont besoin ni des poids ni du GPU. Ils couvrent les budgets,
la confirmation/contradiction, l'expiration, les changements de segment,
les zéros initiaux, la mémoire bornée, les crops, les exports et les interruptions.
`validate-players` inclut maintenant cette suite dans la validation hors réseau.
Le test réel optionnel lit des chiffres synthétiques et contrôle l'abstention
sur un crop vide ; il ne mesure pas la précision sur les matchs.

Les artefacts de validation sont dans `runs/jersey/validation/`. L’essai final
HOU/LAL de 300 frames (10 secondes vidéo, YOLO26 Nano et RTMPose sur RTX 5080,
OCR CPU) a lancé 72 lectures en 48 lots : 2,33 secondes OCR sur 20,30 secondes
au total, 12 crops simultanément en attente (144 Kio). Deux tracks ont fini avec
un numéro confirmé. Ces quantités démontrent le filtrage et les limites de calcul,
pas une précision de reconnaissance. La lecture sur les maillots occultés,
petits ou flous reste limitée ; l'abstention est volontaire.

Validation du 27 septembre 2026 : **21 tests maillot réussis**, modèle CPU réel
inclus ; la version hors réseau sans poids réussit 20 tests et désactive ce test
réel. Le test du lecteur réel passe aussi sur CUDA. Non-régressions : 123 tests
joueurs réussis (10 optionnels désactivés) et 56 tests balle réussis (13 optionnels
désactivés). Les deux environnements passent `uv pip check`.

Le lecteur CUDA seul, sur RTX 5080 avec un lot de quatre crops, atteint environ
109 Mio alloués et 136 Mio réservés par PyTorch (`gpu-memory.json`). Cette mesure
exclut le détecteur, la pose et le contexte CUDA ; elle n'est pas une limite dure
de VRAM. Les poids et sources du lecteur sont locaux. Les futurs bundles de code
joueurs embarquent également le module maillot et ses notices, sans charger ni
inclure automatiquement les poids OCR dans un candidat détecteur.

Ajout du filtre d'effectif (27 septembre 2026) : tests du fichier obligatoire,
de la validation stricte, de l'abstention hors liste même à forte confiance,
des joueurs exclus, des ambiguïtés interéquipes, de la copie figée et des noms
exportés. Aucun entraînement, nouveau modèle ou transfert DVC n'est nécessaire.
