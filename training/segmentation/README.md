# Expérience de segmentation joueurs / arbitres

Travail expérimental isolé : aucune promotion de modèle, aucun fine-tuning et
aucune fusion automatique de PR. La configuration courante reste en détection
bbox : `segmentation.strategy: disabled`, `jersey.mask_policy: disabled`.

Deux chemins partagent le suivi, RTMPose et PARSeq existants :

- `box_prompt` : E-BARD détecte les joueurs et les arbitres ; SAM 2.1 Tiny reçoit
  leurs boîtes comme prompts sur la frame entière. Un seul encodage image, un
  masque par prompt, découpé dans la boîte. Les boîtes, scores et rôles du
  détecteur sont conservés. `segment_referees: false` permet une ablation.
- `direct` : YOLO26 L ou X segmentation produit directement les instances et
  leurs boîtes utilisables par le suivi. Aucun détecteur joueur préalable. Le
  détecteur E-BARD n'est appelé que pour la classe arbitre. Les instances
  personne chevauchant un arbitre à score ≥ 0,25 / IoU ≥ 0,50 sont retirées.
  Les arbitres gardent uniquement leur boîte.

Les checkpoints YOLO26 sont préentraînés COCO : **personne n'est pas joueur**.
Les profils expérimentaux autorisent explicitement le proxy
`non_referee_person_proxy`. Il peut inclure des spectateurs, des personnes du
banc et des arbitres manqués. Cette limite figure dans la provenance et les
faux positifs restent dans le score. Aucun filtrage par les annotations n'est
appliqué aux prédictions. Un futur modèle spécialisé peut fournir sa propre
`class_roles` et conserver `person_role_policy: unknown`.

Les stratégies comparent des configurations complètes. SAM et YOLO n'ont pas
les mêmes poids, architecture, entraînement ni résolution : cette expérience
ne prouve pas qu'une différence est causée par la segmentation seule.

## Installer et utiliser

Les dépendances sont celles du lock joueurs (Ultralytics 8.4.39). L'inférence
exige des checkpoints locaux et n'effectue aucun téléchargement. Installation
explicite, avec SHA-256 épinglés et publication exclusive :

```bash
.venv-players/bin/python -m training.segmentation.setup
.venv-players/bin/python -m pipeline analyze \
  --video datas/train/HOU_LAL/G1/video.mp4 \
  --config configs/segmentation/box-prompt.yaml \
  --output runs/segmentation-example-sam --max-frames 150
.venv-players/bin/python -m pipeline analyze \
  --video datas/train/HOU_LAL/G1/video.mp4 \
  --config configs/segmentation/direct-l.yaml \
  --output runs/segmentation-example-l --max-frames 150
# Remplacer direct-l.yaml par direct-x.yaml pour X.
```

Ces profils désactivent balle et terrain pour isoler la perception des personnes.
Ils n'estiment donc pas la distance. Le bloc segmentation peut aussi être ajouté
au profil complet. Un dossier de sortie neuf est obligatoire.

## Masques et OCR

Chaque observation segmentée porte `segmentation.status`, un éventuel score de
qualité SAM (non calibré) et `mask: {encoding: roi_rle_c, roi, counts}`. `roi`
est une boîte entière `[left, top, right, bottom]` en pixels source, bornes de
fin exclues. `counts` alterne fond / avant-plan, commence par le fond et parcourt
les lignes (ordre C). La somme doit égaler l'aire de la ROI. Un masque vide
conserve la boîte avec statut `empty` ; il ne compte pas comme masque OCR valide.
Le suivi conserve l'association masque / détection et les sorties restent JSON.

Le masque ne remplace pas la pose : il couvre toute la personne, pas seulement
le numéro. Le crop torse utilise les mêmes règles pose/bbox qu'avant.

| `jersey.mask_policy` | Comportement |
| --- | --- |
| `disabled` | Pixels OCR initiaux ; témoin et valeur par défaut |
| `background_neutral` | Fond hors personne à RGB 127 ; repli sur crop initial si masque absent ou couverture insuffisante |
| `require_mask` | Même masquage ; rejet du crop si masque absent ou couverture insuffisante |

Ces politiques concernent le crop **numérique OCR** : sa position vient de la
pose ou de la bbox, et le fond extérieur au masque est neutralisé avant PARSeq.
Les lectures finales conservent aussi `mask_audit` pour rendre le repli visible.
La **couleur d'équipe** utilise séparément le masque du torse dès qu'il est
disponible, pour les groupes anonymes comme pour la consolidation finale.
Les pixels neutralisés ne participent pas aux statistiques de couleur.

La couverture minimale du crop est `min_mask_coverage: 0.5`. Elle ne garantit
pas que tous les chiffres soient conservés. La netteté et le contraste sont
calculés **avant** le masquage ; les bords artificiels ne peuvent améliorer la
qualité du crop. Les crops couleur appliquent le masque dans les deux branches
de l'ablation OCR, puis excluent les pixels neutralisés des votes. Le filtre de chevauchement
existant reste actif. `mask_audit` est journalisé par le lecteur temporel dans `jersey_reads.jsonl`.
La postpasse existante de votes reçoit aussi la politique de masque via la
configuration et conserve ses propres variantes, seuils et règles de sélection ;
son journal v4 sérialise également `mask_audit` pour les nouvelles lectures.

## Protocole et reproduction

Toutes les frames **vérifiées**, train et val, sont admissibles puisque les
segmentateurs ne sont pas fine-tunés sur ces vidéos. Les frames simplement
proposées ne deviennent jamais de la vérité terrain. Les images voisines restent
corrélées ; les scores par vidéo et split doivent être lus avec le total.

```bash
# Nouvel export distinct des données de travail ; annotations source inchangées.
.venv-players/bin/python -m training.players.export \
  --source datas --output runs/segmentation-data-new \
  --config training/players/configs/players_export.yaml

# Figer avant toute inférence. Les chemins relatifs partent de la racine du dépôt.
.venv-players/bin/python -m training.segmentation.evaluate freeze \
  --dataset runs/segmentation-data-new/coco \
  --output reports/segmentation-new

# Une famille à la fois, processus neuf pour isoler la mesure mémoire.
.venv-players/bin/python scripts/run_segmentation_comparison.py \
  --protocol reports/segmentation-new/protocol.json \
  --output runs/segmentation-new

.venv-players/bin/python scripts/segmentation_report.py \
  --runs runs/segmentation-new --report reports/segmentation-new
```

Le protocole inclut les empreintes du code, des poids, des images sélectionnées
et de l'export. Il est vérifié avant et après les essais. Après une modification
du code ou des données, figer un **nouveau** protocole dans un nouveau dossier.
Les sorties brutes privées restent dans `runs/`, avec empreintes dans le rapport.
Les résultats récapitulatifs, courbes et métadonnées des cas sont versionnés.
Les images issues des vidéos et leur page `review/review.html` restent privées
dans `runs/<comparaison>/`, comme les vidéos source.

Détection : AP bbox à IoU .50 et .50:.95 avec un passage à score ≥ .001 ;
précision/rappel/F1 à score ≥ .25 et IoU ≥ .50 avec le véritable appel de
production. Appariement un-à-un, score décroissant. Les métriques ne sont pas
une AP de masque. Les mesures temporelles couvrent perception + retour CPU +
encodage des masques, hors décodage disque, suivi, pose et OCR. Elles ne sont
pas un débit vidéo complet. SAM reste à 1024, détecteur et YOLO à 640 ; les
prétraitements rectangulaires natifs et les réglages numériques sont consignés.

OCR : chaque quinzième frame vérifiée par vidéo, même crop testé brut et masqué,
PARSeq identique. Ce test contrôle la variation des lectures par crop ; il
n'utilise ni les budgets réels, ni l'effectif, ni les votes temporels. Il ne
remplace pas une mesure de précision d'identité de bout en bout. Les lectures
sur des prédictions non appariées aux joueurs sont comptées séparément.

## Annotations de contrôle et scores supplémentaires

Les annotations joueurs existantes évaluent les **boîtes**. Les scores IoU/Dice
masque et précision OCR restent `null` en l'absence de labels alignés et revus.
La confiance OCR n'est jamais utilisée comme vérité terrain. Le gold historique
NBA porte sur d'autres tracks et n'est pas réutilisé par égalité d'ID.

Le rapport génère `runs/<comparaison>/review-template.jsonl`, identifié par frame,
SHA-256 image et `object_id` d'annotation, avec statut `pending`. Après revue :

- `expected_number`: chaîne exacte (`"07"` reste `"07"`), ou `null` pour exiger
  l'abstention lorsque le numéro est illisible ; retirer ce champ pour une
  annotation de masque seule ;
- `person_mask` et `number_mask` : masques RLE dans le même espace source, si
  disponibles ; ils mesurent IoU/Dice et la rétention des pixels du numéro ;
- `review_status: reviewed` : seuls ces objets entrent dans le score.

```bash
.venv-players/bin/python -m training.segmentation.review \
  --dataset runs/segmentation-data-new/coco \
  --run runs/segmentation-new/direct_l \
  --labels /path/to/reviewed.jsonl --output runs/segmentation-new/score-l.json
```

Les détections manquées restent dans le dénominateur. Les objets non annotés ne
sont pas des numéros illisibles par défaut. Comparer les politiques sur les mêmes
labels, sans sélectionner seulement les lectures acceptées. Pour une décision
finale d'identité, ajouter des trajectoires revues, les erreurs d'affectation,
les changements d'ID et le délai de confirmation.

Tests hors réseau :

```bash
.venv-players/bin/python -m unittest discover -s training/segmentation/tests -v
.venv-players/bin/python -m mlops.checks --output runs/segmentation-checks-new
```

Sources des modèles : [SAM 2.1 et prompts bbox](https://docs.ultralytics.com/models/sam-2/),
[segmentation d'instances YOLO](https://docs.ultralytics.com/tasks/segment/).
