# Validation du pipeline — 28 septembre 2026

## Tests automatisés

| Suite | Réussis | Optionnels désactivés |
| --- | ---: | ---: |
| Nouveau pipeline | 29 | 0 |
| Terrain | 21 | 0 |
| Maillots | 27 | 1 |
| Joueurs | 123 | 10 |
| Balle | 53 | 16 |
| **Total** | **253** | **27** |

Exécution avec `.venv-players/bin/python -m unittest discover -s <module>/tests`.
Les modèles réels sont évalués séparément ci-dessous. La compilation Python,
les exemples de configuration, l'installation idempotente du checkpoint E-BARD
et `git diff --check` ont également été vérifiés.

Les tests pipeline contrôlent les distances analytiques (ligne, virage), le
joueur immobile bruité, les temps variables, l'absence de mesure versus zéro,
les ruptures, le changement de point au sol, les conflits d'identité, les rôles,
l'OCR anonyme, les fichiers partiels, le recalcul et l'intégrité des artefacts.
Une vidéo synthétique décodée réellement valide les associations temporelles,
les fenêtres balle aux coupes et l'équivalence séquentiel/parallèle. Un test
vérifie que la pose calculée avant le tracking récupère les bons IDs.

## Modèles réels et protocole

RTX 5080, environnement joueurs CUDA, quatre threads CPU par runtime principal
et worker balle, détecteur E-BARD YOLOv8n à 704, terrain existant à 640, RTMPose,
PARSeq CPU et TrackNet V5 du run local
`1edd81e040564da2bfa36a757fb855f7`. Tous les poids sont locaux à l'inférence.
Tracking BoT-SORT séparé par rôle ; pose calculée en parallèle de la calibration
et du tracking CPU. Cadences et crops OCR fixés en temps vidéo.

Le benchmark utilise les 150 premières images HOU/LAL, soit cinq secondes vidéo,
sans rendu, avec les six capacités actives. Il inclut l'écriture des observations
et contributions. Le chargement est séparé ; les premières inférences ne font
pas l'objet d'un échauffement hors mesure. Deux runs par mode ont été effectués.

| Mode | Traitement 1 | Traitement 2 | Total avec chargement/finalisation |
| --- | ---: | ---: | ---: |
| Séquentiel | 12,055 s | 12,030 s | 15,137 / 15,090 s |
| Parallèle, concurrence GPU 2 | 8,144 s | 8,230 s | 11,154 / 11,192 s |

Sur ces essais : **environ 32 % de temps de traitement en moins**, soit un
facteur de débit d'environ **1,47×**. La réduction du temps total est d'environ
26 %. Ce résultat porte sur cet extrait et ce matériel, pas sur tout match ou GPU.
La recette conserve le réglage configurable ; la limite générique du code reste
une seule tâche GPU, tandis que `configs/analysis.yaml` en autorise deux.

Les observations des deux runs parallèles sont identiques au témoin séquentiel
après retrait des UUID de run et IDs d'observation qui en dérivent : détections,
poses, calibration, OCR, IDs de tracking et balle compris. Les observations
comprennent 1 454 joueurs et 434 arbitres **comptés par image**, pas des individus
uniques. La calibration est disponible sur 135/150 images. Ces comptages ne
mesurent ni précision de détection ni erreur métrique.

## Essais supplémentaires

| Entrée | Frames | Segments | Temps de traitement | Résultat |
| --- | ---: | ---: | ---: | --- |
| HOU/LAL complet, sans roster | 1 842 | 6 | 92,123 s | Terminé, 1 554 frames avec calibration |
| POR/SAS, sans roster | 150 | 1 | 8,268 s | Terminé, 150 frames avec calibration |
| HOU/LAL, avec roster | 150 | 1 | 8,146 s | Terminé, effectif copié et filtre actif |

HOU/LAL complet atteint environ **20 images/s** hors chargement. Le conteneur
annonce 1 843 frames mais le décodeur en restitue 1 842, écart déjà documenté dans
le POC terrain. La nouvelle chaîne publie les frames effectivement décodées et
signale `declared_decoded_frame_count_mismatch`. Ce signal doit être inspecté
pour une nouvelle vidéo, car il peut aussi indiquer une source tronquée.

Le run complet contient 205 sujets anonymes correspondant à des fragments de
tracks ; 131 ont au moins un intervalle métrique mesurable. **Ce ne sont pas
205 joueurs distincts.** La fragmentation et les changements d'ID limitent la
distance attribuable à une personne sans réidentification. Les 12 observations
balle sans contexte correspondent aux deux premières images des six segments.

Sur l'essai roster de cinq secondes, 126 observations OCR ont une identité
ambiguë et les autres restent non résolues : aucun nom unique n'est forcé. Ce
résultat confirme le traitement prudent des numéros partagés, sans démontrer
une bonne couverture nominative sur un match.

Une inspection visuelle de la frame 90 HOU/LAL confirme la présence de boîtes
joueurs et de deux arbitres correctement distingués sur cette image. Les lignes
projetées permettent d'inspecter la calibration, mais cette revue ponctuelle
n'est pas une annotation de vérité terrain ni une validation de distance.

ONNX Runtime a émis des diagnostics cuDNN « No execution plans support the
graph » lors de certaines recherches de kernels. Les appels de pose ont néanmoins
terminé et fourni des résultats ; aucune étape n'a été désactivée dans ces essais.

## Mémoire et limites

Le RSS maximal principal passe d'environ 3 182 Mio sur les runs parallèles courts
à 3 249 Mio sur le clip complet ; le worker balle reste autour de 1 464 Mio.
Les images en vol et le contexte temporel sont bornés. Le résumé des statistiques
croît avec le nombre de tracks/sujets, pas avec la totalité des observations.

Pics d'allocations PyTorch CUDA : environ 396 Mio dans le processus principal
et 461 Mio dans le worker balle. Ces chiffres **excluent ONNX Runtime et les
contextes CUDA** ; ils ne sont ni la VRAM totale ni une réservation matérielle.

La validation technique est acquise pour ces scénarios. Il manque une vérité
terrain annotée de trajectoires métriques pour annoncer une précision de
distance. Les seuils de filtrage restent provisoires. Les faux positifs,
changements d'ID, sauts, positions de pieds et biais de calibration restent des
sources d'erreur. Aucun temps réel à 30 FPS ni distance totale de match n'est
revendiqué.

## Artefacts locaux

Racine : `runs/analysis/validation/`.

- `summary.json` : synthèse des essais réels.
- `optimized-sequential-{1,2}/` et `optimized-gpu2-{1,2}/` : benchmark.
- `full-hou-lal/`, `por-sas/`, `optimized-roster/` : essais supplémentaires.
- Chaque dossier contient `statistics.json`, `observations.jsonl`, `tracks.jsonl`,
  `jersey_reads.jsonl`, `run.json` et `progress.json` ; le roster est présent
  seulement lorsqu'il a été explicitement fourni.
- `parallel-1/preview.jpg` : inspection visuelle ponctuelle précédant la
  séparation du tracking par rôle ; les IDs de cette image ne sont pas ceux
  des runs optimisés.

Des essais intermédiaires sont également conservés. `sequential-1/` a échoué
pendant l'ajout de l'instrumentation CUDA, avant l'inférence ; l'initialisation
du contexte avant la remise à zéro des compteurs corrige cette erreur. Ce run
reste marqué `failed` et n'entre pas dans le benchmark.
