# Inventaire des annotations de balle — 18 octobre 2026

Audit local généré à 2026-10-18T18:59:29.331864+02:00. Lecture des 8 vidéos et des sidecars `.ballann.json`, décodage complet FFmpeg, comparaison de la dernière image lisible avec OpenCV et validation avec le modèle de l’annotateur. Aucune annotation ni aucun export modifié. Aucun entraînement.

## Lecture des catégories

- **Position renseignée** : centre `cx/cy` et rayon. Comprend les positions marquées visibles et les positions estimées marquées occultées.
- **Position inconnue** : une entrée existe, avec `cx/cy/radius = null` et `occluded = true`. L’interface regroupe « occultée » et « hors champ » ; impossible de distinguer ces causes dans ces fichiers.
- **Sans annotation** : aucune entrée pour une image réellement décodable.
- La présence d’une entrée ne prouve pas sa justesse. Ce format ne comporte pas de statut de revue indépendant, de validateur ou de date de validation. Cet audit contrôle les données et leur structure, sans revue visuelle.

## Synthèse par split

| Split | Vidéos | Images décodables | Entrées | Avec position | Position inconnue | Sans annotation | Couverture des entrées |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6 | 12 734 | 11 471 | 9 913 | 1 558 | 1 263 | 90,1 % |
| val | 2 | 3 506 | 3 307 | 2 928 | 379 | 199 | 94,3 % |
| test | 0 | 0 | 0 | 0 | 0 | 0 | — |

Total : **16 240 images**, **12 841 positions**, **1 937 positions inconnues**, **1 462 images sans annotation**. Couverture des entrées : 91,0 % ; couverture des positions : 79,1 %.

## Vidéo par vidéo

| Vidéo | Split | Images décodables | Avec position | Dont occultées estimées | Inconnues | Sans annotation | Couverture des entrées |
|---|---|---:|---:|---:|---:|---:|---:|
| HOU_LAL / G1 | train | 1 842 | 1 482 | 0 | 215 | 145 | 92,1 % |
| MIN_DEN / G3 | train | 1 815 | 1 413 | 0 | 334 | 68 | 96,3 % |
| TOR_CLE / G4 | train | 2 095 | 1 812 | 0 | 283 | 0 | 100,0 % |
| PHI_BOS / G4 | train | 2 720 | 1 825 | 0 | 297 | 598 | 78,0 % |
| ORL_DET / G4 | train | 1 814 | 1 443 | 1 | 189 | 182 | 90,0 % |
| ORL_DET / G5 | train | 2 448 | 1 938 | 1 | 240 | 270 | 89,0 % |
| POR_SAS / G5 | val | 1 642 | 1 421 | 0 | 165 | 56 | 96,6 % |
| PHX_OKC / G4 | val | 1 864 | 1 507 | 0 | 214 | 143 | 92,3 % |

## Images sans annotation : toutes les plages

Indices à partir de zéro ; bornes de frames inclusives. Temps affichés sous forme `[début, fin[` pour couvrir exactement les images concernées. Les durées indiquent du contenu à visionner, pas une estimation du temps de travail.

### train/HOU_LAL/G1 — 145 images à examiner

| Frames inclusives | Nombre | Temps [début, fin[ |
|---|---:|---|
| 400–414 | 15 | 00:13.333 → 00:13.833 |
| 416–437 | 22 | 00:13.867 → 00:14.600 |
| 983–1016 | 34 | 00:32.767 → 00:33.900 |
| 1243–1316 | 74 | 00:41.433 → 00:43.900 |

Positions inconnues déjà enregistrées : 215. Toutes les plages figurent dans `ball_review_ranges.csv`.

### train/MIN_DEN/G3 — 68 images à examiner

| Frames inclusives | Nombre | Temps [début, fin[ |
|---|---:|---|
| 644–675 | 32 | 00:21.467 → 00:22.533 |
| 947–947 | 1 | 00:31.567 → 00:31.600 |
| 956–956 | 1 | 00:31.867 → 00:31.900 |
| 1473–1506 | 34 | 00:49.100 → 00:50.233 |

Positions inconnues déjà enregistrées : 334. Toutes les plages figurent dans `ball_review_ranges.csv`.

### train/TOR_CLE/G4 — 0 images à examiner

Aucune image décodable sans entrée. Les positions inconnues restent à revoir selon leur difficulté et leur utilité.

Positions inconnues déjà enregistrées : 283. Toutes les plages figurent dans `ball_review_ranges.csv`.

### train/PHI_BOS/G4 — 598 images à examiner

| Frames inclusives | Nombre | Temps [début, fin[ |
|---|---:|---|
| 273–449 | 177 | 00:09.100 → 00:15.000 |
| 633–633 | 1 | 00:21.100 → 00:21.133 |
| 640–640 | 1 | 00:21.333 → 00:21.367 |
| 711–1121 | 411 | 00:23.700 → 00:37.400 |
| 1268–1273 | 6 | 00:42.267 → 00:42.467 |
| 1515–1515 | 1 | 00:50.500 → 00:50.533 |
| 2336–2336 | 1 | 01:17.867 → 01:17.900 |

Positions inconnues déjà enregistrées : 297. Toutes les plages figurent dans `ball_review_ranges.csv`.

### train/ORL_DET/G4 — 182 images à examiner

| Frames inclusives | Nombre | Temps [début, fin[ |
|---|---:|---|
| 496–496 | 1 | 00:16.533 → 00:16.567 |
| 610–612 | 3 | 00:20.333 → 00:20.433 |
| 661–662 | 2 | 00:22.033 → 00:22.100 |
| 1019–1191 | 173 | 00:33.967 → 00:39.733 |
| 1695–1697 | 3 | 00:56.500 → 00:56.600 |

Positions inconnues déjà enregistrées : 189. Toutes les plages figurent dans `ball_review_ranges.csv`.

### train/ORL_DET/G5 — 270 images à examiner

| Frames inclusives | Nombre | Temps [début, fin[ |
|---|---:|---|
| 397–502 | 106 | 00:13.233 → 00:16.767 |
| 1144–1199 | 56 | 00:38.133 → 00:40.000 |
| 1361–1464 | 104 | 00:45.367 → 00:48.833 |
| 1724–1724 | 1 | 00:57.467 → 00:57.500 |
| 2444–2445 | 2 | 01:21.467 → 01:21.533 |
| 2447–2447 | 1 | 01:21.567 → 01:21.600 |

Positions inconnues déjà enregistrées : 240. Toutes les plages figurent dans `ball_review_ranges.csv`.

### val/POR_SAS/G5 — 56 images à examiner

| Frames inclusives | Nombre | Temps [début, fin[ |
|---|---:|---|
| 1586–1641 | 56 | 00:52.867 → 00:54.733 |

Positions inconnues déjà enregistrées : 165. Toutes les plages figurent dans `ball_review_ranges.csv`.

### val/PHX_OKC/G4 — 143 images à examiner

| Frames inclusives | Nombre | Temps [début, fin[ |
|---|---:|---|
| 1410–1552 | 143 | 00:47.000 → 00:51.767 |

Positions inconnues déjà enregistrées : 214. Toutes les plages figurent dans `ball_review_ranges.csv`.

## Ordre conseillé pour terminer la revue

1. **Validation POR_SAS/G5** : revoir les frames **1586–1641** (56 images ; 52,867–54,733 s).
2. **Validation PHX_OKC/G4** : revoir les frames **1410–1552** (143 images ; 47,000–51,767 s).
3. **Train PHI_BOS/G4** : 598 images sans entrée, en commençant par **711–1121** (411) puis **273–449** (177).
4. **Train ORL_DET/G5** : 270 images sans entrée ; puis **ORL_DET/G4** : 182 ; **HOU_LAL/G1** : 145 ; **MIN_DEN/G3** : 68.
5. **Revoir les cas inconnus**, en priorité les 379 de validation, puis ceux de train où la balle peut être localisée avec confiance. **TOR_CLE/G4** n’a aucun trou mais conserve 283 positions inconnues.

Il y a **1 462 images sans annotation à examiner**. Leur examen doit décider si une position peut être renseignée ; une balle réellement impossible à localiser peut rester « inconnue ». Il ne faut pas inventer sa position ni convertir automatiquement une inconnue en exemple négatif.

Les **1 937 entrées « position inconnue »** ne sont pas forcément des erreurs ou du travail inachevé. Une revue permet de confirmer ces cas, de récupérer d’éventuelles positions visibles et de documenter les raisons. Le format actuel ne distingue pas « absente », « hors champ » et « occultée ».

## Exports disponibles et séquences temporelles

Les exports existants YOLO, COCO et TrackNet/TOTNet utilisent `unknown_position: exclude` et `occluded_position: include`. Les hashes des 8 sidecars correspondent aux sources enregistrées dans les trois manifests ; les nombres et statuts des lignes `frames.jsonl` sont cohérents. Les octets de toutes les images exportées n’ont pas été revérifiés pour cet inventaire.

- **Train : 9 913 images avec position** ; validation : **2 928**. Les inconnues et les images sans annotation sont exclues.
- **TrackNet SDK : 9 467 triplets train et 2 836 triplets validation**. Une discontinuité dans les positions conservées coupe les séquences. Ces comptes ne tiennent pas compte de coupures caméra à labelliser.
- Aucun exemple explicitement « balle absente » dans ces exports. Les mesures portent sur les positions conservées ; elles ne qualifient pas à elles seules le comportement sur tous les cas négatifs ou inconnus.

| Vidéo | Segments de positions consécutives | Triplets SDK possibles | Plus long segment |
|---|---:|---:|---:|
| train/HOU_LAL/G1 | 38 | 1 410 | 210 images |
| train/MIN_DEN/G3 | 28 | 1 358 | 237 images |
| train/TOR_CLE/G4 | 48 | 1 721 | 140 images |
| train/PHI_BOS/G4 | 53 | 1 725 | 249 images |
| train/ORL_DET/G4 | 31 | 1 383 | 180 images |
| train/ORL_DET/G5 | 35 | 1 870 | 301 images |
| val/POR_SAS/G5 | 21 | 1 379 | 375 images |
| val/PHX_OKC/G4 | 26 | 1 457 | 184 images |

## Contrôles et limites

- **8 documents valides** selon le chargeur de l’annotateur : types, coordonnées finies, centres dans l’image, rayons positifs, indices et métadonnées compatibles. Aucun indice annoté hors des images décodables.
- **2 positions seulement sont marquées occultées/estimées** : `train/ORL_DET/G4`, frame **0** ; `train/ORL_DET/G5`, frame **2134**. Cela ne suffit pas à conclure à l’absence d’autres occultations et limite l’analyse spécifique de cette catégorie.
- **Aucun fichier d’annotation de balle changé depuis l’inventaire du 17 octobre** : les 8 hashes sont identiques.
- Les conteneurs de HOU_LAL/G1, MIN_DEN/G3, TOR_CLE/G4, ORL_DET/G4 et POR_SAS/G5 déclarent chacun une image finale non décodable. Elles ne sont pas comptées comme du travail à faire. En revanche **ORL_DET/G5 frame 2447 est bien décodable** et fait partie des trous.
- Les sidecars gardent des chemins historiques : **PHI_BOS/G4** mentionne un ancien dossier `val` alors qu’il est aujourd’hui dans `train` ; **PHX_OKC/G4** mentionne un ancien dossier `train` alors qu’il est aujourd’hui dans `val`. Le dossier actuel fait autorité pour les exports. Ces chemins ne prouvent pas une fuite, mais la provenance des entraînements antérieurs doit être contrôlée avant de présenter PHX_OKC comme validation jamais vue.
- **Aucun jeu test indépendant** présent. Le préparer à partir d’autres matchs avant le gel final ; ne pas créer le test en découpant simplement des images voisines d’un match utilisé pour entraîner.

## Utiliser l’annotateur

Exemple pour commencer par la validation POR_SAS :

```bash
make annotate-ball VIDEO=datas/val/POR_SAS/G5/video.mp4
```

Le raccourci **f** va à la prochaine image sans entrée ; un clic renseigne le centre ; **↑/↓** ajuste le rayon ; **h** marque la position inconnue ; **v** bascule le statut d’occultation d’une position ; **←/→** navigue image par image. L’enregistrement est automatique. Ne marquer une position que si elle est défendable dans l’image et son contexte.

Après correction des annotations, régénérer les exports utilisés et enregistrer leurs nouvelles identités avant le prochain entraînement. Les comptes ci-dessus devront alors être recalculés.

## Fichiers de cet audit

- `inventory.json` / `inventory.csv` : inventaire source complet, hashes et décodage FFmpeg.
- `ball_validation.json` : contrôle du schéma, du dernier index lisible OpenCV et des séquences.
- `ball_inventory.csv` : tableau des 8 vidéos consacré à la balle.
- `ball_review_ranges.csv` : toutes les plages manquantes et inconnues avec frames, temps et hashes de référence.
- `ball_exports.json` : comparaison des exports existants aux sources actuelles.
