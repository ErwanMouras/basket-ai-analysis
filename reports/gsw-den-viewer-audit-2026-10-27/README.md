# Revue du viewer GSW–DEN — 27 octobre 2026

Analyse concernée : `runs/analysis/gsw-den-2026-10-26-full`.
Les observations, statistiques historiques, poids et vidéo source sont conservés.
Les captures, mesures et relevés détaillés restent locaux dans
`runs/audit-gsw-den-2026-10-27/`.

## Décalage vidéo

La découpe MP4 sans réencodage conserve 134 images de décodage avant le début
visible : premier PTS vidéo à **−5,36 s**. OpenCV et Chromium les ignorent ;
Firefox 152 les présente. À 10 s dans Firefox, l'image correspond à 4,64 s
de la séquence analysée ; à 60 s elle correspond à 54,64 s.

Une copie de lecture est encodée avec **NVENC**, sans images vidéo de PTS
négatif, et conserve les **57 861 images décodées**. `playback.json` lie par
SHA-256 la source, la copie et le nombre d'images de l'analyse. Le serveur
contrôle cette liaison et donne une URL versionnée à la vidéo pour invalider
les anciens buffers du navigateur. Les coordonnées et timestamps des
observations ne sont pas décalés artificiellement.

Contrôle visuel/pixel dans Firefox à 0, 10, 60, 600, 1 200 et 2 300 s : les
images correspondent désormais à la référence OpenCV/Chromium. Les légères
différences de pixels résultent du réencodage et des conversions couleur.
Un contrôle de lecture Firefox à 1× et 2× a vérifié **274 affichages** sans
écart entre le PTS présenté et le timestamp des annotations dessinées.
Une nouvelle inférence E-BARD sur cinq images source donne une IoU moyenne
de correspondance comprise entre 0,987 et 0,995 avec les boîtes enregistrées.

## Jeu placé et orientation

Le run n'avait pas de `scoreboard.json`. L'orientation ne pouvait donc pas
être partagée entre les plans d'un même quart-temps. Un profil explicite
du panneau Peacock est ajouté ; PARSeq lit les scores, le quart-temps et
l'horloge sur GPU. Deux lectures voisines confirment les valeurs.

Sur 4 629 échantillons, **4 391** ont un quart-temps confirmé et **3 829**
un score confirmé. Les quatre quarts disposent d'une orientation ancrée
dans les contacts observés, sans supposer automatiquement une alternance.
Les changements de score restent des observations du bandeau TV.

Le temps classé passe de **395,60 s** dans le relevé initial de l'API à
**766,48 s (12 min 46 s)**. Il mesure les intervalles vidéo de jeu placé
effectivement attribuables, et non le temps officiel de jeu. Les intervalles
restants sont explicitement ventilés :

| Motif | Secondes vidéo |
| --- | ---: |
| Transition ou arrêt / contact arbitre | 511,48 |
| Contact avec un joueur d'équipe non attribuée | 381,12 |
| Calibration indisponible | 347,32 |
| Indices insuffisants pour installer une phase | 156,12 |
| Orientation indisponible | 117,16 |
| Contact disputé | 20,96 |
| Coupes et discontinuités | 13,80 |

Ces catégories expliquent les exclusions du calcul ; elles ne constituent
pas des annotations humaines de temps de jeu. Aucun seuil n'a été abaissé
pour augmenter artificiellement la couverture.

## Tentatives non localisées

« Non localisé » signifie qu'une proposition ne peut pas être placée sur la
carte. Cela ne signifie pas « tir hors jeu placé ». Les propositions restent
comptées et conservent leur catégorie probable / à vérifier.

Deux corrections s'appliquent :

- L'orientation héritée du quart-temps est publiée aussi pour les plans
  sans vote local, afin que les cartes de tirs puissent l'utiliser.
- L'origine utilise les positions au sol du tireur observées juste avant
  le lâcher, même lorsque le contact main-ballon n'est visible que brièvement.
  Elle reste bornée à 0,25 s, au même plan, à une track unique de la même
  équipe et à une position stable. Le ballon en vol n'est pas projeté au sol.

| Équipe | Propositions | Localisées avant → après | Non localisées avant → après |
| --- | ---: | ---: | ---: |
| Golden State | 63 | 19 → 37 | 44 → 26 |
| Denver | 49 | 12 → 32 | 37 → 17 |

Pour Denver, les 17 exclusions restantes sont : 9 positions au sol absentes,
5 instables, 2 incompatibles avec la moitié d'attaque confirmée et 1 orientation
indisponible. Le viewer affiche ces motifs. Cela ne valide pas indépendamment
la précision du détecteur de tirs ni l'exactitude des statistiques sportives.

## GPU prioritaire et repli CPU

Les profils d'analyse privilégient désormais CUDA pour les six modèles.
La seconde passe OCR ne force plus `device: cpu`. Les échecs de device peuvent
déclencher un repli CPU explicite et tracé ; les erreurs de modèle ou de données
restent des erreurs. Un manque temporaire de mémoire OCR peut utiliser le
lecteur CPU puis revenir sur GPU. Les opérations OpenCV, le suivi, les votes
et la sérialisation restent des opérations CPU.

La progression distingue maintenant lecture des couleurs, OCR et résolution
des identités ; les timings séparent perception, votes et résolution.

Validation : 68 tests pipeline, 107 tests viewer, 31 tests maillots dont un
optionnel ignoré, 24 tests JavaScript et scénarios navigateur. Un essai réel
de 30 images a terminé avec les six modèles et la consolidation OCR sur CUDA,
sans repli. L'OCR réel a également été testé sur GPU et en repli CPU avec
CUDA simulé indisponible. Les résultats du run historique ne sont pas
présentés comme un nouveau benchmark de débit GPU.
