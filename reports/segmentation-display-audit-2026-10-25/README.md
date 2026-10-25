# Audit du rendu des masques — passage à 51,60 s

Le studio avait un défaut temporel après une recherche en pause : `seeked`
remplaçait le timestamp de l’image présentée par la frame **la plus proche** de
`currentTime`. À 51,619 s, le navigateur montre encore l’image 1548 (51,600 s),
mais le studio dessinait les masques 1549 (51,633 s), soit une image d’avance.
La navigation image par image dessinait aussi la destination avant le décodage.

Le correctif utilise `requestVideoFrameCallback.mediaTime` pour choisir la frame
présentée. La recherche efface les calques jusqu’à sa présentation ; une réponse
réseau tardive ne peut pas restaurer les masques de l’image précédente. Le repli
sans cette API choisit la frame dont l’intervalle contient `currentTime` et attend
`seeked`. Les contrôles par catégorie restent indépendants.

## Mesures sur le run complet HOU–LAL

Toutes les valeurs figurent dans [measurements.json](measurements.json).

| Contrôle | Résultat |
| --- | --- |
| Masques canvas vs décodage Python indépendant | 91 / 91 identiques pixel par pixel, sur 7 frames |
| Recherche en pause, image native du navigateur comparée aux frames OpenCV voisines | Avant : 8 / 11 mauvais indices ; après : 0 / 11 |
| Passage exact 51,600 s / recherche 51,619 s | Image 1548 dans les deux cas après correction |
| Lecture à 1× et 2×, 60 callbacks par vitesse | 120 / 120 calques visibles associés au PTS présenté |
| Navigation d’une image après 51,619 s, 0,190 s et 30,019 s | Indices 1549, 6 et 901 corrects |
| Réponses API retardées de 250 ms après vidage du cache | Aucun ancien masque affiché ; image 900 restaurée à son arrivée |
| Navigateur sans callback de présentation | 0,190 s → image 5 ; 51,619 s → image 1548 |
| Relance indépendante de SAM sur l’image 1548 et ses bbox stockées | 9 / 9 masques identiques dans leur ROI ; IoU de reproduction = 1 |
| Régressions | 14 tests JavaScript, 8 tests serveur et syntaxe JavaScript réussis |

Les images natives ont été réduites à 480×270 puis comparées par erreur moyenne
absolue RGB/BGR aux cinq candidates OpenCV autour de l’ancien indice. Le meilleur
indice retrouve la même image source avant/après ; l’erreur résiduelle de couleur
(environ 2,4–2,6 / 255) vient des décodages distincts. Ce contrôle est indépendant
du choix d’index du studio. Le parcours complet des PTS ffprobe et des observations
ne révèle pas de décalage source permanent : 1 842 images dans les deux parcours.

## Ce qui reste dans les prédictions

À 51,600 s exactement, aucun décalage temporel n’a été mesuré avant correction.
La relance SAM et le décodage navigateur reproduisent les prédictions enregistrées :
les contours imparfaits visibles sur cette image ne sont pas causés par le RLE.
Les masques sont calculés indépendamment par image, sans suivi temporel du contour.
Leur rôle joueur/arbitre vient du détecteur, pas de SAM.

Le pipeline conserve seulement la partie de chaque masque à l’intérieur de la
bbox. Sur cette image, cinq masques perdent respectivement 10, 6, 31, 82 et 4 pixels
par rapport à la sortie SAM complète. Cela peut créer une coupure au bord d’une
boîte, mais représente ici moins de 0,6 % de chacun des cinq masques. Ce n’est
pas une mesure d’erreur contre une vérité terrain : l’audit établit la fidélité
du rendu, pas la qualité de segmentation. Les éventuelles erreurs de silhouette,
de catégorie ou d’inclusion d’une autre personne demandent une revue des masques.

Le correctif porte uniquement sur le viewer : aucune inférence complète ni lecture
de maillot n’a été recalculée. L’évaluation de stratégie reste dans le
[rapport de comparaison](../segmentation-comparison-2026-10-25/README.md).
Scripts diagnostiques, captures vidéo et masques détaillés restent localement dans
`runs/segmentation-display-audit/`. Les métriques et empreintes seules sont versionnées.

Pour vérifier les régressions :

```bash
node --test viewer/test_app.js
node --check viewer/app.js
python -m unittest viewer.test_server -v
```
