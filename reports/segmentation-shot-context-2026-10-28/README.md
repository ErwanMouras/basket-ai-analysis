# Masques de maillots et attribution des tirs — GSW–DEN, 28 octobre 2026

Run : `runs/analysis/gsw-den-2026-10-26-full`.
Preuves locales : `runs/segmentation-shot-context-2026-10-28/`.

## Couleur d'équipe et numéro

La couleur finale était calculée sur le rectangle du torse, sans filtrer ses
pixels par la segmentation. Le regroupement anonyme pendant l'analyse avait
la même limite. Les deux chemins utilisent désormais le masque du joueur :
seuls ses pixels dans le cœur du torse contribuent aux votes. Le fond neutralisé
n'entre pas dans les statistiques HSV. Il faut au moins seize pixels et 50 %
de couverture ; une segmentation trop petite provoque une abstention. Le
rectangle brut reste un repli déclaré lorsque la segmentation est absente.

Le masque ne localise pas lui-même les chiffres : la pose ou la bbox choisit
le rectangle du numéro, puis `jersey.mask_policy: background_neutral` neutralise
ses pixels hors personne avant PARSeq. Ce traitement était déjà actif sur
GSW–DEN. Cette politique conserve un repli brut si le masque est absent ou
insuffisant ; `require_mask` impose au contraire une abstention. Les nouvelles
lectures de la postpasse sérialisent aussi `mask_audit`, comme le lecteur
pendant l'analyse et la récupération ciblée.

Contrôle sur le « 27 » de Murray, frame 750 / track 95 : trois variantes de crop
ont `mask_audit.status: applied` et une couverture de 100 %. PARSeq retrouve 27
avec des scores de 0,9962, 0,9953 et 0,9631. Cette inférence est exécutée en FP32
sur **CUDA:0 / RTX 5080**, sans repli CPU. Ce contrôle ponctuel ne mesure pas
l'exactitude de l'OCR sur tous les joueurs du match.

Les votes de couleur du match ont été recalculés dans une copie avec les
masques enregistrés, puis les identités et distances rejouées. Sur les
**498 138 observations avec ID de track**, toutes possèdent une segmentation ;
aucun repli bbox n'est utilisé pour leurs couleurs. **109 majorités d'équipe**
changent : 73 deviennent inconnues, 30 deviennent attribuées et six changent
d'équipe. Les observations nommées passent de **133 850 à 136 981**, soit un
net de +3 131 (4 396 ajouts, 1 265 retraits lorsque la preuve de couleur ou une
liaison n'est plus suffisante). Les tracks ayant au moins un nom passent de
882 à 895 ; huit anciennes tracks deviennent anonymes. Aucun nom existant
n'est remplacé par un autre.

Les bbox, poses, masques, IDs, plans et détections sont identiques frame par
frame. Les lectures OCR numériques, leurs scores et leurs pixels source sont
conservés ; seuls leurs repères d'épisode sont recalés sur les nouveaux épisodes
de couleur. `evidence_version: 6` distingue ce traitement. La migration v5 → v6
réutilise l'OCR existant et ne relance pas de modèle. Le décodage, le calcul de
couleur et le rejeu JSON/statistiques utilisent le CPU ; le contrôle PARSeq
utilise le GPU. Les sauvegardes précédentes sont dans `backup-before-apply/`.

## Pourquoi deux vidéos

**Oui : la copie dédiée au viewer corrige le début de la vidéo.**
`GSW_DEN.mp4` contient 134 images de présentation négative. Le premier paquet
vidéo a un PTS de **−5,36 s** et un DTS de −5,44 s, malgré un début déclaré du
flux à zéro. Chromium/OpenCV et Firefox ne présentent pas ces images de la
même façon. La copie `GSW_DEN.viewer.mp4`, encodée avec **h264_nvenc**, commence
au PTS zéro et contient les **57 861 images utilisées par l'analyse**, contre
57 995 images déclarées dans l'original. Ce fichier évite le décalage entre
annotations et lecture, sans modifier la vidéo source ni déplacer les données.

`playback.json` lie les deux fichiers par SHA-256 et conserve la provenance de
l'encodeur. Le [rapport de lecture précédent](../gsw-den-viewer-audit-2026-10-27/README.md)
détaille les vérifications Chromium/Firefox.

## Tir attribué à un défenseur

Le cas de Curry à **104,52 s (1 min 44,52 s), frame 2613** présente une origine
au sol x = 26,44 m, alors que Golden State attaque dans le sens décroissant sur
ce plan du premier quart-temps. Le contact main-ballon utilisé par le détecteur
peut être celui d'un défenseur qui conteste le tir. La trajectoire plausible ne
suffit donc pas à attribuer le lâcher à ce joueur.

Une validation s'exécute maintenant après l'estimation des directions et
**avant** les compteurs, cartes et listes individuelles. Elle annule le tireur
si une origine fiable se trouve dans sa moitié défensive, avec une marge de
0,5 m autour du milieu, ou si une phase installée confirmée de l'adversaire
couvre l'instant dans le même plan. Une phase passée, future, d'un autre plan
ou un simple contact ne suffit pas. Une orientation ou position absente ne
crée pas de contradiction artificielle.

L'événement conserve sa trajectoire et son statut ; `shooter` devient inconnu
et il disparaît des statistiques individuelles et des cartes d'équipe. Les
champs `attribution.reason`, `previous_shooter`, `previous_origin` et
`team_direction` gardent la preuve du refus. Aucun autre joueur ne reçoit
l'événement sur la seule base de sa proximité.

Après recalcul des couleurs et des phases, **14 attributions** sont annulées :
douze pour demi-terrain défensif et deux pour attaque installée de l'adversaire.
Les **223 propositions** restent présentes. Le compteur individuel de Curry
passe de **cinq à quatre tirs probables** ; l'événement de 104,52 s reste dans
la liste générale sous « Tireur non attribué » et quitte sa revue individuelle.

Les tests couvrent les deux orientations, la marge au milieu, les origines
absentes, les périodes non couvertes, les limites de plans, la conservation des
événements et l'idempotence de la correction. Les suites navigateur vérifient
également les listes individuelles et les compteurs. Les JSON et captures de
contrôle du vrai run sont enregistrés dans le dossier local de preuves.

Validation : 73 tests pipeline, 37 tests jersey (un optionnel ignoré), 52 tests
tirs/cartes/serveur et sept suites navigateur. Les artefacts recalculés passent
le contrôle de contrats et de hashes. Le viewer sur le port 8765 a été relancé
et le parcours de revue individuelle de Curry a été contrôlé sur la vraie vidéo.
