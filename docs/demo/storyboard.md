# Conduite de démonstration

## Préparation visuelle

Master README / entretien : **1920 × 1080**, interface française, navigateur
sans barre d’outils, zoom 100 %, lecteur en pause au départ. LinkedIn : **1080 ×
1350**, composition dédiée ; recadrer les éléments utiles plutôt que réduire
un écran desktop entier jusqu’à rendre ses textes illisibles.

Fond charbon, orange du viewer, titres condensés. Une seule idée par plan, une
ou deux lignes de texte au maximum. La vidéo doit rester compréhensible sans
voix off. Ne pas utiliser la bande-son TV ; le montage exporté est silencieux.
La voix ci-dessous reste une proposition à enregistrer avec sa propre voix.

Les captures d’écran et leurs recadrages montrent les sorties existantes.
Les titres du montage sont ajoutés autour de ces captures, sans modifier les
détections, statuts, positions ou valeurs de l’interface.

## Montage principal — 36 secondes

Les temps « montage » sont différents des temps de la vidéo source.

| Montage | Plan et geste | Texte à l’écran | Intention |
| --- | --- | --- | --- |
| 00–06 s | Même action brute puis annotée, source autour de 16,80 s. Garder le ballon et les joueurs visibles. | **Du match aux données.** | Faire comprendre la transformation avant de parler de technique. |
| 06–12 s | Ligne du tir à 16,80 s, frame 504, puis « Voir le tir » et retour en pause à 16,20 s. | **Un tir devient un point d’entrée.** | Montrer une fonction utile et immédiatement vérifiable. |
| 12–18 s | Équipe LAL : cartes Présence et Tirs alignées. Montrer le contrôle Attaque/Défense sous Présence. | **Lire l’espace. Retrouver les tentatives.** | Passer d’une action à une lecture collective. |
| 18–24 s | Schéma simple : perception → suivi/identité → projection → artefacts/viewer. | **Un pipeline complet, des sorties inspectables.** | Rendre visible le travail d’ingénierie derrière le viewer. |
| 24–30 s | Contraster le probable à 16,80 s et le candidat à 47,57 s. Afficher les statuts d’origine. | **L’incertitude fait partie du résultat.** | Donner une raison de faire confiance à la démarche. |
| 30–36 s | Retour aux deux cartes, URL du dépôt et question finale. | **Quel serait votre premier test de robustesse ?** | Ouvrir une conversation précise avec les équipes IA. |

### Voix off proposée

> J’ai construit Vyneros pour explorer une vidéo de basket à partir de ses données.
> Les joueurs, le ballon et le terrain sont analysés avant la lecture.
> Un clic ramène au départ d’un tir probable.
> Les cartes donnent ensuite une lecture spatiale de l’extrait.
> Derrière le viewer, les modèles et les sorties sont versionnés.
> Les cas incertains restent à vérifier.
> Quel serait votre premier test de robustesse ?

Environ 75 mots. Faire une lecture naturelle chronométrée avant l’enregistrement ;
ajuster le débit ou le montage, sans accélérer artificiellement la voix.
Ne pas doubler à l’oral chaque libellé déjà visible.

### Variante courte — 12 secondes

0–4 : brut → annoté. 4–8 : « Voir le tir » → pause avant le départ.
8–12 : les deux cartes + nom du projet. Une seule phrase :
**« D’une vidéo de basket à des actions que l’on peut explorer. »**
Cette variante sert d’aperçu du README ou d’introduction en entretien ; elle ne
remplace pas l’explication du traitement batch.

## Moments source retenus

| Source | Repère | Ce que l’on montre | Ce que l’on dit |
| --- | --- | --- | --- |
| 16,80 s · frame 504 | `shot-3`, probable, Luke Kennard #10 | Action principale avec segmentation / ballon / terrain ; bbox et pose dans les détails | « Proposition de tir probable attribuée dans cet extrait. » |
| 16,20 s | Destination du clic « Voir le tir » pour `shot-3` | Relecture en pause 0,6 s avant le départ estimé | « On revient au contexte avant le relâchement. » |
| 47,57 s · frame 1427 | `shot-7`, candidat, Marcus Smart #36 | Carte / ligne « À vérifier », sans changer le statut | « Une trajectoire compatible reste à vérifier. » |
| 13,30–14,60 s | Plans rapprochés et changements de plan | Exemple complémentaire de calibration indisponible | « Une projection absente ne doit pas produire une distance inventée. » |
| 37,50 s · frame 1125 | `shot-6`, probable sans identité nominative | Démo complémentaire de non-attribution | « L’événement peut être proposé sans nommer un joueur. » |

Les contrôles visuels de développement sont dans
[la revue des tirs](../../reports/shots-2026-10-26/README.md).
Ils ne sont pas une annotation exhaustive et indépendante.
Revoir la séquence en mouvement avant de conserver un plan : les captures seules
ne suffisent pas à qualifier la qualité d’une trajectoire ou d’un masque.

## Démo guidée en entretien — 3 minutes

| Temps | Manipulation | Message |
| --- | --- | --- |
| 00:00–00:20 | Montrer une action brute puis les calques | « Je pars d’une vidéo et d’un effectif optionnel pour construire des artefacts analysables. » |
| 00:20–00:50 | Lecture, pause, une frame, masque, bbox | « La synchronisation suit l’image effectivement présentée, y compris après un seek. » |
| 00:50–01:15 | Tir de 16,80 s → Voir le tir | « Le détecteur propose une fenêtre à revoir, pas un résultat marqué/raté. » |
| 01:15–01:40 | Luke Kennard → déplacements et couverture | « Ce sont des mesures sur les intervalles observés. Le hors-champ reste inconnu. » |
| 01:40–02:10 | Équipe → présence/tirs → Défense | « Le filtre ne change que la présence. La carte des tirs couvre tout l’extrait. » |
| 02:10–02:35 | Candidat de 47,57 s ou identité absente à 37,50 s | « Je montre où le système doit s’abstenir ou demander une revue. » |
| 02:35–03:00 | Schéma + configuration et manifeste | « Les choix de modèle, les hashes, les évaluations et les tests rendent ce résultat retraçable. » |

La démonstration charge des résultats préparés. Le run réel traite 1 842 frames
en 318,49 s de traitement, soit 5,78 images/s pour ce profil. La lecture de la
vidéo à vitesse normale ne mesure pas le débit d’inférence.

## Trois séquences complémentaires

**Ballon, 15–20 s.** Partir de 16,10–19,70 s. Montrer l’image et la localisation
observée. Faire ensuite un zoom de montage clairement identifiable sur le ballon.
Présenter le F1 de localisation seulement avec le protocole : 2 627 frames
communes de validation, tolérance 15 px source, pas de score indépendant.

**Géométrie, 15–20 s.** Passer de la projection disponible à un changement de
plan autour de 13,30 s. Faire remarquer que les points et mesures ne doivent pas
continuer dans un repère invalide. Vérifier la frame effectivement affichée et
la calibration du run avant la prise finale.

**Du modèle au produit, 20–30 s.** Configuration figée → `run.json` → observation
→ viewer. Utiliser le schéma pour orienter, puis montrer un vrai champ et son
usage. Pas de faux terminal ni de ligne de commande simulant une inférence live.

## Répétition et repli

Lancer le viewer à l’avance et attendre le chargement des observations et des
heatmaps. Conserver les clips exportés et les captures sur le disque de la
machine de présentation. Si une manipulation ne répond pas, passer au média
exporté en le présentant comme un enregistrement. Ne pas lancer les modèles
pendant une démo de trois minutes.

Sur un petit écran, utiliser la vidéo et les captures cadrées plutôt que
réduire toute l’interface. En entretien technique, garder la configuration et
le registre des preuves à une navigation du viewer.
