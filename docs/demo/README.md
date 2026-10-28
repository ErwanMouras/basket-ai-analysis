# Vyneros — kit de démonstration et de lancement

Préparé le 27 octobre 2026 pour des **recruteurs et équipes IA / vision**.
Version de départ : PR #8, commit `5f6f0b6`.

**Promesse : transformer une vidéo de basket en actions que l’on peut explorer et vérifier.**
Le premier écran doit montrer cette transformation. Les modèles, les métriques
et la stack viennent ensuite, en réponse à ce que le spectateur vient de voir.

## Livrables

| Usage | Livrable | Fonction |
| --- | --- | --- |
| Préparer le tournage | [Storyboard et conduite de démo](storyboard.md) | Montage de 36 s, démonstration de 3 min, moments source et voix off |
| Publier sur LinkedIn | [Post, commentaire et relances](linkedin.md) | Texte prêt à copier, angle technique accessible, question de discussion |
| Défendre les affirmations | [Registre des preuves](evidence.md) | Origine des chiffres, limites et formulations exactes |
| Régénérer les médias | `scripts/capture_viewer_demo.cjs` et `scripts/build_demo_pack.cjs` | Captures du viewer courant, clip réel, vidéo, sous-titres, carrousel PDF et aperçus publics |
| Explorer les résultats locaux | `release/demo-2026-10-27/index.html` | Galerie et fichiers à joindre ; chemin volontairement ignoré par Git |

## Le parti pris

1. **Un avant/après sur la même image.** Montrer ce qu’apporte l’analyse à une
   action de basket identifiable. Pas de terminal, de logo animé ou de liste de
   modèles pour ouvrir la vidéo.
2. **Une interaction qui a une conséquence.** « Voir le tir » ramène avant le
   relâchement, en pause. Le spectateur comprend immédiatement l’usage.
3. **Une lecture spatiale lisible.** Deux cartes alignées : présence d’un côté,
   origines estimées des tirs de l’autre. Les filtres Attaque/Défense restent
   associés à la présence.
4. **Une preuve d’ingénierie.** Expliquer le traitement batch, les artefacts et
   les mesures de couverture. Détailler les outils dans le dépôt et en entretien.
5. **Une incertitude visible.** Montrer « À vérifier » sur un exemple réel : une
   interface d’analyse doit permettre d’interroger les prédictions.

L’effet recherché est : « Je comprends ce que cela permet de faire ; je veux
voir comment c’est construit. » Le taux d’engagement n’est pas une promesse.

## Deux ensembles de médias distincts

**Démo réelle locale.** Les images proviennent du run terminé
`hou-lal-sam-yolo26l-ball`. Les captures et vidéos dérivées restent dans
`release/`, conformément à l’organisation actuelle du dépôt qui garde les
images NBA locales. Le montage est une proposition à visionner avant diffusion.
Il ne réexécute aucun modèle et ne modifie aucun résultat.

**Aperçus publics du README.** Les nouvelles images sous `imagesReadMe/` utilisent
uniquement le sample synthétique scripté. Leur légende l’indique. Elles servent
à présenter et essayer l’interface ; elles ne démontrent pas la qualité des modèles.
Le README renvoie au scénario réel sans inventer une URL de vidéo publiée.

## Reproduire

Depuis deux terminaux, lancer les serveurs si leurs ports ne sont pas déjà utilisés :

```bash
python -m viewer.server
python -m viewer.server --port 8766 --runs-root examples/demo/runs --video-root examples/demo/videos
```

Puis, depuis la racine :

```bash
npm --prefix viewer ci
node viewer/node_modules/playwright/cli.js install chromium
node scripts/capture_viewer_demo.cjs
node scripts/build_demo_pack.cjs
```

La capture produit aussi `viewer-walkthrough.mp4` : un enregistrement continu
avec les vraies interactions du viewer, accompagné des temps de chapitres.
Le build utilise FFmpeg/FFprobe, Chromium et les polices déjà présentes dans le
viewer. Il produit un manifeste avec le commit du viewer, les hashes des médias,
de la source et des observations, les moments sélectionnés et les valeurs lues
dans le run. Les options du script de capture permettent de changer `--output`,
`--run`, `--real-url`, `--sample-url` ou `--mode public|real|walkthrough|all`.
Les exemples réels sont vérifiés par frame et statut : changer de run nécessite
d’adapter et revoir le storyboard, plutôt que d’utiliser les mêmes timecodes.

## Ordre de diffusion proposé

| Moment | Contenu | Question à laquelle il répond |
| --- | --- | --- |
| Lancement | Vidéo native + post principal | « À quoi sert le projet ? » |
| Après les premiers retours | Réponses avec exemples concrets | « Comment gères-tu tel cas ? » |
| Quelques jours plus tard | Carrousel technique ou analyse d’un échec | « Qu’as-tu appris en le construisant ? » |
| Candidature / entretien | Démo guidée de 3 min et README | « Qu’as-tu effectivement implémenté et vérifié ? » |

Ne pas publier la vidéo et le carrousel comme deux contenus identiques le même
jour. Le second doit apporter une explication nouvelle. Choisir l’heure selon
ses propres habitudes et données d’audience ; aucune heure magique n’est supposée.

Mesurer les visites du dépôt et du profil, les échanges techniques utiles et les
prises de contact, puis les vues et réactions comme contexte. Pour comparer une
future accroche, garder le même objectif et noter les différences de contenu et
d’audience : deux publications ne constituent pas un test contrôlé.

## Références de format

Les recommandations éditoriales suivent le [guide de partage LinkedIn](https://content.linkedin.com/content/dam/help/linkedin/en-us/LinkedIn-Sharing-Guide.pdf) :
une entrée visuelle rapide, une démonstration concrète et une invitation à discuter.
Le choix de 36 s est éditorial, pas une règle de classement.
Les exports proposés sont un master 16:9 et une version LinkedIn 4:5 avec du texte
lisible et des sous-titres. Les [spécifications officielles de partage vidéo](https://www.linkedin.com/help/linkedin/answer/a7494039)
acceptent ces rapports d’image et permettent d’ajouter une miniature et un fichier
de sous-titres. Sources consultées le 27 octobre 2026.
