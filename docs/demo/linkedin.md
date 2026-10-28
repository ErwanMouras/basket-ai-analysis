# Publication LinkedIn — version recruteurs / équipes IA

## Post principal, prêt à copier

Une vidéo de basket. Un clic pour revenir au départ d’un tir.

C’est le parcours que j’ai voulu construire avec Vyneros : passer d’images de match à des actions que l’on peut explorer, puis vérifier.

Dans la démo :
→ joueurs et ballon suivis dans la vidéo ;
→ positions projetées sur le terrain ;
→ propositions de tirs accessibles depuis une timeline ;
→ cartes de présence et de tirs par équipe.

Le défi qui m’a le plus intéressé : faire tenir ces briques ensemble.

Synchroniser les masques avec la frame réellement affichée. Gérer les changements de caméra. Relier les numéros de maillot à un effectif. Interrompre une mesure quand les données deviennent insuffisantes.

J’ai aussi construit le chemin autour des modèles : données versionnées, expériences et évaluations suivies, configurations traçables, tests automatisés et viewer conteneurisé.

Le traitement se fait en amont. La vidéo montre l’exploration des résultats, avec des tirs « probables » et d’autres « à vérifier ». Les résultats marqué/raté ne sont pas déduits.

Le code, les protocoles et les limites sont documentés ici :
https://github.com/ErwanMouras/basket-ai-analysis

Si vous deviez tester ce système sur un nouveau match, que mettriez-vous à l’épreuve en premier : l’identité des joueurs, la géométrie du terrain ou les changements de caméra ?

#ComputerVision #MachineLearning #MLOps

## Média à joindre

**Choix principal :** `release/demo-2026-10-27/linkedin-36s.mp4` avec sa miniature
`linkedin-cover.png`. Ajouter `linkedin-36s.fr.srt` si la plateforme le propose.
Les mots de montage sont déjà lisibles sans audio. Visionner le fichier complet
avant de le joindre : il contient des images du match de développement, conservées
localement dans ce dépôt.

**Alternative :** `carousel.pdf` et ses six pages PNG. Garder la question finale
et remplacer « Dans la démo » par « Dans ces captures ». Les pages du carrousel
présentent des résultats existants, sans laisser entendre une démonstration live.
Choisir l’un des deux formats pour cette publication.

Le lien est dans le corps du post pour être facile à trouver. Le commentaire
ci-dessous apporte des précisions ; son emplacement ne suppose aucun avantage
algorithmique. Ne pas solliciter des commentaires par mot-clé pour distribuer
un dépôt déjà accessible.

## Premier commentaire proposé

Quelques détails pour celles et ceux qui veulent regarder sous le capot :

• Le viewer lit des artefacts d’analyse : il ne relance pas les modèles pendant la lecture.
• Le profil montré combine détection de personnes, SAM 2.1, YOLO26L pour le ballon, pose, calibration du terrain et OCR des maillots.
• Les compteurs joueurs retiennent les tirs probables. La carte d’équipe distingue aussi les candidats à vérifier.
• La prochaine étape d’évaluation est de mesurer la robustesse sur des matchs indépendants du développement.

Les protocoles et les résultats détaillés sont dans les sections Results et Model lifecycle du README.

## Accroches alternatives

**Plus produit :** « Retrouver un tir dans une vidéo de basket, puis regarder où l’équipe joue. C’est ce que permet cette version de Vyneros. »

**Plus ingénierie :** « Le modèle localisait le ballon. Il restait à construire tout ce qui permet d’exploiter cette prédiction. »

L’accroche principale est retenue pour sa correspondance directe avec le premier
plan. Les variantes servent à une future publication avec un angle différent,
pas à republier immédiatement le même contenu.

## Réponses utiles aux commentaires

**« C’est du temps réel ? »**
« Le run montré est traité en batch. Pour ce profil, 1 842 frames ont demandé
318,49 s de traitement sur RTX 5080. La lecture fluide vient ensuite, sur les
résultats enregistrés. »

**« Comment sais-tu qu’un tir est réussi ? »**
« Je ne le déduis pas actuellement. Le système propose des tentatives à partir
du ballon, des mains, de l’arc et du contexte. Le résultat marqué/raté reste inconnu. »

**« Quelle précision ? »**
« La localisation du ballon a un F1 de 0,9451 sur les 2 627 frames communes de
validation du protocole publié, avec une tolérance de 15 pixels source. Ce
chiffre ne mesure ni les tirs ni la qualité globale du système. Il reste à
confirmer la généralisation sur des matchs indépendants. »

**« Tu as entraîné tous les modèles ? »**
« Le projet combine des poids existants, des adaptations et des entraînements,
notamment pour la comparaison des modèles ballon. La provenance et les
configurations sont documentées ; je distingue l’intégration de la contribution
aux modèles eux-mêmes. »

**« Qu’est-ce qui est le plus fragile ? »**
« Les occultations, les changements de caméra, les pieds hors champ et les
numéros ambigus. Les cas manquants restent visibles dans la couverture ou le
statut plutôt que d’être complétés. Je veux maintenant les évaluer sur des
matchs non utilisés pendant le développement. »

Adapter ces réponses à la question réellement posée, ajouter une capture ou un
lien précis si utile et dire quand on ne sait pas. Ne pas poster de faux retours
d’expérience, témoignages ou résultats de diffusion.

## Publication suivante : un apprentissage concret

Angle proposé : **« Un masque correct sur la mauvaise frame reste un résultat faux. »**
Montrer le problème de synchronisation vidéo, la logique de frame présentée et
le test qui évite la régression. C’est un sujet distinct de la vidéo principale,
pertinent pour une équipe vision qui doit intégrer ses sorties dans un produit.
Base documentaire : [audit de segmentation](../../reports/segmentation-display-audit-2026-10-25/README.md)
et [guide du viewer](../../viewer/README.md).
