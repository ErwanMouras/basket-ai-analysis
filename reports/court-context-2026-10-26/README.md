# Classifier Attaque / Défense avec les côtés du quart-temps

> Étape historique : le [modèle de jeu installé](../settled-play-2026-10-26/README.md)
> remplace ce complément par un état neutre après changement de possession, exclu du viewer et des cartes.

Les côtés connus servent désormais **à attribuer la situation**, en complément
des contacts avec le ballon. Exemple sur le QT1 de HOU–LAL : Houston attaque à
gauche, les Lakers à droite. Un regroupement stable des deux équipes dans la
partie droite permet donc d’estimer **Lakers en attaque / Houston en défense**,
même si le ballon n’est pas attribuable.

Le viewer affiche cette origine sous la vidéo : **« LAL en attaque · zone du
terrain »**, ou **« … · ballon »** pour les périodes fondées sur les contacts.
Les heatmaps utilisent ces classifications et distinguent leurs contributions
en secondes de vidéo. La PR reste ouverte pour revue, sans merge.

## Règle et priorités

1. Apprendre le sens du QT exclusivement avec les contacts et la disposition
   des défenseurs, comme dans l’[étape précédente](../quarter-score-2026-10-26/README.md).
   Les votes spatiaux ne contribuent pas à leur propre repère.
2. Rechercher un regroupement calibré : deux joueurs distincts par équipe au
   minimum, médianes des deux équipes dans le même tiers extérieur (≤35 % ou
   ≥65 % de la longueur), et 80 % des joueurs localisés dans le même demi-terrain,
   avec une marge de 5 % près du milieu. Les joueurs non attribués participent au
   contrôle des positions hors du regroupement, sans devenir une équipe connue.
3. Confirmer pendant au moins 0,35 s sur trois images, avec 90 % d’accord dans
   une fenêtre de 0,8 s. Aucun partage d’indices entre coupes, exclusions,
   frames manquantes, grands trous temporels ou QT différents.
4. Attribuer l’attaque à l’équipe dont le panier attaqué se trouve de ce côté,
   uniquement si la période n’était pas déjà classée par le ballon.
5. Bloquer la déduction près d’un contact avec l’équipe contraire (±1 s), lors
   d’un contact actuel avec un arbitre, une identité inconnue ou disputée, et
   autour d’une mise à jour du score (2 s avant / 0,75 s après).

Un angle caméra confirmé comme inversé utilise son propre sens. Sans QT établi,
sans orientation exploitable ou en transition près du milieu, cette déduction
n’est pas utilisée. Les observations, annotations et poids restent inchangés ;
le calcul reste en bibliothèque standard dans le viewer.

## Comparaison sur HOU–LAL

Source SHA256 `e841e7d9c584884d8db0dc7202795f590560f3de159c5a104e01ebadeadde8b9`,
mêmes observations et même bandeau dérivé que l’étape précédente, sur 61,37 s.

| Méthode | Temps classé | Ballon | Zone du terrain | Non classé |
| --- | ---: | ---: | ---: | ---: |
| Contacts + côtés conservés, sans complément spatial | 34,80 s | 34,80 s | 0 s | 26,57 s |
| Complément spatial sans exclusion autour du score | 40,73 s | 34,80 s | 5,93 s | 20,63 s |
| Version retenue, avec exclusion autour du score | **40,23 s** | **34,80 s** | **5,43 s** | **21,13 s** |

Les attributions existantes issues du ballon sont conservées. Le complément
retenu ajoute **5,43 s**, soit environ 15,6 % de temps classé en plus sur cet
extrait. Cela mesure une couverture, pas un gain de précision.

Sur les 21 [points de développement initiaux](../attack-defense-2026-10-26/review-checkpoints.json),
19 étaient attribuables et deux étaient des gros plans : les décisions passent
de **10 à 12**, les 12 correspondent au contexte attendu, et les deux gros plans
restent non classés. Voir l’[expérience complète](experiment.json).

### Vérifier les nouvelles décisions, pas seulement les anciens points

Un second contrôle a ciblé huit instants dans les intervalles ajoutés par le
prototype. À **32,25 s**, Houston prépare une remise en jeu après un panier des
Lakers : les deux équipes restent près du panier droit, donc la seule zone
occupée attribuait à tort l’attaque aux Lakers. Le score observé est mis à jour
plus tard. L’exclusion autour de cette mise à jour retire cette déduction de
0,5 s, tout en laissant intactes les périodes classées par le ballon.

| Sur les huit instants ciblés | Conformes | Erreurs observées | Abstentions |
| --- | ---: | ---: | ---: |
| Complément spatial sans exclusion autour du score | 7 | 1 | 0 |
| Version retenue | **7** | **0** | **1** |

Les [points ciblés](targeted-checkpoints.json) et leur [comparaison](targeted-experiment.json)
sont conservés. Ils ont servi au réglage : ces deux contrôles sont **des audits
de développement, pas des validations indépendantes**, et leurs décomptes ne
s’additionnent pas comme des observations indépendantes.

La zone peut encore se tromper lors de transitions ou changements de contrôle
non observés. Une mise à jour du score peut être retardée ou manquer. Les arrêts,
ralentis et possessions officielles ne sont pas identifiés automatiquement.
L’origine affichée permet de repérer et discuter les périodes déduites du terrain.

## Tester et reproduire

Dans le viewer local, ouvrir HOU–LAL et se placer vers **15 s** : le libellé affiche
une attaque Lakers issue de la zone. Dans les heatmaps, les situations Attaque /
Défense intègrent le complément ; la ligne « Sur cette plage » sépare ballon et
zone. À **32,25 s**, la déduction spatiale du prototype reste désormais écartée.

```bash
.venv-players/bin/python -m viewer.phase_experiment \
  --run runs/analysis/hou-lal-sam-yolo26l-ball \
  --review-checkpoints reports/attack-defense-2026-10-26/review-checkpoints.json \
  --output /tmp/court-context-experiment.json

.venv-players/bin/python -m viewer.phase_experiment \
  --run runs/analysis/hou-lal-sam-yolo26l-ball \
  --review-checkpoints reports/court-context-2026-10-26/targeted-checkpoints.json \
  --output /tmp/court-context-targeted.json
```

L’outil compare les contacts, les côtés seuls, le complément spatial et sa
variante sans exclusion du score, sans inférence ML ni modification des runs.
Les seuils et la provenance source sont inclus dans les résultats.

## Validation

Tests spécifiques : longue absence du ballon, attribution opposée dans l’autre
moitié, absence de QT, repère non appris, changement de QT, contact contraire
confirmé ou isolé, arbitre/identité inconnue/contact disputé, milieu de terrain,
calibration absente, tracks dupliqués, joueurs hors regroupement, coupes et
exclusions, angle inversé, changement de score, conservation des temps et
contributions dans les heatmaps, timeline compacte et API.

**436 tests Python réussis, 28 optionnels ignorés** (464 découverts),
**29 tests JavaScript réussis**. Le navigateur vérifie aussi l’origine affichée
en FR/EN et les durées par source. Les suites existantes du lecteur, des cinq
heatmaps, des exports et du responsive restent exécutées.
