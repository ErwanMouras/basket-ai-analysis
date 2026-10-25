# Périmètre de la version POC

Le produit est un outil batch d'inspection de clips NBA pour un analyste et un
évaluateur technique. Il transforme une vidéo et un roster facultatif en
observations, positions au sol, distances observées et interface de contrôle.
Le cas d'usage de cette version est une analyse traçable et inspectable ; aucun
gain de temps utilisateur n'a encore été mesuré.

La release démontre deux parcours : consultation de résultats synthétiques
partageables sans GPU, et inférence réelle avec les modèles locaux. Le viewer
consulte des analyses terminées. La version ne fournit pas de service public
d'upload ou de streaming en direct.

Le matériel des benchmarks GPU est une RTX 5080. L'image CPU d'inférence a aussi
été exercée sans réseau sur une courte entrée réelle, avec les six capacités.
Ces essais n'établissent ni un matériel minimum ni un débit généralisable.

Les statistiques sont limitées aux intervalles effectivement visibles et valides.
Les identités ambiguës, projections impossibles et ruptures restent explicites.
Le périmètre n'inclut pas distance totale de match garantie, hauteur du ballon,
tirs, possessions, autres géométries de terrain ou réidentification globale fiable.

Les critères techniques sont : données et sorties traçables, intégrité vérifiée,
inférence sans téléchargement implicite, exemple portable, régressions contrôlées,
promotion et retour arrière audités. Les exigences de qualité physique et
d'identification restent à fixer et mesurer avec une vérité terrain indépendante.
