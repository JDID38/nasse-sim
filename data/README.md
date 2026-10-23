# Données cartographiques

© les contributeurs OpenStreetMap. Données sous [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/). Voir [OpenStreetMap : copyright et attribution](https://www.openstreetmap.org/copyright).

`paris_roads.json` contient 19 529 voies et les métadonnées de l'extrait local. Date indiquée : **20 octobre 2026**, sans garantie d'actualité de chaque voie. Le fichier publié conserve seulement `nom`, `date`, `source`, `bbox` et `w`. Les autres couches du fichier historique ne sont pas utilisées et ne sont pas publiées.

Dans `w`, chaque voie contient sa classe puis des coordonnées latitude/longitude entières en degrés multipliés par 100 000. Les classes contrôlent la largeur rasterisée dans `osm_map.py`, pas la largeur réelle disponible pour une foule.

`paris_grid.npz` est le cache raster dérivé, sous les mêmes conditions ODbL. Il reproduit le découpage v2. L'échelle cartographique nominale est de 10 m par case ; ni le mouvement d'un groupe ni la durée d'un tour ne sont calibrés physiquement.

`CropSampler` exclut les centres d'entraînement à moins de 1 000 m des trois lieux de test. Le test porte néanmoins sur la même ville ; cela ne valide pas la généralisation à d'autres villes ou dispositifs policiers.

Pour recalculer une grille dans une copie de travail, déplacer `paris_grid.npz` puis appeler `osm_map.big_grid()`. Documenter toute modification des données ou paramètres.
