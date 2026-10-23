# Contribuer

Les échanges et contributions en français ou en anglais sont bienvenus.

## Parcours court

1. Lire le README, `docs/MODELE.md` et `docs/ROADMAP.md`.
2. Ouvrir une issue avant une modification importante du modèle ou de la récompense : question, hypothèses et critères de comparaison.
3. Forker le dépôt, créer une branche, installer `requirements.txt` dans un environnement virtuel.
4. Modifier un sujet à la fois et ajouter un test ciblé lorsque le comportement change.
5. Exécuter `python -m unittest discover -v` et un petit entraînement v2.
6. Ouvrir une pull request décrivant le problème, le changement, les vérifications et les limites.

## Pour une expérience

Enregistrer version du code, configuration, graines, versions Python/PyTorch/NumPy, matériel et empreintes des checkpoints. Séparer entraînement, validation et test. Comparer des règles simples sur des conditions appariées. Rapporter les échecs, le contrôle sans ouverture, les scores par scénario et la densité ; publier seulement le meilleur score masque la variabilité.

Un changement de définition de la réussite ou de difficulté exige un protocole identifié. Ne pas comparer directement les pourcentages v1 et v2. Un effet simulé ne doit pas être présenté comme une recommandation de terrain.

Les gros résultats et modèles vont dans une release ou un dépôt de données, avec manifeste SHA256 et fiche descriptive, pas dans un commit. Aucun service payant ne doit être nécessaire aux tests. L'automatisation publique ne crée ni ne détruit de machines cloud.

## Données et sources

Proposer des sources publiques, datées et attribuées. Préciser ce qui est observé, inféré ou choisi arbitrairement. Ne pas téléverser d'images identifiantes ou de données personnelles pour calibrer le modèle. Respecter les licences, dont l'ODbL des données cartographiques.

## Licence des contributions

Les contributions sont proposées sous la licence applicable au fichier : MIT pour le code et la documentation propres au projet, et les conditions spécifiques des données tierces lorsque concernées. Signaler toute dépendance ou donnée provenant d'un tiers.
