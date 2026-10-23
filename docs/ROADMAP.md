# Travaux ouverts

Priorité à la validité et aux échecs observables, pas à la maximisation d'un score isolé.

## 1. Communication locale et incertaine

Remplacer la diffusion globale par une portée configurable et des relais explicites. Tester retard, pertes, expiration et informations devenues fausses après fermeture. Comparer à la v2 avec graines appariées et rendre visibles les messages dans les replays.

Critère : les résultats doivent préciser ce qui reste du gain lorsque la transmission globale disparaît, y compris les effets sur la densité.

## 2. Compromis sorties / densité

Étudier plusieurs pénalités et contraintes de densité, le débit aux passages et les congestions après sortie. Présenter une courbe de compromis plutôt qu'un vainqueur unique. Garder séparés proxy de densité et risque physique non validé.

## 3. Validation empirique documentée

Construire une fiche de sources publiques : date d'observation, lieu, mécanisme décrit, qualité, incertitudes et paramètres éventuellement identifiables. Ne pas confondre date de publication, période observée et actualité. Ne pas collecter de données identifiantes de manifestants.

## 4. Reprise exacte et reproductibilité

Sauvegarder et restaurer les états aléatoires, l'environnement et l'optimiseur ; ajouter un test de continuation. Enregistrer matériel et dépendances. Comparer entraînement continu et interrompu avant de proposer `--resume`.

## 5. Visualisation et revue indépendante

Un lecteur de replays avec légende, contrôle du temps, messages, sorties et densité faciliterait la critique. Documenter l'accessibilité et le fonctionnement hors ligne. Une revue des hypothèses et des erreurs du modèle est une contribution à part entière.

Pour commencer sans GPU : améliorer la documentation, proposer un cas d'échec, ajouter une source correctement contextualisée, vérifier un invariant ou travailler sur le lecteur de replays.
