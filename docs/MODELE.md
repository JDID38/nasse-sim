# Modèle v2

`ExitEnv` représente un cordon fermé sur une grille de rues. La géométrie est artificielle ; ce n'est pas le relevé d'un dispositif policier identifié.

## Ce qu'un groupe peut faire et observer

Sept actions : attendre, quatre directions, demander un passage, partager une ouverture effectivement visible. Une case de cordon fermé est infranchissable. Les observations comprennent une fenêtre locale de rayon 7, la position propre du groupe, le temps et la dernière information reçue. Les futurs passages ne sont pas étiquetés en v2.

La vue locale ne calcule pas l'occultation par les bâtiments. Un message peut atteindre tous les groupes après un retard, avec pertes et expiration. Ce sont deux simplifications importantes. Le réseau est sans mémoire récurrente ; les messages sont gérés par l'environnement.

## Réactions expérimentales du cordon

| Scénario | Règle |
|---|---|
| `sealed` | Jamais d'ouverture : contrôle d'impossibilité. |
| `delayed` | Un passage ouvre tardivement puis reste ouvert. |
| `responsive` | Une demande proche d'un passage possible peut recevoir une réponse positive retardée. |
| `relocating` | Des ouvertures temporaires alternent entre des emplacements fixes. |

Les délais, probabilités et débits sont des hypothèses. Le cordon ne se déplace pas physiquement ; gaz, blessures, charges et contrôles d'identité ne sont pas représentés. La légalité d'une situation réelle n'est pas déduite par le code.

## Apprentissage et résultat

PPO, réseau partagé par les groupes. `SymmetricExitPolicy` moyenne quatre vues tournées et réaligne les actions : une rotation du problème doit tourner les probabilités de déplacement. La réussite v2 exige trois tours consécutifs hors du cordon. Un retour dedans réinitialise le compteur ; le groupe réussi est retiré et reçoit une prime unique.

La récompense pénalise aussi le temps, les tentatives bloquées et la densité excessive. Le proxy de densité n'est pas une estimation de blessure. Le retrait des groupes réussis ignore les congestions et interceptions ultérieures.

## Comparaisons

Les trois entraînements utilisent les graines 0, 1 et 2. La sélection utilise 2027, le test 2029. `local_explore` mémorise les cases visitées et demande parfois un passage près d'un cordon ; `random_legal` choisit des déplacements accessibles. Ces références n'accèdent pas aux passages cachés, aux calendriers ou à la carte globale. La règle avec mémoire et le réseau sans récurrence n'ont pas la même architecture.

La v1 récompensait l'arrivée au bord de carte et marquait les contacts possibles. Ses pourcentages ne mesurent pas la même réussite que la v2.
