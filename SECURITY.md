# Sécurité

Ne pas publier de clé, jeton, donnée personnelle ou détail exploitable dans une issue publique. Utiliser **Security → Report a vulnerability** si le signalement privé est activé ; à défaut, utiliser les fonctions de signalement de GitHub sans joindre de secret.

Les modèles publiés sont accompagnés d'empreintes. Préférer le script de téléchargement du dépôt et le chargement restreint `weights_only=True`. Les fichiers PyTorch de sources inconnues ne doivent pas être chargés avec la désérialisation complète.

Le projet est expérimental et ne garantit pas la sécurité physique. Un problème de validité scientifique doit être décrit dans une issue de méthode, sans données personnelles.
