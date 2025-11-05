# Robot Path Planning with Potential Fields

Ce projet implémente un algorithme de planification de trajectoire pour robot utilisant la méthode des champs de potentiel. Cette approche permet à un robot de naviguer de manière autonome vers un objectif tout en évitant les obstacles.

## 🚀 Fonctionnalités

- Navigation autonome vers un objectif
- Évitement d'obstacles dynamique
- Visualisation des champs de force
- Optimisation de trajectoire
- Gestion des collisions entre obstacles

## � Évolution de l'optimisation

Le processus d'optimisation utilise JAX pour minimiser une fonction de perte qui prend en compte :
- La distance à la ligne droite idéale
- Les collisions potentielles entre obstacles
- La distance à l'objectif

Voici l'évolution de l'optimisation à travers différentes étapes :

### Étape 1 : Configuration initiale
![Étape 1](step1.png)
*Configuration initiale avec les obstacles placés aléatoirement*

### Étape 2 : Premières itérations
![Étape 2](step2.png)
*Les obstacles commencent à se déplacer pour optimiser la trajectoire*

### Étape 3 : Optimisation avancée
![Étape 3](step3.png)
*Les obstacles s'organisent pour créer un passage plus fluide*

### Étape 4 : Configuration finale
![Étape 4](step4.png)
*Configuration optimale obtenue après convergence*

## �🛠️ Technologies utilisées

- Python 3.x
- JAX (pour l'accélération des calculs et l'optimisation)
- NumPy (pour les calculs numériques)
- Matplotlib (pour la visualisation)

## 📋 Prérequis

```bash
pip install jax jaxlib numpy matplotlib
```

## 💡 Principe de fonctionnement

Le système utilise deux types de forces :
1. **Forces répulsives** : générées par les obstacles pour les éviter
2. **Forces attractives** : générées par l'objectif pour guider le robot

L'algorithme calcule en continu :
- La trajectoire optimale
- Les forces d'interaction entre les obstacles
- Le champ de potentiel global

## 🔧 Structure du code

- `get_forces()` : Calcule les forces répulsives et attractives
- `loss()` : Fonction de perte pour l'optimisation de la trajectoire
- `affichage()` : Visualisation de la trajectoire et des champs de force

## 📊 Visualisation

Le projet inclut des fonctionnalités de visualisation permettant d'afficher :
- La trajectoire calculée
- Les champs de force vectoriels
- La position des obstacles
- Le point de départ et d'arrivée

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Committer vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📝 License

Ce projet est sous licence MIT.
