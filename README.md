# 🌍 Projet Big Data – Prédiction avec PySpark

Ce projet utilise **PySpark** dans un environnement **Jupyter Notebook sous Docker** pour traiter des données, entraîner des modèles de machine learning, et effectuer des prédictions sur les niveaux de pollution de l'air (CO, NO2, O3).

---

## 🚀 Objectifs du projet

- Nettoyer et analyser un jeu de données environnemental.
- Entraîner un modèle de machine learning (RandomForest).
- Prédire les concentrations de CO, NO2, et O3 à partir de mesures capteurs.
- Déployer et exécuter le projet via un conteneur Docker PySpark.
---

### 🧠 Technologies utilisées
- PySpark pour la manipulation des données massives.

- Scikit-learn pour l'entraînement du modèle.

- Docker pour l'environnement isolé.

- Jupyter Notebook pour l'exécution interactive.

- Git & GitHub pour la gestion de version.

- Pandas, NumPy, Matplotlib pour le traitement et la visualisation.


# 🧪 Présentation des Polluants
**🔸 O₃ — Ozone troposphérique**
Origine : Formé par réaction photochimique entre des polluants (NOₓ, COV) et les rayons UV du soleil.

Effets : Irritations respiratoires, impact sur la végétation.

Caractéristique : Polluant secondaire, présent surtout lors des épisodes de pollution estivale.

**🔸 NO₂ — Dioxyde d’azote**
Origine : Combustion des carburants (trafic routier, industries).

Effets : Provoque des inflammations des voies respiratoires et contribue à la formation d’ozone.

Caractéristique : Indicateur principal de la pollution urbaine.

**🔸 CO — Monoxyde de carbone**
Origine : Résulte d’une combustion incomplète (véhicules, chauffage domestique).

Effets : Réduit la capacité du sang à transporter l’oxygène.

Caractéristique : Inodore, incolore, très toxique à forte concentration.

---
# 🛠️ Tâches Réalisées
**1. Préparation des données**
Gestion des valeurs manquantes

Détection et traitement des outliers

Normalisation/standardisation

**2. Modélisation**

Entraînement de modèles de régression (régression linéaire, forêts aléatoires, etc.)

Évaluation via des métriques telles que RMSE et R²

**3. Visualisation des Résultats**

Comparaison des niveaux réels vs prédits

Graphiques de tendances et distribution des polluants

**4. Tests et Documentation**

Rédaction d’un rapport de résultats et conclusions

---

# 📈 Compétences Développées
Manipulation de données réelles (nettoyage, traitement des anomalies)

Implémentation et évaluation de modèles de régression

Interprétation et visualisation des résultats

Rédaction technique et communication scientifique

---


## 🐳 Utilisation de Docker

Voici les étapes à suivre pour exécuter le projet dans un conteneur Docker basé sur l’image `jupyter/pyspark-notebook`.

### ✅ 1. Lancer le conteneur Docker


docker run -p 8888:8888 -v /c/projet_bigdata:/home/jovyan/work --name fraud_detection jupyter/pyspark-notebook

*** Explication:
* -v /c/projet_bigdata:/home/jovyan/work : Monte le dossier local dans le conteneur.

* /home/jovyan/work : Répertoire de travail dans le conteneur.

📁 Résultat : Tous les fichiers de /c/projet_bigdata sont visibles dans Jupyter sous le dossier work.

### ✅ 2. Se connecter au conteneur Docker
docker exec -it fraud_detection bash

### ✅ 3. Aller dans le répertoire de travail à l’intérieur du conteneur
cd /home/jovyan/work

### ✅ 4. Exécuter le script Python
python MachineLearning_.py

---
🧠 Le script sera exécuté ligne par ligne dans l’environnement PySpark.

Jupyter sera accessible à l’adresse affichée dans le terminal (ex. http://127.0.0.1:8888).

### Contenu du projet
- MachineLearning_.ipynb : Notebook principal contenant les analyses.

- MachineLearning_.py : Script Python exécutable automatiquement.

- random_forest_model.pkl : Modèle sauvegardé.

- README.md : Instructions du projet.



