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
---

## 🧪 Modèles de Prédiction et Résultats

Dans ce projet, deux approches ont été testées pour prédire les niveaux de pollution (CO, NO2, O3) :

### ✅ 1. Approche 1 – **Model Simpl_OutPut**
Chaque polluant (CO, NO2, O3) est prédit indépendamment à l'aide de modèles distincts.

- **O3 (Ozone)**
  - 🔍 Modèle : K-Nearest Neighbors (KNN)
  - 📈 Résultats : `RMSE = 0.0508`, `R² = 0.9113`

- **NO2 (Dioxyde d’azote)**
  - 🔍 Modèle : Random Forest
  - 📈 Résultats : `RMSE = 0.0567`, `R² = 0.8048`

- **CO (Monoxyde de carbone)**
  - 🔍 Modèle : K-Nearest Neighbors (KNN)
  - 📈 Résultats : `RMSE = 0.0522`, `R² = 0.9063`

---

### ✅ 2. Approche 2 – **Model Multi-OutPut (MultiOutputRegressor)**
Un seul modèle est entraîné pour prédire simultanément les trois variables (CO, NO2, O3).

📊 **Performances des différents algorithmes testés avec MultiOutput :**

| Modèle                       | RMSE        | R²          |
|-----------------------------|-------------|-------------|
| Random Forest               | 0.0496      | 0.8658      |
| Gradient Boosting           | 0.0540      | 0.8412      |
| Support Vector Machine (SVM)| 0.0556      | 0.8304      |
| K-Nearest Neighbors (KNN)   | 0.0517      | 0.8545      |
| ElasticNet                  | 0.1401      | -0.00003    |
| XGBoost                     | 0.0512      | 0.8587      |


### 🎯 Conclusion

- L'approche **SimplOutPut** offre des performances légèrement meilleures pour chaque variable ciblée, notamment grâce à l'adaptation du modèle au type de variable.
- L’approche **Multi_Output** est plus compacte et efficace en termes de code et de gestion des données, tout en maintenant des performances globales élevées.

Les deux approches sont donc complémentaires selon le contexte d’utilisation (précision vs simplicité).
### ✅ Choix Final du Modèle

Après comparaison des deux approches (modèles simples vs. modèle multi-sortie) et analyse des résultats, j'ai opté pour les modèles suivants pour les prédictions finales :

- 🔹 **O3 (Ozone)** : **K-Nearest Neighbors (KNN)**, pour ses très bonnes performances (`R² = 0.9113`)
- 🔹 **CO (Monoxyde de carbone)** : **K-Nearest Neighbors (KNN)**, avec un 'R² = 0.9063'
- 🔹 **NO2 (Dioxyde d’azote)** : **Random Forest** pour la prédiction de **NO2**, car c’est le modèle qui a donné les meilleurs résultats avec précision (`RMSE = 0.0567`)  pour cette variable dans l’approche par modèle simple.

---

### 🎯 Conclusion

- L'approche **SimplOutPut** offre des performances légèrement meilleures pour chaque variable ciblée, notamment grâce à l'adaptation du modèle au type de variable.
- L’approche **Multi_OutPut** est plus compacte et efficace en termes de code et de gestion des données, tout en maintenant des performances globales élevées.

Les deux approches sont donc complémentaires selon le contexte d’utilisation (précision vs simplicité).



## 🐳 Utilisation de Docker

Voici les étapes à suivre pour exécuter le projet dans un conteneur Docker basé sur l’image `jupyter/pyspark-notebook`.

### ✅ 1. Lancer le conteneur Docker


docker run -p 8888:8888 -v /c/projet_bigdata:/home/jovyan/work --name Pred_Air jupyter/pyspark-notebook

*** Explication:
* -v /c/projet_bigdata:/home/jovyan/work : Monte le dossier local dans le conteneur.

* /home/jovyan/work : Répertoire de travail dans le conteneur.

📁 Résultat : Tous les fichiers de /c/projet_bigdata sont visibles dans Jupyter sous le dossier work.

### ✅ 2. Se connecter au conteneur Docker
docker exec -it Pred_Air bash

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



