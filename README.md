# ⚙️ PredictMaint – Maintenance prédictive par IA

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

Projet PFA – **Yousra Erraki** · EMSI Rabat (Ingénierie Informatique, IA & Data Science)

**Objectif :** anticiper les pannes de machines industrielles (presses hydrauliques, tours CNC, ponts roulants…) à partir des **données capteurs** (températures, vibrations, pression d'huile, charge, vitesse, consommation) et des informations sur les pièces, afin de planifier la maintenance au bon moment.

![Accueil](screenshots/accueil.png)

---

## 📁 Contenu du dépôt

| Fichier | Description |
|---|---|
| `PFA.ipynb` | Notebook : préparation des données, rééquilibrage des classes, entraînement et comparaison de 5 modèles |
| `app.py` | Application Streamlit : diagnostic d'une machine, exploration des données, performance du modèle |
| `requirements.txt` | Dépendances Python |
| `screenshots/` | Captures d'écran de l'application |

## 🧠 Démarche

1. **Données** : historique de sorties de pièces et de mesures capteurs, avec une variable cible `breakdown_flag` (panne / pas de panne). Environ 10 % de pannes : jeu de données **déséquilibré**.
2. **Préparation** : rééquilibrage (3 non-pannes pour 1 panne), extraction de variables temporelles (mois, saison), encodage des variables catégorielles, normalisation pour SVM, KNN et ANN.
3. **Modélisation** : comparaison de 5 algorithmes, avec gestion du déséquilibre (`class_weight`, SMOTE pour KNN).
4. **Application** : intégration du modèle dans une interface Streamlit avec un **seuil d'alerte réglable**.

## 📊 Résultats (jeu de test, scores pondérés)

| Modèle | Accuracy | Précision | Rappel | F1-score |
|---|---|---|---|---|
| SVM (RBF) | 58,78 % | 74,22 % | 58,78 % | 61,35 % |
| KNN (+ SMOTE) | 57,58 % | 71,74 % | 57,58 % | 60,31 % |
| Decision Tree | 57,43 % | 73,45 % | 57,43 % | 60,02 % |
| **Random Forest** | **71,86 %** | 66,81 % | **71,86 %** | 68,10 % |
| ANN (128-64-32) | 67,09 % | 71,08 % | 67,09 % | **68,57 %** |

**Analyse :**
- L'**ANN** et le **Random Forest** obtiennent les meilleurs F1-scores (≈ 68 %).
- Le **Random Forest** a été retenu pour l'application : F1 quasi identique, meilleure accuracy, entraînement rapide et **interprétable** (importance des variables).
- Les modèles avec classes équilibrées (SVM, Decision Tree) détectent davantage de pannes (rappel de la classe « panne » ≈ 76-77 %) mais génèrent plus de fausses alertes : c'est le compromis classique de la maintenance prédictive, que l'application permet de régler avec le **seuil d'alerte**.

## 🖥️ L'application

- **Diagnostic** : saisie de l'état d'une machine → jauge de risque de panne, verdict et capteurs à surveiller
- **Exploration** : taux de panne par machine et par famille de pièce, distribution des capteurs (panne vs normal)
- **Performance** : accuracy, précision, rappel, F1, matrice de confusion, variables les plus influentes

![Diagnostic](screenshots/diagnostic.png)
![Exploration](screenshots/exploration.png)

## 🚀 Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Placer `DataPiece.csv` (séparateur `;`) dans le même dossier que `app.py`, ou le charger depuis la barre latérale.

## 🔭 Pistes d'amélioration

- Exploiter la dimension temporelle (moyennes glissantes, tendances des capteurs par machine)
- Tester XGBoost / LightGBM et optimiser les hyperparamètres
- Optimiser le seuil selon le coût réel d'une panne ratée vs une fausse alerte
- Déployer l'application (Streamlit Community Cloud) et l'alimenter en temps réel (IoT)

## 🛠️ Technologies

Python · Pandas · NumPy · Scikit-learn · Imbalanced-learn · TensorFlow/Keras · Streamlit · Plotly

## 👩‍💻 Auteure

**Yousra Erraki** – Étudiante ingénieure en IA & Data Science (EMSI Rabat)
[LinkedIn](https://www.linkedin.com/in/yousra-erraki-885b20221) · [GitHub](https://github.com/yousraerraki) · [Portfolio](https://yousra-erraki-portfolio.netlify.app)
