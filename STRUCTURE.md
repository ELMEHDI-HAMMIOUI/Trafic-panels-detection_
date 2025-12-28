# Structure du Projet - Détection de Panneaux de Signalisation

## 📁 Organisation des Fichiers

### Fichiers Python (.py) - Modules Réutilisables

Tous les fichiers Python sont dans le dossier `src/` :

1. **`src/__init__.py`** - Module d'initialisation
2. **`src/data_loader.py`** - Chargement du dataset GTSRB
   - Classe `GTSRBDataLoader` : charge les images et labels
3. **`src/preprocessing.py`** - Prétraitement des images
   - Classe `ImagePreprocessor` : normalisation, augmentation, amélioration du contraste
4. **`src/model.py`** - Modèles de classification
   - Classe `TrafficSignClassifier` : CNN et ResNet
5. **`src/detector.py`** - Détection en temps réel
   - Classe `RealTimeDetector` : détection avec webcam
6. **`src/utils.py`** - Fonctions utilitaires
   - Visualisation, évaluation, matrices de confusion

### Notebooks Jupyter (.ipynb) - Exploration Interactive

Tous les notebooks sont dans le dossier `notebooks/` :

1. **`01_data_exploration.ipynb`** 
   - Exploration du dataset GTSRB
   - Visualisation des images
   - Analyse de la distribution des classes

2. **`02_preprocessing.ipynb`**
   - Tests de prétraitement
   - Normalisation et augmentation
   - Amélioration du contraste

3. **`03_model_training.ipynb`**
   - Entraînement des modèles
   - Évaluation des performances
   - Comparaison CNN vs ResNet

4. **`04_real_time_detection.ipynb`**
   - Test de détection sur images
   - Détection en temps réel avec webcam

### Fichiers Principaux

- **`main.py`** - Script principal pour exécution en ligne de commande
- **`requirements.txt`** - Dépendances Python
- **`README.md`** - Documentation complète du projet

## 🚀 Utilisation

### Avec les Notebooks (Recommandé pour l'apprentissage)
```bash
jupyter notebook
```
Puis ouvrez les notebooks dans l'ordre (01 → 04)

### Avec les Scripts Python
```bash
# Entraîner un modèle
python main.py train --model-type cnn --epochs 50

# Détection en temps réel
python main.py detect --model-path models/traffic_sign_cnn.h5
```

## 📂 Structure Complète

```
trafic panel/
├── src/                          # Fichiers Python (.py)
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── detector.py
│   └── utils.py
│
├── notebooks/                    # Notebooks Jupyter (.ipynb)
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_real_time_detection.ipynb
│
├── data/                         # Dataset GTSRB
│   ├── Train/
│   └── Test/
│
├── models/                        # Modèles entraînés
│
├── main.py                       # Script principal
├── requirements.txt              # Dépendances
├── README.md                      # Documentation
└── STRUCTURE.md                  # Ce fichier
```

## 💡 Différence entre .py et .ipynb

- **Fichiers .py** : Code réutilisable, modules, classes → Utilisés par les notebooks et le script principal
- **Notebooks .ipynb** : Exploration interactive, visualisation, expérimentation → Pour comprendre et tester

