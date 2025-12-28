# Détection et Classification des Panneaux de Signalisation Routière

Système de vision artificielle pour identifier automatiquement les panneaux de signalisation sur les routes (STOP, limitation de vitesse, sens interdit, etc.).

## 📋 Description

Ce projet implémente un système complet de détection et classification des panneaux de signalisation routière en utilisant :
- **Dataset** : German Traffic Sign Recognition Benchmark (GTSRB)
- **Technologies** : TensorFlow/Keras, OpenCV, YOLO/SSD/Faster R-CNN
- **Déploiement** : Webcam ou caméra embarquée en temps réel

## 🎯 Objectifs Pédagogiques

- Apprentissage supervisé sur un dataset d'images réelles
- Comprendre la segmentation et la classification d'objets
- Mettre en œuvre la détection d'objets en temps réel

## 📁 Structure du Projet

```
trafic panel/
├── src/                    # Modules Python réutilisables (.py)
│   ├── __init__.py
│   ├── data_loader.py     # Chargement du dataset GTSRB
│   ├── preprocessing.py   # Prétraitement des images
│   ├── model.py           # Modèles de classification (CNN, ResNet)
│   ├── detector.py        # Détection en temps réel
│   └── utils.py           # Fonctions utilitaires
│
├── notebooks/              # Notebooks Jupyter (.ipynb)
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_real_time_detection.ipynb
│
├── data/                   # Dataset GTSRB
│   ├── Train/
│   └── Test/
│
├── models/                 # Modèles entraînés sauvegardés
│
├── main.py                 # Script principal
├── requirements.txt        # Dépendances Python
└── README.md              # Ce fichier
```

## 🚀 Installation

### 1. Créer un environnement Conda

```bash
conda create -n traffic_signs python=3.9
conda activate traffic_signs
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Télécharger le Dataset GTSRB

```bash
python main.py download
```

Ou téléchargez manuellement depuis : https://benchmark.ini.rub.de/gtsrb_dataset.html

Extrayez les fichiers dans le dossier `data/` :
- `data/Train/` (dossiers 00000, 00001, etc.)
- `data/Test/` (images + Test.csv)

## 💻 Utilisation

### Entraîner un modèle

```bash
# Modèle CNN simple
python main.py train --model-type cnn --epochs 50

# Modèle ResNet (transfer learning)
python main.py train --model-type resnet --epochs 30
```

### Détection en temps réel

```bash
python main.py detect --model-path models/traffic_sign_cnn.h5 --camera 0
```

### Utiliser les Notebooks Jupyter

```bash
jupyter notebook
```

Puis ouvrez les notebooks dans l'ordre :
1. `01_data_exploration.ipynb` - Exploration du dataset
2. `02_preprocessing.ipynb` - Prétraitement des images
3. `03_model_training.ipynb` - Entraînement des modèles
4. `04_real_time_detection.ipynb` - Détection en temps réel

## 📚 Modules Python (.py)

### `src/data_loader.py`
- Classe `GTSRBDataLoader` pour charger le dataset
- Méthodes : `load_train_data()`, `load_test_data()`, `get_class_names()`

### `src/preprocessing.py`
- Classe `ImagePreprocessor` pour le prétraitement
- Normalisation, augmentation de données, amélioration du contraste

### `src/model.py`
- Classe `TrafficSignClassifier` pour créer et entraîner les modèles
- Support pour CNN simple et ResNet (transfer learning)

### `src/detector.py`
- Classe `RealTimeDetector` pour la détection en temps réel
- Utilise OpenCV pour la capture vidéo

### `src/utils.py`
- Fonctions utilitaires : visualisation, évaluation, etc.

## 📓 Notebooks Jupyter (.ipynb)

Les notebooks permettent une exploration interactive :
- **01_data_exploration.ipynb** : Analyse et visualisation du dataset
- **02_preprocessing.ipynb** : Tests de prétraitement et augmentation
- **03_model_training.ipynb** : Expérimentation avec différents modèles
- **04_real_time_detection.ipynb** : Tests de détection en temps réel

## 🔧 Configuration

Modifiez les paramètres dans `main.py` ou utilisez les arguments en ligne de commande :
- `--data-path` : Chemin vers le dataset
- `--model-type` : Type de modèle (cnn/resnet)
- `--epochs` : Nombre d'époques
- `--camera` : Index de la caméra

## 📊 Résultats Attendus

- **Accuracy** : > 95% sur le dataset de test
- **Temps réel** : Détection à ~15-30 FPS selon le matériel
- **Classes** : 43 classes de panneaux de signalisation allemands

## 🛠️ Technologies Utilisées

- **TensorFlow/Keras** : Deep Learning
- **OpenCV** : Traitement d'image et capture vidéo
- **NumPy/Pandas** : Manipulation de données
- **Matplotlib/Seaborn** : Visualisation
- **Scikit-learn** : Préprocessing et évaluation

## 📝 Notes

- Le dataset GTSRB contient 43 classes de panneaux allemands
- Les modèles peuvent être adaptés pour d'autres types de panneaux
- Pour une meilleure performance, utilisez une GPU pour l'entraînement

## 🤝 Contribution

N'hésitez pas à améliorer le projet en ajoutant :
- Support pour YOLO/SSD/Faster R-CNN
- Détection multi-panneaux simultanés
- Interface web avec Streamlit/Flask
- Export pour déploiement mobile

## 📄 Licence

Ce projet est à des fins éducatives.

