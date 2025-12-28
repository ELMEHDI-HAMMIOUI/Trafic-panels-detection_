# 🚀 Guide d'Exécution - Détection de Panneaux de Signalisation

## 📋 Prérequis

### 1. Vérifier Python
```bash
python --version
# Doit être Python 3.8 ou supérieur
```

### 2. Installer les Dépendances

**Option A : Avec pip (recommandé)**
```bash
# Activer votre environnement conda si vous en avez un
conda activate votre_env  # ou créer un nouvel environnement

# Installer les dépendances
pip install -r requirements.txt
```

**Option B : Avec conda**
```bash
conda install --file requirements.txt
```

**Option C : Installation manuelle des principales**
```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn Pillow jupyter
```

### 3. Vérifier l'Installation
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

## 🎯 Méthode 1 : Utiliser les Notebooks Jupyter (Recommandé pour l'apprentissage)

### Étape 1 : Lancer Jupyter
```bash
# Depuis le dossier racine du projet
jupyter notebook
```

### Étape 2 : Ouvrir les Notebooks dans l'Ordre

1. **`01_data_exploration.ipynb`** - Explorer le dataset
   - Vérifie que le dataset est présent
   - Visualise les images
   - Analyse la distribution des classes

2. **`02_preprocessing.ipynb`** - Prétraiter les images
   - Teste la normalisation
   - Explore l'augmentation de données
   - Prépare les données pour l'entraînement

3. **`03_model_training.ipynb`** - Entraîner le modèle
   - Crée un modèle CNN
   - Entraîne le modèle
   - Évalue les performances
   - Sauvegarde le modèle

4. **`04_real_time_detection.ipynb`** - Détection en temps réel
   - Charge un modèle entraîné
   - Teste sur des images
   - Lance la détection avec webcam

### ⚠️ Important pour les Notebooks
- Exécutez les cellules **dans l'ordre**
- Attendez que chaque cellule se termine avant de passer à la suivante
- Les notebooks sont dans `notebooks/`, donc les chemins sont relatifs à ce dossier

## 🎯 Méthode 2 : Utiliser le Script Principal (Plus Rapide)

### Étape 1 : Entraîner un Modèle

**Modèle CNN simple :**
```bash
python main.py train --model-type cnn --epochs 50
```

**Modèle ResNet (meilleure performance) :**
```bash
python main.py train --model-type resnet --epochs 30
```

**Avec options personnalisées :**
```bash
python main.py train --data-path data --model-type cnn --epochs 100
```

### Étape 2 : Détection en Temps Réel

**Avec webcam (caméra par défaut) :**
```bash
python main.py detect --model-path models/traffic_sign_cnn.h5
```

**Avec une caméra spécifique :**
```bash
python main.py detect --model-path models/traffic_sign_cnn.h5 --camera 1
```

### Étape 3 : Obtenir de l'Aide
```bash
python main.py --help
python main.py train --help
python main.py detect --help
```

## 📊 Structure des Commandes

### Commande Train
```bash
python main.py train [OPTIONS]

Options:
  --data-path PATH      Chemin vers le dataset (défaut: data)
  --model-type TYPE     Type de modèle: cnn ou resnet (défaut: cnn)
  --epochs N            Nombre d'époques (défaut: 50)
```

### Commande Detect
```bash
python main.py detect [OPTIONS]

Options:
  --model-path PATH     Chemin vers le modèle (défaut: models/traffic_sign_cnn.h5)
  --camera N           Index de la caméra (défaut: 0)
```

### Commande Download
```bash
python main.py download [OPTIONS]

Options:
  --data-path PATH      Où télécharger le dataset (défaut: data)
```

## 🔍 Vérification Rapide

### Test 1 : Vérifier les Imports
```bash
python -c "from src.data_loader import GTSRBDataLoader; print('✅ OK')"
python -c "from src.model import TrafficSignClassifier; print('✅ OK')"
python -c "from src.detector import RealTimeDetector; print('✅ OK')"
```

### Test 2 : Vérifier le Dataset
```bash
python -c "from pathlib import Path; print('Dataset existe:', Path('data/Train').exists())"
```

### Test 3 : Vérifier les Chemins
```bash
# Depuis le dossier racine du projet
python -c "from pathlib import Path; print('Racine:', Path.cwd()); print('Data:', Path('data').exists()); print('Src:', Path('src').exists())"
```

## 🐛 Résolution de Problèmes

### Problème 1 : "ModuleNotFoundError"
**Solution :** Installez les dépendances
```bash
pip install -r requirements.txt
```

### Problème 2 : "No module named 'cv2'"
**Solution :** Installez OpenCV
```bash
pip install opencv-python
```

### Problème 3 : "No module named 'tensorflow'"
**Solution :** Installez TensorFlow
```bash
pip install tensorflow
# Ou pour GPU
pip install tensorflow-gpu
```

### Problème 4 : Erreur de Chemin dans les Notebooks
**Solution :** Assurez-vous d'exécuter les notebooks depuis Jupyter (pas directement)
- Lancez `jupyter notebook` depuis le dossier racine
- Ouvrez les notebooks depuis l'interface Jupyter

### Problème 5 : "Dataset not found"
**Solution :** Le dataset doit être dans `data/`
```bash
# Vérifier la structure
ls data/Train/  # Doit contenir des dossiers 00000, 00001, etc.
ls data/Test/   # Doit contenir des fichiers .ppm et Test.csv
```

### Problème 6 : Erreur GPU
**Solution :** TensorFlow utilisera le CPU si GPU n'est pas disponible
- C'est normal, l'entraînement sera juste plus lent
- Pour GPU, installez `tensorflow-gpu` et les drivers CUDA

## 📝 Exemple Complet d'Exécution

### Scénario : Entraîner et Tester

```bash
# 1. Installer les dépendances (une seule fois)
pip install -r requirements.txt

# 2. Vérifier que le dataset existe
python main.py download

# 3. Entraîner un modèle CNN
python main.py train --model-type cnn --epochs 50

# 4. Tester la détection
python main.py detect --model-path models/traffic_sign_cnn.h5
```

## 🎓 Pour les Débutants

### Première Exécution Recommandée

1. **Installez les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

2. **Lancez Jupyter**
   ```bash
   jupyter notebook
   ```

3. **Ouvrez `01_data_exploration.ipynb`**
   - Cliquez sur "Run" pour chaque cellule
   - Vérifiez que tout fonctionne

4. **Continuez avec les autres notebooks dans l'ordre**

## ⚡ Pour les Utilisateurs Expérimentés

### Exécution Rapide
```bash
# Tout en une ligne
pip install -r requirements.txt && python main.py train --epochs 50 && python main.py detect
```

### Script d'Automatisation
Créez un fichier `run.sh` (Linux/Mac) ou `run.bat` (Windows) :
```bash
#!/bin/bash
pip install -r requirements.txt
python main.py train --model-type cnn --epochs 50
python main.py detect --model-path models/traffic_sign_cnn.h5
```

## 📞 Support

Si vous rencontrez des problèmes :
1. Vérifiez que toutes les dépendances sont installées
2. Vérifiez que le dataset est présent dans `data/`
3. Vérifiez que vous êtes dans le bon dossier (racine du projet)
4. Consultez les messages d'erreur pour plus de détails

---

**Bon entraînement ! 🚦**

