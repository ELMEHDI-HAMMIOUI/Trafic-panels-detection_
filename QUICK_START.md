# ⚡ Démarrage Rapide

## 🚀 En 3 Étapes

### 1️⃣ Installer les Dépendances
```bash
pip install -r requirements.txt
```

### 2️⃣ Vérifier que Tout Fonctionne
```bash
python test_setup.py
```

### 3️⃣ Choisir votre Méthode

**Option A : Notebooks (Recommandé pour apprendre)**
```bash
jupyter notebook
# Puis ouvrez notebooks/01_data_exploration.ipynb
```

**Option B : Script Principal (Plus rapide)**
```bash
# Entraîner
python main.py train --model-type cnn --epochs 50

# Détecter
python main.py detect --model-path models/traffic_sign_cnn.h5
```

## 📋 Checklist Rapide

- [ ] Python 3.8+ installé
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Dataset présent dans `data/` (déjà présent ✅)
- [ ] Test réussi (`python test_setup.py`)

## 🎯 Commandes Essentielles

```bash
# Vérifier l'installation
python test_setup.py

# Entraîner un modèle
python main.py train

# Détection en temps réel
python main.py detect

# Lancer Jupyter
jupyter notebook
```

## 📚 Documentation Complète

- **Guide détaillé** : `HOW_TO_RUN.md`
- **Structure du projet** : `STRUCTURE.md`
- **Documentation** : `README.md`

---

**C'est parti ! 🚦**

