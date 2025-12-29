# 📁 Guide des Scripts de Détection

Ce projet contient maintenant **3 scripts séparés** pour chaque fonctionnalité de détection.

## 📋 Scripts Disponibles

### 1. `test_image_upload.py` - Détection sur Image Locale

**Usage :**
```bash
python test_image_upload.py
```

**Fonctionnalités :**
- Charge une image depuis votre ordinateur
- Analyse et détecte les panneaux
- Sauvegarde le résultat dans `detection_result.jpg`
- Affiche le résultat

**Exemple :**
```bash
python test_image_upload.py
# Entrez le chemin: data/Test/00000.ppm
# ou
# Entrez le chemin: C:/Users/HP/Pictures/traffic_sign.jpg
```

---

### 2. `test_image_url.py` - Détection sur Image depuis URL

**Usage :**
```bash
python test_image_url.py
```

**Fonctionnalités :**
- Télécharge une image depuis une URL
- Analyse et détecte les panneaux
- Sauvegarde le résultat dans `detection_result.jpg`
- Affiche le résultat
- Supprime automatiquement le fichier temporaire

**Exemple :**
```bash
python test_image_url.py
# Entrez l'URL: https://example.com/traffic_sign.jpg
```

---

### 3. `test_live_detection.py` - Détection en Temps Réel

**Usage :**
```bash
python test_live_detection.py
```

**Fonctionnalités :**
- Détection en temps réel avec webcam
- Appuyez sur 'q' pour quitter
- Choisissez l'index de la caméra (0 par défaut)

**Exemple :**
```bash
python test_live_detection.py
# Index de la caméra (appuyez Entrée pour 0): 0
```

---

## 🔧 Résolution Automatique des Chemins

Tous les scripts **trouvent automatiquement le modèle** dans le dossier `models/` :
- Cherche `models/traffic_sign_cnn.h5`
- Si non trouvé, cherche `models/traffic_sign_resnet.h5`
- Affiche un message d'erreur clair si aucun modèle n'est trouvé

## ✅ Avantages de la Séparation

1. **Plus Simple** : Chaque script fait une seule chose
2. **Plus Clair** : Facile de comprendre ce que fait chaque script
3. **Plus Facile à Maintenir** : Modifications isolées
4. **Chemins Corrigés** : Gestion automatique des chemins relatifs/absolus

## 📝 Exemples d'Utilisation

### Exemple 1 : Tester une Image Locale
```bash
python test_image_upload.py
# Entrez: data/Test/00000.ppm
```

### Exemple 2 : Tester depuis URL
```bash
python test_image_url.py
# Entrez: https://e7.pngegg.com/pngimages/31/875/png-clipart-germany-traffic-sign.png
```

### Exemple 3 : Détection Live
```bash
python test_live_detection.py
# Appuyez Entrée pour utiliser la caméra 0
# Appuyez 'q' pour quitter
```

## 🐛 Résolution de Problèmes

### Problème : "Aucun modèle trouvé"

**Solution :**
```bash
# Entraînez d'abord un modèle
python main.py train --model-type cnn --epochs 50
```

### Problème : "Fichier non trouvé" (test_image_upload.py)

**Solution :**
- Utilisez un chemin absolu : `C:/Users/HP/Pictures/image.jpg`
- Ou un chemin relatif depuis le dossier du projet : `data/Test/00000.ppm`

### Problème : "Erreur de téléchargement" (test_image_url.py)

**Solution :**
- Vérifiez que l'URL est accessible
- Vérifiez que l'URL commence par `http://` ou `https://`
- Vérifiez votre connexion Internet

### Problème : "Impossible d'ouvrir la caméra" (test_live_detection.py)

**Solution :**
- Vérifiez que votre webcam est connectée
- Essayez un autre index de caméra (1, 2, etc.)
- Vérifiez que la caméra n'est pas utilisée par un autre programme

## 📊 Comparaison avec main.py

| Fonctionnalité | Script Dédié | main.py |
|---------------|--------------|---------|
| Image Locale | ✅ `test_image_upload.py` | ✅ `test --image chemin` |
| URL | ✅ `test_image_url.py` | ✅ `test --image URL` |
| Live | ✅ `test_live_detection.py` | ✅ `detect` |
| Interface | ✅ Interactive | ❌ Ligne de commande |

## 💡 Recommandations

- **Pour débuter** : Utilisez les scripts dédiés (plus simples)
- **Pour automatiser** : Utilisez `main.py` avec des scripts batch
- **Pour Jupyter** : Utilisez les notebooks dans `notebooks/`

---

**Les scripts sont prêts à être utilisés !** 🚀

