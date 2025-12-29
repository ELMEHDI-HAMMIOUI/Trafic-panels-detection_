# 🖥️ Guide : Utiliser une URL depuis le Terminal

## 🎯 Utilisation Basique

### Tester une Image depuis une URL

```bash
python main.py test --image https://example.com/traffic_sign.jpg
```

### Tester une Image Locale

```bash
python main.py test --image data/Test/00000.ppm
```

### Sauvegarder le Résultat

```bash
python main.py test --image https://example.com/image.jpg --output result.jpg
```

## 📋 Exemples Complets

### Exemple 1 : Image depuis URL
```bash
python main.py test --image "https://e7.pngegg.com/pngimages/31/875/png-clipart-germany-traffic-sign-overtaking-almanya-daki-otoyollar-driving-driving-truck-logo.png"
```

### Exemple 2 : Image Locale
```bash
python main.py test --image "data/Test/00000.ppm"
```

### Exemple 3 : Avec Sauvegarde
```bash
python main.py test --image "https://example.com/sign.jpg" --output "detection_result.jpg"
```

### Exemple 4 : Modèle Personnalisé
```bash
python main.py test --model-path models/traffic_sign_resnet.h5 --image "https://example.com/image.jpg"
```

## 🔧 Options Disponibles

### Mode `test`
- `--image` : **OBLIGATOIRE** - Chemin vers l'image ou URL
- `--model-path` : Chemin vers le modèle (défaut: `models/traffic_sign_cnn.h5`)
- `--output` : Chemin pour sauvegarder le résultat (optionnel)

### Exemples avec Options

```bash
# Utiliser un modèle ResNet
python main.py test --model-path models/traffic_sign_resnet.h5 --image "https://example.com/image.jpg"

# Sauvegarder le résultat
python main.py test --image "data/Test/00000.ppm" --output "my_result.png"

# Tout ensemble
python main.py test --model-path models/traffic_sign_resnet.h5 --image "https://example.com/image.jpg" --output "result.jpg"
```

## 📝 Format des URLs

Les URLs doivent commencer par :
- `http://` ou `https://`
- Exemple valide : `https://example.com/image.jpg`
- Exemple invalide : `example.com/image.jpg` (manque http://)

## 🖼️ Formats d'Images Supportés

- `.jpg` / `.jpeg`
- `.png`
- `.ppm`
- `.bmp`

## ⚠️ Notes Importantes

1. **Le modèle doit exister** : Entraînez d'abord un modèle avec `python main.py train`

2. **L'image est téléchargée temporairement** : Si vous utilisez une URL, l'image est téléchargée dans un fichier temporaire qui est supprimé après traitement

3. **Affichage** : Si vous ne spécifiez pas `--output`, le résultat s'affichera avec matplotlib (nécessite un affichage graphique)

4. **Sauvegarde** : Utilisez `--output` pour sauvegarder le résultat dans un fichier

## 🐛 Résolution de Problèmes

### Erreur : "Le modèle n'existe pas"
```bash
# Entraînez d'abord un modèle
python main.py train --model-type cnn --epochs 50
```

### Erreur : "Vous devez spécifier --image"
```bash
# N'oubliez pas l'option --image
python main.py test --image "votre_url_ou_chemin"
```

### Erreur : "Impossible de charger l'image"
- Vérifiez que l'URL est accessible
- Vérifiez que le chemin local est correct
- Vérifiez que l'image est dans un format supporté

## 💡 Astuces

### Tester Plusieurs Images
```bash
# Créez un script batch (Windows)
@echo off
python main.py test --image "https://example.com/image1.jpg" --output "result1.jpg"
python main.py test --image "https://example.com/image2.jpg" --output "result2.jpg"
python main.py test --image "https://example.com/image3.jpg" --output "result3.jpg"
```

### Utiliser avec des Chemins Absolus
```bash
# Windows
python main.py test --image "C:\Users\HP\Pictures\traffic_sign.jpg"

# Linux/Mac
python main.py test --image "/home/user/images/traffic_sign.jpg"
```

## 📊 Comparaison : Terminal vs Jupyter

| Fonctionnalité | Terminal | Jupyter |
|---------------|----------|---------|
| URL | ✅ `--image URL` | ✅ Widget |
| Fichier Local | ✅ `--image chemin` | ✅ Widget |
| Sauvegarde | ✅ `--output` | ❌ Affichage seulement |
| Batch Processing | ✅ Script | ❌ Manuel |
| Interface | ❌ Ligne de commande | ✅ Interactif |

---

**Utilisez le terminal pour automatiser et traiter plusieurs images !** 🚀

