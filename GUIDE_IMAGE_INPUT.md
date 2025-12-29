# 📸 Guide : Charger une Image dans Jupyter

Ce guide explique comment charger une image dans le notebook pour tester la détection de panneaux.

## 🎯 Méthodes Disponibles

Le notebook `04_real_time_detection.ipynb` propose **4 méthodes** pour charger une image :

### Méthode 1 : Upload de Fichier (Recommandé) 📁

**La plus simple et intuitive !**

1. Exécutez la cellule "Méthode 1"
2. Cliquez sur le bouton "Upload" qui apparaît
3. Sélectionnez votre image depuis votre ordinateur
4. L'image sera automatiquement analysée et les résultats affichés

**Formats supportés** : `.jpg`, `.jpeg`, `.png`, `.ppm`, `.bmp`

### Méthode 2 : Chemin Manuel 📝

**Pour les images déjà sur votre ordinateur**

1. Exécutez la cellule "Méthode 2"
2. Entrez le chemin vers votre image dans le champ texte
   - Exemple : `C:/Users/HP/Pictures/traffic_sign.jpg`
   - Ou chemin relatif : `../data/Test/00000.ppm`
3. Cliquez sur "Charger et Détecter"
4. Les résultats s'affichent automatiquement

**Astuce** : Vous pouvez utiliser des chemins relatifs depuis le dossier du notebook

### Méthode 3 : Téléchargement depuis URL 🌐

**Pour tester avec des images depuis Internet**

1. Exécutez la cellule "Méthode 3"
2. Entrez l'URL d'une image
   - Exemple : `https://example.com/traffic_sign.jpg`
3. Cliquez sur "Télécharger et Détecter"
4. L'image sera téléchargée, analysée, puis supprimée automatiquement

**Note** : L'image doit être accessible publiquement

### Méthode 4 : Image du Dataset 🗂️

**Pour tester rapidement avec une image du dataset GTSRB**

1. Exécutez la cellule "Méthode 4"
2. L'image par défaut (`../data/Test/00000.ppm`) sera chargée
3. Vous pouvez modifier le chemin dans le code si besoin

## 💻 Exemple de Code Simple

Si vous préférez écrire votre propre code, voici un exemple simple :

```python
# Charger une image
image_path = "chemin/vers/votre/image.jpg"
result = detector.detect_from_image(image_path)

# Afficher
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Résultat de la Détection")
plt.axis('off')
plt.show()
```

## 🔧 Installation des Widgets (si nécessaire)

Si les widgets ne fonctionnent pas, installez `ipywidgets` :

```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

Ou dans JupyterLab :

```bash
pip install ipywidgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## 📋 Checklist

Avant d'utiliser les méthodes d'input :

- [ ] Le modèle est entraîné et sauvegardé dans `models/`
- [ ] Le détecteur est initialisé (cellule "1. Charger le Modèle")
- [ ] Les dépendances sont installées (`pip install -r requirements.txt`)
- [ ] `ipywidgets` est installé pour les méthodes 1, 2, 3

## 🐛 Résolution de Problèmes

### Problème : "ModuleNotFoundError: No module named 'ipywidgets'"
**Solution** :
```bash
pip install ipywidgets
```

### Problème : Les widgets ne s'affichent pas
**Solution** :
```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

### Problème : "Image not found"
**Solution** : Vérifiez que le chemin est correct
- Utilisez des chemins absolus : `C:/Users/HP/Pictures/image.jpg`
- Ou des chemins relatifs depuis le notebook : `../data/Test/image.ppm`

### Problème : "Model not found"
**Solution** : Entraînez d'abord un modèle avec le notebook `03_model_training.ipynb`

## 🎓 Conseils

1. **Commencez par la Méthode 1** : C'est la plus simple et ne nécessite pas de connaître les chemins
2. **Testez avec différentes images** : Panneaux réels, images du dataset, etc.
3. **Vérifiez la qualité** : Les images trop petites ou floues peuvent donner de mauvais résultats
4. **Utilisez des images claires** : Le modèle fonctionne mieux avec des panneaux bien visibles

## 📸 Exemples d'Images à Tester

- Images du dataset GTSRB (déjà présentes)
- Photos de panneaux de signalisation réels
- Images depuis Internet (avec la méthode URL)
- Captures d'écran de vidéos

---

**Bon test ! 🚦**

