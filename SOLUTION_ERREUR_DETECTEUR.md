# 🔧 Solution : Erreur "detector is not defined"

## ❌ Le Problème

Vous obtenez cette erreur :
```
NameError: name 'detector' is not defined
```

## ✅ La Solution

**Le détecteur n'a pas été initialisé !** Vous devez d'abord exécuter la cellule qui charge le modèle.

### Étapes à Suivre

1. **Exécutez la Cellule 1** (Imports)
   - C'est la première cellule avec les imports
   - Cliquez dessus et appuyez sur `Shift + Enter`

2. **Exécutez la Cellule 3** : "1. Charger le Modèle"
   - ⚠️ **CETTE CELLULE EST OBLIGATOIRE !**
   - Elle doit afficher : `✅ Détecteur initialisé avec succès!`
   - Si vous voyez une erreur, lisez la section "Problèmes Courants" ci-dessous

3. **Ensuite** : Vous pouvez utiliser n'importe quelle méthode (1, 2, 3 ou 4)

## 📋 Ordre d'Exécution Correct

```
Cellule 1 (Imports) → Shift + Enter
    ↓
Cellule 3 ("1. Charger le Modèle") → Shift + Enter
    ↓
✅ Détecteur initialisé !
    ↓
Maintenant vous pouvez utiliser :
- Méthode 1 (Upload)
- Méthode 2 (Chemin)
- Méthode 3 (URL) ← C'est celle que vous voulez !
- Méthode 4 (Dataset)
```

## 🐛 Problèmes Courants

### Problème 1 : "Le modèle n'existe pas"

**Message d'erreur** :
```
❌ ERREUR: Le modèle ../models/traffic_sign_cnn.h5 n'existe pas!
```

**Solution** :
1. Entraînez d'abord un modèle avec le notebook `03_model_training.ipynb`
2. Ou modifiez `MODEL_PATH` dans la cellule 1 pour pointer vers un modèle existant

### Problème 2 : "ModuleNotFoundError"

**Message d'erreur** :
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution** :
```bash
pip install -r requirements.txt
```

### Problème 3 : Le détecteur n'est toujours pas défini après exécution

**Vérifications** :
1. Assurez-vous que la cellule 3 s'est exécutée **sans erreur**
2. Vérifiez que vous voyez le message `✅ Détecteur initialisé avec succès!`
3. Si vous voyez une erreur, lisez le message et corrigez le problème

## 💡 Astuce : Vérifier que le Détecteur est Initialisé

Ajoutez cette cellule pour vérifier :

```python
# Vérification
if 'detector' in globals() and detector is not None:
    print("✅ Détecteur est initialisé et prêt!")
else:
    print("❌ Détecteur n'est pas initialisé. Exécutez la cellule '1. Charger le Modèle'")
```

## 🎯 Exemple Complet

Voici comment utiliser la méthode 3 (URL) correctement :

```python
# 1. D'abord, exécutez la cellule 1 (Imports)
# 2. Ensuite, exécutez la cellule 3 ("1. Charger le Modèle")
#    Vous devriez voir : ✅ Détecteur initialisé avec succès!

# 3. Maintenant, exécutez la cellule 9 (Méthode 3)
# 4. Entrez une URL dans le champ texte
# 5. Cliquez sur "Télécharger et Détecter"
```

## 📝 Notes Importantes

- ⚠️ **Toujours exécuter la cellule 3 avant les autres méthodes**
- ✅ Le détecteur reste initialisé pour toute la session Jupyter
- 🔄 Si vous redémarrez le kernel, vous devez réexécuter la cellule 3

---

**Si le problème persiste**, vérifiez que :
1. Toutes les dépendances sont installées
2. Le modèle existe dans `models/traffic_sign_cnn.h5`
3. Vous exécutez les cellules dans le bon ordre

