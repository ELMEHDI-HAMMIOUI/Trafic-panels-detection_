"""
Module utilitaire avec fonctions auxiliaires
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from typing import Tuple, List


def plot_training_history(history, save_path: str = None):
    """
    Affiche l'historique d'entraînement
    
    Args:
        history: Historique retourné par model.fit()
        save_path: Chemin pour sauvegarder le graphique (optionnel)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Graphique sauvegardé dans {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], save_path: str = None):
    """
    Affiche la matrice de confusion
    
    Args:
        y_true: Vraies étiquettes
        y_pred: Prédictions
        class_names: Liste des noms de classes
        save_path: Chemin pour sauvegarder le graphique (optionnel)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Matrice de confusion sauvegardée dans {save_path}")
    
    plt.show()


def display_sample_images(images: np.ndarray, labels: np.ndarray, 
                         class_names: dict, num_samples: int = 9):
    """
    Affiche un échantillon d'images avec leurs labels
    
    Args:
        images: Tableau d'images
        labels: Tableau de labels
        class_names: Dictionnaire des noms de classes
        num_samples: Nombre d'images à afficher
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        axes[i].imshow(images[idx])
        class_name = class_names.get(labels[idx], f"Classe {labels[idx]}")
        axes[i].set_title(f"Classe: {class_name}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                  class_names: List[str]):
    """
    Évalue le modèle et affiche les métriques
    
    Args:
        model: Modèle à évaluer
        X_test: Images de test
        y_test: Labels de test
        class_names: Liste des noms de classes
    """
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Rapport de classification
    print("\n" + "="*50)
    print("RAPPORT DE CLASSIFICATION")
    print("="*50)
    print(classification_report(y_test, y_pred_classes, 
                                target_names=[str(c) for c in class_names]))
    
    # Matrice de confusion
    plot_confusion_matrix(y_test, y_pred_classes, class_names)
    
    # Accuracy globale
    accuracy = np.mean(y_pred_classes == y_test)
    print(f"\nAccuracy globale: {accuracy:.4f}")


def download_gtsrb_dataset(download_path: str = "data"):
    """
    Aide à télécharger le dataset GTSRB
    Note: Cette fonction nécessite que l'utilisateur télécharge manuellement
    le dataset depuis https://benchmark.ini.rub.de/gtsrb_dataset.html
    
    Args:
        download_path: Chemin où placer le dataset
    """
    print("="*60)
    print("TÉLÉCHARGEMENT DU DATASET GTSRB")
    print("="*60)
    print("\nLe dataset GTSRB doit être téléchargé manuellement.")
    print("Instructions:")
    print("1. Visitez: https://benchmark.ini.rub.de/gtsrb_dataset.html")
    print("2. Téléchargez les fichiers suivants:")
    print("   - GTSRB-Training_fixed.zip")
    print("   - GTSRB-Test_fixed.zip")
    print("3. Extrayez-les dans le dossier:", download_path)
    print("4. La structure devrait être:")
    print(f"   {download_path}/")
    print("   ├── Train/")
    print("   │   ├── 00000/")
    print("   │   ├── 00001/")
    print("   │   └── ...")
    print("   └── Test/")
    print("       ├── 00000.ppm")
    print("       ├── 00001.ppm")
    print("       └── Test.csv")
    print("\n" + "="*60)

