"""
Script pour tester la détection sur une image uploadée (fichier local)
"""

import sys
import os
from pathlib import Path

# Ajouter le dossier src au path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

import cv2
import matplotlib.pyplot as plt
from src.data_loader import GTSRBDataLoader
from src.detector import RealTimeDetector


def find_model():
    """Trouve le modèle disponible"""
    models_dir = project_root / "models"
    
    # Chercher les modèles disponibles
    model_files = [
        models_dir / "traffic_sign_cnn.h5",
        models_dir / "traffic_sign_resnet.h5",
    ]
    
    for model_path in model_files:
        if model_path.exists():
            return str(model_path)
    
    return None


def main():
    print("="*60)
    print("DÉTECTION SUR IMAGE (FICHIER LOCAL)")
    print("="*60)
    
    # Trouver le modèle
    model_path = find_model()
    if model_path is None:
        print("\n❌ ERREUR: Aucun modèle trouvé!")
        print("Modèles recherchés dans: models/")
        print("\nPour résoudre:")
        print("1. Entraînez un modèle avec: python main.py train")
        print("2. Ou placez un modèle dans le dossier models/")
        return
    
    print(f"\n✅ Modèle trouvé: {model_path}")
    
    # Demander le chemin de l'image
    print("\n" + "-"*60)
    image_path = input("Entrez le chemin vers l'image: ").strip()
    
    # Supprimer les guillemets si présents
    image_path = image_path.strip('"').strip("'")
    
    if not image_path:
        print("❌ Aucun chemin fourni")
        return
    
    # Vérifier si le fichier existe
    if not os.path.exists(image_path):
        print(f"❌ Erreur: Le fichier '{image_path}' n'existe pas")
        print("\nAstuce: Utilisez un chemin absolu ou relatif depuis ce dossier")
        return
    
    try:
        # Charger les noms de classes
        data_path = project_root / "data"
        loader = GTSRBDataLoader(str(data_path))
        class_names = loader.get_class_names()
        
        # Créer le détecteur
        print("\nChargement du détecteur...")
        detector = RealTimeDetector(model_path, class_names)
        print("✅ Détecteur chargé")
        
        # Détecter
        print(f"\nAnalyse de l'image: {image_path}")
        result = detector.detect_from_image(image_path)
        
        # Sauvegarder le résultat
        output_path = project_root / "detection_result.jpg"
        cv2.imwrite(str(output_path), result)
        print(f"✅ Résultat sauvegardé dans: {output_path}")
        
        # Afficher
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(14, 8))
        plt.imshow(result_rgb)
        plt.title("Résultat de la Détection", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

