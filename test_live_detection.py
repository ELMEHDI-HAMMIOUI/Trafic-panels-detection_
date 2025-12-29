"""
Script pour la détection en temps réel avec webcam
"""

import sys
import os
from pathlib import Path

# Ajouter le dossier src au path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

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
    print("DÉTECTION EN TEMPS RÉEL (WEBCAM)")
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
    
    # Demander l'index de la caméra
    print("\n" + "-"*60)
    camera_input = input("Index de la caméra (0 pour webcam par défaut, appuyez Entrée pour 0): ").strip()
    
    try:
        camera_index = int(camera_input) if camera_input else 0
    except ValueError:
        print("⚠️  Index invalide, utilisation de la caméra 0")
        camera_index = 0
    
    try:
        # Charger les noms de classes
        data_path = project_root / "data"
        loader = GTSRBDataLoader(str(data_path))
        class_names = loader.get_class_names()
        
        # Créer le détecteur
        print("\nChargement du détecteur...")
        detector = RealTimeDetector(model_path, class_names)
        print("✅ Détecteur chargé")
        
        print("\n" + "="*60)
        print("DÉTECTION EN COURS...")
        print("Appuyez sur 'q' pour quitter")
        print("="*60)
        
        # Lancer la détection
        detector.run_detection(camera_index)
        
    except KeyboardInterrupt:
        print("\n\n✅ Détection interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

