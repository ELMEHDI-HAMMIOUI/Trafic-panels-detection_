"""
Script pour améliorer radicalement la détection en ajoutant des filtres stricts
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import cv2
import numpy as np
from src.detector import RealTimeDetector
from src.data_loader import GTSRBDataLoader

def test_improved_detection(image_path: str, confidence: float = 0.75):
    """
    Test avec des filtres très stricts pour éliminer les faux positifs
    """
    print("="*70)
    print("DÉTECTION AMÉLIORÉE AVEC FILTRES STRICTS")
    print("="*70)
    
    # Charger le modèle
    loader = GTSRBDataLoader("data")
    class_names = loader.get_class_names()
    
    # Trouver le modèle
    model_paths = [
        "models/traffic_sign_cnn.h5",
        "models/traffic_sign_resnet.h5"
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if not model_path:
        print("❌ Aucun modèle trouvé. Entraînez d'abord le modèle.")
        return
    
    print(f"✅ Modèle trouvé: {model_path}")
    
    # Créer le détecteur
    detector = RealTimeDetector(model_path, class_names)
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Impossible de charger l'image: {image_path}")
        return
    
    print(f"✅ Image chargée: {image.shape[1]}x{image.shape[0]}")
    print(f"✅ Seuil de confiance: {confidence}")
    
    # Détecter avec un seuil très élevé
    detections = detector.detect_signs_in_frame(image, confidence_threshold=confidence)
    
    print(f"\n📊 Résultats:")
    print(f"   Nombre de détections: {len(detections)}")
    
    if len(detections) == 0:
        print("\n⚠️  Aucune détection trouvée avec ce seuil strict.")
        print("   Cela signifie que le modèle n'est pas assez confiant.")
        print("   Solutions:")
        print("   1. Réentraîner le modèle avec plus d'époques")
        print("   2. Utiliser un dataset de panneaux français")
        print("   3. Le modèle actuel n'est pas adapté aux panneaux français")
    else:
        print("\n✅ Détections trouvées:")
        for i, (class_id, conf, (x, y, w, h)) in enumerate(detections, 1):
            class_name = class_names.get(class_id, f"Classe {class_id}")
            print(f"   {i}. {class_name}: {conf:.2f} à ({x}, {y}, {w}x{h})")
    
    # Dessiner les détections
    result = detector.draw_detections(image.copy(), detections)
    
    # Sauvegarder
    output_path = "detection_improved.jpg"
    cv2.imwrite(output_path, result)
    print(f"\n✅ Résultat sauvegardé: {output_path}")
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test avec détection améliorée")
    parser.add_argument("--image", type=str, required=True, help="Chemin vers l'image")
    parser.add_argument("--confidence", type=float, default=0.75, help="Seuil de confiance (défaut: 0.75)")
    
    args = parser.parse_args()
    test_improved_detection(args.image, args.confidence)

