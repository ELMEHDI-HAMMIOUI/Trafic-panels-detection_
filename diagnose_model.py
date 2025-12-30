"""
Script de diagnostic pour comprendre pourquoi le modèle ne détecte pas
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import cv2
import tensorflow as tf
from src.data_loader import GTSRBDataLoader
from src.model import TrafficSignClassifier
from src.preprocessing import ImagePreprocessor

def diagnose_model(model_path: str):
    """
    Diagnostique les problèmes du modèle
    
    Args:
        model_path: Chemin vers le modèle
    """
    print("="*70)
    print("DIAGNOSTIC DU MODÈLE")
    print("="*70)
    
    # 1. Vérifier que le modèle existe
    if not Path(model_path).exists():
        print(f"❌ ERREUR: Le modèle {model_path} n'existe pas!")
        return
    
    print(f"\n✅ Modèle trouvé: {model_path}")
    
    # 2. Charger le modèle
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Modèle chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return
    
    # 3. Vérifier l'architecture
    print(f"\n📊 Architecture du modèle:")
    print(f"  Nombre de couches: {len(model.layers)}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    
    # 4. Vérifier le nombre de classes
    num_classes_model = model.output_shape[-1]
    print(f"  Nombre de classes dans le modèle: {num_classes_model}")
    
    # 5. Charger les données pour vérifier
    print(f"\n📁 Vérification du dataset:")
    loader = GTSRBDataLoader("data")
    
    try:
        X, y = loader.load_train_data(img_size=(64, 64))
        num_classes_data = len(set(y))
        print(f"  Images chargées: {len(X)}")
        print(f"  Classes dans les données: {num_classes_data}")
        
        if num_classes_model != num_classes_data:
            print(f"\n❌ PROBLÈME DÉTECTÉ!")
            print(f"   Le modèle a {num_classes_model} classes mais les données ont {num_classes_data} classes!")
            print(f"   Solution: Réentraînez le modèle avec le bon nombre de classes")
            return
        else:
            print(f"  ✅ Nombre de classes correspond")
    except Exception as e:
        print(f"  ⚠️  Impossible de charger les données: {e}")
    
    # 6. Tester avec une image du dataset
    print(f"\n🧪 Test avec une image du dataset:")
    try:
        # Charger une image de test
        test_image_path = Path("data/Test/00000.ppm")
        if test_image_path.exists():
            img = cv2.imread(str(test_image_path))
            if img is not None:
                # Redimensionner et normaliser
                img_resized = cv2.resize(img, (64, 64))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_normalized = img_rgb.astype(np.float32) / 255.0
                img_input = np.expand_dims(img_normalized, axis=0)
                
                # Prédiction
                prediction = model.predict(img_input, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class])
                
                print(f"  Image testée: {test_image_path}")
                print(f"  Classe prédite: {predicted_class}")
                print(f"  Confiance: {confidence:.4f}")
                print(f"  Top 3 prédictions:")
                
                top3_indices = np.argsort(prediction[0])[-3:][::-1]
                for i, idx in enumerate(top3_indices, 1):
                    conf = float(prediction[0][idx])
                    class_names = loader.get_class_names()
                    class_name = class_names.get(idx, f"Classe {idx}")
                    print(f"    {i}. Classe {idx} ({class_name}): {conf:.4f}")
                
                if confidence < 0.5:
                    print(f"\n  ⚠️  Confiance faible ({confidence:.4f})")
                    print(f"     Le modèle n'est pas sûr de sa prédiction")
            else:
                print(f"  ❌ Impossible de charger l'image")
        else:
            print(f"  ⚠️  Image de test non trouvée: {test_image_path}")
    except Exception as e:
        print(f"  ❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
    
    # 7. Vérifier les noms de classes
    print(f"\n📝 Vérification des noms de classes:")
    class_names = loader.get_class_names()
    print(f"  Nombre de noms définis: {len(class_names)}")
    print(f"  Classes manquantes: ", end="")
    missing = [i for i in range(num_classes_model) if i not in class_names]
    if missing:
        print(f"{missing}")
        print(f"  ⚠️  Certaines classes n'ont pas de nom défini!")
    else:
        print(f"Aucune")
        print(f"  ✅ Toutes les classes ont un nom")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC TERMINÉ")
    print("="*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnostique les problèmes du modèle")
    parser.add_argument("--model-path", type=str, default="models/traffic_sign_cnn.h5",
                       help="Chemin vers le modèle")
    
    args = parser.parse_args()
    diagnose_model(args.model_path)

