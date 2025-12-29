"""
Script pour tester la détection sur une image depuis une URL
"""

import sys
import os
import urllib.request
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
    print("DÉTECTION SUR IMAGE (URL)")
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
    
    # Demander l'URL
    print("\n" + "-"*60)
    url = input("Entrez l'URL de l'image: ").strip()
    
    # Supprimer les guillemets si présents
    url = url.strip('"').strip("'")
    
    if not url:
        print("❌ Aucune URL fournie")
        return
    
    # Vérifier que c'est une URL valide
    if not url.startswith(('http://', 'https://')):
        print("❌ Erreur: L'URL doit commencer par http:// ou https://")
        return
    
    temp_path = project_root / "temp_url_image.jpg"
    
    try:
        # Télécharger l'image
        print(f"\nTéléchargement depuis: {url}")
        urllib.request.urlretrieve(url, str(temp_path))
        print("✅ Image téléchargée")
        
        # Charger les noms de classes
        data_path = project_root / "data"
        loader = GTSRBDataLoader(str(data_path))
        class_names = loader.get_class_names()
        
        # Créer le détecteur
        print("\nChargement du détecteur...")
        detector = RealTimeDetector(model_path, class_names)
        print("✅ Détecteur chargé")
        
        # Détecter
        print(f"\nAnalyse de l'image...")
        result = detector.detect_from_image(str(temp_path))
        
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
        
    except urllib.error.URLError as e:
        print(f"\n❌ Erreur de téléchargement: {e}")
        print("Vérifiez que l'URL est accessible et valide")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Nettoyer le fichier temporaire
        if temp_path.exists():
            os.remove(str(temp_path))
            print("\n🧹 Fichier temporaire supprimé")


if __name__ == "__main__":
    main()

