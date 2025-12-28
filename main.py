"""
Script principal pour la détection et classification des panneaux de signalisation
"""

import argparse
import os
import sys
from pathlib import Path

# Ajouter le dossier src au path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_loader import GTSRBDataLoader
from src.preprocessing import ImagePreprocessor
from src.model import TrafficSignClassifier
from src.detector import RealTimeDetector
from src.utils import plot_training_history, evaluate_model, download_gtsrb_dataset


def train_model(data_path: str, model_type: str = "cnn", epochs: int = 50):
    """
    Entraîne un modèle de classification
    
    Args:
        data_path: Chemin vers le dataset GTSRB
        model_type: Type de modèle ("cnn" ou "resnet")
        epochs: Nombre d'époques
    """
    print("="*60)
    print("ENTRAÎNEMENT DU MODÈLE")
    print("="*60)
    
    # Charger les données
    print("\n1. Chargement des données...")
    loader = GTSRBDataLoader(data_path)
    X, y = loader.load_train_data(img_size=(64, 64))
    print(f"   {len(X)} images chargées")
    print(f"   {len(set(y))} classes différentes")
    
    # Prétraitement
    print("\n2. Prétraitement des données...")
    preprocessor = ImagePreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        X, y, test_size=0.2, normalize=True
    )
    print(f"   Train: {len(X_train)} images")
    print(f"   Test: {len(X_test)} images")
    
    # Créer le modèle
    print(f"\n3. Création du modèle ({model_type})...")
    num_classes = len(set(y))
    classifier = TrafficSignClassifier(num_classes=num_classes)
    
    if model_type == "cnn":
        model = classifier.create_cnn_model()
    elif model_type == "resnet":
        model = classifier.create_resnet_model()
    else:
        raise ValueError("model_type doit être 'cnn' ou 'resnet'")
    
    model.summary()
    
    # Entraîner
    print("\n4. Entraînement du modèle...")
    history = classifier.train(
        X_train, y_train,
        X_test, y_test,
        epochs=epochs,
        batch_size=32
    )
    
    # Sauvegarder
    print("\n5. Sauvegarde du modèle...")
    os.makedirs("models", exist_ok=True)
    model_path = f"models/traffic_sign_{model_type}.h5"
    classifier.save_model(model_path)
    
    # Évaluer
    print("\n6. Évaluation du modèle...")
    class_names = list(range(num_classes))
    evaluate_model(model, X_test, y_test, class_names)
    
    # Afficher l'historique
    plot_training_history(history, save_path="models/training_history.png")
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ!")
    print("="*60)


def run_detection(model_path: str, camera_index: int = 0):
    """
    Lance la détection en temps réel
    
    Args:
        model_path: Chemin vers le modèle entraîné
        camera_index: Index de la caméra
    """
    print("="*60)
    print("DÉTECTION EN TEMPS RÉEL")
    print("="*60)
    
    # Charger les noms de classes
    loader = GTSRBDataLoader("data")
    class_names = loader.get_class_names()
    
    # Créer le détecteur
    detector = RealTimeDetector(model_path, class_names)
    
    # Lancer la détection
    try:
        detector.run_detection(camera_index)
    except KeyboardInterrupt:
        print("\nDétection interrompue par l'utilisateur")
    except Exception as e:
        print(f"\nErreur: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Détection et classification des panneaux de signalisation"
    )
    
    parser.add_argument(
        "mode",
        choices=["train", "detect", "download"],
        help="Mode d'exécution: train, detect, ou download"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help="Chemin vers le dataset GTSRB (défaut: data)"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["cnn", "resnet"],
        default="cnn",
        help="Type de modèle (défaut: cnn)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Nombre d'époques pour l'entraînement (défaut: 50)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/traffic_sign_cnn.h5",
        help="Chemin vers le modèle pour la détection"
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Index de la caméra (défaut: 0)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model(args.data_path, args.model_type, args.epochs)
    
    elif args.mode == "detect":
        if not os.path.exists(args.model_path):
            print(f"Erreur: Le modèle {args.model_path} n'existe pas.")
            print("Entraînez d'abord un modèle avec: python main.py train")
            return
        run_detection(args.model_path, args.camera)
    
    elif args.mode == "download":
        download_gtsrb_dataset(args.data_path)


if __name__ == "__main__":
    main()

