"""
Script de test pour vérifier que tout est correctement configuré
"""

import sys
from pathlib import Path

def test_imports():
    """Teste que tous les modules peuvent être importés"""
    print("=" * 60)
    print("TEST DES IMPORTS")
    print("=" * 60)
    
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        from src.data_loader import GTSRBDataLoader
        print("[OK] data_loader.py")
    except Exception as e:
        print(f"[ERREUR] data_loader.py: {e}")
        return False
    
    try:
        from src.preprocessing import ImagePreprocessor
        print("[OK] preprocessing.py")
    except Exception as e:
        print(f"[ERREUR] preprocessing.py: {e}")
        return False
    
    try:
        from src.model import TrafficSignClassifier
        print("[OK] model.py")
    except Exception as e:
        print(f"[ERREUR] model.py: {e}")
        return False
    
    try:
        from src.detector import RealTimeDetector
        print("[OK] detector.py")
    except Exception as e:
        print(f"[ERREUR] detector.py: {e}")
        return False
    
    try:
        from src.utils import plot_training_history, evaluate_model
        print("[OK] utils.py")
    except Exception as e:
        print(f"[ERREUR] utils.py: {e}")
        return False
    
    return True

def test_structure():
    """Teste que la structure du projet est correcte"""
    print("\n" + "=" * 60)
    print("TEST DE LA STRUCTURE")
    print("=" * 60)
    
    base = Path(__file__).parent
    required = {
        "src/": ["__init__.py", "data_loader.py", "preprocessing.py", 
                 "model.py", "detector.py", "utils.py"],
        "notebooks/": ["01_data_exploration.ipynb", "02_preprocessing.ipynb",
                      "03_model_training.ipynb", "04_real_time_detection.ipynb"],
        "": ["main.py", "requirements.txt", "README.md"]
    }
    
    all_ok = True
    for folder, files in required.items():
        folder_path = base / folder if folder else base
        for file in files:
            file_path = folder_path / file
            if file_path.exists():
                print(f"[OK] {folder}{file}")
            else:
                print(f"[ERREUR] {folder}{file} - MANQUANT")
                all_ok = False
    
    return all_ok

def test_dataset():
    """Teste que le dataset est présent"""
    print("\n" + "=" * 60)
    print("TEST DU DATASET")
    print("=" * 60)
    
    base = Path(__file__).parent
    train_path = base / "data" / "Train"
    test_path = base / "data" / "Test"
    
    if train_path.exists():
        classes = [d for d in train_path.iterdir() if d.is_dir()]
        print(f"[OK] Train/ trouve avec {len(classes)} classes")
    else:
        print("[ERREUR] Train/ non trouve")
        return False
    
    if test_path.exists():
        test_files = list(test_path.glob("*.ppm"))
        csv_file = test_path / "Test.csv"
        print(f"[OK] Test/ trouve avec {len(test_files)} images")
        if csv_file.exists():
            print("[OK] Test.csv trouve")
        else:
            print("[ATTENTION] Test.csv non trouve (optionnel)")
    else:
        print("[ERREUR] Test/ non trouve")
        return False
    
    return True

def test_dependencies():
    """Teste que les dépendances principales sont installées"""
    print("\n" + "=" * 60)
    print("TEST DES DÉPENDANCES")
    print("=" * 60)
    
    dependencies = {
        "numpy": "np",
        "pandas": "pd",
        "matplotlib": "plt",
        "cv2": "cv2",
        "tensorflow": "tf",
        "sklearn": "sklearn",
        "seaborn": "sns",
        "PIL": "PIL"
    }
    
    all_ok = True
    for module, alias in dependencies.items():
        try:
            if module == "cv2":
                import cv2
                print(f"[OK] {module} (version: {cv2.__version__})")
            elif module == "tensorflow":
                import tensorflow as tf
                print(f"[OK] {module} (version: {tf.__version__})")
            elif module == "PIL":
                from PIL import Image
                print(f"[OK] {module} (Pillow)")
            else:
                __import__(module)
                print(f"[OK] {module}")
        except ImportError:
            print(f"[ERREUR] {module} - NON INSTALLE")
            all_ok = False
    
    return all_ok

def main():
    """Exécute tous les tests"""
    print("\n" + "VERIFICATION DU PROJET" + "\n")
    
    results = {
        "Structure": test_structure(),
        "Imports": test_imports(),
        "Dataset": test_dataset(),
        "Dépendances": test_dependencies()
    }
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "[OK]" if result else "[ERREUR]"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("SUCCES ! TOUS LES TESTS SONT PASSES !")
        print("Le projet est pret a etre utilise.")
    else:
        print("ATTENTION: CERTAINS TESTS ONT ECHOUE")
        print("Consultez les messages ci-dessus pour plus de details.")
        if not results["Dépendances"]:
            print("\nSolution: pip install -r requirements.txt")
    print("=" * 60 + "\n")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

