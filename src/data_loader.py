"""
Module pour charger et gérer le dataset GTSRB (German Traffic Sign Recognition Benchmark)
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict


class GTSRBDataLoader:
    """Classe pour charger et préparer le dataset GTSRB"""
    
    def __init__(self, data_path: str):
        """
        Initialise le chargeur de données
        
        Args:
            data_path: Chemin vers le dossier contenant le dataset GTSRB
        """
        self.data_path = Path(data_path)
        self.train_path = self.data_path / "Train"
        self.test_path = self.data_path / "Test"
        
    def load_train_data(self, img_size: Tuple[int, int] = (64, 64)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Charge les données d'entraînement
        
        Args:
            img_size: Taille des images à redimensionner (width, height)
            
        Returns:
            Tuple de (images, labels)
        """
        images = []
        labels = []
        
        if not self.train_path.exists():
            raise FileNotFoundError(f"Le dossier {self.train_path} n'existe pas")
        
        # Parcourir tous les dossiers de classes
        for class_folder in sorted(self.train_path.iterdir()):
            if class_folder.is_dir():
                class_id = int(class_folder.name)
                
                # Lire toutes les images de cette classe
                for img_file in class_folder.glob("*.ppm"):
                    img = cv2.imread(str(img_file))
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        labels.append(class_id)
        
        return np.array(images), np.array(labels)
    
    def load_test_data(self, csv_file: str = "Test.csv", img_size: Tuple[int, int] = (64, 64)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Charge les données de test
        
        Args:
            csv_file: Nom du fichier CSV contenant les métadonnées de test
            img_size: Taille des images à redimensionner
            
        Returns:
            Tuple de (images, labels)
        """
        images = []
        labels = []
        
        csv_path = self.test_path / csv_file
        if not csv_path.exists():
            raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas")
        
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            img_path = self.test_path / row['Path']
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.resize(img, img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(row['ClassId'])
        
        return np.array(images), np.array(labels)
    
    def get_class_names(self) -> Dict[int, str]:
        """
        Retourne le dictionnaire complet des noms de classes GTSRB (43 classes)
        
        Returns:
            Dictionnaire des noms de classes {id: nom}
        """
        # Noms complets des 43 classes GTSRB
        class_names = {
            0: "Limite de vitesse 20",
            1: "Limite de vitesse 30",
            2: "Limite de vitesse 50",
            3: "Limite de vitesse 60",
            4: "Limite de vitesse 70",
            5: "Limite de vitesse 80",
            6: "Fin limite de vitesse 80",
            7: "Limite de vitesse 100",
            8: "Limite de vitesse 120",
            9: "Depassement interdit",
            10: "Depassement interdit pour camions",
            11: "Priorite a droite",
            12: "Route prioritaire",
            13: "Cedez le passage",
            14: "STOP",
            15: "Sens interdit",
            16: "Interdiction aux camions",
            17: "Interdiction d'entree",
            18: "Attention generale",
            19: "Virage dangereux a gauche",
            20: "Virage dangereux a droite",
            21: "Double virage",
            22: "Route bosselee",
            23: "Route glissante",
            24: "Retrécissement de route a droite",
            25: "Travaux",
            26: "Feux de signalisation",
            27: "Passage pietons",
            28: "Enfants traversant",
            29: "Velos traversant",
            30: "Attention glace/neige",
            31: "Animaux sauvages traversant",
            32: "Fin de toutes les limitations",
            33: "Tourner a droite obligatoire",
            34: "Tourner a gauche obligatoire",
            35: "Tout droit obligatoire",
            36: "Tout droit ou a droite obligatoire",
            37: "Tout droit ou a gauche obligatoire",
            38: "Tenir la droite",
            39: "Tenir la gauche",
            40: "Rond-point obligatoire",
            41: "Fin de depassement interdit",
            42: "Fin de depassement interdit pour camions",
        }
        return class_names

