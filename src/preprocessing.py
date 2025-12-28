"""
Module pour le prétraitement des images de panneaux de signalisation
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class ImagePreprocessor:
    """Classe pour prétraiter les images avant l'entraînement"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
    
    def normalize_images(self, images: np.ndarray) -> np.ndarray:
        """
        Normalise les images entre 0 et 1
        
        Args:
            images: Tableau d'images
            
        Returns:
            Images normalisées
        """
        return images.astype(np.float32) / 255.0
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applique des transformations d'augmentation de données
        
        Args:
            image: Image à augmenter
            
        Returns:
            Image augmentée
        """
        # Rotation aléatoire
        angle = np.random.randint(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Translation aléatoire
        tx = np.random.randint(-5, 5)
        ty = np.random.randint(-5, 5)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))
        
        # Luminosité aléatoire
        brightness = np.random.uniform(0.7, 1.3)
        image = cv2.convertScaleAbs(image, alpha=1, beta=brightness*50-50)
        
        return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Améliore le contraste de l'image
        
        Args:
            image: Image à traiter
            
        Returns:
            Image avec contraste amélioré
        """
        # Conversion en LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Application de CLAHE sur le canal L
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Fusion des canaux
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def detect_sign(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Détecte un panneau dans l'image en utilisant la détection de contours
        
        Args:
            image: Image à analyser
            
        Returns:
            Image du panneau détecté ou None
        """
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Détection de contours
        edges = cv2.Canny(gray, 50, 150)
        
        # Recherche de contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Trouver le plus grand contour (probablement le panneau)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Vérifier si le contour est suffisamment grand
            if cv2.contourArea(largest_contour) > 100:
                # Obtenir le rectangle englobant
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Extraire la région du panneau
                sign = image[y:y+h, x:x+w]
                return sign
        
        return None
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, 
                    normalize: bool = True) -> Tuple:
        """
        Prépare les données pour l'entraînement
        
        Args:
            X: Images
            y: Labels
            test_size: Proportion des données de test
            normalize: Si True, normalise les images
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        # Encoder les labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Diviser en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Normaliser si demandé
        if normalize:
            X_train = self.normalize_images(X_train)
            X_test = self.normalize_images(X_test)
        
        return X_train, X_test, y_train, y_test

