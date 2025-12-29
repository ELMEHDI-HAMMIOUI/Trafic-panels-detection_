"""
Module pour la détection en temps réel des panneaux de signalisation
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import tensorflow as tf


class RealTimeDetector:
    """Classe pour la détection en temps réel avec webcam"""
    
    def __init__(self, model_path: str, class_names: dict, input_size: Tuple[int, int] = (64, 64)):
        """
        Initialise le détecteur
        
        Args:
            model_path: Chemin vers le modèle entraîné (peut être relatif ou absolu)
            class_names: Dictionnaire des noms de classes
            input_size: Taille d'entrée du modèle
        """
        # Convertir en Path pour gérer les chemins relatifs/absolus
        from pathlib import Path
        model_path_obj = Path(model_path)
        
        # Si le chemin est relatif et n'existe pas, essayer depuis le dossier models/
        if not model_path_obj.is_absolute() and not model_path_obj.exists():
            # Essayer depuis le dossier du projet
            project_root = Path(__file__).parent.parent
            alt_path = project_root / model_path
            if alt_path.exists():
                model_path = str(alt_path)
            else:
                # Essayer directement dans models/
                alt_path = project_root / "models" / model_path_obj.name
                if alt_path.exists():
                    model_path = str(alt_path)
        
        # Vérifier que le fichier existe
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Modèle non trouvé: {model_path}\n"
                f"Vérifiez que le chemin est correct ou entraînez un modèle avec: python main.py train"
            )
        
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
        self.input_size = input_size
        self.cap = None
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Prétraite une frame pour la classification
        
        Args:
            frame: Frame de la webcam
            
        Returns:
            Frame prétraitée
        """
        # Redimensionner
        resized = cv2.resize(frame, self.input_size)
        
        # Normaliser
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def detect_signs_in_frame(self, frame: np.ndarray) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        """
        Détecte les panneaux dans une frame
        
        Args:
            frame: Frame de la webcam
            
        Returns:
            Liste de (classe, confiance, bbox)
        """
        # Conversion en RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Détection de contours pour trouver les panneaux
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilatation pour connecter les contours
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Trouver les contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filtrer les petits contours
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extraire la région du panneau
                roi = rgb_frame[y:y+h, x:x+w]
                
                if roi.size > 0:
                    # Prétraiter
                    processed = self.preprocess_frame(roi)
                    
                    # Prédiction
                    prediction = self.model.predict(
                        np.expand_dims(processed, axis=0),
                        verbose=0
                    )
                    
                    class_id = np.argmax(prediction[0])
                    confidence = float(prediction[0][class_id])
                    
                    if confidence > 0.5:  # Seuil de confiance
                        detections.append((class_id, confidence, (x, y, w, h)))
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Tuple]) -> np.ndarray:
        """
        Dessine les détections sur la frame
        
        Args:
            frame: Frame originale
            detections: Liste des détections
            
        Returns:
            Frame avec les détections dessinées
        """
        for class_id, confidence, (x, y, w, h) in detections:
            # Dessiner le rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Obtenir le nom de la classe
            class_name = self.class_names.get(class_id, f"Classe {class_id}")
            
            # Dessiner le label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Rectangle pour le texte
            cv2.rectangle(
                frame,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                (0, 255, 0),
                -1
            )
            
            # Texte
            cv2.putText(
                frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return frame
    
    def run_detection(self, camera_index: int = 0):
        """
        Lance la détection en temps réel
        
        Args:
            camera_index: Index de la caméra (0 pour webcam par défaut)
        """
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la caméra {camera_index}")
        
        print("Détection en cours... Appuyez sur 'q' pour quitter")
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Détecter les panneaux
            detections = self.detect_signs_in_frame(frame)
            
            # Dessiner les détections
            frame = self.draw_detections(frame, detections)
            
            # Afficher la frame
            cv2.imshow('Détection de Panneaux de Signalisation', frame)
            
            # Quitter avec 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def detect_from_image(self, image_path: str) -> np.ndarray:
        """
        Détecte les panneaux dans une image
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Image avec les détections dessinées
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        detections = self.detect_signs_in_frame(image)
        result = self.draw_detections(image.copy(), detections)
        
        return result

