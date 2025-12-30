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
    
    def detect_signs_in_frame(self, frame: np.ndarray, confidence_threshold: float = 0.7) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        """
        Détecte les panneaux dans une frame avec plusieurs méthodes
        
        Args:
            frame: Frame de la webcam
            confidence_threshold: Seuil de confiance minimum (défaut: 0.3, plus bas = plus de détections)
            
        Returns:
            Liste de (classe, confiance, bbox)
        """
        detections = []
        h, w = frame.shape[:2]
        
        # Méthode 1: Détection par sliding window (plus robuste)
        detections.extend(self._sliding_window_detection(frame, confidence_threshold))
        
        # Méthode 2: Détection par contours (si sliding window ne trouve rien)
        if len(detections) == 0:
            detections.extend(self._contour_detection(frame, confidence_threshold))
        
        # Méthode 3: Détection sur toute l'image (si rien n'est trouvé)
        if len(detections) == 0:
            detections.extend(self._full_image_detection(frame, confidence_threshold))
        
        # Supprimer les doublons et garder les meilleures détections
        detections = self._non_max_suppression(detections)
        
        return detections
    
    def _sliding_window_detection(self, frame: np.ndarray, confidence_threshold: float) -> List[Tuple]:
        """Détection par fenêtre glissante (désactivée par défaut - trop de fausses détections)"""
        # Cette méthode est désactivée car elle génère trop de fausses détections
        # On utilise uniquement la détection par contours qui est plus précise
        return []
    
    def _has_traffic_sign_colors(self, roi: np.ndarray) -> bool:
        """
        Vérifie si la région contient des couleurs typiques de panneaux de signalisation
        Les panneaux ont généralement du rouge, bleu, blanc, ou jaune vif
        Version améliorée avec seuils plus stricts
        """
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return False
        
        # Convertir en HSV pour une meilleure détection de couleur
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        
        # Masques pour les couleurs de panneaux (seuils plus stricts)
        # Rouge (pour STOP, interdiction) - plus strict
        red_lower1 = np.array([0, 100, 100])  # Saturation et valeur plus élevées
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
        
        # Bleu (pour obligation) - plus strict
        blue_lower = np.array([100, 100, 100])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Jaune (pour avertissement) - plus strict
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Blanc (pour limitation de vitesse) - plus strict
        white_lower = np.array([0, 0, 220])  # Plus lumineux
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Compter les pixels de chaque couleur
        red_pixels = np.sum(red_mask > 0)
        blue_pixels = np.sum(blue_mask > 0)
        yellow_pixels = np.sum(yellow_mask > 0)
        white_pixels = np.sum(white_mask > 0)
        
        total_pixels = roi.shape[0] * roi.shape[1]
        
        # Au moins 10% de l'image doit avoir une couleur de panneau (augmenté de 5%)
        min_color_ratio = 0.10
        has_red = red_pixels / total_pixels > min_color_ratio
        has_blue = blue_pixels / total_pixels > min_color_ratio
        has_yellow = yellow_pixels / total_pixels > min_color_ratio
        has_white = white_pixels / total_pixels > min_color_ratio
        
        # Vérifier aussi qu'il n'y a pas trop de vert (feuilles)
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_pixels = np.sum(green_mask > 0)
        green_ratio = green_pixels / total_pixels
        
        # Si plus de 30% de vert, c'est probablement du feuillage
        if green_ratio > 0.30:
            return False
        
        return has_red or has_blue or has_yellow or has_white
    
    def _contour_detection(self, frame: np.ndarray, confidence_threshold: float) -> List[Tuple]:
        """Détection par contours améliorée avec filtres de couleur"""
        detections = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Détection de contours avec seuils optimisés
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Utiliser un seul seuil Canny bien calibré
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilatation modérée pour connecter les contours
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Trouver les contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Seuil d'aire BEAUCOUP plus élevé pour éviter les petits détails (feuilles, etc.)
            min_area = max(2000, (h * w) * 0.005)  # Au moins 0.5% de l'image (augmenté)
            max_area = (h * w) * 0.25  # Maximum 25% de l'image (réduit)
            
            if min_area < area < max_area:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Vérifier le ratio d'aspect (les panneaux sont souvent carrés ou rectangulaires)
                aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
                
                # Ratio plus strict pour les panneaux
                if 0.5 < aspect_ratio < 2.0:  # Plus flexible pour panneaux rectangulaires
                    # Vérifier la taille minimale (BEAUCOUP augmentée)
                    if w_rect > 80 and h_rect > 80:  # Beaucoup plus grand pour éviter les petits objets
                        # Extraire la région du panneau avec un peu de padding
                        padding = 5
                        x_start = max(0, x - padding)
                        y_start = max(0, y - padding)
                        x_end = min(w, x + w_rect + padding)
                        y_end = min(h, y + h_rect + padding)
                        
                        roi = rgb_frame[y_start:y_end, x_start:x_end]
                        
                        if roi.size > 0 and roi.shape[0] > 30 and roi.shape[1] > 30:
                            # FILTRE IMPORTANT : Vérifier les couleurs de panneau
                            if not self._has_traffic_sign_colors(roi):
                                continue  # Ignorer si pas de couleurs de panneau (feuilles, etc.)
                            
                            # Prétraiter
                            processed = self.preprocess_frame(roi)
                            
                            # Prédiction
                            prediction = self.model.predict(
                                np.expand_dims(processed, axis=0),
                                verbose=0
                            )
                            
                            class_id = np.argmax(prediction[0])
                            confidence = float(prediction[0][class_id])
                            
                            # Seuil de confiance plus élevé + vérifier que c'est vraiment un panneau
                            if confidence > confidence_threshold:
                                # Vérifier que la deuxième meilleure prédiction est bien inférieure
                                sorted_pred = np.sort(prediction[0])[::-1]
                                if len(sorted_pred) > 1:
                                    # La différence entre la meilleure et la deuxième doit être significative
                                    confidence_diff = sorted_pred[0] - sorted_pred[1]
                                    # Augmenter BEAUCOUP le seuil de différence pour éviter les mauvaises classifications
                                    if confidence_diff > 0.35:  # Au moins 35% de différence (augmenté de 0.25)
                                        detections.append((class_id, confidence, (x, y, w_rect, h_rect)))
                                else:
                                    detections.append((class_id, confidence, (x, y, w_rect, h_rect)))
        
        return detections
    
    def _full_image_detection(self, frame: np.ndarray, confidence_threshold: float) -> List[Tuple]:
        """Détection sur toute l'image (dernier recours)"""
        detections = []
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Redimensionner l'image entière à la taille du modèle
        resized = cv2.resize(rgb_frame, self.input_size)
        processed = self.preprocess_frame(resized)
        
        # Prédiction
        prediction = self.model.predict(
            np.expand_dims(processed, axis=0),
            verbose=0
        )
        
        class_id = np.argmax(prediction[0])
        confidence = float(prediction[0][class_id])
        
        if confidence > confidence_threshold:
            # Utiliser toute l'image comme bounding box
            detections.append((class_id, confidence, (0, 0, w, h)))
        
        return detections
    
    def _region_based_detection(self, frame: np.ndarray, confidence_threshold: float) -> List[Tuple]:
        """Détection par zones d'intérêt (amélioration)"""
        detections = []
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Diviser l'image en grille et tester chaque zone
        grid_size = 3  # 3x3 = 9 zones
        cell_w = w // grid_size
        cell_h = h // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                x_start = j * cell_w
                y_start = i * cell_h
                x_end = min(x_start + cell_w, w)
                y_end = min(y_start + cell_h, h)
                
                # Extraire la zone
                roi = rgb_frame[y_start:y_end, x_start:x_end]
                
                if roi.size > 0 and roi.shape[0] > 20 and roi.shape[1] > 20:
                    # Prétraiter
                    processed = self.preprocess_frame(roi)
                    
                    # Prédiction
                    prediction = self.model.predict(
                        np.expand_dims(processed, axis=0),
                        verbose=0
                    )
                    
                    class_id = np.argmax(prediction[0])
                    confidence = float(prediction[0][class_id])
                    
                    if confidence > confidence_threshold:
                        detections.append((class_id, confidence, (x_start, y_start, x_end-x_start, y_end-y_start)))
        
        return detections
    
    def _non_max_suppression(self, detections: List[Tuple], iou_threshold: float = 0.3) -> List[Tuple]:
        """Supprime les détections qui se chevauchent trop (amélioré)"""
        if len(detections) == 0:
            return detections
        
        # Trier par confiance (décroissant)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        
        filtered = []
        for det in detections:
            class_id, confidence, (x, y, w, h) = det
            
            # Vérifier le chevauchement avec les détections déjà gardées
            overlap = False
            for kept_class, kept_conf, (kx, ky, kw, kh) in filtered:
                # Calculer IoU (Intersection over Union)
                intersection_x = max(0, min(x + w, kx + kw) - max(x, kx))
                intersection_y = max(0, min(y + h, ky + kh) - max(y, ky))
                intersection = intersection_x * intersection_y
                
                area1 = w * h
                area2 = kw * kh
                union = area1 + area2 - intersection
                
                if union > 0:
                    iou = intersection / union
                    # Si chevauchement significatif, garder seulement la meilleure
                    if iou > iou_threshold:
                        overlap = True
                        break
            
            if not overlap:
                filtered.append(det)
        
        # Limiter le nombre de détections (garder seulement les meilleures)
        max_detections = 5
        return filtered[:max_detections]
    
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
            
            # Convertir en ASCII pour éviter les problèmes d'encodage OpenCV
            class_name_ascii = class_name.encode('ascii', 'ignore').decode('ascii')
            
            # Dessiner le label
            label = f"{class_name_ascii}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Rectangle pour le texte
            cv2.rectangle(
                frame,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                (0, 255, 0),
                -1
            )
            
            # Texte (utiliser putText avec encodage correct)
            # Convertir le label en ASCII pour éviter les problèmes d'encodage
            label_ascii = label.encode('ascii', 'ignore').decode('ascii')
            cv2.putText(
                frame,
                label_ascii,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return frame
    
    def run_detection(self, camera_index: int = 0, confidence_threshold: float = 0.7):
        """
        Lance la détection en temps réel avec la webcam
        
        Args:
            camera_index: Index de la caméra (0 pour webcam par défaut)
            confidence_threshold: Seuil de confiance pour la détection (défaut: 0.7)
        """
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la caméra {camera_index}")
        
        # Configurer la résolution de la caméra (optionnel, pour améliorer les performances)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Détection en cours... Appuyez sur 'q' pour quitter")
        print(f"Seuil de confiance: {confidence_threshold}")
        print("Contrôles: '+' pour augmenter, '-' pour réduire le seuil")
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("⚠️  Impossible de lire la frame de la caméra")
                break
            
            # Détecter les panneaux avec le seuil de confiance
            detections = self.detect_signs_in_frame(frame, confidence_threshold=confidence_threshold)
            
            # Dessiner les détections
            frame = self.draw_detections(frame, detections)
            
            # Afficher le nombre de détections sur la frame
            info_text = f"Detections: {len(detections)} | Seuil: {confidence_threshold:.2f}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Appuyez 'q' pour quitter, '+'/- pour ajuster", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Afficher la frame
            cv2.imshow('Détection de Panneaux de Signalisation', frame)
            
            # Quitter avec 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                # Augmenter le seuil avec '+'
                confidence_threshold = min(0.95, confidence_threshold + 0.05)
                print(f"Seuil augmenté à: {confidence_threshold:.2f}")
            elif key == ord('-'):
                # Réduire le seuil avec '-'
                confidence_threshold = max(0.1, confidence_threshold - 0.05)
                print(f"Seuil réduit à: {confidence_threshold:.2f}")
            
            frame_count += 1
        
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Détection terminée ({frame_count} frames traitées)")
    
    def detect_from_image(self, image_path: str, confidence_threshold: float = 0.7) -> np.ndarray:
        """
        Détecte les panneaux dans une image
        
        Args:
            image_path: Chemin vers l'image
            confidence_threshold: Seuil de confiance minimum (défaut: 0.3)
            
        Returns:
            Image avec les détections dessinées
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        print(f"Analyse de l'image: {image_path}")
        print(f"Taille de l'image: {image.shape[1]}x{image.shape[0]}")
        print(f"Seuil de confiance utilisé: {confidence_threshold}")
        
        # Essayer plusieurs méthodes de détection
        detections = []
        
        # Méthode 1: Détection par contours
        detections.extend(self._contour_detection(image, confidence_threshold))
        
        # Méthode 2: Si rien trouvé, essayer sur toute l'image
        if len(detections) == 0:
            print("  Aucune détection par contours, essai sur toute l'image...")
            detections.extend(self._full_image_detection(image, confidence_threshold))
        
        # Méthode 3: Si toujours rien, essayer avec sliding window sur zones d'intérêt
        if len(detections) == 0:
            print("  Essai avec détection par zones...")
            detections.extend(self._region_based_detection(image, confidence_threshold))
        
        # Supprimer les doublons
        detections = self._non_max_suppression(detections)
        
        print(f"Nombre de détections trouvées: {len(detections)}")
        if len(detections) > 0:
            for i, (class_id, confidence, (x, y, w, h)) in enumerate(detections, 1):
                class_name = self.class_names.get(class_id, f"Classe {class_id}")
                # Utiliser ASCII pour éviter les problèmes d'encodage
                class_name_ascii = class_name.encode('ascii', 'ignore').decode('ascii')
                print(f"  Détection {i}: {class_name_ascii} (confiance: {confidence:.2f}) à ({x}, {y}, {w}x{h})")
        else:
            print("⚠️  Aucune détection trouvée.")
            print("   Suggestions:")
            print("   - Vérifiez que l'image contient bien un panneau visible")
            print("   - Essayez avec une image du dataset: data/Test/00000.ppm")
            print("   - Réduisez le seuil: python test_image_upload.py (modifiez confidence_threshold=0.3)")
        
        result = self.draw_detections(image.copy(), detections)
        
        return result

