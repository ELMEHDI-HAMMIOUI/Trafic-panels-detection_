"""
Module pour créer et entraîner les modèles de classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional
import os


class TrafficSignClassifier:
    """Classe pour créer et gérer le modèle de classification"""
    
    def __init__(self, num_classes: int, input_shape: Tuple[int, int, int] = (64, 64, 3)):
        """
        Initialise le classifieur
        
        Args:
            num_classes: Nombre de classes à classifier
            input_shape: Forme des images d'entrée (height, width, channels)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
    
    def create_cnn_model(self) -> keras.Model:
        """
        Crée un modèle CNN simple pour la classification
        
        Returns:
            Modèle Keras compilé
        """
        model = models.Sequential([
            # Première couche de convolution
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Deuxième couche de convolution
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Troisième couche de convolution
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Aplatissement
            layers.Flatten(),
            
            # Couches fully connected
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            
            # Couche de sortie
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def create_resnet_model(self) -> keras.Model:
        """
        Crée un modèle basé sur ResNet pour une meilleure performance
        
        Returns:
            Modèle Keras compilé
        """
        # Utiliser ResNet50 pré-entraîné
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Geler les couches de base
        base_model.trainable = False
        
        # Ajouter des couches personnalisées
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32,
              callbacks: Optional[list] = None) -> keras.callbacks.History:
        """
        Entraîne le modèle
        
        Args:
            X_train: Images d'entraînement
            y_train: Labels d'entraînement
            X_val: Images de validation
            y_val: Labels de validation
            epochs: Nombre d'époques
            batch_size: Taille du batch
            callbacks: Liste de callbacks Keras
            
        Returns:
            Historique d'entraînement
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été créé. Appelez create_cnn_model() ou create_resnet_model() d'abord.")
        
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001
                )
            ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Prédit la classe d'une image
        
        Args:
            image: Image à classifier
            
        Returns:
            Tuple (classe_prédite, probabilité)
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été chargé ou créé.")
        
        # Préparer l'image
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Prédiction
        predictions = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return predicted_class, confidence
    
    def save_model(self, filepath: str):
        """
        Sauvegarde le modèle
        
        Args:
            filepath: Chemin où sauvegarder le modèle
        """
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Modèle sauvegardé dans {filepath}")
    
    def load_model(self, filepath: str):
        """
        Charge un modèle sauvegardé
        
        Args:
            filepath: Chemin vers le modèle sauvegardé
        """
        self.model = keras.models.load_model(filepath)
        print(f"Modèle chargé depuis {filepath}")

