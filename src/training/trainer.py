import tensorflow as tf
import numpy as np
from pathlib import Path
import time
import logging

from src.config.config import Config
from src.models.unet import create_unet_model
from src.models.losses import get_loss
from src.data.data_loader import DataLoader
from src.training.callbacks import CustomCallbacks
from src.utils.visualization import Visualizer

class Trainer:
    def __init__(self, model_name="vessel_segmentation"):
        """
        Initialise le trainer
        Args:
            model_name: Nom du modèle pour la sauvegarde
        """
        self.model_name = model_name
        self.model = None
        self.data_loader = DataLoader()
        self.history = None
        
        # Créer les dossiers nécessaires
        Config.create_directories()

    def setup_model(self, loss_type='bce'):
        """
        Configure le modèle avec l'optimizer et la fonction de perte
        Args:
            loss_type: Type de fonction de perte à utiliser
        """
        print("Configuration du modèle...")
        
        # Activer la précision mixte si configuré
        if Config.USE_MIXED_PRECISION:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Créer le modèle U-Net
        self.model = create_unet_model()
        
        # Obtenir la fonction de perte
        loss_function = get_loss(loss_type)
        
        # Définir l'optimizer avec le learning rate du config
        optimizer = tf.keras.optimizers.Adam(learning_rate=Config.INITIAL_LEARNING_RATE)
        
        # Métriques
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
        
        # Compiler le modèle
        self.model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=metrics
        )
        
        # Afficher le résumé du modèle
        self.model.summary()

    def train(self):
        """
        Entraîne le modèle
        Returns:
            Historique d'entraînement
        """
        print("\nChargement des données...")
        train_dataset, val_dataset = self.data_loader.get_train_val_datasets()
        
        print("\nDémarrage de l'entraînement...")
        start_time = time.time()
        
        # Obtenir les callbacks
        callbacks = CustomCallbacks.get_callbacks(self.model_name)
        
        # Entraînement
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=Config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        # Sauvegarder l'historique
        self.history = history
        
        # Calculer et afficher le temps d'entraînement
        training_time = time.time() - start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        
        print(f"\nEntraînement terminé en {hours}h {minutes}min")
        
        # Sauvegarder le modèle final
        self.save_model('final')
        
        return history

    def evaluate(self, test_dataset=None):
        """
        Évalue le modèle
        Args:
            test_dataset: Dataset de test optionnel
        Returns:
            Métriques d'évaluation
        """
        if test_dataset is None:
            test_dataset = self.data_loader.get_test_dataset()
        
        print("\nÉvaluation du modèle...")
        results = self.model.evaluate(test_dataset, verbose=1)
        
        # Créer un dictionnaire des résultats
        metrics_dict = {}
        for metric_name, value in zip(self.model.metrics_names, results):
            metrics_dict[metric_name] = value
            print(f"{metric_name}: {value:.4f}")
        
        return metrics_dict

    def predict(self, dataset):
        """
        Fait des prédictions
        Args:
            dataset: Dataset pour les prédictions
        Returns:
            Prédictions du modèle
        """
        print("\nGénération des prédictions...")
        return self.model.predict(dataset, verbose=1)

    def save_model(self, suffix=''):
        """
        Sauvegarde le modèle
        Args:
            suffix: Suffixe pour le nom du fichier
        """
        save_path = Config.MODELS_DIR / f"{self.model_name}_{suffix}.keras"
        self.model.save(save_path)
        print(f"\nModèle sauvegardé: {save_path}")

    def load_model(self, model_path):
        """
        Charge un modèle sauvegardé
        Args:
            model_path: Chemin vers le modèle
        """
        print(f"\nChargement du modèle: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("Modèle chargé avec succès!")

    def save_training_visualizations(self):
        """
        Sauvegarde les visualisations d'entraînement
        """
        if self.history is None:
            print("Pas d'historique d'entraînement disponible")
            return
            
        visualizer = Visualizer(Config.OUTPUT_DIR / 'visualizations')
        
        # Plot de l'historique d'entraînement
        visualizer.plot_training_history(
            self.history,
            save_path=Config.OUTPUT_DIR / 'visualizations' / 'training_history.png'
        )
        
        print("\nVisualisations sauvegardées dans:", Config.OUTPUT_DIR / 'visualizations')