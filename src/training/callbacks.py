import tensorflow as tf
import numpy as np
from pathlib import Path
from src.config.config import Config

class CustomCallbacks:
    """Classe pour gérer tous les callbacks d'entraînement"""
    
    @staticmethod
    def get_callbacks(model_name):
        """
        Crée et retourne tous les callbacks nécessaires
        Args:
            model_name: Nom du modèle pour la sauvegarde
        Returns:
            Liste des callbacks
        """
        callbacks = []
        
        # ModelCheckpoint - Sauvegarde le meilleur modèle
        checkpoint_path = Config.MODELS_DIR / f"{model_name}_best.keras"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=Config.MODEL_CHECKPOINT['monitor'],
            mode=Config.MODEL_CHECKPOINT['mode'],
            save_best_only=Config.MODEL_CHECKPOINT['save_best_only'],
            save_weights_only=Config.MODEL_CHECKPOINT['save_weights_only'],
            verbose=Config.MODEL_CHECKPOINT['verbose']
        )
        callbacks.append(checkpoint)
        
        # EarlyStopping - Arrête l'entraînement si pas d'amélioration
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=Config.EARLY_STOPPING['monitor'],
            mode=Config.EARLY_STOPPING['mode'],
            patience=Config.EARLY_STOPPING['patience'],
            restore_best_weights=Config.EARLY_STOPPING['restore_best_weights'],
            min_delta=Config.EARLY_STOPPING['min_delta'],
            verbose=Config.EARLY_STOPPING['verbose']
        )
        callbacks.append(early_stopping)
        
        # ReduceLROnPlateau - Réduit le learning rate sur plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=Config.REDUCE_LR['monitor'],
            mode=Config.REDUCE_LR['mode'],
            factor=Config.REDUCE_LR['factor'],
            patience=Config.REDUCE_LR['patience'],
            min_lr=Config.REDUCE_LR['min_lr'],
            min_delta=Config.REDUCE_LR['min_delta'],
            verbose=Config.REDUCE_LR['verbose']
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard_dir = Config.LOGS_DIR / 'tensorboard' / model_name
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=Config.TENSORBOARD['histogram_freq'],
            write_graph=Config.TENSORBOARD['write_graph'],
            write_images=Config.TENSORBOARD['write_images'],
            update_freq=Config.TENSORBOARD['update_freq'],
            profile_batch=Config.TENSORBOARD['profile_batch']
        )
        callbacks.append(tensorboard)
        
        # CSV Logger - Sauvegarde les métriques dans un fichier CSV
        csv_path = Config.LOGS_DIR / f"{model_name}_training_log.csv"
        csv_logger = tf.keras.callbacks.CSVLogger(
            str(csv_path),
            separator=',',
            append=True
        )
        callbacks.append(csv_logger)
        
        return callbacks

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Callback pour afficher le progrès détaillé de l'entraînement"""
    
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        print('Démarrage de l\'entraînement...')

    def on_epoch_begin(self, epoch, logs=None):
        print(f'\nEpoque {epoch+1}/{self.epochs}')
        self.seen = 0
        self.loss_sum = 0
        
    def on_batch_end(self, batch, logs=None):
        self.seen += logs.get('size', 0)
        self.loss_sum += logs.get('loss', 0) * logs.get('size', 0)
        if batch % 10 == 0:  # Afficher tous les 10 batches
            loss = self.loss_sum / max(self.seen, 1)
            print(f'Batch {batch}, Loss: {loss:.4f}', end='\r')

    def on_epoch_end(self, epoch, logs=None):
        metrics = []
        for metric, value in logs.items():
            metrics.append(f'{metric}: {value:.4f}')
        print(f"\nEpoque {epoch+1}: {', '.join(metrics)}")

class ValidationVisualizerCallback(tf.keras.callbacks.Callback):
    """Callback pour visualiser les prédictions sur l'ensemble de validation"""
    
    def __init__(self, validation_data, num_samples=4):
        super().__init__()
        self.validation_data = validation_data
        self.num_samples = num_samples
        
        # Créer le dossier pour sauvegarder les visualisations
        self.output_dir = Config.OUTPUT_DIR / 'visualizations' / 'validation'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # Visualiser toutes les N époques
        if (epoch + 1) % 5 == 0:  # Toutes les 5 époques
            # Prendre des échantillons aléatoires
            for i in range(min(self.num_samples, len(self.validation_data[0]))):
                # Faire une prédiction
                image = self.validation_data[0][i:i+1]
                true_mask = self.validation_data[1][i:i+1]
                pred_mask = self.model.predict(image)
                
                # Sauvegarder la comparaison
                self._save_comparison(
                    epoch + 1,
                    image[0],
                    true_mask[0],
                    pred_mask[0],
                    sample_index=i
                )
    
    def _save_comparison(self, epoch, image, true_mask, pred_mask, sample_index):
        """Sauvegarde une comparaison entre la vérité terrain et la prédiction"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Image originale
        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        # Vérité terrain
        axes[1].imshow(true_mask[..., 0], cmap='gray')
        axes[1].set_title('Vérité terrain')
        axes[1].axis('off')
        
        # Prédiction
        axes[2].imshow(pred_mask[..., 0], cmap='gray')
        axes[2].set_title('Prédiction')
        axes[2].axis('off')
        
        # Sauvegarder
        save_path = self.output_dir / f'epoch_{epoch}_sample_{sample_index}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

class MetricsLogger(tf.keras.callbacks.Callback):
    """Callback pour logger les métriques personnalisées"""
    
    def __init__(self):
        super().__init__()
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Enregistrer toutes les métriques
        for metric, value in logs.items():
            if metric not in self.history:
                self.history[metric] = []
            self.history[metric].append(value)
            
        # Sauvegarder les métriques dans un fichier numpy
        metrics_path = Config.LOGS_DIR / 'metrics_history.npy'
        np.save(str(metrics_path), self.history)