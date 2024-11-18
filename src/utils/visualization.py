import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path
from src.config.config import Config

class Visualizer:
    def __init__(self, output_dir=None):
        """
        Initialise le visualiseur
        Args:
            output_dir: Dossier de sortie pour les images (optionnel)
        """
        self.output_dir = Path(output_dir) if output_dir else Config.OUTPUT_DIR / 'visualizations'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_segmentation_results(self, image, pred_mask, true_mask=None, save_path=None):
        """
        Affiche les résultats de segmentation
        Args:
            image: Image d'entrée
            pred_mask: Masque prédit
            true_mask: Masque de vérité terrain (optionnel)
            save_path: Chemin pour sauvegarder l'image
        """
        # Déterminer le nombre de sous-plots
        n_plots = 2  # Image + prédiction
        if true_mask is not None:
            n_plots += 1  # Ajouter vérité terrain
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        
        # Image originale
        axes[0].imshow(image)
        axes[0].set_title('Image Originale')
        axes[0].axis('off')
        
        # Prédiction
        axes[1].imshow(pred_mask[..., 0], cmap='gray')
        axes[1].set_title('Prédiction')
        axes[1].axis('off')
        
        if true_mask is not None:
            # Vérité terrain
            axes[2].imshow(true_mask[..., 0], cmap='gray')
            axes[2].set_title('Vérité Terrain')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_training_history(self, history, save_path=None):
        """Reste de la méthode inchangée"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot des pertes
        ax1.plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Evolution de la perte')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot des métriques
        metrics = [k for k in history.history.keys() 
                  if not k.startswith('val_') and k != 'loss']
        for metric in metrics:
            ax2.plot(history.history[metric], label=f'Train {metric}')
            if f'val_{metric}' in history.history:
                ax2.plot(history.history[f'val_{metric}'], 
                        label=f'Val {metric}', linestyle='--')
        ax2.set_title('Evolution des métriques')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def create_overlay(image, mask, alpha=0.5):
        """Reste de la méthode inchangée"""
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        mask_rgb = np.zeros((*mask.shape, 3))
        mask_rgb[..., 1] = mask  # Canal vert pour les vaisseaux
        
        overlay = image * (1 - alpha) + mask_rgb * alpha
        return np.clip(overlay, 0, 1)