from src.training.trainer import Trainer
from src.config.config import Config
import tensorflow as tf
from src.utils.visualization import Visualizer

def main():
    # Configuration GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Créer et configurer le trainer
    trainer = Trainer(model_name="vessel_segmentation")
    
    # Setup du modèle avec la fonction de perte combinée
    trainer.setup_model(loss_type='combined')
    
    # Entraînement
    history = trainer.train()
    
    # Évaluation
    trainer.evaluate()
    
    # Sauvegarder les visualisations
    
    visualizer = Visualizer()
    visualizer.plot_training_history(history)

if __name__ == "__main__":
    main()