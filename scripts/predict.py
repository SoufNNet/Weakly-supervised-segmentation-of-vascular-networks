import tensorflow as tf
import argparse
from pathlib import Path

from src.config.config import Config
from src.training.trainer import Trainer
from src.utils.prediction_utils import load_images, create_dataset, save_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Prédiction de segmentation des vaisseaux')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Chemin vers le modèle entraîné (.keras)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Dossier contenant les images à prédire')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Dossier de sortie pour les prédictions')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Taille du batch pour les prédictions')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configuration GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Erreur GPU: {e}")
    
    # Définir le dossier de sortie
    output_dir = args.output_dir or Config.OUTPUT_DIR / 'predictions'
    
    try:
        # Charger les images
        images, image_paths = load_images(args.input_dir)
        
        # Créer le dataset
        dataset = create_dataset(images, args.batch_size)
        
        # Charger et utiliser le modèle
        trainer = Trainer()
        trainer.load_model(args.model_path)
        predictions = trainer.predict(dataset)
        
        # Sauvegarder les résultats
        save_predictions(predictions, image_paths, output_dir)
        
        print("\nPrédictions terminées avec succès!")
        print(f"Résultats sauvegardés dans: {output_dir}")
        
    except Exception as e:
        print(f"\nErreur lors de la prédiction: {str(e)}")
        raise e

if __name__ == "__main__":
    main()