import os
import tensorflow as tf
import argparse
import logging
from datetime import datetime
from pathlib import Path

from src.config.config import Config
from src.training.trainer import Trainer
from src.utils.metrics import get_metrics
from src.utils.visualization import Visualizer



def setup_logging():
    """Configure le logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def setup_device():
    """Configure l'environnement pour utiliser GPU si disponible, sinon CPU"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Activer la croissance de la mémoire GPU pour éviter l'allocation complète
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            if Config.USE_MIXED_PRECISION:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            logging.info(f"GPU(s) disponible(s): {len(gpus)}")
            for gpu in gpus:
                logging.info(f"  - {gpu.device_type}: {gpu.name}")
        except RuntimeError as e:
            logging.error(f"Erreur lors de la configuration GPU: {e}")
            logging.warning("Basculement sur CPU.")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        logging.warning("Aucun GPU détecté. L'entraînement sera effectué sur CPU.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def parse_args():
    """Parse les arguments en ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Entraînement segmentation vaisseaux sanguins DRIVE dataset'
    )
    
    # Paramètres d'entraînement
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS,
                       help='Nombre d\'époques')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                       help='Taille du batch')
    parser.add_argument('--learning_rate', type=float, default=Config.INITIAL_LEARNING_RATE,
                       help='Taux d\'apprentissage')
    
    # Paramètres du modèle
    parser.add_argument('--model_name', type=str, default='drive_segmentation',
                       help='Nom du modèle pour la sauvegarde')
    parser.add_argument('--loss_type', type=str, default='combined',
                       choices=['combined', 'dice', 'custom'],
                       help='Type de fonction de perte à utiliser')
    
    # Paramètres de prétraitement
    parser.add_argument('--use_augmentation', type=bool, default=True,
                       help='Utiliser la data augmentation')
    parser.add_argument('--use_clahe', type=bool, default=True,
                       help='Utiliser CLAHE pour le prétraitement')
    
    # Paramètres de reprise d'entraînement
    parser.add_argument('--resume', type=str, default=None,
                       help='Chemin vers un modèle pour reprendre l\'entraînement')
    
    return parser.parse_args()

def update_config(args):
    """Met à jour la configuration avec les arguments"""
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.INITIAL_LEARNING_RATE = args.learning_rate
    Config.PREPROCESSING_PARAMS['clahe']['enabled'] = args.use_clahe
    
    if not args.use_augmentation:
        Config.AUGMENTATION_PARAMS = None

def main():
    # Configuration du logging
    log_file = setup_logging()
    logging.info(f"Logs sauvegardés dans: {log_file}")
    
    try:
        # Parser les arguments
        args = parse_args()
        
        # Mettre à jour la configuration
        update_config(args)
        
        # Configuration GPU ou CPU
        setup_device()
        
        # Valider l'environnement
        Config.validate_paths()
        Config.create_directories()
        
        # Afficher la configuration
        Config.print_info()
        
        # Initialiser le trainer
        trainer = Trainer(model_name=args.model_name)
        
        # Charger un modèle existant si demandé
        if args.resume:
            logging.info(f"Reprise de l'entraînement à partir de: {args.resume}")
            trainer.load_model(args.resume)
        else:
            logging.info("Configuration du nouveau modèle")
            trainer.setup_model(loss_type=args.loss_type)
        
        # Entraînement
        logging.info("Démarrage de l'entraînement")
        history = trainer.train()
        
        # Évaluation
        logging.info("Évaluation du modèle")
        test_results = trainer.evaluate()
        
        # Sauvegarder les visualisations
        logging.info("Génération des visualisations")
        trainer.save_training_visualizations()
        
        # Résumé final
        logging.info("\nRésultats finaux:")
        for metric_name, value in test_results.items():
            logging.info(f"{metric_name}: {value:.4f}")
        
        logging.info("\nEntraînement terminé avec succès!")
        logging.info(f"Modèle sauvegardé dans: {Config.MODELS_DIR}")
        logging.info(f"Visualisations dans: {Config.OUTPUT_DIR/'visualizations'}")
        
    except Exception as e:
        logging.error(f"Erreur lors de l'entraînement: {str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
