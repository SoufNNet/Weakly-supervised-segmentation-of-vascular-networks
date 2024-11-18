import tensorflow as tf
import numpy as np
import argparse
from pathlib import Path
import json
import pandas as pd

from src.config.config import Config
from src.training.trainer import Trainer
from src.utils.prediction_utils import load_images, create_dataset
from src.utils.metrics import get_metrics
from src.utils.visualization import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation du modèle de segmentation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Chemin vers le modèle à évaluer')
    parser.add_argument('--test_images', type=str, required=True,
                        help='Dossier contenant les images de test')
    parser.add_argument('--test_masks', type=str, required=True,
                        help='Dossier contenant les masques de test')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Dossier pour sauvegarder les résultats')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Taille du batch pour l\'évaluation')
    return parser.parse_args()

def compute_metrics_per_image(y_true, y_pred):
    """
    Calcule les métriques pour chaque image
    """
    metrics = {}
    
    # Dice coefficient
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    metrics['dice'] = (2. * intersection + 1e-6) / (union + 1e-6)
    
    # IoU (Intersection over Union)
    metrics['iou'] = (intersection + 1e-6) / (union - intersection + 1e-6)
    
    # Précision et Recall
    true_positives = np.sum(y_true * y_pred)
    false_positives = np.sum((1 - y_true) * y_pred)
    false_negatives = np.sum(y_true * (1 - y_pred))
    
    metrics['precision'] = true_positives / (true_positives + false_positives + 1e-6)
    metrics['recall'] = true_positives / (true_positives + false_negatives + 1e-6)
    
    # F1-Score
    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-6)
    
    return metrics

def evaluate_model(model, test_dataset, test_images, test_masks, output_dir):
    """
    Évalue le modèle et sauvegarde les résultats
    """
    # Prédictions sur l'ensemble de test
    print("\nGénération des prédictions...")
    predictions = model.predict(test_dataset)
    
    # Initialisation des résultats
    all_metrics = []
    global_metrics = {metric: [] for metric in ['dice', 'iou', 'precision', 'recall', 'f1_score']}
    
    # Visualiseur pour sauvegarder les résultats
    visualizer = Visualizer(output_dir)
    
    print("\nÉvaluation des prédictions...")
    for idx, (image, true_mask, pred_mask) in enumerate(zip(test_images, test_masks, predictions)):
        # Calcul des métriques pour cette image
        metrics = compute_metrics_per_image(true_mask[..., 0], pred_mask[..., 0])
        metrics['image_id'] = idx
        all_metrics.append(metrics)
        
        # Mise à jour des métriques globales
        for metric in global_metrics:
            global_metrics[metric].append(metrics[metric])
        
        # Sauvegarder la visualisation
        save_path = Path(output_dir) / f'comparison_{idx}.png'
        visualizer.plot_segmentation_results(
            image=image,
            pred_mask=pred_mask,
            true_mask=true_mask,
            save_path=save_path
        )
    
    # Calcul des statistiques globales
    results = {
        'global_metrics': {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
            for metric, values in global_metrics.items()
        },
        'per_image_metrics': all_metrics
    }
    
    # Sauvegarder les résultats
    results_file = Path(output_dir) / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Créer un DataFrame pour une meilleure visualisation
    df = pd.DataFrame(all_metrics)
    df_path = Path(output_dir) / 'metrics_summary.csv'
    df.to_csv(df_path, index=False)
    
    # Afficher le résumé
    print("\nRésultats de l'évaluation:")
    print("-" * 50)
    for metric, stats in results['global_metrics'].items():
        print(f"\n{metric.upper()}:")
        for stat_name, value in stats.items():
            print(f"  {stat_name}: {value:.4f}")
    
    return results

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
    output_dir = args.output_dir or Config.OUTPUT_DIR / 'evaluation'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Charger les données de test
        print("Chargement des données de test...")
        test_images, _ = load_images(args.test_images)
        test_masks, _ = load_images(args.test_masks)
        
        # Créer le dataset
        test_dataset = create_dataset(test_images, args.batch_size)
        
        # Charger le modèle
        print("\nChargement du modèle...")
        trainer = Trainer()
        trainer.load_model(args.model_path)
        
        # Évaluer le modèle
        results = evaluate_model(
            model=trainer.model,
            test_dataset=test_dataset,
            test_images=test_images,
            test_masks=test_masks,
            output_dir=output_dir
        )
        
        print(f"\nRésultats sauvegardés dans: {output_dir}")
        
    except Exception as e:
        print(f"\nErreur lors de l'évaluation: {str(e)}")
        raise e

if __name__ == "__main__":
    main()