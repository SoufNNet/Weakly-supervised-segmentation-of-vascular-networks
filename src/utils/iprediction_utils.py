import tensorflow as tf
import numpy as np
from pathlib import Path
from src.data.preprocessor import Preprocessor
from src.utils.visualization import Visualizer

def load_images(input_dir):
    """
    Charge et prétraite les images d'un dossier
    Args:
        input_dir: Chemin du dossier contenant les images
    Returns:
        Liste des images prétraitées et leurs chemins
    """
    input_dir = Path(input_dir)
    images = []
    image_paths = []
    
    valid_extensions = ['.tif', '.png', '.jpg', '.jpeg']
    
    for ext in valid_extensions:
        image_paths.extend(input_dir.glob(f'*{ext}'))
    
    if not image_paths:
        raise ValueError(f"Aucune image trouvée dans {input_dir}")
    
    print(f"Chargement de {len(image_paths)} images...")
    
    for img_path in sorted(image_paths):
        try:
            image = Preprocessor.preprocess_image(img_path)
            images.append(image)
        except Exception as e:
            print(f"Erreur lors du chargement de {img_path.name}: {str(e)}")
    
    return np.array(images), image_paths

def create_dataset(images, batch_size):
    """
    Crée un tf.data.Dataset à partir des images
    Args:
        images: Liste des images
        batch_size: Taille du batch
    Returns:
        tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def save_predictions(predictions, image_paths, output_dir):
    """
    Sauvegarde les prédictions
    Args:
        predictions: Masques prédits
        image_paths: Chemins des images originales
        output_dir: Dossier de sortie
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSauvegarde des prédictions dans {output_dir}")
    
    visualizer = Visualizer(output_dir)
    
    for i, (pred, img_path) in enumerate(zip(predictions, image_paths)):
        try:
            # Charger l'image originale
            original_image = Preprocessor.preprocess_image(img_path)
            
            # Sauvegarder la visualisation
            save_path = output_dir / f"{img_path.stem}_prediction.png"
            visualizer.plot_segmentation_results(
                image=original_image,
                pred_mask=pred,
                save_path=save_path
            )
            
            # Sauvegarder le masque binaire
            mask_path = output_dir / f"{img_path.stem}_mask.npy"
            np.save(str(mask_path), pred)
            
            print(f"Traité {i+1}/{len(predictions)}: {img_path.name}")
            
        except Exception as e:
            print(f"Erreur lors du traitement de {img_path.name}: {str(e)}")