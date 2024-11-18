from pathlib import Path

class Config:
    # Chemins des dossiers
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / 'data' / 'DRIVE'
    OUTPUT_DIR = BASE_DIR / 'outputs'
    MODELS_DIR = OUTPUT_DIR / 'models'
    LOGS_DIR = OUTPUT_DIR / 'logs'
    
    # Chemins des données
    TRAIN_IMAGES_DIR = DATA_DIR / 'training' / 'images'        # Images .tif
    TRAIN_MASKS_GT_DIR = DATA_DIR / 'training' / '1st_manual'  # Ground truth .gif
    TRAIN_MASKS_FOV_DIR = DATA_DIR / 'training' / 'mask'      # FOV masks .gif
    
    TEST_IMAGES_DIR = DATA_DIR / 'test' / 'images'            # Images .tif
    TEST_MASKS_FOV_DIR = DATA_DIR / 'test' / 'mask'          # FOV masks .gif
    
    # Paramètres des images
    IMAGE_HEIGHT = 584  # Dimension originale DRIVE
    IMAGE_WIDTH = 565   # Dimension originale DRIVE
    TARGET_HEIGHT = 576  # Multiple de 16 pour U-Net
    TARGET_WIDTH = 576  # Multiple de 16 pour U-Net
    IMAGE_CHANNELS = 3
    MASK_CHANNELS = 1

    # Paramètres d'entraînement
    RANDOM_SEED = 42
    BATCH_SIZE = 2  # Adapté pour GPU 2GB
    EPOCHS = 150
    VALIDATION_SPLIT = 0.2
    INITIAL_LEARNING_RATE = 1e-4
    MIN_LEARNING_RATE = 1e-7
    
    # Paramètres U-Net
    UNET_FILTERS = [32, 64, 128, 256, 512]  # Réduit pour GPU 2GB
    UNET_KERNEL_SIZE = 3
    UNET_PADDING = 'same'
    UNET_ACTIVATION = 'relu'
    UNET_FINAL_ACTIVATION = 'sigmoid'
    UNET_DROPOUT_RATE = 0.2
    USE_BATCH_NORM = True
    
    # Paramètres de fonction de perte
    LOSS_BETA = 0.0005
    LOSS_EPSILON = 0.01
    LOSS_KAPPA = 1e-8
    COMBINE_LOSS_WEIGHTS = {
        'dice_loss': 0.5,
        'custom_loss': 0.5
    }
    
    # Paramètres de data augmentation
    AUGMENTATION_PARAMS = {
        'rotation_range': 15,        # Rotation limitée
        'width_shift_range': 0.05,   # Translation horizontale
        'height_shift_range': 0.05,  # Translation verticale
        'zoom_range': 0.1,          # Zoom
        'horizontal_flip': True,     # Flip horizontal autorisé
        'vertical_flip': False,      # Pas de flip vertical
        'fill_mode': 'reflect',      # Mode de remplissage
        'brightness_range': [0.8, 1.2]  # Variation de luminosité
    }
    
    # Paramètres de prétraitement
    PREPROCESSING_PARAMS = {
        'normalize': True,
        'clahe': {
            'enabled': True,
            'clip_limit': 2.0,
            'tile_grid_size': (8, 8)
        },
        'histogram_equalization': False,
        'gamma_correction': False
    }
    
    # Paramètres de monitoring
    MONITOR_METRIC = 'val_dice_coefficient'
    MONITOR_MODE = 'max'
    
    # Configuration des callbacks
    EARLY_STOPPING = {
        'monitor': MONITOR_METRIC,
        'mode': MONITOR_MODE,
        'patience': 20,
        'restore_best_weights': True,
        'min_delta': 0.001,
        'verbose': 1
    }
    
    MODEL_CHECKPOINT = {
        'monitor': MONITOR_METRIC,
        'mode': MONITOR_MODE,
        'save_best_only': True,
        'save_weights_only': False,
        'verbose': 1
    }
    
    REDUCE_LR = {
        'monitor': MONITOR_METRIC,
        'mode': MONITOR_MODE,
        'factor': 0.5,
        'patience': 10,
        'min_lr': MIN_LEARNING_RATE,
        'min_delta': 0.001,
        'verbose': 1
    }
    
    TENSORBOARD = {
        'histogram_freq': 1,
        'write_graph': True,
        'write_images': False,
        'update_freq': 'epoch',
        'profile_batch': 0
    }
    
    # Paramètres de performance
    USE_MIXED_PRECISION = True      # Pour GPU
    CACHE_DATASET = True           # Mettre en cache le dataset
    PREFETCH_BUFFER_SIZE = 2       # Nombre de batches à précharger
    NUM_PARALLEL_CALLS = 4         # Nombre de threads pour le chargement
    
    @classmethod
    def create_directories(cls):
        """Crée les répertoires nécessaires"""
        dirs_to_create = [
            cls.MODELS_DIR,
            cls.LOGS_DIR,
            cls.OUTPUT_DIR / 'predictions',
            cls.OUTPUT_DIR / 'visualizations',
            cls.LOGS_DIR / 'tensorboard'
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
            
    @classmethod
    def validate_paths(cls):
        """Valide l'existence des chemins requis"""
        required_dirs = [
            cls.TRAIN_IMAGES_DIR,
            cls.TRAIN_MASKS_GT_DIR,    # Vérité terrain
            cls.TRAIN_MASKS_FOV_DIR,   # Masques FOV
            cls.TEST_IMAGES_DIR,
            cls.TEST_MASKS_FOV_DIR
        ]
        
        missing_dirs = []
        for directory in required_dirs:
            if not directory.exists():
                missing_dirs.append(directory)
                
        if missing_dirs:
            raise FileNotFoundError(
                "Dossiers manquants:\n" + 
                "\n".join([f"- {d}" for d in missing_dirs])
            )
    
    @classmethod
    def print_info(cls):
        """Affiche les informations de configuration"""
        print("\nConfiguration du projet DRIVE:")
        print("-" * 50)
        print(f"Images d'entraînement: {cls.TRAIN_IMAGES_DIR}")
        print(f"Vérité terrain: {cls.TRAIN_MASKS_GT_DIR}")
        print(f"Masques FOV: {cls.TRAIN_MASKS_FOV_DIR}")
        print(f"\nDimensions:")
        print(f"- Originales: {cls.IMAGE_HEIGHT}x{cls.IMAGE_WIDTH}")
        print(f"- Cible: {cls.TARGET_HEIGHT}x{cls.TARGET_WIDTH}")
        print(f"\nParamètres d'entraînement:")
        print(f"- Batch size: {cls.BATCH_SIZE}")
        print(f"- Epochs: {cls.EPOCHS}")
        print(f"- Learning rate: {cls.INITIAL_LEARNING_RATE}")
        print(f"- Architecture U-Net: {cls.UNET_FILTERS}")
        print(f"\nAugmentation activée: {bool(cls.AUGMENTATION_PARAMS)}")
        print(f"CLAHE activé: {cls.PREPROCESSING_PARAMS['clahe']['enabled']}")
        print(f"Métrique surveillée: {cls.MONITOR_METRIC}")
        print(f"Mixed Precision: {cls.USE_MIXED_PRECISION}")
        print("-" * 50)
    
    @classmethod
    def get_model_path(cls, model_name):
        """Retourne le chemin complet du modèle"""
        return cls.MODELS_DIR / f"{model_name}.keras"