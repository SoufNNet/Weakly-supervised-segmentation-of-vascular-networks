import tensorflow as tf
from src.config.config import Config

class CustomSegmentationLoss(tf.keras.losses.Loss):
    def __init__(self, beta=Config.LOSS_BETA, epsilon=Config.LOSS_EPSILON, kappa=Config.LOSS_KAPPA):
        """
        Initialise la fonction de perte personnalisée
        Loss = BCE + Σ(y_pred + κε)|∇²It|² + β(ε|∇y_pred|² + 1/4ε(y_pred-1)²)
        """
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.kappa = kappa

    def compute_gradient(self, y):
        """Calcule le gradient spatial"""
        dy = tf.image.sobel_edges(y)
        return dy

    def compute_laplacian(self, y):
        """Calcule le laplacien (∇²)"""
        laplacian_kernel = tf.constant([
            [[0.0,  1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0,  1.0, 0.0]]
        ], dtype=tf.float32)
        
        laplacian_kernel = tf.reshape(laplacian_kernel, [3, 3, 1, 1])
        
        if len(y.shape) == 3:
            y = tf.expand_dims(y, axis=-1)
            
        return tf.nn.conv2d(y, laplacian_kernel, strides=[1, 1, 1, 1], padding='SAME')

    def call(self, y_true, y_pred):
        """Calcule la perte totale"""
        # Binary Cross Entropy
        bce = -tf.reduce_mean(
            y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)) + 
            (1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-7, 1.0))
        )
        
        # Terme du laplacien
        laplacian = self.compute_laplacian(y_pred)
        laplacian_term = tf.reduce_mean(
            (y_pred + self.kappa * self.epsilon) * tf.square(laplacian)
        )
        
        # Terme de régularisation
        gradient = self.compute_gradient(y_pred)
        gradient_magnitude = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=-1))
        
        reg_term = self.beta * tf.reduce_mean(
            self.epsilon * tf.square(gradient_magnitude) +
            1/(4 * self.epsilon) * tf.square(y_pred - 1.0)
        )
        
        return bce + laplacian_term + reg_term

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6):
        """Initialise la Dice Loss"""
        super().__init__()
        self.smooth = smooth

    def call(self, y_true, y_pred):
        """Calcule la Dice Loss"""
        y_true = tf.cast(y_true, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        sum_tensors = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        dice = (2 * intersection + self.smooth) / (sum_tensors + self.smooth)
        return 1 - dice

class BCELoss(tf.keras.losses.Loss):
    def __init__(self):
        """Initialise la BCE Loss"""
        super().__init__()

    def call(self, y_true, y_pred):
        """Calcule la BCE Loss"""
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)

class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, weights={'custom': 1.0, 'dice': 1.0}):
        """
        Combine plusieurs pertes avec des poids
        Args:
            weights: dictionnaire des poids pour chaque perte
        """
        super().__init__()
        self.weights = weights
        self.custom_loss = CustomSegmentationLoss()
        self.dice_loss = DiceLoss()

    def call(self, y_true, y_pred):
        """Calcule la perte combinée pondérée"""
        total_loss = 0
        if 'custom' in self.weights:
            total_loss += self.weights['custom'] * self.custom_loss(y_true, y_pred)
        if 'dice' in self.weights:
            total_loss += self.weights['dice'] * self.dice_loss(y_true, y_pred)
        return total_loss

def get_loss(loss_type='combined', **kwargs):
    """
    Factory function pour créer la fonction de perte
    Args:
        loss_type: Type de perte ('custom', 'dice', 'bce', 'combined')
        **kwargs: Arguments supplémentaires pour la fonction de perte
    Returns:
        Instance de la fonction de perte
    """
    losses = {
        'custom': CustomSegmentationLoss,
        'dice': DiceLoss,
        'bce': BCELoss,
        'combined': CombinedLoss
    }
    
    if loss_type not in losses:
        raise ValueError(f"Type de perte non reconnu. Choix possibles: {list(losses.keys())}")
    
    return losses[loss_type](**kwargs)