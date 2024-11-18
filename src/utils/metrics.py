import tensorflow as tf
from tensorflow.keras.metrics import Metric

class DiceCoefficient(Metric):
    def __init__(self, name='dice_coefficient', smooth=1e-6, **kwargs):
        """
        Coefficient de Dice pour la segmentation
        Args:
            name: Nom de la métrique
            smooth: Terme de lissage pour éviter la division par zéro
        """
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.dice_sum = self.add_weight(name='dice_sum', initializer='zeros')
        self.total_batches = self.add_weight(name='total_batches', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        self.dice_sum.assign_add(dice)
        self.total_batches.assign_add(1.)

    def result(self):
        return self.dice_sum / self.total_batches

    def reset_state(self):
        self.dice_sum.assign(0.)
        self.total_batches.assign(0.)

class IoU(Metric):
    def __init__(self, name='iou', smooth=1e-6, **kwargs):
        """
        Intersection over Union (Jaccard Index)
        Args:
            name: Nom de la métrique
            smooth: Terme de lissage pour éviter la division par zéro
        """
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.iou_sum = self.add_weight(name='iou_sum', initializer='zeros')
        self.total_batches = self.add_weight(name='total_batches', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        self.iou_sum.assign_add(iou)
        self.total_batches.assign_add(1.)

    def result(self):
        return self.iou_sum / self.total_batches

    def reset_state(self):
        self.iou_sum.assign(0.)
        self.total_batches.assign(0.)

class Sensitivity(Metric):
    def __init__(self, name='sensitivity', **kwargs):
        """
        Sensibilité (Recall, True Positive Rate)
        """
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred > 0.5, tf.bool)
        
        true_positives = tf.cast(tf.logical_and(y_true, y_pred), tf.float32)
        false_negatives = tf.cast(tf.logical_and(y_true, tf.logical_not(y_pred)), tf.float32)
        
        self.true_positives.assign_add(tf.reduce_sum(true_positives))
        self.false_negatives.assign_add(tf.reduce_sum(false_negatives))

    def result(self):
        return self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_positives.assign(0.)
        self.false_negatives.assign(0.)

class Specificity(Metric):
    def __init__(self, name='specificity', **kwargs):
        """
        Spécificité (True Negative Rate)
        """
        super().__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred > 0.5, tf.bool)
        
        true_negatives = tf.cast(tf.logical_and(tf.logical_not(y_true), tf.logical_not(y_pred)), tf.float32)
        false_positives = tf.cast(tf.logical_and(tf.logical_not(y_true), y_pred), tf.float32)
        
        self.true_negatives.assign_add(tf.reduce_sum(true_negatives))
        self.false_positives.assign_add(tf.reduce_sum(false_positives))

    def result(self):
        return self.true_negatives / (self.true_negatives + self.false_positives + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_negatives.assign(0.)
        self.false_positives.assign(0.)

class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        """
        F1 Score (moyenne harmonique de la précision et du recall)
        """
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

def get_metrics():
    """
    Retourne la liste complète des métriques pour l'évaluation
    Returns:
        Liste des métriques à utiliser pour l'entraînement et l'évaluation
    """
    return [
        DiceCoefficient(name='dice_coefficient'),
        IoU(name='iou'),
        Sensitivity(name='sensitivity'),
        Specificity(name='specificity'),
        F1Score(name='f1_score'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]

# Export explicite
__all__ = [
    'DiceCoefficient',
    'IoU',
    'Sensitivity',
    'Specificity',
    'F1Score',
    'get_metrics'
]