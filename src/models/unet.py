import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, 
    MaxPooling2D, 
    Conv2DTranspose, 
    concatenate, 
    Dropout,
    BatchNormalization,
    ReLU
)
from src.config.config import Config

class UNet:
    def __init__(self):
        """Initialise l'architecture U-Net"""
        self.input_shape = (Config.TARGET_HEIGHT, Config.TARGET_WIDTH, Config.IMAGE_CHANNELS)
        self.filters = Config.UNET_FILTERS
        self.kernel_size = Config.UNET_KERNEL_SIZE
        self.padding = Config.UNET_PADDING
        self.activation = Config.UNET_ACTIVATION
        self.final_activation = Config.UNET_FINAL_ACTIVATION
        self.dropout_rate = Config.UNET_DROPOUT_RATE

    def conv_block(self, inputs, filters):
        """Block de convolution double"""
        # Première convolution
        x = Conv2D(
            filters=filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            kernel_initializer='he_normal'
        )(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Deuxième convolution
        x = Conv2D(
            filters=filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            kernel_initializer='he_normal'
        )(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    def encoder_block(self, inputs, filters):
        """Block d'encodeur"""
        # Convolutions
        conv = self.conv_block(inputs, filters)
        # Pooling
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        # Dropout optionnel
        if self.dropout_rate > 0:
            pool = Dropout(self.dropout_rate)(pool)
        return conv, pool

    def decoder_block(self, inputs, skip_features, filters):
        """Block de décodeur"""
        # Convolution transposée pour l'upsampling
        x = Conv2DTranspose(
            filters=filters,
            kernel_size=2,
            strides=2,
            padding=self.padding
        )(inputs)
        
        # Redimensionner skip_features si nécessaire pour correspondre à x
        if x.shape[1] != skip_features.shape[1] or x.shape[2] != skip_features.shape[2]:
            skip_features = tf.image.resize(
                skip_features,
                size=(x.shape[1], x.shape[2]),
                method='nearest'
            )
        
        # Concaténation
        x = concatenate([x, skip_features])
        
        # Double convolution
        x = self.conv_block(x, filters)
        
        return x

    def build(self):
        """Construit le modèle U-Net complet"""
        # Input
        inputs = Input(shape=self.input_shape)

        # Liste pour stocker les skip connections
        skip_features = []

        # Encoder
        x = inputs
        for filters in self.filters[:-1]:
            conv, x = self.encoder_block(x, filters)
            skip_features.append(conv)

        # Bridge
        x = self.conv_block(x, self.filters[-1])
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        # Decoder
        skip_features.reverse()
        for skip, filters in zip(skip_features, self.filters[-2::-1]):
            x = self.decoder_block(x, skip, filters)

        # Output
        outputs = Conv2D(
            filters=Config.MASK_CHANNELS,
            kernel_size=1,
            activation=self.final_activation
        )(x)

        # Créer le modèle
        model = Model(inputs, outputs, name='U-Net')
        
        return model

def create_unet_model():
    """Fonction utilitaire pour créer le modèle"""
    unet = UNet()
    model = unet.build()
    return model