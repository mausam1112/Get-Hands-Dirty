import keras
from keras import layers
from config import setup


def augment():
    return keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(setup.IMAGE_SIZE[0], setup.IMAGE_SIZE[1]),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data-augmentation",
    )
