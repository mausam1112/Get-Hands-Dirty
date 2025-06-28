from keras import layers
from typing import List


def mlp(x, hidden_units: List[int], dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation="gelu")(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
