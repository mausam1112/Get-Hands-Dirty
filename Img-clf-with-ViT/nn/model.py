import keras
from config import setup
from keras import layers
from nn.blocks import mlp
from nn.layers import data, patch


def vit_classifier(num_class: int):
    data_augemntation: keras.Sequential = data.augment()
    # ----------------------------------
    inputs = keras.Input(shape=(*setup.IMAGE_SIZE, setup.CHANNEL))
    augmented = data_augemntation(inputs)

    patches = patch.Patches(setup.patch_size)(augmented)
    encoded_patches = patch.PatchEncoder(setup.num_patches, setup.projection_dim)(
        patches
    )

    # multiple layers of the Transformer block
    for _ in range(setup.transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attn_output = layers.MultiHeadAttention(
            num_heads=setup.num_heads, key_dim=setup.projection_dim, dropout=0.1
        )(x1, x1)

        # skip connection 1
        x2 = layers.Add()([attn_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, setup.transformer_units, dropout_rate=0.1)

        # skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # create tensor of shape [batch_size, projection_dim]
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, setup.mlp_head_units, 0.5)

    logits = layers.Dense(num_class)(features)

    return keras.Model(inputs=inputs, outputs=logits)
