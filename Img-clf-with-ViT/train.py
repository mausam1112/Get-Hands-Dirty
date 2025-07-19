import keras
import os
from config import setup


def get_callbacks(suffix):
    callbacks = []

    checkpoint_dir = os.path.join(setup.MODEL_PATH, f"{setup.PREFIX}_{suffix}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_filepath = os.path.join(checkpoint_dir, "checkpoint.weights.h5")
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )
    callbacks.append(checkpoint_callback)

    callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=5))

    return callbacks


def run_experiment(model_nn: keras.Model, train_ds, val_ds, test_ds, suffix):
    optimizer = keras.optimizers.AdamW(
        learning_rate=setup.learning_rate, weight_decay=setup.weight_decay
    )

    model_nn.compile(
        optimizer=optimizer,  # type: ignore
        # loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            # keras.metrics.SparseCategoricalAccuracy(name="Accuracy"),
            # keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            keras.metrics.BinaryAccuracy()
        ],
    )

    callbacks = get_callbacks(suffix)

    history = model_nn.fit(
        train_ds, validation_data=val_ds, callbacks=callbacks, epochs=2
    )

    _, accuracy = model_nn.evaluate(test_ds)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history
