"""Custom Convolutional Neural Network (CNN) for pneumonia detection."""

import tensorflow as tf

from src import config


def build_cnn(input_shape=tuple(config.IMG_SIZE), dropout_rate: float = config.DROPOUT_RATE) -> tf.keras.Model:
    """Build a small, fully custom CNN for binary chest X-ray classification.

    Architecture:
        Conv2D -> BatchNorm -> ReLU -> MaxPool (x3)
        Conv2D -> BatchNorm -> ReLU -> MaxPool (x3)
        Conv2D -> BatchNorm -> ReLU -> MaxPool (x3)
        GlobalAveragePooling2D
        Dense(128) -> Dropout -> Dense(1, sigmoid)

    The model is compiled by the training pipeline; this function only builds
    the architecture.
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Custom_CNN")
    return model


def compile_cnn(model: tf.keras.Model, learning_rate: float = config.LEARNING_RATE) -> tf.keras.Model:
    """Compile the custom CNN with binary cross-entropy and Adam."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model
