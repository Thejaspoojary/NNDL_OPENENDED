"""Transfer learning with ImageNet pretrained MobileNetV2 for pneumonia detection."""

import tensorflow as tf

from src import config


def _resolve_mobilenet_v2():
    """Resolve the MobileNetV2 class and preprocessing function across
    TensorFlow / Keras versions."""
    try:
        from tensorflow.keras.applications import MobileNetV2 as Cls  # Keras 3 style

        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        return Cls, preprocess_input
    except Exception:
        pass
    try:
        import tensorflow.keras.applications.mobilenet_v2 as module

        return module.MobileNetV2, module.preprocess_input
    except Exception:
        pass
    try:
        import keras.applications.mobilenet_v2 as module

        return module.MobileNetV2, module.preprocess_input
    except Exception as exc:
        raise ImportError(
            "MobileNetV2 could not be loaded from TensorFlow/Keras. "
            "Make sure TensorFlow is installed correctly."
        ) from exc


def build_mobilenetv2(
    input_shape=tuple(config.IMG_SIZE),
    fine_tune: bool = False,
    fine_tune_top_layers: int = config.FINE_TUNE_TOP_LAYERS,
    dropout_rate: float = config.DROPOUT_RATE,
) -> tf.keras.Model:
    """Build a MobileNetV2 transfer learning model for binary X-ray classification.

    The ImageNet pretrained backbone has its original classification head
    removed; a small classifier head on top is trained on the pneumonia dataset.

    If ``fine_tune`` is True, the top ``fine_tune_top_layers`` of the backbone
    are unfrozen and fine tuned with a low learning rate.
    """
    BaseCls, preprocess_input = _resolve_mobilenet_v2()
    base = BaseCls(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = base.input
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_CNN")

    if fine_tune:
        for layer in base.layers:
            layer.trainable = True
        # Keep only the last `fine_tune_top_layers` backbone layers trainable.
        n = len(base.layers)
        for layer in base.layers[: max(0, n - fine_tune_top_layers)]:
            layer.trainable = False
        model.trainable = True

    return model


def compile_mobilenetv2(model: tf.keras.Model, learning_rate: float = config.LEARNING_RATE) -> tf.keras.Model:
    """Compile the MobileNetV2 transfer learning model."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def get_mobilenetv2_preprocess_input():
    """Return the MobileNetV2 preprocessing function."""
    _, preprocess_input = _resolve_mobilenet_v2()
    return preprocess_input
