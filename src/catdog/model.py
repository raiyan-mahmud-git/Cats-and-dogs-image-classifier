"""CNN model architectures: baseline, GAP, and modernized from-scratch CNN."""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from catdog.config import Config


def build_data_augmentation_layer() -> layers.Layer:
    """Build in-model data augmentation pipeline.

    Design explanation:
    Placing augmentation layers inside the Keras model ensures transformations run
    efficiently on the computation device (GPU/CPU) during training (`training=True`)
    and automatically deactivate during validation and inference (`training=False`).
    """
    return models.Sequential(
        [
            layers.RandomFlip("horizontal", name="aug_flip"),
            layers.RandomRotation(0.083, fill_mode="nearest", name="aug_rotation"),  # ~30 deg
            layers.RandomTranslation(
                height_factor=0.1,
                width_factor=0.1,
                fill_mode="nearest",
                name="aug_translation",
            ),
            layers.RandomZoom(0.2, fill_mode="nearest", name="aug_zoom"),
        ],
        name="data_augmentation",
    )


def build_baseline_model(config: Config) -> models.Model:
    """Construct and compile the original freeCodeCamp baseline CNN.

    Architecture:
    3x Conv2D + MaxPool2D -> Flatten (36,992 features) -> Dense(512) -> Dropout(0.5) -> Dense(1).
    Total parameters: ~19,034,177 (~72.6 MB).
    """
    inputs = layers.Input(
        shape=(config.img_height, config.img_width, config.channels),
        name="image_input",
    )

    augmented = build_data_augmentation_layer()(inputs)
    rescaled = layers.Rescaling(1.0 / 255.0, name="rescaling")(augmented)

    x = layers.Conv2D(32, (3, 3), activation="relu", name="conv1")(rescaled)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(512, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_gap_model(config: Config) -> models.Model:
    """Ablation Variant 1: Baseline replaced with GlobalAveragePooling2D.

    Design explanation:
    Replacing the 19M-parameter Flatten+Dense layer with GlobalAveragePooling2D
    shrinks spatial feature maps into channel averages, dropping parameter count
    from ~19,000,000 down to ~421,000 without sacrificing representational capacity.
    """
    inputs = layers.Input(
        shape=(config.img_height, config.img_width, config.channels),
        name="image_input",
    )

    augmented = build_data_augmentation_layer()(inputs)
    rescaled = layers.Rescaling(1.0 / 255.0, name="rescaling")(augmented)

    x = layers.Conv2D(32, (3, 3), activation="relu", name="conv1")(rescaled)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    x = layers.Conv2D(256, (3, 3), activation="relu", name="conv4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="gap_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_improved_cnn(
    config: Config,
    use_batchnorm: bool = True,
    l2_reg: float = 1e-4,
    dropout_rate: float = 0.3,
) -> models.Model:
    """Stage 2: Production-grade from-scratch CNN.

    Design explanation:
    Uses `use_bias=False` Conv2D followed by `BatchNormalization` to stabilize internal
    feature distributions and accelerate convergence. Spatial maps are pooled via
    `GlobalAveragePooling2D`, and mild L2 regularization with tuned Dropout prevents overfitting.
    """
    inputs = layers.Input(
        shape=(config.img_height, config.img_width, config.channels),
        name="image_input",
    )

    augmented = build_data_augmentation_layer()(inputs)
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(augmented)

    # 4 Convolutional blocks
    filters_list = [32, 64, 128, 256]
    for i, filters in enumerate(filters_list, start=1):
        x = layers.Conv2D(
            filters,
            (3, 3),
            padding="same",
            use_bias=not use_batchnorm,
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
            name=f"conv{i}",
        )(x)
        if use_batchnorm:
            x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D((2, 2), name=f"pool{i}")(x)

    # Global pooling & head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(
        128,
        use_bias=not use_batchnorm,
        kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
        name="dense_head",
    )(x)
    if use_batchnorm:
        x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.ReLU(name="relu_dense")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="improved_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_model(
    config: Config,
    learning_rate: float = 1e-3,
) -> models.Model:
    """Stage 3: Pretrained MobileNetV2 Transfer Learning Model.

    Design explanation:
    Pretrained MobileNetV2 requires inputs normalized to [-1, 1] (via `Rescaling(1/127.5, offset=-1.0)`).
    The base feature extractor is initially frozen (`trainable=False`), and called with `training=False`
    so its internal BatchNormalization layers preserve ImageNet running statistics during training.
    """
    inputs = layers.Input(
        shape=(config.img_height, config.img_width, config.channels),
        name="image_input",
    )

    augmented = build_data_augmentation_layer()(inputs)
    # MobileNetV2 specific preprocessing: [0, 255] -> [-1.0, 1.0]
    rescaled = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenet_rescaling",
    )(augmented)

    from tensorflow.keras.applications import MobileNetV2

    base_model = MobileNetV2(
        input_shape=(config.img_height, config.img_width, config.channels),
        include_top=False,
        weights="imagenet",
        name="mobilenetv2_base",
    )
    base_model.trainable = False

    # Call with training=False to preserve frozen BN weights
    x = base_model(rescaled, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_head")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="transfer_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
