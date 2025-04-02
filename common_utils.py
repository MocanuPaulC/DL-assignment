import hashlib
import os
import pandas as pd
import tensorflow as tf
from keras import layers, models
import tensorflow as tf
import numpy as np
from pathlib import Path
from collections import defaultdict
import optuna
from tensorflow.keras import layers, models


def get_unique_image_shapes():
    image_paths = list(Path("../raw_data2/face_age").rglob("*.png"))
    shapes = set()

    for path in image_paths:
        img = tf.io.read_file(str(path))
        img = tf.image.decode_image(img, channels=3)
        shapes.add(tuple(img.shape))

    return shapes

def file_hash(path, algo='md5'):
    hasher = hashlib.new(algo)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_unique_image_paths(directory="../raw_data2/face_age"):
    image_paths = list(Path(directory).rglob("*.png"))
    hash_to_paths = defaultdict(list)

    for path in image_paths:
        h = file_hash(path)
        hash_to_paths[h].append(path)

    # Only keep hashes that appear once
    unique_paths = [
        paths[0]
        for paths in hash_to_paths.values()
        if len(paths) == 1
    ]

    return unique_paths


def load_images_from_paths(paths_tensor,target_tensor, channels, ratio=1.0,batch_size=64):

    if ratio < 1.0:
        paths_tensor = paths_tensor[:int(ratio * len(paths_tensor))]
        target_tensor = target_tensor[:int(ratio * len(target_tensor))]
    def parse_image(path, target):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=channels, expand_animations=False)
        label= tf.one_hot(target, 13)
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
        return img, label

    # Create dataset from DataFrame columns (paths and age_bin)
    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, target_tensor))
    dataset = dataset.map(parse_image)
    dataset = dataset.batch(batch_size)
    return dataset


def build_image_dataframe(paths):
    data = []

    for path in paths:
        # Extract the age from the parent folder name
        age_str = path.parent.name
        try:
            age = int(age_str)
            data.append({
                "path": str(path),
                "age": age
            })
        except ValueError:
            
            return ValueError

    return pd.DataFrame(data)


def split_data(df):
    """
    Splits the input DataFrame into training, validation, and test sets
    based on image paths (no image data is loaded).

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    num_samples = len(df)
    indices = np.random.permutation(num_samples)

    test_split = int(0.1 * num_samples)
    test_idx = indices[:test_split]
    remaining_idx = indices[test_split:]

    val_split = int(0.2 * len(remaining_idx))
    val_idx = remaining_idx[:val_split]
    train_idx = remaining_idx[val_split:]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def bin_ages(df):
    bins = [
        (1, 2, "Infants (1–2)"),
        (3, 5, "Toddlers (3–5)"),
        (6, 8, "Children (6–8)"),
        (9, 12, "Pre-teens (9–12)"),
        (13, 17, "Teens (13–17)"),
        (18, 24, "Young Adults (18–24)"),
        (25, 34, "Adults (25–34)"),
        (35, 44, "Mid Adults (35–44)"),
        (45, 54, "Mature Adults (45–54)"),
        (55, 64, "Older Adults (55–64)"),
        (65, 74, "Seniors (65–74)"),
        (75, 84, "Elderly (75–84)"),
        (85, 150, "Oldest (85+)"),
    ]

    def assign_bin_info(age):
        for i, (low, high, label) in enumerate(bins):
            if low <= age <= high:
                return i, label
        return None, None

    df[["age_bin", "age_bin_label"]] = df["age"].apply(
        lambda age: pd.Series(assign_bin_info(age))
    )
    df["age_bin"] = pd.Categorical(df["age_bin"])

    return df

def build_cnn_model(
    channels=3,         # Number of image channels (e.g., 3 for RGB)
    dropout_rate=0,  # Dropout rate applied after each dense layer (set to 0 to disable)
    task="regression",       # "regression" or "classification"
    num_classes=None,        # Required if task == "classification"
    num_conv_layers=3,       # Number of convolutional blocks
    conv_filters=None,       # List of filter sizes for each conv block; if None, defaults to increasing powers of 2
    kernel_size=3,           # Kernel size for all conv layers
    activation="relu",       # Activation function for conv and dense layers
    num_dense_layers=1,      # Number of fully connected (dense) layers after the conv blocks
    dense_units=None,        # List of unit counts for dense layers; if None, defaults to 128 per dense layer
    output_activation='softmax'  # Activation function for output layer

):
    # Set default filters if none provided
    if conv_filters is None:
        conv_filters = [32 * (2 ** i) for i in range(num_conv_layers)]
    # Set default dense units if none provided
    if dense_units is None:
        dense_units = [128] * num_dense_layers

    inputs = layers.Input(shape=(200,200,channels))
    x = inputs


    # Build convolutional blocks (keep the rest of your code)
    for i in range(num_conv_layers):
        x = layers.Conv2D(conv_filters[i], kernel_size, padding="same", activation=activation)(x)
        x = layers.BatchNormalization()(x)
        
    # Flatten feature maps
    # TODO: Add GlobalAveragePooling2D layer instead of Flatten with option from contructor
    x = layers.GlobalAveragePooling2D()(x)

    # Add fully connected (dense) layers
    for units in dense_units:
        x = layers.Dense(units, activation=activation)(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    # Final output layer: regression uses a single linear neuron;
    # classification uses a softmax output with num_classes neurons.
    if task == "regression":
        outputs = layers.Dense(1, activation="linear")(x)
    elif task == "classification":
        if num_classes is None:
            raise ValueError("num_classes must be provided for classification task.")
        outputs = layers.Dense(num_classes, activation=output_activation)(x)
    else:
        raise ValueError("task must be either 'regression' or 'classification'.")

    return models.Model(inputs=inputs, outputs=outputs)




def build_sequential_cnn_model(
    channels=3,         # Number of image channels (e.g., 3 for RGB)
    dropout_rate=0,     # Dropout rate applied after each dense layer (set to 0 to disable)
    task="regression",  # "regression" or "classification"
    num_classes=None,   # Required if task == "classification"
    num_conv_layers=3,  # Number of convolutional blocks
    conv_filters=None,  # List of filter sizes for each conv block; if None, defaults to increasing powers of 2
    kernel_size=3,      # Kernel size for all conv layers
    activation="relu",  # Activation function for conv and dense layers
    num_dense_layers=1, # Number of fully connected (dense) layers after the conv blocks
    dense_units=None,   # List of unit counts for dense layers; if None, defaults to 128 per dense layer
    output_activation='softmax'  # Activation function for output layer
):
    # Set default filters if none provided
    if conv_filters is None:
        conv_filters = [32 * (2 ** i) for i in range(num_conv_layers)]
    # Set default dense units if none provided
    if dense_units is None:
        dense_units = [128] * num_dense_layers

    model = models.Sequential()
    # Define the input shape in the first layer
    model.add(layers.Input(shape=(200, 200, channels)))

    # Build convolutional blocks
    for i in range(num_conv_layers):
        model.add(layers.Conv2D(conv_filters[i], kernel_size, padding="same", activation=activation))
        model.add(layers.BatchNormalization())

    # Global Average Pooling instead of Flatten
    model.add(layers.GlobalAveragePooling2D())

    # Add fully connected (dense) layers
    for units in dense_units:
        model.add(layers.Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Final output layer
    if task == "regression":
        model.add(layers.Dense(1, activation="linear"))
    elif task == "classification":
        if num_classes is None:
            raise ValueError("num_classes must be provided for classification task.")
        model.add(layers.Dense(num_classes, activation=output_activation))
    else:
        raise ValueError("task must be either 'regression' or 'classification'.")

    return model

def build_model_from_config(config):
    conv_filters = [config['base_filters'] * (2 ** i)
                    for i in range(config['num_conv_layers'])]
    # Build the model
    model = build_cnn_model(
        task=config['task'],
        channels=config['channels'],
        num_classes=config['num_classes'],
        num_conv_layers=config['num_conv_layers'],
        conv_filters=conv_filters,
        kernel_size=config['kernel_size'],
        activation=config['activation'],
        use_pooling=config.get('use_pooling', False),  # From enforced rule
        num_dense_layers=config['num_dense_layers'],
        dense_units=[config['dense_units']] * config['num_dense_layers'],
        dropout_rate=config['dropout_rate'],
        pool_size=config['pool_size'], # Fixed for simplicity
        output_activation=config['output_activation']
    )

    return model
