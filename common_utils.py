import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import Callback, EarlyStopping
import matplotlib.pyplot as plt
from IPython.display import clear_output


import os
import glob

import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from PIL import Image
import hashlib

class LiveLossPlot(Callback):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.epochs = []
        self.losses = []
        self.val_losses = []

    def on_train_begin(self, logs=None):
        self.epochs = []
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        clear_output(wait=True)
        plt.figure(figsize=(8, 5))
        plt.plot(self.epochs, self.losses, label="Training Loss")
        plt.plot(self.epochs, self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss for model: {self.model_name}")
        plt.legend()
        plt.show()

# workaround to make sure the early stopping does have fair enough steps, but only after 60 epochs
class DelayedEarlyStopping(EarlyStopping):
    # delay = check only after 60 epochs
    def __init__(self, delay=60, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay

    def on_epoch_end(self, epoch, logs=None):
        #
        if epoch >= self.delay:
            super().on_epoch_end(epoch, logs)



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


def augment_images(unique_paths):
    rotation_angles = [20, 40, -20, -40]
    new_paths = []

    for path in unique_paths:
        try:
            img = Image.open(path)
        except Exception as e:
            print(f"Error opening {path}: {e}")
            continue

        folder = path.parent
        name = path.stem
        ext = path.suffix

        for angle in rotation_angles:
            rotated = img.rotate(angle, expand=True)
            new_filename = f"{name}_rot{angle}{ext}"
            new_path = folder / new_filename
            rotated.save(new_path)
            new_paths.append(new_path)

        mirrored = img.transpose(Image.FLIP_LEFT_RIGHT)
        mirror_filename = f"{name}_mirror{ext}"
        mirror_path = folder / mirror_filename
        mirrored.save(mirror_path)
        new_paths.append(mirror_path)

        for angle in rotation_angles:
            rotated_mirrored = mirrored.rotate(angle, expand=True)
            new_filename = f"{name}_mirror_rot{angle}{ext}"
            new_path = folder / new_filename
            rotated_mirrored.save(new_path)
            new_paths.append(new_path)

    return new_paths


def load_tensors_from_paths_csv(paths_csv):
    paths_train_df, paths_val_df, paths_test_df = split_data(paths_csv)

    # Create TensorFlow constants from the DataFrame columns
    train_filenames = tf.constant(list(paths_train_df['path']))
    train_labels = tf.constant(list(paths_train_df['age_bin']))
    train_labels_regr = tf.constant(list(paths_train_df['age']))

    val_filenames = tf.constant(list(paths_val_df['path']))
    val_labels = tf.constant(list(paths_val_df['age_bin']))
    val_labels_regr = tf.constant(list(paths_val_df['age']))

    test_filenames = tf.constant(list(paths_test_df['path']))
    test_labels = tf.constant(list(paths_test_df['age_bin']))
    test_labels_regr = tf.constant(list(paths_test_df['age']))

    # Return a nested dictionary for easy access
    return {
        'train': {'filenames': train_filenames, 'labels': train_labels,'labels_regr':train_labels_regr},
        'val': {'filenames': val_filenames, 'labels': val_labels,'labels_regr':val_labels_regr},
        'test': {'filenames': test_filenames, 'labels': test_labels,'labels_regr':test_labels_regr},
    }

def load_images_from_paths(paths_tensor,target_tensor=None,resize=192, channels=3, ratio=1.0,batch_size=256,class_count=7,task="classification"):

    if target_tensor is None and task!="autoencoder":
        raise ValueError("target_tensor must be provided for classification or regression tasks.")
    if ratio < 1.0:
        paths_tensor = paths_tensor[:int(ratio * len(paths_tensor))]
        target_tensor = target_tensor[:int(ratio * len(target_tensor))]

    debug_var="rando"
    def parse_image(path, target):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=channels, expand_animations=False)
        img = tf.image.resize(img, [resize, resize])
        label= tf.one_hot(target, class_count)
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]

        return img, label

    def parse_image_reg(path,target):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=channels, expand_animations=False)
        img = tf.image.resize(img, [200, 200])
        label = tf.cast(target, tf.float32)  
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
        return img, label

    def parse_image_autoencoder(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=channels, expand_animations=False)
        img = tf.image.resize(img, [resize, resize])
        img = tf.cast(img, tf.float32) / 255.0  # Normalize
        return img, img  # input and output are the same

    if task=="autoencoder":
        dataset = tf.data.Dataset.from_tensor_slices(paths_tensor)
        dataset= dataset.map(parse_image_autoencoder)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, target_tensor))
        dataset = dataset.map(parse_image if task=='classification' else parse_image_reg)
    dataset = dataset.batch(batch_size)
    return dataset


def build_image_dataframe(paths):
    data = []

    for path in paths:
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

def bin_ages_7(df):
    bins = [
        (1, 4, "Infant/Toddler (1–4)"),
        (5, 12, "Children (5–12)"),
        (13, 19, "Teens (13–19)"),
        (20, 34, "Young Adults (20–34)"),
        (35, 54, "Middle Adults (35–54)"),
        (55, 74, "Older Adults (55–74)"),
        (75, 150, "Seniors (75+)"),
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
    channels=3,
    dropout_rate=0.5,
    task="classification",
    num_classes=13,
    conv_filters=None,
    kernel_size=3,
    activation="relu",
    dense_units=[128],
    output_activation='softmax',
    batch_norm=False,
    batch_norm_dense=False,
    use_skip=False,
    l2_reg=0.0

):
    inputs = layers.Input(shape=(200, 200, channels))
    x = inputs

    for filters in conv_filters:
        block_input = x
        x = layers.Conv2D(
            filters, kernel_size, padding="same", activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(x)

        if batch_norm:
            x = layers.BatchNormalization()(x)

        x = layers.AveragePooling2D(pool_size=(2, 2))(x)

        if use_skip:
            # create skip branch: apply pooling to block_input to match spatial dims
            skip = layers.AveragePooling2D(pool_size=(2, 2))(block_input)

            # if channel numbers don't match, adjust with a 1x1 convolution
            if skip.shape[-1] != filters:
                skip = layers.Conv2D(
                    filters, kernel_size=1, padding="same",
                    kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None
                )(skip)

            # add the skip connection
            x = layers.Add()([x, skip])

    x = layers.GlobalAveragePooling2D()(x)

    for units in dense_units:
        x = layers.Dense(
            units, activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(x)
        if batch_norm_dense:
            x = layers.BatchNormalization()(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    if task == "regression":
        outputs = layers.Dense(1, activation=output_activation,kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(x)
    elif task == "classification":
        outputs = layers.Dense(num_classes, activation=output_activation,kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(x)
    else:
        raise ValueError("task must be either 'regression' or 'classification'.")

    model = models.Model(inputs, outputs)
    return model



def get_base_name(file_path):

    file_name = os.path.basename(file_path)
    base, _ = os.path.splitext(file_name)
    # split at underscore if present and return the first part( aka the base name of the file)
    return base.split('_')[0]


def split_blocks(df):
    """
    Returns:
        tuple: (train_block1, val_block1, train_block2, val_block2, test_block)
    """
    df = df.copy()
    df['base_name'] = df['path'].apply(get_base_name)

    unique_bases = df['base_name'].unique()
    # MAKING IT RANDOM
    np.random.shuffle(unique_bases)

    total_groups = len(unique_bases)
    test_count = int(0.1 * total_groups)

    test_bases = unique_bases[:test_count]
    remaining_bases = unique_bases[test_count:]

    #50% of whats left
    mid = int(0.5 * len(remaining_bases))
    block1_bases = remaining_bases[:mid]
    block2_bases = remaining_bases[mid:]

    # train - val split 80-20
    def split_train_val(bases):
        train_count = int(0.8 * len(bases))
        return bases[:train_count], bases[train_count:]

    train_block1_bases, val_block1_bases = split_train_val(block1_bases)
    train_block2_bases, val_block2_bases = split_train_val(block2_bases)

    train_block1 = df[df['base_name'].isin(train_block1_bases)].reset_index(drop=True)
    val_block1 = df[df['base_name'].isin(val_block1_bases)].reset_index(drop=True)
    train_block2 = df[df['base_name'].isin(train_block2_bases)].reset_index(drop=True)
    val_block2 = df[df['base_name'].isin(val_block2_bases)].reset_index(drop=True)
    test_block = df[df['base_name'].isin(test_bases)].reset_index(drop=True)

    return train_block1, val_block1, train_block2, val_block2, test_block


def add_augmentations(train_df, debug=False):
    augmented_rows = []

    for i, row in train_df.iterrows():
        raw_path = row['path']
        directory = os.path.dirname(raw_path)
        base = get_base_name(raw_path)

        pattern = os.path.join(directory, f"{base}_*.png")
        matching_files = glob.glob(pattern)
        if debug:
            print(f"Row {i}: using pattern: {pattern}")
            print(f"  Found: {matching_files}")

        if not matching_files:
            matching_files = [raw_path]

        for aug_path in matching_files:
            new_row = row.copy()
            new_row['path'] = aug_path
            augmented_rows.append(new_row)
        augmented_rows.append(row)
    return pd.DataFrame(augmented_rows)


def build_autoencoder(
        input_shape=(192, 192, 3),
        conv_filters=[32, 64, 128, 256, 512],
        kernel_size=3,
        activation="relu",
        batch_norm=True,
        l2_reg=1e-4,
):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # enoder part
    for i, filters in enumerate(conv_filters):
        x = layers.Conv2D(filters, (kernel_size, kernel_size), padding='same',
                          activation=activation,
                          kernel_regularizer=regularizers.l2(l2_reg))(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        pool_padding = 'same' if i < 2 else 'valid'
        x = layers.AveragePooling2D((2, 2), padding=pool_padding)(x)

    # the latent layer for later extraction
    encoded = layers.AveragePooling2D((2, 2), padding='valid', name='encoded')(x)

    # decoder
    x = encoded
    for i, filters in enumerate(reversed(conv_filters)):
        x = layers.Conv2DTranspose(filters, (kernel_size, kernel_size),
                                   strides=(2, 2), padding='same',
                                   activation=activation,
                                   kernel_regularizer=regularizers.l2(l2_reg))(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)

    outputs = layers.Conv2DTranspose(input_shape[-1], (kernel_size, kernel_size),
                                     strides=(2, 2), padding='same',
                                     activation="sigmoid",
                                     kernel_regularizer=regularizers.l2(l2_reg))(x)

    return models.Model(inputs, outputs)

def build_transfer_model_from_autoencoder(encoder, config, num_classes=13):
    input_tensor = layers.Input(shape=(192, 192, 3))  

    x = encoder(input_tensor, training=False)  
    x = layers.GlobalAveragePooling2D()(x)


    for i, units in enumerate(config["dense_units"]):
        x = layers.Dense(units, activation="relu")(x)
        if config.get("batch_norm_dense", False):
            x = layers.BatchNormalization(name=f"batch_norm_dense_{i}")(x)
        if config.get("dropout_rate", 0) > 0:
            x = layers.Dropout(config["dropout_rate"])(x)

    output = layers.Dense(num_classes, activation="softmax",
                          kernel_regularizer=regularizers.l2(config["l2_reg"]))(x)

    model = models.Model(inputs=input_tensor, outputs=output)
    return model


