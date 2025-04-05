import hashlib
from PIL import Image
import pandas as pd
from keras import layers, models
import tensorflow as tf
import numpy as np
from pathlib import Path
from collections import defaultdict
from tensorflow.keras import layers, models

import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow.keras.callbacks import Callback,EarlyStopping
from tensorflow.keras import layers, models, regularizers

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
    # Define rotation angles: positive for anticlockwise, negative for clockwise
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

        # Augmentations on the original image (rotations)
        for angle in rotation_angles:
            rotated = img.rotate(angle, expand=True)
            new_filename = f"{name}_rot{angle}{ext}"
            new_path = folder / new_filename
            rotated.save(new_path)
            new_paths.append(new_path)

        # Create mirrored image and save
        mirrored = img.transpose(Image.FLIP_LEFT_RIGHT)
        mirror_filename = f"{name}_mirror{ext}"
        mirror_path = folder / mirror_filename
        mirrored.save(mirror_path)
        new_paths.append(mirror_path)

        # Apply rotations on the mirrored image
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

def load_images_from_paths(paths_tensor,target_tensor, channels, ratio=1.0,batch_size=256,class_count=13,normalize=True,task="classification"):
    # label = tf.one_hot(label, num_classes)
    if ratio < 1.0:
        paths_tensor = paths_tensor[:int(ratio * len(paths_tensor))]
        target_tensor = target_tensor[:int(ratio * len(target_tensor))]

    debug_var="rando"
    def parse_image(path, target):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=channels, expand_animations=False)
        img = tf.image.resize(img, [200, 200])    
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

    # Create dataset from DataFrame columns (paths and age_bin)
    dataset = tf.data.Dataset.from_tensor_slices((paths_tensor, target_tensor))
    dataset = dataset.map(parse_image if task=='classification' else parse_image_reg)
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
    channels=3,         # Number of image channels (e.g., 3 for RGB)
    dropout_rate=0.5,     # Dropout rate applied after each dense layer (set to 0 to disable)
    task="classification",  # "regression" or "classification"
    num_classes=13,   # Required if task == "classification"
    conv_filters=None,  # List of filter sizes for each conv block; if None, defaults to increasing powers of 2
    kernel_size=3,      # Kernel size for all conv layers
    activation="relu",  # Activation function for conv and dense layers
    dense_units=[128],   # List of unit counts for dense layers; if None, defaults to 128 per dense layer
    output_activation='softmax',  # Activation function for output layer
    batch_norm=False,
    batch_norm_dense=False,
    use_skip=False,
    l2_reg=0.0

):
    # Define model input
    inputs = layers.Input(shape=(200, 200, channels))
    x = inputs

    for filters in conv_filters:
        # Save the input to the block for skip connection
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
        outputs = layers.Dense(1, activation="linear",kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(x)
    elif task == "classification":
        outputs = layers.Dense(num_classes, activation=output_activation,kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(x)
    else:
        raise ValueError("task must be either 'regression' or 'classification'.")

    model = models.Model(inputs, outputs)
    return model



def build_sequential_cnn_model(
    channels=3,         # Number of image channels (e.g., 3 for RGB)
    dropout_rate=0.5,     # Dropout rate applied after each dense layer (set to 0 to disable)
    task="classification",  # "regression" or "classification"
    num_classes=13,   # Required if task == "classification"
    conv_filters=None,  # List of filter sizes for each conv block; if None, defaults to increasing powers of 2
    kernel_size=3,      # Kernel size for all conv layers
    activation="relu",  # Activation function for conv and dense layers
    dense_units=[128],   # List of unit counts for dense layers; if None, defaults to 128 per dense layer
    output_activation='softmax',  # Activation function for output layer
    batch_norm=False,
    batch_norm_dense=False,
):

    model = models.Sequential()
    # Define the input shape in the first layer
    model.add(layers.Input(shape=(200, 200, channels)))

    # Build convolutional blocks
    for i in range(len(conv_filters)):
        model.add(layers.Conv2D(conv_filters[i], kernel_size, activation=activation))
        if batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.AveragePooling2D(pool_size=(2,2)))

    # Global Average Pooling instead of Flatten
    model.add(layers.GlobalAveragePooling2D())

    # Add fully connected (dense) layers
    for units in dense_units:
        model.add(layers.Dense(units, activation=activation))
        if batch_norm_dense:
            model.add(layers.BatchNormalization())
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

