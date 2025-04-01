# optuna_worker.py
import sys
import os
import tensorflow as tf
import optuna
import pandas as pd

gpu_id = int(sys.argv[1])  # move this above all TF logic
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[gpu_id], True)

import shutil

# === IMPORT YOUR OWN MODULES HERE ===
# These must be implemented:
# - build_model_from_config
# - load_images_from_paths
# - train_filenames_tensor, train_labels_tensor, val_filenames_tensor, val_labels_tensor
from common_utils import (
    build_model_from_config,
    load_images_from_paths,
    split_data

)

# Converting the filenames and target class labels into lists for augmented train and test datasets.
image_paths_csv = pd.read_csv('./processed_data/image_paths.csv')
paths_train_df, paths_val_df, paths_test_df = split_data(image_paths_csv)


train_filenames_list = list(paths_train_df['path'])
train_labels_list = list(paths_train_df['age_bin'])

train_filenames_tensor = tf.constant(train_filenames_list)
train_labels_tensor = tf.constant(train_labels_list)

val_filenames_list = list(paths_val_df['path'])
val_labels_list = list(paths_val_df['age_bin'])

val_filenames_tensor = tf.constant(val_filenames_list)
val_labels_tensor = tf.constant(val_labels_list)

def objective_wrapper(gpu_id):
    def objective(trial):
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)

        config = {
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'channels': trial.suggest_categorical('channels', [1, 3]),
            'num_conv_layers': trial.suggest_categorical('num_conv_layers', [3, 4, 5]),
            'base_filters': trial.suggest_categorical('base_filters', [32, 64, 128]),
            'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
            'activation': trial.suggest_categorical('activation', ['relu', 'swish']),
            'num_dense_layers': trial.suggest_categorical('num_dense_layers', [1, 2, 3]),
            'dense_units': trial.suggest_categorical('dense_units', [128, 256]),
            'num_classes': 13,
            'dropout_rate': trial.suggest_categorical('dropout_rate', [0.3, 0.5, 0.7]),
            'output_activation': trial.suggest_categorical('output_activation', ['softmax', 'sigmoid']),
            'pool_size': trial.suggest_categorical('pool_size', [2, 3]),
            'task': 'classification',
        }

        model_name = "_".join([
            f"{config['channels']}ch",
            f"conv{config['num_conv_layers']}",
            f"k{config['kernel_size']}",
            config['activation'],
            f"dense{config['num_dense_layers']}x{config['dense_units']}",
            f"drop{config['dropout_rate']}",
            f"out_{config['output_activation']}"
        ])
        trial.set_user_attr('model_name', model_name)

        model = build_model_from_config(config)
        model.compile(optimizer='lion', loss='categorical_crossentropy', metrics=['accuracy'])

        train_dataset = load_images_from_paths(train_filenames_tensor, train_labels_tensor,
                                               config['channels'], 1, config['batch_size'])
        val_dataset = load_images_from_paths(val_filenames_tensor, val_labels_tensor,
                                             config['channels'], 1, config['batch_size'])

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=50,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
        )

        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_name}_trial_{trial.number}.keras")
        model.save(model_path)

        trial.set_user_attr('model_path', model_path)
        val_acc = max(history.history['val_accuracy'])
        print(f"Trial {trial.number} | Model: {model_name} | Val Acc: {val_acc:.4f}")
        return val_acc

    return objective

if __name__ == "__main__":
    gpu_id = int(sys.argv[1])
    n_trials = int(sys.argv[2])
    study_name = sys.argv[3]
    storage_url = sys.argv[4]

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.optimize(objective_wrapper(gpu_id), n_trials=n_trials)
