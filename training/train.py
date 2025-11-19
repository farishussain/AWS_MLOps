
#!/usr/bin/env python3
"""
Production Training Script for Vertex AI Custom Training
========================================================

This script trains multiple ML models on the Iris dataset and saves the best model
to Google Cloud Storage. Designed to run in Vertex AI Custom Training jobs.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
from pathlib import Path
import logging
import tempfile

# ML libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import GridSearchCV

# Google Cloud
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ML models on Iris dataset')

    # Data arguments
    parser.add_argument('--data-bucket', type=str, required=True,
                       help='GCS bucket containing processed data')
    parser.add_argument('--data-version', type=str, default='latest',
                       help='Version of processed data to use')

    # Training arguments
    parser.add_argument('--models', type=str, default='all',
                       choices=['all', 'sklearn', 'tensorflow'],
                       help='Which models to train')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                       help='Enable hyperparameter tuning')

    # Output arguments
    parser.add_argument('--output-bucket', type=str, required=True,
                       help='GCS bucket for saving trained models')
    parser.add_argument('--model-version', type=str, default=None,
                       help='Version tag for saved models')

    # TensorFlow arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs for TensorFlow model')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for TensorFlow training')

    return parser.parse_args()

def load_data_from_gcs(bucket_name, version='latest'):
    """Load processed data from GCS."""
    logger.info(f"Loading data from gs://{bucket_name}")

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Find version to use
    if version == 'latest':
        # Find latest version
        blobs = bucket.list_blobs(prefix="processed_data/v")
        versions = set()
        for blob in blobs:
            parts = blob.name.split('/')
            if len(parts) >= 2 and parts[1].startswith('v'):
                versions.add(parts[1])

        if not versions:
            raise ValueError("No processed data versions found")
        version = sorted(versions)[-1]

    logger.info(f"Using data version: {version}")

    # Load datasets
    datasets = {}
    for split in ['train', 'validation', 'test']:
        blob_path = f"processed_data/{version}/iris_{split}.npz"
        blob = bucket.blob(blob_path)

        if blob.exists():
            with tempfile.NamedTemporaryFile() as temp_file:
                blob.download_to_filename(temp_file.name)
                with np.load(temp_file.name) as data:
                    datasets[split] = {
                        'X': data['X'],
                        'y': data['y'],
                        'feature_names': data['feature_names'],
                        'target_names': data['target_names']
                    }
            logger.info(f"Loaded {split} data: {datasets[split]['X'].shape}")
        else:
            raise FileNotFoundError(f"Data file not found: gs://{bucket_name}/{blob_path}")

    return datasets, version

def train_sklearn_models(X_train, y_train, X_val, y_val, tune_hyperparameters=False):
    """Train scikit-learn models."""
    logger.info("Training scikit-learn models")

    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm': SVC(random_state=42, probability=True),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'gradient_boosting': GradientBoostingClassifier(random_state=42)
    }

    trained_models = {}
    results = []

    for name, model in models.items():
        logger.info(f"Training {name}")

        if tune_hyperparameters and name == 'random_forest':
            # Hyperparameter tuning for Random Forest
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            }

            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
            )

            # Combine train and validation for hyperparameter tuning
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])

            grid_search.fit(X_combined, y_combined)
            model = grid_search.best_estimator_
            logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
        else:
            model.fit(X_train, y_train)

        # Evaluate
        val_accuracy = accuracy_score(y_val, model.predict(X_val))

        trained_models[name] = model
        results.append({
            'model': name,
            'val_accuracy': val_accuracy
        })

        logger.info(f"{name} validation accuracy: {val_accuracy:.4f}")

    return trained_models, results

def train_tensorflow_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=16):
    """Train TensorFlow model."""
    logger.info("Training TensorFlow model")

    # Create model
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(len(np.unique(y_train)), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train with callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=20, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"TensorFlow model validation accuracy: {val_accuracy:.4f}")

    return model, val_accuracy

def save_models_to_gcs(models_dict, tf_model, bucket_name, version, data_version):
    """Save trained models to GCS."""
    logger.info(f"Saving models to gs://{bucket_name}")

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Save sklearn models
    for name, model in models_dict.items():
        with tempfile.NamedTemporaryFile() as temp_file:
            pickle.dump(model, open(temp_file.name, 'wb'))

            blob_path = f"models/v{version}/sklearn/{name}.pkl"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(temp_file.name)

            logger.info(f"Uploaded {name} model to gs://{bucket_name}/{blob_path}")

    # Save TensorFlow model
    if tf_model is not None:
        with tempfile.NamedTemporaryFile(suffix='.keras') as temp_file:
            tf_model.save(temp_file.name)

            blob_path = f"models/v{version}/tensorflow/model.keras"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(temp_file.name)

            logger.info(f"Uploaded TensorFlow model to gs://{bucket_name}/{blob_path}")

    # Save metadata
    metadata = {
        'version': version,
        'training_date': datetime.now().isoformat(),
        'data_version': data_version,
        'models': list(models_dict.keys()) + (['tensorflow'] if tf_model else [])
    }

    blob_path = f"models/v{version}/metadata.json"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(json.dumps(metadata, indent=2))

    logger.info(f"Uploaded metadata to gs://{bucket_name}/{blob_path}")

    return version

def main():
    """Main training function."""
    args = parse_args()

    logger.info("Starting Vertex AI Custom Training Job")
    logger.info(f"Arguments: {vars(args)}")

    # Load data
    datasets, data_version = load_data_from_gcs(args.data_bucket, args.data_version)

    X_train = datasets['train']['X']
    y_train = datasets['train']['y']
    X_val = datasets['validation']['X']
    y_val = datasets['validation']['y']

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")

    # Set model version
    if args.model_version is None:
        model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        model_version = args.model_version

    # Train models
    trained_models = {}
    tf_model = None

    if args.models in ['all', 'sklearn']:
        sklearn_models, sklearn_results = train_sklearn_models(
            X_train, y_train, X_val, y_val, args.tune_hyperparameters
        )
        trained_models.update(sklearn_models)

    if args.models in ['all', 'tensorflow']:
        tf_model, tf_accuracy = train_tensorflow_model(
            X_train, y_train, X_val, y_val, args.epochs, args.batch_size
        )

    # Save models
    saved_version = save_models_to_gcs(
        trained_models, tf_model, args.output_bucket, model_version, data_version
    )

    logger.info(f"Training completed! Models saved with version: {saved_version}")

    # Output for Vertex AI
    print(f"MODEL_VERSION={saved_version}")
    print(f"DATA_VERSION={data_version}")

if __name__ == '__main__':
    main()
