"""Persistence helpers for trained model bundles."""

from __future__ import annotations

from datetime import datetime

import joblib


def build_model_metadata(model) -> dict:
    return {
        "model_name": model.model_name,
        "feature_names": model.feature_names,
        "train_time": model.train_time,
        "train_samples": model.X_train.shape[0],
        "feature_count": model.X_train.shape[1],
        "best_params": model.best_params,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_model_bundle(model, model_path):
    metadata = build_model_metadata(model)
    joblib.dump({"model": model.model, "metadata": metadata}, model_path)
    return model_path


def load_model_bundle(model_path):
    return joblib.load(model_path)
