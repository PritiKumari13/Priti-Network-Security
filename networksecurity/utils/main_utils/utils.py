
import os
import sys
import pickle

import yaml
import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


# --------------------------------------------------
# Read YAML File
# --------------------------------------------------

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def convert_to_python_types(value):
    """Convert NumPy objects into standard Python objects."""

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return [
            convert_to_python_types(item)
            for item in value.tolist()
        ]

    if isinstance(value, dict):
        return {
            key: convert_to_python_types(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            convert_to_python_types(item)
            for item in value
        ]

    return value
# --------------------------------------------------
# Write YAML File
# --------------------------------------------------

def write_yaml_file(
    file_path: str,
    content: object,
    replace: bool = False
) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        directory = os.path.dirname(file_path) or "."
        os.makedirs(directory, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(
    convert_to_python_types(content),
    file,
    sort_keys=False
)

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# --------------------------------------------------
# Save NumPy Array
# --------------------------------------------------

def save_numpy_array_data(
    file_path: str,
    array: np.ndarray
) -> None:
    try:
        directory = os.path.dirname(file_path) or "."
        os.makedirs(directory, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# --------------------------------------------------
# Save Python Object
# --------------------------------------------------

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered save_object method")

        directory = os.path.dirname(file_path) or "."
        os.makedirs(directory, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Exited save_object method")

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# --------------------------------------------------
# Load Python Object
# --------------------------------------------------

def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File does not exist: {file_path}"
            )

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# --------------------------------------------------
# Load NumPy Array
# --------------------------------------------------

def load_numpy_array_data(file_path: str) -> np.ndarray:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File does not exist: {file_path}"
            )

        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e


# --------------------------------------------------
# Evaluate and Select Machine Learning Models
# --------------------------------------------------

def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    param
):
    try:
        if not models:
            raise ValueError("No models provided for evaluation.")

        report = {}
        best_models = {}

        # Stratified cross-validation on training data only
        stratified_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=42
        )

        for model_name, model in models.items():

            logging.info(
                f"Starting hyperparameter tuning: {model_name}"
            )

            param_grid = param.get(model_name, {})

            # Grid Search using training data only
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="f1_macro",
                cv=stratified_cv,
                n_jobs=-1,
                refit=True,
                error_score="raise"
            )

            grid_search.fit(X_train, y_train)

            # Retrieve best estimator
            best_model = grid_search.best_estimator_

            best_models[model_name] = best_model

            # Store cross-validation macro F1 score
            report[model_name] = float(
                grid_search.best_score_
            )

            logging.info(
                f"Best parameters for {model_name}: "
                f"{grid_search.best_params_}"
            )

            logging.info(
                f"Cross-validation macro F1 for {model_name}: "
                f"{grid_search.best_score_}"
            )

        # Update original model dictionary with tuned estimators
        models.update(best_models)

        # X_test and y_test are intentionally not used here.
        # They are reserved for independent final evaluation
        # inside ModelTrainer.train_model().

        logging.info(
            f"Model evaluation report: {report}"
        )

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
