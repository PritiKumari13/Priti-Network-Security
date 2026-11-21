
import os
import sys
from urllib.parse import urlparse

import mlflow
from dotenv import load_dotenv

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)

from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)


# --------------------------------------------------
# MLflow Configuration
# --------------------------------------------------

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

if os.getenv("MLFLOW_TRACKING_USERNAME"):
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv(
        "MLFLOW_TRACKING_USERNAME"
    )

if os.getenv("MLFLOW_TRACKING_PASSWORD"):
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv(
        "MLFLOW_TRACKING_PASSWORD"
    )


# --------------------------------------------------
# Model Trainer Class
# --------------------------------------------------

class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    # --------------------------------------------------
    # MLflow Tracking
    # --------------------------------------------------

    def track_mlflow(
        self,
        best_model,
        classificationmetric,
        dataset_type
    ):
        try:
            tracking_url_type_store = urlparse(
                mlflow.get_tracking_uri()
            ).scheme

            run_name = f"{dataset_type}_evaluation"

            with mlflow.start_run(run_name=run_name):

                metrics = {
                    "accuracy": classificationmetric.accuracy,

                    "macro_precision": classificationmetric.macro_precision_score,
                    "macro_recall": classificationmetric.macro_recall_score,
                    "macro_f1_score": classificationmetric.macro_f1_score,

                    # Class 0: Phishing
                    "phishing_precision": classificationmetric.phishing_precision_score,
                    "phishing_recall": classificationmetric.phishing_recall_score,
                    "phishing_f1_score": classificationmetric.phishing_f1_score,

                    # Class 1: Legitimate
                    "legitimate_precision": classificationmetric.legitimate_precision_score,
                    "legitimate_recall": classificationmetric.legitimate_recall_score,
                    "legitimate_f1_score": classificationmetric.legitimate_f1_score,

                    # Confusion matrix
                    "actual_phishing_predicted_phishing":
                        classificationmetric.actual_phishing_predicted_phishing,

                    "actual_phishing_predicted_legitimate":
                        classificationmetric.actual_phishing_predicted_legitimate,

                    "actual_legitimate_predicted_phishing":
                        classificationmetric.actual_legitimate_predicted_phishing,

                    "actual_legitimate_predicted_legitimate":
                        classificationmetric.actual_legitimate_predicted_legitimate,
                }

                mlflow.log_metrics(metrics)

                mlflow.set_tag("dataset_split", dataset_type)
                mlflow.set_tag("class_0", "Phishing")
                mlflow.set_tag("class_1", "Legitimate")
                mlflow.set_tag(
                    "model_type",
                    type(best_model).__name__
                )

                confusion_matrix_data = {
                    "labels": ["Phishing (0)", "Legitimate (1)"],
                    "matrix": [
                        [
                            classificationmetric.actual_phishing_predicted_phishing,
                            classificationmetric.actual_phishing_predicted_legitimate
                        ],
                        [
                            classificationmetric.actual_legitimate_predicted_phishing,
                            classificationmetric.actual_legitimate_predicted_legitimate
                        ]
                    ],
                    "orientation": "Rows = actual, columns = predicted"
                }

                mlflow.log_dict(
                    confusion_matrix_data,
                    "evaluation/confusion_matrix.json"
                )

                # Register model only during training evaluation
                if dataset_type == "train":

                    trusted_types = [
                        "sklearn.tree._tree.Tree"
                    ]

                    if tracking_url_type_store != "file":

                        mlflow.sklearn.log_model(
                            sk_model=best_model,
                            name="model",
                            registered_model_name="NetworkSecurityModel",
                            skops_trusted_types=trusted_types,
                        )

                    else:

                        mlflow.sklearn.log_model(
                            sk_model=best_model,
                            name="model",
                            skops_trusted_types=trusted_types,
                        )

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    # --------------------------------------------------
    # Model Training and Evaluation
    # --------------------------------------------------

    def train_model(self, X_train, y_train, x_test, y_test):

        try:

            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {

                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"]
                },

                "Random Forest": {
                    "n_estimators": [8, 16, 32, 128, 256]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },

                "Logistic Regression": {},

                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            # Evaluate candidate models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            if not model_report:
                raise ValueError("Model evaluation returned no results.")

            # Select best model
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name}")
            logging.info(f"Best model score: {best_model_score}")

            # Training predictions
            y_train_pred = best_model.predict(X_train)

            classification_train_metric = get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred,
            )

            logging.info(
                f"Training metrics: {classification_train_metric}"
            )

            self.track_mlflow(
                best_model,
                classification_train_metric,
                dataset_type="train",
            )

            # Testing predictions
            y_test_pred = best_model.predict(x_test)

            classification_test_metric = get_classification_score(
                y_true=y_test,
                y_pred=y_test_pred,
            )

            logging.info(
                f"Testing metrics: {classification_test_metric}"
            )

            self.track_mlflow(
                best_model,
                classification_test_metric,
                dataset_type="test",
            )

            # Load preprocessing object
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            # Save trained model
            model_dir_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )

            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=network_model,
            )

            # Save final estimator
            os.makedirs("final_model", exist_ok=True)

            save_object(
                file_path="final_model/model.pkl",
                obj=best_model
            )

            # Create training artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )

            logging.info(
                f"Model trainer artifact: {model_trainer_artifact}"
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    # --------------------------------------------------
    # Initiate Model Trainer Pipeline
    # --------------------------------------------------

    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:

            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )

            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            model_trainer_artifact = self.train_model(
                x_train,
                y_train,
                x_test,
                y_test,
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
