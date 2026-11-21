from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassificationMetricArtifact:
    accuracy: float

    macro_precision_score: float
    macro_recall_score: float
    macro_f1_score: float

    phishing_precision_score: float
    phishing_recall_score: float
    phishing_f1_score: float

    legitimate_precision_score: float
    legitimate_recall_score: float
    legitimate_f1_score: float

    actual_phishing_predicted_phishing: int
    actual_phishing_predicted_legitimate: int
    actual_legitimate_predicted_phishing: int
    actual_legitimate_predicted_legitimate: int

    # Backward compatibility with existing model_trainer.py
    @property
    def f1_score(self):
        return self.legitimate_f1_score

    @property
    def precision_score(self):
        return self.legitimate_precision_score

    @property
    def recall_score(self):
        return self.legitimate_recall_score


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact
