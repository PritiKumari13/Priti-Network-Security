
import sys

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from networksecurity.entity.artifact_entity import (
    ClassificationMetricArtifact,
)

from networksecurity.exception.exception import (
    NetworkSecurityException,
)


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        # Label mapping:
        # 0 = Phishing
        # 1 = Legitimate

        accuracy = accuracy_score(y_true, y_pred)

        # Calculate metrics separately for both classes
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0
        )

        # Calculate macro-average metrics
        macro_precision, macro_recall, macro_f1, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        )

        # Confusion matrix: rows = actual, columns = predicted
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )

        classification_metric = ClassificationMetricArtifact(
            accuracy=float(accuracy),

            macro_precision_score=float(macro_precision),
            macro_recall_score=float(macro_recall),
            macro_f1_score=float(macro_f1),

            phishing_precision_score=float(precision[0]),
            phishing_recall_score=float(recall[0]),
            phishing_f1_score=float(f1[0]),

            legitimate_precision_score=float(precision[1]),
            legitimate_recall_score=float(recall[1]),
            legitimate_f1_score=float(f1[1]),

            actual_phishing_predicted_phishing=int(cm[0, 0]),
            actual_phishing_predicted_legitimate=int(cm[0, 1]),
            actual_legitimate_predicted_phishing=int(cm[1, 0]),
            actual_legitimate_predicted_legitimate=int(cm[1, 1]),
        )

        return classification_metric

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
