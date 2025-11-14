"""
Quality monitoring and evaluation metrics.

Computes accuracy, confusion matrix, and extracts interesting examples
for continuous quality monitoring and improvement.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
from collections import defaultdict


class QualityMetrics:
    """
    Compute evaluation metrics for threat detection model.

    Supports:
    - Overall accuracy
    - Per-class precision/recall
    - Confusion matrix
    - False positive/negative extraction
    """

    def __init__(self):
        self.classes = ["benign", "suspicious", "malicious"]

    def compute_accuracy(self, y_true: List[str], y_pred: List[str]) -> float:
        """
        Compute overall accuracy.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Label lists must have same length")

        if len(y_true) == 0:
            return 0.0

        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)

    def compute_confusion_matrix(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Compute confusion matrix.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Nested dictionary: confusion_matrix[true_label][pred_label] = count
        """
        matrix = {true_class: {pred_class: 0 for pred_class in self.classes}
                  for true_class in self.classes}

        for true, pred in zip(y_true, y_pred):
            if true in self.classes and pred in self.classes:
                matrix[true][pred] += 1

        return matrix

    def compute_per_class_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Compute precision, recall, and F1 for each class.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Dictionary mapping class name to metrics dict
        """
        metrics = {}

        for cls in self.classes:
            # True positives, false positives, false negatives
            tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
            fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
            fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)

            # Compute metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(1 for true in y_true if true == cls)
            }

        return metrics

    def extract_mistakes(self,
                        df: pd.DataFrame,
                        true_col: str = "true_label",
                        pred_col: str = "prediction") -> Dict[str, pd.DataFrame]:
        """
        Extract interesting mistakes for review.

        Args:
            df: DataFrame with predictions and ground truth
            true_col: Column name for true labels
            pred_col: Column name for predictions

        Returns:
            Dictionary mapping mistake type to DataFrame of examples
        """
        mistakes = {}

        # False negatives (missed threats)
        # True = malicious, Predicted = benign or suspicious
        fn_malicious = df[
            (df[true_col] == "malicious") &
            (df[pred_col].isin(["benign", "suspicious"]))
        ]
        mistakes["false_negatives_malicious"] = fn_malicious

        # False positives (false alarms)
        # True = benign, Predicted = malicious
        fp_benign = df[
            (df[true_col] == "benign") &
            (df[pred_col] == "malicious")
        ]
        mistakes["false_positives_malicious"] = fp_benign

        # Suspicious misclassifications
        suspicious_errors = df[
            (df[true_col] == "suspicious") &
            (df[pred_col] != "suspicious")
        ]
        mistakes["suspicious_errors"] = suspicious_errors

        # All incorrect predictions
        all_errors = df[df[true_col] != df[pred_col]]
        mistakes["all_errors"] = all_errors

        return mistakes

    def generate_summary_report(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Dictionary with all computed metrics
        """
        accuracy = self.compute_accuracy(y_true, y_pred)
        confusion = self.compute_confusion_matrix(y_true, y_pred)
        per_class = self.compute_per_class_metrics(y_true, y_pred)

        return {
            "overall_accuracy": accuracy,
            "confusion_matrix": confusion,
            "per_class_metrics": per_class,
            "total_samples": len(y_true)
        }


def evaluate_predictions(predictions_df: pd.DataFrame,
                        true_col: str = "true_label",
                        pred_col: str = "prediction") -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """
    Convenience function to evaluate a DataFrame of predictions.

    Args:
        predictions_df: DataFrame with true and predicted labels
        true_col: Column name for ground truth
        pred_col: Column name for predictions

    Returns:
        Tuple of (metrics_summary, mistakes_dict)
    """
    metrics = QualityMetrics()

    y_true = predictions_df[true_col].tolist()
    y_pred = predictions_df[pred_col].tolist()

    summary = metrics.generate_summary_report(y_true, y_pred)
    mistakes = metrics.extract_mistakes(predictions_df, true_col, pred_col)

    return summary, mistakes
