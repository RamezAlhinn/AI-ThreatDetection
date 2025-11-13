"""
Storage layer for predictions and labeled data.

Provides simple CSV-based persistence with easy path to SQLite if needed.
Stores all predictions for quality monitoring and continuous improvement.
"""

import os
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd


class PredictionStore:
    """
    Storage for threat detection predictions.

    Uses CSV format for simplicity and portability.
    Can be easily swapped for SQLite or other backends.
    """

    def __init__(self, storage_path: str = "data/predictions.csv"):
        """
        Initialize the prediction store.

        Args:
            storage_path: Path to CSV file for storing predictions
        """
        self.storage_path = storage_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(storage_path):
            self._initialize_storage()

    def _initialize_storage(self):
        """Create the CSV file with appropriate headers."""
        headers = [
            "id",
            "timestamp",
            "log",
            "prediction",
            "confidence",
            "explanation",
            "recommended_action",
            "human_label"
        ]

        with open(self.storage_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

    def save_prediction(self, result: Dict[str, Any], human_label: Optional[str] = None) -> int:
        """
        Save a single prediction to storage.

        Args:
            result: Prediction result dictionary
            human_label: Optional ground truth label for evaluation

        Returns:
            ID of the saved prediction
        """
        # Read existing data to get next ID
        existing_data = self.load_all_predictions()
        next_id = len(existing_data) + 1

        # Prepare row
        row = {
            "id": next_id,
            "timestamp": result.get("timestamp", datetime.utcnow().isoformat()),
            "log": result.get("log", ""),
            "prediction": result.get("prediction", ""),
            "confidence": result.get("confidence", 0.0),
            "explanation": result.get("explanation", ""),
            "recommended_action": result.get("recommended_action", ""),
            "human_label": human_label or ""
        }

        # Append to CSV
        with open(self.storage_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        return next_id

    def save_predictions_batch(self, results: List[Dict[str, Any]]) -> List[int]:
        """
        Save multiple predictions to storage.

        Args:
            results: List of prediction result dictionaries

        Returns:
            List of IDs for saved predictions
        """
        ids = []
        for result in results:
            pred_id = self.save_prediction(result)
            ids.append(pred_id)
        return ids

    def load_all_predictions(self) -> pd.DataFrame:
        """
        Load all stored predictions.

        Returns:
            DataFrame with all predictions
        """
        if not os.path.exists(self.storage_path):
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.storage_path)
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def load_labeled_data(self, labeled_data_path: str = "data/labeled_logs.csv") -> pd.DataFrame:
        """
        Load labeled dataset for evaluation.

        Args:
            labeled_data_path: Path to CSV with ground truth labels

        Returns:
            DataFrame with labeled data
        """
        if not os.path.exists(labeled_data_path):
            raise FileNotFoundError(f"Labeled data not found at {labeled_data_path}")

        df = pd.read_csv(labeled_data_path)
        return df

    def clear_predictions(self):
        """Clear all stored predictions (for testing/reset)."""
        self._initialize_storage()


def create_store(storage_path: Optional[str] = None) -> PredictionStore:
    """
    Create and return a prediction store instance.

    Args:
        storage_path: Optional custom storage path

    Returns:
        PredictionStore instance
    """
    from config import settings
    path = storage_path or settings.storage_path
    return PredictionStore(storage_path=path)
