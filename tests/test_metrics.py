"""
Unit tests for metrics module.

Tests accuracy calculation, confusion matrix, and mistake extraction.
"""

import pytest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from metrics import QualityMetrics, evaluate_predictions


class TestQualityMetrics:
    """Test suite for QualityMetrics class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = QualityMetrics()
    
    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        y_true = ["benign", "suspicious", "malicious"]
        y_pred = ["benign", "suspicious", "malicious"]
        
        accuracy = self.metrics.compute_accuracy(y_true, y_pred)
        assert accuracy == 1.0
    
    def test_accuracy_half(self):
        """Test accuracy with 50% correct predictions."""
        y_true = ["benign", "benign", "malicious", "malicious"]
        y_pred = ["benign", "suspicious", "malicious", "benign"]
        
        accuracy = self.metrics.compute_accuracy(y_true, y_pred)
        assert accuracy == 0.5
    
    def test_accuracy_empty(self):
        """Test accuracy with empty lists."""
        y_true = []
        y_pred = []
        
        accuracy = self.metrics.compute_accuracy(y_true, y_pred)
        assert accuracy == 0.0
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        y_true = ["benign", "benign", "malicious", "suspicious"]
        y_pred = ["benign", "suspicious", "malicious", "suspicious"]
        
        confusion = self.metrics.compute_confusion_matrix(y_true, y_pred)
        
        # Check structure
        assert "benign" in confusion
        assert "suspicious" in confusion["benign"]
        
        # Check counts
        assert confusion["benign"]["benign"] == 1  # True benign → Pred benign
        assert confusion["benign"]["suspicious"] == 1  # True benign → Pred suspicious
        assert confusion["malicious"]["malicious"] == 1  # True malicious → Pred malicious
    
    def test_per_class_metrics(self):
        """Test per-class precision, recall, F1."""
        y_true = ["benign", "benign", "malicious", "malicious"]
        y_pred = ["benign", "benign", "malicious", "benign"]
        
        metrics = self.metrics.compute_per_class_metrics(y_true, y_pred)
        
        # Benign: 2 TP, 0 FP, 0 FN → Precision=1.0, Recall=1.0
        assert metrics["benign"]["precision"] == 1.0
        assert metrics["benign"]["recall"] == 1.0
        
        # Malicious: 1 TP, 0 FP, 1 FN → Precision=1.0, Recall=0.5
        assert metrics["malicious"]["precision"] == 1.0
        assert metrics["malicious"]["recall"] == 0.5
    
    def test_extract_mistakes(self):
        """Test mistake extraction from DataFrame."""
        data = {
            "log": ["log1", "log2", "log3"],
            "true_label": ["malicious", "benign", "suspicious"],
            "prediction": ["benign", "malicious", "benign"]
        }
        df = pd.DataFrame(data)
        
        mistakes = self.metrics.extract_mistakes(df)
        
        # Should catch false negative (missed malicious)
        assert len(mistakes["false_negatives_malicious"]) == 1
        
        # Should catch false positive (benign → malicious)
        assert len(mistakes["false_positives_malicious"]) == 1
        
        # Total errors
        assert len(mistakes["all_errors"]) == 3  # All predictions wrong
    
    def test_summary_report(self):
        """Test comprehensive summary report generation."""
        y_true = ["benign", "suspicious", "malicious"]
        y_pred = ["benign", "suspicious", "malicious"]
        
        report = self.metrics.generate_summary_report(y_true, y_pred)
        
        assert "overall_accuracy" in report
        assert "confusion_matrix" in report
        assert "per_class_metrics" in report
        assert "total_samples" in report
        
        assert report["overall_accuracy"] == 1.0
        assert report["total_samples"] == 3
