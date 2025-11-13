"""
Core threat detection agent logic.

Orchestrates the threat detection pipeline:
1. Receive log entries
2. Call LLM for classification
3. Structure results with metadata
4. Return actionable threat intelligence
"""

from typing import List, Dict, Any
from datetime import datetime

from .llm_client import create_client


class ThreatDetectionAgent:
    """
    AI-powered threat detection agent.

    Analyzes security logs using LLM inference and produces
    structured threat assessments with explanations.
    """

    def __init__(self, use_mock: bool = None):
        """
        Initialize the threat detection agent.

        Args:
            use_mock: Override config for mock mode (None = use config)
        """
        self.llm_client = create_client(use_mock=use_mock)

    def analyze_log(self, log_entry: str) -> Dict[str, Any]:
        """
        Analyze a single log entry for security threats.

        Args:
            log_entry: The log line to analyze

        Returns:
            Dictionary containing:
            - timestamp: ISO format timestamp of analysis
            - log: Original log entry
            - prediction: Threat classification
            - confidence: Confidence score
            - explanation: Human-readable reasoning
            - recommended_action: Suggested next steps
        """
        # Call LLM for classification
        result = self.llm_client.classify_log(log_entry)

        # Add metadata
        result["timestamp"] = datetime.utcnow().isoformat()
        result["log"] = log_entry

        return result

    def analyze_logs_batch(self, log_entries: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple log entries in batch.

        Args:
            log_entries: List of log lines to analyze

        Returns:
            List of analysis results (one per log entry)
        """
        results = []

        for log_entry in log_entries:
            result = self.analyze_log(log_entry)
            results.append(result)

        return results

    def get_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute summary statistics from a batch of results.

        Args:
            results: List of analysis results

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {
                "total_logs": 0,
                "benign_count": 0,
                "suspicious_count": 0,
                "malicious_count": 0,
                "avg_confidence": 0.0
            }

        total = len(results)
        benign = sum(1 for r in results if r["prediction"] == "benign")
        suspicious = sum(1 for r in results if r["prediction"] == "suspicious")
        malicious = sum(1 for r in results if r["prediction"] == "malicious")
        avg_conf = sum(r["confidence"] for r in results) / total

        return {
            "total_logs": total,
            "benign_count": benign,
            "suspicious_count": suspicious,
            "malicious_count": malicious,
            "benign_pct": (benign / total) * 100,
            "suspicious_pct": (suspicious / total) * 100,
            "malicious_pct": (malicious / total) * 100,
            "avg_confidence": avg_conf
        }
