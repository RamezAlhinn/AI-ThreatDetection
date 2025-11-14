"""
LLM Client abstraction for threat detection.

Provides two modes:
1. Mock Mode: Deterministic responses based on pattern matching (no API calls)
2. Real Mode: Calls Hugging Face Inference API with configured model

This abstraction allows reliable demos and easy framework switching.
"""

import re
import json
import requests
from typing import Optional, Dict, Any
from datetime import datetime

from config import settings, is_mock_mode, has_api_key


class LLMClient:
    """
    Abstraction layer for LLM inference.

    Supports two implementations:
    - MockLLM: Pattern-based deterministic responses
    - RealLLM: Hugging Face Inference API calls
    """

    def __init__(self, use_mock: Optional[bool] = None):
        """
        Initialize LLM client.

        Args:
            use_mock: Override config setting for mock mode (None = use config)
        """
        self.use_mock = use_mock if use_mock is not None else is_mock_mode()

        if not self.use_mock and not has_api_key():
            print("⚠️  Warning: Real mode requested but no API key found. Falling back to mock mode.")
            self.use_mock = True

    def classify_log(self, log_entry: str) -> Dict[str, Any]:
        """
        Classify a security log entry as benign, suspicious, or malicious.

        Args:
            log_entry: The log line to analyze

        Returns:
            Dictionary with keys:
            - prediction: "benign", "suspicious", or "malicious"
            - confidence: float between 0 and 1
            - explanation: str describing the reasoning
            - recommended_action: str suggesting next steps
        """
        if self.use_mock:
            return self._mock_classify(log_entry)
        else:
            return self._real_classify(log_entry)

    def _mock_classify(self, log_entry: str) -> Dict[str, Any]:
        """
        Mock classification using pattern matching.

        This enables offline demos and deterministic behavior for testing.
        """
        log_lower = log_entry.lower()

        # Pattern matching for malicious indicators
        malicious_patterns = [
            (r"sql.*injection", "SQL injection attempt detected"),
            (r"select.*from.*where.*'1'='1'", "SQL injection pattern in query"),
            (r"privilege.?escalation", "Unauthorized privilege escalation"),
            (r"chmod.*777.*shadow", "Critical file permission modification"),
            (r"port.?scan", "Network port scanning activity"),
            (r"code.?execution", "Code execution attempt"),
            (r"buffer.?overflow", "Buffer overflow attack pattern"),
            (r"directory.?traversal|\.\.\/", "Directory traversal attempt"),
            (r"system\(.*\$", "Command injection pattern"),
        ]

        # Pattern matching for suspicious indicators
        suspicious_patterns = [
            (r"failed.*attempts?=[3-9]|attempts?=1[0-9]", "Multiple failed login attempts"),
            (r"config.?change", "Configuration file modification"),
            (r"user.?creation.*backdoor", "Suspicious user account creation"),
            (r"unauthorized", "Unauthorized access attempt"),
            (r"failed.*password.*attempts?=2", "Repeated authentication failures"),
        ]

        # Check for malicious patterns
        for pattern, reason in malicious_patterns:
            if re.search(pattern, log_lower):
                return {
                    "prediction": "malicious",
                    "confidence": 0.92,
                    "explanation": f"{reason}. This is a high-severity security event requiring immediate attention.",
                    "recommended_action": "URGENT: Block source IP, isolate affected systems, initiate incident response protocol, preserve logs for forensic analysis."
                }

        # Check for suspicious patterns
        for pattern, reason in suspicious_patterns:
            if re.search(pattern, log_lower):
                return {
                    "prediction": "suspicious",
                    "confidence": 0.78,
                    "explanation": f"{reason}. This activity warrants further investigation to rule out malicious intent.",
                    "recommended_action": "INVESTIGATE: Review associated logs, check user activity history, monitor for escalation, consider alerting security team."
                }

        # Default to benign for normal activity
        return {
            "prediction": "benign",
            "confidence": 0.95,
            "explanation": "Standard operational activity with no malicious indicators detected. Log entry shows normal user behavior patterns.",
            "recommended_action": "NONE: Continue monitoring. No action required."
        }

    def _real_classify(self, log_entry: str) -> Dict[str, Any]:
        """
        Real classification using Hugging Face Inference API.

        Calls a free-tier LLM to analyze the log entry.
        """
        # Construct the prompt for the LLM
        prompt = self._build_prompt(log_entry)

        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {settings.hf_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": settings.llm_max_tokens,
                "temperature": settings.llm_temperature,
                "return_full_text": False
            }
        }

        try:
            # Make the API call
            response = requests.post(
                settings.hf_api_url,
                headers=headers,
                json=payload,
                timeout=settings.llm_timeout
            )

            # Check for errors
            if response.status_code == 503:
                # Model is loading, fall back to mock
                print("⚠️  Model is loading on HF servers. Using mock response.")
                return self._mock_classify(log_entry)

            response.raise_for_status()

            # Parse the response
            result = response.json()

            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            else:
                generated_text = str(result)

            # Parse the LLM output
            return self._parse_llm_response(generated_text, log_entry)

        except requests.exceptions.Timeout:
            print("⚠️  API request timed out. Using mock response.")
            return self._mock_classify(log_entry)

        except requests.exceptions.RequestException as e:
            print(f"⚠️  API request failed: {e}. Using mock response.")
            return self._mock_classify(log_entry)

        except Exception as e:
            print(f"⚠️  Unexpected error: {e}. Using mock response.")
            return self._mock_classify(log_entry)

    def _build_prompt(self, log_entry: str) -> str:
        """
        Build the prompt for LLM-based log classification.

        This uses a structured prompt with clear instructions and output format.
        """
        prompt = f"""You are a cybersecurity expert analyzing system logs for threats.

Classify the following log entry as either "benign", "suspicious", or "malicious".
Provide a brief explanation and recommended action.

Log Entry:
{log_entry}

Respond in this exact format:
PREDICTION: [benign/suspicious/malicious]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [Brief explanation of your reasoning]
RECOMMENDED_ACTION: [What should be done about this event]

Your response:"""

        return prompt

    def _parse_llm_response(self, llm_output: str, log_entry: str) -> Dict[str, Any]:
        """
        Parse the LLM's text response into structured format.

        Falls back to mock classification if parsing fails.
        """
        try:
            # Extract fields using regex
            prediction_match = re.search(r"PREDICTION:\s*(benign|suspicious|malicious)", llm_output, re.IGNORECASE)
            confidence_match = re.search(r"CONFIDENCE:\s*(0?\.\d+|1\.0|1)", llm_output)
            explanation_match = re.search(r"EXPLANATION:\s*(.+?)(?=\nRECOMMENDED_ACTION:|\n\n|$)", llm_output, re.DOTALL)
            action_match = re.search(r"RECOMMENDED_ACTION:\s*(.+?)(?=\n\n|$)", llm_output, re.DOTALL)

            if not prediction_match:
                raise ValueError("Could not parse prediction from LLM response")

            prediction = prediction_match.group(1).lower()
            confidence = float(confidence_match.group(1)) if confidence_match else 0.75
            explanation = explanation_match.group(1).strip() if explanation_match else "LLM analysis completed"
            recommended_action = action_match.group(1).strip() if action_match else "Review manually"

            return {
                "prediction": prediction,
                "confidence": confidence,
                "explanation": explanation,
                "recommended_action": recommended_action
            }

        except Exception as e:
            print(f"⚠️  Failed to parse LLM response: {e}. Using mock classification.")
            return self._mock_classify(log_entry)


# Convenience function for quick access
def create_client(use_mock: Optional[bool] = None) -> LLMClient:
    """Create and return an LLM client instance."""
    return LLMClient(use_mock=use_mock)
