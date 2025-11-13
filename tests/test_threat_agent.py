"""
Unit tests for threat detection agent.

Tests agent logic in mock mode to ensure deterministic behavior.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from threat_agent import ThreatDetectionAgent


class TestThreatDetectionAgent:
    """Test suite for ThreatDetectionAgent class."""
    
    def setup_method(self):
        """Set up test fixtures with mock mode enabled."""
        self.agent = ThreatDetectionAgent(use_mock=True)
    
    def test_analyze_benign_log(self):
        """Test classification of benign log entry."""
        log = "2024-01-15 10:23:41 INFO user=alice action=login ip=192.168.1.100 status=success"
        
        result = self.agent.analyze_log(log)
        
        assert result["prediction"] == "benign"
        assert result["log"] == log
        assert "timestamp" in result
        assert "explanation" in result
        assert "recommended_action" in result
    
    def test_analyze_sql_injection(self):
        """Test classification of SQL injection attempt."""
        log = "ERROR user=unknown action=sql_query query=\"SELECT * FROM users WHERE '1'='1'\" ip=198.51.100.42"
        
        result = self.agent.analyze_log(log)
        
        assert result["prediction"] == "malicious"
        assert "SQL injection" in result["explanation"]
        assert result["confidence"] > 0.8
    
    def test_analyze_privilege_escalation(self):
        """Test classification of privilege escalation."""
        log = "CRITICAL user=root action=privilege_escalation ip=203.0.113.67 command=\"chmod 777 /etc/shadow\""
        
        result = self.agent.analyze_log(log)
        
        assert result["prediction"] == "malicious"
        assert "privilege" in result["explanation"].lower()
    
    def test_analyze_failed_login_attempts(self):
        """Test classification of multiple failed login attempts."""
        log = "WARN user=charlie action=login ip=203.0.113.45 status=failed attempts=5"
        
        result = self.agent.analyze_log(log)
        
        assert result["prediction"] in ["suspicious", "malicious"]
        assert "failed" in result["explanation"].lower() or "attempts" in result["explanation"].lower()
    
    def test_analyze_batch(self):
        """Test batch analysis of multiple logs."""
        logs = [
            "INFO user=alice action=login status=success",
            "ERROR query=\"SELECT * FROM users WHERE '1'='1'\"",
            "CRITICAL action=privilege_escalation"
        ]
        
        results = self.agent.analyze_logs_batch(logs)
        
        assert len(results) == 3
        assert results[0]["prediction"] == "benign"
        assert results[1]["prediction"] == "malicious"
        assert results[2]["prediction"] == "malicious"
    
    def test_summary_stats(self):
        """Test summary statistics calculation."""
        results = [
            {"prediction": "benign", "confidence": 0.9},
            {"prediction": "benign", "confidence": 0.95},
            {"prediction": "malicious", "confidence": 0.85},
        ]
        
        stats = self.agent.get_summary_stats(results)
        
        assert stats["total_logs"] == 3
        assert stats["benign_count"] == 2
        assert stats["malicious_count"] == 1
        assert stats["suspicious_count"] == 0
        assert stats["avg_confidence"] == pytest.approx((0.9 + 0.95 + 0.85) / 3)
    
    def test_summary_stats_empty(self):
        """Test summary statistics with empty results."""
        stats = self.agent.get_summary_stats([])
        
        assert stats["total_logs"] == 0
        assert stats["avg_confidence"] == 0.0
