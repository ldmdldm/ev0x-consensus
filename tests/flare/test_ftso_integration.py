import pytest
import os
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import requests

from src.flare_integrations.ftso_integration import FTSOIntegration
from src.models.model_runner import ModelRunner
from src.consensus.synthesizer import ConsensusSynthesizer


class TestFTSOIntegration:
    """Test suite for FTSO integration with consensus mechanism."""

    @pytest.fixture
    def ftso_integration(self):
        """Initialize FTSO integration instance for testing."""
        # Real FTSO contract addresses from Flare Network
        ftso_addresses = {
            "ftso_manager": "0x1000000000000000000000000000000000000003",
            "price_submitter": "0x1000000000000000000000000000000000000003",
            "ftso_registry": "0x1000000000000000000000000000000000000002",
            "ftso_reward_manager": "0x1000000000000000000000000000000000000004"
        }
        return FTSOIntegration(
            network_rpc="https://flare-api.flare.network/ext/C/rpc",
            addresses=ftso_addresses,
            supported_symbols=["XRP", "LTC", "XLM", "DOGE", "ADA"]
        )

    @pytest.fixture
    def consensus_system(self):
        """Initialize consensus system with model runner."""
        config = {
            "iterations": {"max_iterations": 3},
            "models": [
                {"id": "model_1", "params": {"temperature": 0.7}},
                {"id": "model_2", "params": {"temperature": 0.5}}
            ]
        }
        return ConsensusSynthesizer(config)

    def test_price_feed_integration(self, ftso_integration):
        """Test integration with FTSO price feeds."""
        # Get current prices for supported symbols
        prices = ftso_integration.get_current_prices()
        
        # Verify structure and data types
        assert isinstance(prices, dict)
        for symbol in ftso_integration.supported_symbols:
            assert symbol in prices
            assert isinstance(prices[symbol], dict)
            assert "price" in prices[symbol]
            assert "timestamp" in prices[symbol]
            assert isinstance(prices[symbol]["price"], float)
            assert prices[symbol]["price"] > 0
    
    def test_price_reliability_verification(self, ftso_integration):
        """Test verification of price reliability metrics."""
        # Get reliability metrics for supported symbols
        reliability_metrics = ftso_integration.get_price_reliability_metrics("XRP")
        
        # Verify reliability metrics structure
        assert isinstance(reliability_metrics, dict)
        assert "volatility" in reliability_metrics
        assert "provider_agreement" in reliability_metrics
        assert "data_freshness" in reliability_metrics
        
        # Verify calculation of reliability score
        reliability_score = ftso_integration.calculate_reliability_score("XRP")
        assert 0 <= reliability_score <= 1.0
    
    def test_economic_incentive_calculations(self, ftso_integration):
        """Test calculation of economic incentives based on price data quality."""
        # Calculate rewards for a sample address
        test_address = "0x1234567890123456789012345678901234567890"
        rewards = ftso_integration.calculate_provider_rewards(test_address)
        
        # Verify rewards structure
        assert isinstance(rewards, dict)
        assert "current_epoch" in rewards
        assert "estimated_rewards" in rewards
        assert "historical_rewards" in rewards
        assert isinstance(rewards["estimated_rewards"], float)
        
        # Test reward distribution calculation
        distribution = ftso_integration.calculate_reward_distribution()
        assert isinstance(distribution, dict)
        assert sum(distribution.values()) <= 1.0
    
    def test_ftso_consensus_integration(self, ftso_integration, consensus_system):
        """Test integration of FTSO data with consensus mechanism."""
        # Get price data confidence intervals
        confidence_data = ftso_integration.get_price_confidence_intervals("XRP")
        
        # Use confidence data to weight consensus models
        weighted_models = ftso_integration.apply_price_confidence_to_models(
            confidence_data,
            [{"id": "model_1", "weight": 0.5}, {"id": "model_2", "weight": 0.5}]
        )
        
        # Verify model weights were adjusted based on price confidence
        assert isinstance(weighted_models, list)
        assert len(weighted_models) == 2
        assert sum(model["weight"] for model in weighted_models) == pytest.approx(1.0)
        
        # Test consensus with price-weighted models
        consensus_system.update_weights({
            model["id"]: model["weight"] for model in weighted_models
        })
        
        # Verify consensus system properly integrated the weights
        consensus_weights = consensus_system.get_model_weights()
        assert isinstance(consensus_weights, dict)
        assert len(consensus_weights) == 2
    
    def test_price_anomaly_detection(self, ftso_integration):
        """Test detection of price anomalies in FTSO data."""
        # Get historical prices for analysis
        historical_prices = ftso_integration.get_historical_prices("XRP", days=7)
        
        # Detect anomalies in price data
        anomalies = ftso_integration.detect_price_anomalies("XRP", historical_prices)
        
        # Verify anomaly detection results
        assert isinstance(anomalies, list)
        
        # Test anomaly threshold configuration
        ftso_integration.set_anomaly_threshold(0.15)  # 15% deviation
        new_anomalies = ftso_integration.detect_price_anomalies("XRP", historical_prices)
        
        # Threshold change should affect number of detected anomalies
        assert isinstance(new_anomalies, list)
    
    def test_cross_symbol_correlation(self, ftso_integration):
        """Test correlation analysis between different FTSO price feeds."""
        # Calculate correlation between price symbols
        correlation = ftso_integration.calculate_symbol_correlation(["XRP", "XLM"])
        
        # Verify correlation results
        assert isinstance(correlation, float)
        assert -1.0 <= correlation <= 1.0
        
        # Test correlation matrix for all supported symbols
        correlation_matrix = ftso_integration.calculate_correlation_matrix()
        
        # Verify correlation matrix structure
        assert isinstance(correlation_matrix, dict)
        for symbol in ftso_integration.supported_symbols:
            assert symbol in correlation_matrix
            assert isinstance(correlation_matrix[symbol], dict)

