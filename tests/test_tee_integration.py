"""Tests for TEE integration with main application."""

import unittest
from unittest.mock import patch, MagicMock
from main import EvolutionaryConsensusSystem

class TestTEEIntegration(unittest.TestCase):
    """Test cases for TEE integration with main application."""

    @patch('src.tee.attestation.TEEAttestationManager.verify_environment')
    @patch('src.tee.attestation.TEEAttestationManager.get_tee_type')
    @patch('src.tee.attestation.TEEAttestationManager.get_attestation')
    def test_tee_initialization(self, mock_get_attestation, mock_get_tee_type, mock_verify_environment):
        """Test TEE initialization in main application."""
        # Mock successful TEE environment
        mock_verify_environment.return_value = True
        mock_get_tee_type.return_value = "TDX"
        mock_get_attestation.return_value = {"quote": "test_quote"}

        # Initialize system
        system = EvolutionaryConsensusSystem()

        # Verify TEE status
        self.assertTrue(system.tee_status["verified"])
        self.assertEqual(system.tee_status["type"], "TDX")
        self.assertTrue(system.tee_status["attestation_available"])

    @patch('src.tee.attestation.TEEAttestationManager.verify_environment')
    @patch('src.tee.attestation.TEEAttestationManager.get_tee_type')
    @patch('src.tee.attestation.TEEAttestationManager.get_attestation')
    def test_tee_initialization_failure(self, mock_get_attestation, mock_get_tee_type, mock_verify_environment):
        """Test TEE initialization failure handling."""
        # Mock failed TEE environment
        mock_verify_environment.return_value = False
        mock_get_tee_type.return_value = "UNKNOWN"
        mock_get_attestation.return_value = None

        # Initialize system
        system = EvolutionaryConsensusSystem()

        # Verify TEE status reflects failure
        self.assertFalse(system.tee_status["verified"])
        self.assertEqual(system.tee_status["type"], "NONE")
        self.assertFalse(system.tee_status["attestation_available"])

    @patch('src.tee.attestation.TEEAttestationManager.verify_environment')
    @patch('src.tee.attestation.TEEAttestationManager.get_tee_type')
    @patch('src.tee.attestation.TEEAttestationManager.export_attestation')
    def test_attestation_export(self, mock_export_attestation, mock_get_tee_type, mock_verify_environment):
        """Test attestation export during initialization."""
        # Mock successful TEE environment
        mock_verify_environment.return_value = True
        mock_get_tee_type.return_value = "SEV"
        mock_export_attestation.return_value = True

        # Initialize system
        system = EvolutionaryConsensusSystem()

        # Verify export was attempted
        mock_export_attestation.assert_called_once()

    @patch('src.tee.attestation.TEEAttestationManager.verify_environment')
    async def test_tee_status_in_query_response(self, mock_verify_environment):
        """Test TEE status inclusion in query responses."""
        # Mock successful TEE environment
        mock_verify_environment.return_value = True

        # Initialize system
        system = EvolutionaryConsensusSystem()

        # Process a test query
        response = await system.process_query("test query")

        # Verify TEE status is included in response metadata
        self.assertIn("tee_status", response["meta"])
        self.assertIsInstance(response["meta"]["tee_status"], dict)
        self.assertIn("verified", response["meta"]["tee_status"])


if __name__ == '__main__':
    unittest.main()

