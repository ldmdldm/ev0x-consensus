"""Tests for TEE Attestation functionality."""

import unittest
from unittest.mock import patch, MagicMock
from src.tee.attestation import TEEAttestationManager
import json
from pathlib import Path
import requests
from requests.exceptions import Timeout, RequestException

class TestTEEAttestation(unittest.TestCase):
    """Test cases for TEE Attestation Manager."""

    def setUp(self):
        """Set up test cases."""
        self.tee_manager = TEEAttestationManager()

    @patch('requests.get')
    def test_verify_environment(self, mock_get):
        """Test environment verification."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        self.assertTrue(self.tee_manager.verify_environment())
        
        # Mock failed response
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        self.assertFalse(self.tee_manager.verify_environment())

    @patch('requests.get')
    def test_get_tee_type(self, mock_get):
        """Test TEE type detection."""
        # Mock TDX response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ["TDX"]
        mock_get.return_value = mock_response
        
        self.assertEqual(self.tee_manager.get_tee_type(), "TDX")
        
        # Mock SEV response
        mock_response.json.return_value = ["SEV"]
        mock_get.return_value = mock_response
        
        self.assertEqual(self.tee_manager.get_tee_type(), "SEV")
        
        # Mock unknown response
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        self.assertEqual(self.tee_manager.get_tee_type(), "UNKNOWN")

    @patch('requests.get')
    def test_get_attestation(self, mock_get):
        """Test attestation retrieval."""
        test_attestation = {"quote": "test_quote", "signature": "test_signature"}
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = test_attestation
        mock_get.return_value = mock_response
        
        attestation = self.tee_manager.get_attestation(force_refresh=True)
        self.assertEqual(attestation, test_attestation)
        
        # Test caching
        mock_get.reset_mock()
        cached_attestation = self.tee_manager.get_attestation(force_refresh=False)
        self.assertEqual(cached_attestation, test_attestation)
        mock_get.assert_not_called()

    @patch('requests.post')
    def test_get_vtpm_quote(self, mock_post):
        """Test vTPM quote retrieval."""
        test_quote = {"pcr": "test_pcr", "signature": "test_signature"}
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = test_quote
        mock_post.return_value = mock_response
        
        quote = self.tee_manager.get_vtpm_quote()
        self.assertEqual(quote, test_quote)

    @patch('src.tee.attestation.TEEAttestationManager.get_attestation')
    def test_export_attestation(self, mock_get_attestation):
        """Test attestation export."""
        test_attestation = {"quote": "test_quote", "signature": "test_signature"}
        mock_get_attestation.return_value = test_attestation
        
        # Test successful export
        test_path = "/tmp/test_attestation.json"
        result = self.tee_manager.export_attestation(test_path)
        self.assertTrue(result)
        
        # Verify file contents
        with open(test_path, 'r') as f:
            saved_attestation = json.load(f)
        self.assertEqual(saved_attestation, test_attestation)
        
        # Clean up test file
        # Clean up test file
        Path(test_path).unlink()

    @patch('requests.get')
    def test_verify_environment_timeout(self, mock_get):
        """Test environment verification with timeout."""
        mock_get.side_effect = Timeout("Connection timed out")
        self.assertFalse(self.tee_manager.verify_environment())

    @patch('requests.get')
    def test_verify_environment_connection_error(self, mock_get):
        """Test environment verification with connection error."""
        mock_get.side_effect = RequestException("Connection failed")
        self.assertFalse(self.tee_manager.verify_environment())

    @patch('requests.get')
    def test_get_attestation_invalid_response(self, mock_get):
        """Test attestation retrieval with invalid response."""
        # Mock invalid JSON response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        
        attestation = self.tee_manager.get_attestation(force_refresh=True)
        self.assertIsNone(attestation)

    @patch('requests.post')
    def test_get_vtpm_quote_non_200_response(self, mock_post):
        """Test vTPM quote retrieval with non-200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_post.return_value = mock_response
        
        quote = self.tee_manager.get_vtpm_quote()
        self.assertIsNone(quote)

    def test_export_attestation_invalid_path(self):
        """Test attestation export with invalid path."""
        # Test with invalid directory path
        invalid_path = "/nonexistent/directory/attestation.json"
        result = self.tee_manager.export_attestation(invalid_path)
        self.assertFalse(result)

    @patch('requests.get')
    def test_get_tee_type_connection_error(self, mock_get):
        """Test TEE type detection with connection error."""
        mock_get.side_effect = RequestException("Connection failed")
        self.assertEqual(self.tee_manager.get_tee_type(), "UNKNOWN")

    @patch('requests.get')
    def test_multiple_attestation_refreshes(self, mock_get):
        """Test multiple attestation refreshes."""
        # Mock different responses for each call
        attestation1 = {"quote": "quote1", "signature": "sig1"}
        attestation2 = {"quote": "quote2", "signature": "sig2"}
        
        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = attestation1
        
        mock_response2 = MagicMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = attestation2
        
        mock_get.side_effect = [mock_response1, mock_response2]
        
        # First attestation
        first_attestation = self.tee_manager.get_attestation(force_refresh=True)
        self.assertEqual(first_attestation, attestation1)
        
        # Second attestation with force refresh
        second_attestation = self.tee_manager.get_attestation(force_refresh=True)
        self.assertEqual(second_attestation, attestation2)

if __name__ == '__main__':
    unittest.main()
