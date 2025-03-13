"""TEE Attestation module for Confidential Computing."""

import os
import json
import logging
import requests
from typing import Dict, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class TEEAttestationManager:
    """Manages TEE attestation for Confidential Space."""
    
    def __init__(self):
        """Initialize the TEE Attestation Manager."""
        self.tee_endpoint = "http://metadata.google.internal/computeMetadata/v1/instance/confidential-space"
        self.headers = {"Metadata-Flavor": "Google"}
        self._cached_attestation = None
    
    def get_tee_type(self) -> str:
        """
        Determine the type of TEE (TDX or SEV).
        
        Returns:
            str: The TEE type ('TDX', 'SEV', or 'UNKNOWN')
        """
        try:
            response = requests.get(
                f"{self.tee_endpoint}/capabilities",
                headers=self.headers
            )
            if response.status_code == 200:
                capabilities = response.json()
                if "TDX" in capabilities:
                    return "TDX"
                elif "SEV" in capabilities:
                    return "SEV"
        except Exception as e:
            logger.error(f"Error getting TEE type: {e}")
        return "UNKNOWN"
    
    def get_attestation(self, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get TEE attestation from the Confidential Space.
        
        Args:
            force_refresh: Force refresh of cached attestation
        
        Returns:
            Optional[Dict[str, Any]]: Attestation data or None if failed
        """
        if self._cached_attestation and not force_refresh:
            return self._cached_attestation
        
        try:
            response = requests.get(
                f"{self.tee_endpoint}/attestation",
                headers=self.headers
            )
            if response.status_code == 200:
                self._cached_attestation = response.json()
                return self._cached_attestation
        except Exception as e:
            logger.error(f"Error getting attestation: {e}")
        return None
    
    def verify_environment(self) -> bool:
        """
        Verify that we're running in a valid Confidential Space environment.
        
        Returns:
            bool: True if valid Confidential Space environment
        """
        try:
            # Check if we're in a Confidential Space environment
            response = requests.get(
                self.tee_endpoint,
                headers=self.headers
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error verifying environment: {e}")
            return False
    
    def get_vtpm_quote(self) -> Optional[Dict[str, Any]]:
        """
        Get a vTPM quote from the Confidential Space.
        
        Returns:
            Optional[Dict[str, Any]]: vTPM quote data or None if failed
        """
        try:
            response = requests.post(
                f"{self.tee_endpoint}/vtpm/quote",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error getting vTPM quote: {e}")
        return None
    
    def export_attestation(self, output_path: str) -> bool:
        """
        Export attestation data to a file.
        
        Args:
            output_path: Path to save attestation data
        
        Returns:
            bool: True if export successful
        """
        attestation = self.get_attestation(force_refresh=True)
        if attestation:
            try:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(attestation, f, indent=2)
                return True
            except Exception as e:
                logger.error(f"Error exporting attestation: {e}")
        return False

