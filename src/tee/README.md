# TEE Attestation System

## Overview
The TEE (Trusted Execution Environment) Attestation System provides a robust implementation for verifying and managing confidential computing environments, supporting both Intel TDX and AMD SEV platforms.

## Features
- Automatic detection of TEE environment
- Support for Intel TDX and AMD SEV
- vTPM quote generation and verification
- Attestation data caching and export
- Integration with Google Cloud Confidential Space

## Usage

### Basic Usage
```python
from src.tee.attestation import TEEAttestationManager

# Initialize the TEE manager
tee_manager = TEEAttestationManager()

# Verify the environment
if tee_manager.verify_environment():
    # Get TEE type
    tee_type = tee_manager.get_tee_type()
    print(f"Running in {tee_type} environment")

    # Get attestation
    attestation = tee_manager.get_attestation()
    if attestation:
        print("Attestation successful")

    # Get vTPM quote
    quote = tee_manager.get_vtpm_quote()
    if quote:
        print("vTPM quote retrieved")
```

### Integration with Main Application
The TEE Attestation System is automatically initialized with the EvolutionaryConsensusSystem:
```python
system = EvolutionaryConsensusSystem()

# Check TEE status
tee_status = system.tee_status
if tee_status["verified"]:
    print(f"Running in verified {tee_status['type']} environment")
```

## API Reference

### TEEAttestationManager

#### `verify_environment() -> bool`
Verifies if the current environment is a valid Confidential Space environment.

#### `get_tee_type() -> str`
Returns the type of TEE environment ("TDX", "SEV", or "UNKNOWN").

#### `get_attestation(force_refresh: bool = False) -> Optional[Dict[str, Any]]`
Retrieves attestation data from the Confidential Space environment.

#### `get_vtpm_quote() -> Optional[Dict[str, Any]]`
Retrieves a vTPM quote for remote attestation.

#### `export_attestation(output_path: str) -> bool`
Exports attestation data to a file for external verification.

## Testing
Run the test suite:
```bash
PYTHONPATH=$PYTHONPATH:. python3 -m unittest tests/test_tee_attestation.py -v
```

## Security Considerations
- Always verify attestation signatures
- Keep attestation data secure and encrypted
- Regularly refresh attestation data
- Monitor for environmental changes

