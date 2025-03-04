# ev0x Deployment Guide

This guide provides instructions for deploying the ev0x system to Google Cloud Confidential VMs with Trusted Execution Environment (TEE) support.

## Requirements

- Google Cloud Platform account with appropriate permissions
- Google Cloud SDK installed and configured
- Docker installed for local development and testing
- Access to Google Container Registry or Artifact Registry

## TEE Environment Specifications

The ev0x system runs within a Trusted Execution Environment (TEE) on Google Cloud Confidential VMs with the following specifications:

- **CPU Types**: Intel Trust Domain Extensions (TDX) or AMD Secure Encrypted Virtualization (SEV)
- **Attestation**: vTPM attestations required
- **Machine Type**: n2d-standard-2 (unless technical requirements dictate otherwise)
- **Optional Acceleration**: NVIDIA H100 support for GPU acceleration

## Deployment Architecture

The deployment consists of:

1. Confidential VM instances running the ev0x container
2. Cloud Storage for persistent data
3. Cloud Monitoring for observability
4. BigQuery for dataset access
5. Optional integration with Flare's TeeV1Verifier and TeeV1Interface smart contracts

## Deploying to Google Cloud Confidential VMs

### 1. Build and Push the Docker Image

```bash
# Clone the repository
git clone https://github.com/your-org/ev0x.git
cd ev0x

# Build the Docker image
docker build -t gcr.io/your-project-id/ev0x:latest .

# Push to Google Container Registry
docker push gcr.io/your-project-id/ev0x:latest
```

### 2. Create a Confidential VM Instance

```bash
# Create a Confidential VM with AMD SEV
gcloud compute instances create ev0x-instance \
--machine-type=n2d-standard-2 \
--confidential-compute \
--maintenance-policy=

