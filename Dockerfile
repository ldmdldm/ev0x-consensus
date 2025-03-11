# ev0x: Evolutionary Model Consensus Mechanism
# Simplified Dockerfile for TEE-compatible container with better macOS compatibility
# Build with:
# docker build -t gcr.io/verifiable-ai-hackathon/ev0x:latest .

# Single-stage build for simplified dependencies
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies for TPM and TEE support
RUN apt-get update && apt-get install -y \
    curl \
    libtpm2-pkcs11-1 \
    libtss2-dev \
    tpm2-tools \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies directly
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up TEE and confidential computing directories
RUN mkdir -p /opt/confidential-space /app/tee_data

# Copy application code
COPY . .

# Set simplified environment variables
ENV PYTHONUNBUFFERED=1 \
    TPM_INTERFACE_TYPE=dev \
    ATTESTATION_VERIFIER_SERVICE="https://confidentialcomputing.googleapis.com" \
    PYTHONPATH="/app:${PYTHONPATH}"

# Expose port for API server
EXPOSE 8080

# Add healthcheck - checks API server health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set entry point and command
ENTRYPOINT ["python"]
CMD ["src/api/server.py", "--config", "/app/config/production.yml"]
