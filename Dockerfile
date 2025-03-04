# ev0x: Evolutionary Model Consensus Mechanism
# Production Dockerfile for TEE-compatible container

# Builder stage
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies for TPM and TEE support
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libtpm2-pkcs11-1 \
    libtpm2-tss-dev \
    tpm2-tools \
    gcc \
    python3-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip tools for efficient dependency management
RUN pip install --no-cache-dir pip==23.1.2 wheel setuptools

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt

# Compile TEE attestation tools if needed
RUN mkdir -p /app/attestation

# Runtime stage
FROM python:3.10-slim AS runtime

# Add security labels
LABEL org.opencontainers.image.source="https://github.com/organization/ev0x"
LABEL org.opencontainers.image.description="Evolutionary Model Consensus Mechanism in TEE"
LABEL org.opencontainers.image.licenses="Proprietary"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libtpm2-pkcs11-1 \
    tpm2-tools \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy wheels from builder
COPY --from=builder /app/wheels /app/wheels

# Install Python packages
RUN pip install --no-cache-dir /app/wheels/*.whl

# Create non-root user for security
RUN useradd -m appuser -u 1000 -s /bin/bash -G tss

# Set up TEE and confidential computing directories
RUN mkdir -p /opt/confidential-space /app/tee_data && \
    chown -R appuser:appuser /opt/confidential-space /app/tee_data

# Copy application code
COPY --chown=appuser:appuser . .

# Apply security hardening
RUN chmod -R 550 /app/src && \
    chmod -R 770 /app/tee_data && \
    find /app -type f -name "*.py" -exec chmod 440 {} \; && \
    find /app/src -type f -name "*.py" -exec chmod 550 {} \; && \
    chmod 550 /app/src/api/server.py

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TPM_INTERFACE_TYPE=dev \
    ATTESTATION_VERIFIER_SERVICE="https://confidentialcomputing.googleapis.com" \
    GOOGLE_CLOUD_PROJECT="" \
    PATH="/app:/app/src/scripts:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    HOME="/home/appuser"

# Add persistent volume for model data if needed
VOLUME ["/app/tee_data"]

# Expose port for API server
EXPOSE 8080

# Add healthcheck - checks API server health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set entry point and command
ENTRYPOINT ["python"]
CMD ["src/api/server.py", "--config", "/app/config/production.json"]
