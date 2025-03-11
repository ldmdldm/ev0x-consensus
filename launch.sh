#!/bin/bash
set -e

# ev0x Launch Script
# This script prepares the environment and launches the ev0x system

echo "==== ev0x System Launch ===="
echo "Starting environment setup..."

# Create necessary directories
mkdir -p logs
mkdir -p data/models
mkdir -p data/outputs

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Setup virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Check for TEE capabilities
echo "Verifying TEE environment..."
if python3 -c "from src.tee.confidential_vm import TEEVerifier; exit(0 if TEEVerifier().verify_environment() else 1)"; then
    echo "TEE environment verified successfully."
else
    echo "Warning: Running without TEE verification. Production deployment requires TEE."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Load production configuration
export EV0X_ENV=production
export EV0X_CONFIG_PATH="$(pwd)/config/production.yml"
echo "Using configuration: $EV0X_CONFIG_PATH"

# Initialize Flare integrations
echo "Initializing Flare ecosystem integrations..."
python3 -c "from src.integrations.flare import initialize_integrations; initialize_integrations()"

# Start the system
echo "Starting ev0x system..."
python3 src/api/server.py &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
echo "Server logs available at logs/server.log"

# Register cleanup for graceful shutdown
trap "echo 'Shutting down...'; kill $SERVER_PID; exit" SIGINT SIGTERM

echo "==== ev0x System Running ===="
echo "Access the dashboard at: http://localhost:5000/dashboard"
echo "API endpoints available at: http://localhost:5000/api/v1"
echo "Press Ctrl+C to stop the server"

# Keep the script running
wait $SERVER_PID

