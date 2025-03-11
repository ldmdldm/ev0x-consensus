#!/bin/bash
set -e

# Script to build the ev0x Docker image and push it to Google Container Registry
# This is a prerequisite for deploying to Confidential Space

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: docker is not installed. Please install it first."
    exit 1
fi

# Set variables
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-verifiable-ai-hackathon}"
IMAGE_NAME="ev0x"
IMAGE_TAG=$(date +%Y%m%d-%H%M%S)
GCR_HOSTNAME="gcr.io"
FULL_IMAGE_NAME="${GCR_HOSTNAME}/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
LATEST_TAG="${GCR_HOSTNAME}/${PROJECT_ID}/${IMAGE_NAME}:latest"

echo "=== Building and pushing Docker image for ev0x ==="
echo "Project ID: ${PROJECT_ID}"
echo "Image: ${FULL_IMAGE_NAME}"

# Authenticate with Google Cloud
echo "=== Authenticating with Google Cloud ==="
gcloud auth configure-docker --quiet

# Build the Docker image
echo "=== Building Docker image ==="
docker build -t ${FULL_IMAGE_NAME} -f Dockerfile .

# Tag the image as latest
echo "=== Tagging image as latest ==="
docker tag ${FULL_IMAGE_NAME} ${LATEST_TAG}

# Push the image to Google Container Registry
echo "=== Pushing image to Google Container Registry ==="
docker push ${FULL_IMAGE_NAME}
docker push ${LATEST_TAG}

echo "=== Successfully built and pushed ${FULL_IMAGE_NAME} ==="
echo "=== Also tagged and pushed as ${LATEST_TAG} ==="

# Print information for deployment
echo ""
echo "=== Deployment Information ==="
echo "To deploy this image to Confidential Space, use the following image reference:"
echo "${FULL_IMAGE_NAME}"
echo "or use the latest tag:"
echo "${LATEST_TAG}"

echo ""
echo "=== Next Steps ==="
echo "1. Use the deploy_tee.sh script to deploy this image to Confidential Space"
echo "2. Ensure your service account has the necessary permissions"
echo ""

exit 0

