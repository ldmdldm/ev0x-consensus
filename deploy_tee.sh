#!/bin/bash
set -e

# Colors for better output readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}    Ev0x Deployment to GCP Confidential Space with SIA${NC}"
echo -e "${BLUE}==================================================${NC}"

# Configuration variables
PROJECT_ID="verifiable-ai-hackathon"
ZONE="us-central1-a"
INSTANCE_NAME="ev0x-confidential-space"
MACHINE_TYPE="n2d-standard-4"  # n2d series supports AMD SEV
VM_IMAGE="confidential-space-250101"  # Latest Confidential Space image
VM_IMAGE_PROJECT="confidential-space-images"
BOOT_DISK_SIZE="50GB"
SERVICE_ACCOUNT_NAME="confidential-sa"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed. Please install it first.${NC}"
    echo "Visit https://cloud.google.com/sdk/docs/install for installation instructions."
    exit 1
fi

# Check if user is authenticated
GCLOUD_AUTH=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null)
if [ -z "$GCLOUD_AUTH" ]; then
    echo -e "${RED}Error: You are not authenticated with gcloud.${NC}"
    echo "Please run 'gcloud auth login' first."
    exit 1
fi

# Check if .env file exists and contains OPEN_ROUTER_API_KEY
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found in the current directory.${NC}"
    exit 1
fi

if ! grep -q "OPEN_ROUTER_API_KEY" .env; then
    echo -e "${RED}Error: OPEN_ROUTER_API_KEY not found in .env file.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites met!${NC}"

# Set the Google Cloud project
echo -e "${YELLOW}Setting Google Cloud project to $PROJECT_ID...${NC}"
gcloud config set project $PROJECT_ID

# Check if user has adequate permissions in the GCP project
echo -e "${YELLOW}Checking user permissions...${NC}"
if ! gcloud projects describe $PROJECT_ID &>/dev/null; then
    echo -e "${RED}Error: You don't have permission to access project $PROJECT_ID.${NC}"
    exit 1
fi

# Check if user can create VMs in the project
if ! gcloud compute instances list --project=$PROJECT_ID --quiet &>/dev/null; then
    echo -e "${RED}Error: You don't have permission to list compute instances in project $PROJECT_ID.${NC}"
    echo -e "Please ensure you have at least the 'compute.viewer' role in this project."
    exit 1
fi

# Check if the service account exists, create it if it doesn't
echo -e "${YELLOW}Checking if service account exists...${NC}"
if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT --project=$PROJECT_ID &>/dev/null; then
    echo -e "${YELLOW}Service account $SERVICE_ACCOUNT doesn't exist. Creating it...${NC}"
    if ! gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
        --display-name="Ev0x Confidential Space Service Account" \
        --project=$PROJECT_ID; then
        echo -e "${RED}Error: Failed to create service account. Please check your permissions.${NC}"
        echo -e "You need 'iam.serviceAccounts.create' permission (usually part of roles/iam.serviceAccountAdmin)."
        exit 1
    fi
    
    echo -e "${GREEN}✓ Service account created successfully!${NC}"
else
    echo -e "${GREEN}✓ Service account already exists!${NC}"
fi

# Grant necessary roles to the service account for Confidential Space with SIA
echo -e "${YELLOW}Skipping granting roles to service account (assuming it already has necessary permissions)...${NC}"

# List of roles that would typically be needed (for documentation purposes)
REQUIRED_ROLES=(
    "roles/confidentialcomputing.workloadUser"
    "roles/logging.logWriter"
    "roles/monitoring.metricWriter"
    "roles/storage.objectViewer"
)

# For informational purposes only, not actually granting permissions
echo -e "${YELLOW}The service account would normally need these roles:${NC}"
for ROLE in "${REQUIRED_ROLES[@]}"; do
    echo -e "${BLUE}  - $ROLE${NC}"
done

echo -e "${YELLOW}Continuing deployment assuming service account already has necessary permissions...${NC}"

# Create cloud-init.yaml for Confidential Space configuration
echo -e "${YELLOW}Creating cloud-init.yaml configuration for Confidential Space...${NC}"
cat > cloud-init.yaml << 'EOF'
#cloud-config

write_files:
- path: /etc/customer/manifest.json
  permissions: 0644
  owner: root
  content: |
    {
      "containers": [
        {
          "name": "ev0x",
          "image": "gcr.io/${PROJECT_ID}/ev0x:latest",
          "env": [
            "TEE_ENVIRONMENT=1"
          ],
          "volumeMounts": [
            {
              "mountPath": "/etc/ev0x",
              "name": "ev0x-config"
            }
          ]
        }
      ],
      "volumes": [
        {
          "name": "ev0x-config",
          "secret": {
            "secretName": "ev0x-config"
          }
        }
      ]
    }

- path: /etc/customer/tee-attestation.json
  permissions: 0644
  owner: root
  content: |
    {
      "env": {
        "TEE_ATTESTATION": "true",
        "TEE_SIA_ENABLED": "true",
        "OPENROUTER_API_KEY": "${OPEN_ROUTER_API_KEY}"
      }
    }
EOF

# Check if the VM instance already exists
echo -e "${YELLOW}Checking if VM instance already exists...${NC}"
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --quiet &>/dev/null; then
    echo -e "${GREEN}✓ VM instance '$INSTANCE_NAME' already exists! Skipping creation step.${NC}"
    # Get the external IP address of the existing VM for later use
    EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
    # Skip initialization wait for existing VMs
    echo -e "${GREEN}✓ Skipping initialization wait for existing VM.${NC}"
else
    # Create a GCP Confidential Space instance with Sensitive Information Access (SIA)
    echo -e "${YELLOW}Creating Confidential Space with SIA support...${NC}"
    gcloud compute instances create $INSTANCE_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --confidential-compute \
  --maintenance-policy=TERMINATE \
  --image=$VM_IMAGE \
  --image-project=$VM_IMAGE_PROJECT \
  --boot-disk-size=$BOOT_DISK_SIZE \
  --service-account=$SERVICE_ACCOUNT \
  --scopes=cloud-platform \
  --metadata="tee-env-svn=1,tee-containers-manifest-path=gs://${PROJECT_ID}-tee-container/manifest.json" \
  --metadata-from-file=user-data=cloud-init.yaml

    # Wait for VM to be fully initialized - only for newly created VMs
    echo -e "${YELLOW}Waiting for new VM to initialize...${NC}"
    VM_READY=false
    MAX_ATTEMPTS=30
    ATTEMPT=0

    while [ "$VM_READY" = false ] && [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        ATTEMPT=$((ATTEMPT+1))
        echo -e "${YELLOW}Checking VM status (attempt $ATTEMPT/$MAX_ATTEMPTS)...${NC}"
        
        if gcloud compute instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE 2>/dev/null | grep -q "Finished running startup scripts"; then
            VM_READY=true
            echo -e "${GREEN}✓ VM is initialized and ready!${NC}"
        else
            echo -e "${YELLOW}VM still initializing. Waiting 10 seconds...${NC}"
            sleep 10
        fi
    done

    if [ "$VM_READY" = false ]; then
        echo -e "${YELLOW}VM initialization is taking longer than expected. Continuing deployment...${NC}"
    fi
fi

# Create a temporary deployment directory
DEPLOY_DIR=$(mktemp -d)
echo -e "${YELLOW}Created temporary deployment directory: $DEPLOY_DIR${NC}"

# Copy necessary files to the deployment directory
echo -e "${YELLOW}Preparing application files for deployment...${NC}"
rsync -av --exclude=".git" --exclude=".venv" --exclude="__pycache__" ./* $DEPLOY_DIR/

# Create startup script for the VM
cat > $DEPLOY_DIR/startup.sh << 'EOF'
#!/bin/bash
set -e

# Update and install dependencies
apt-get update
apt-get install -y python3-pip python3-venv git

# Set up application directory
mkdir -p /opt/ev0x
cd /opt/ev0x

# Copy files from the temp directory
cp -r /tmp/deploy/* .

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables for TEE
export TEE_ENVIRONMENT=1
echo "TEE_ENVIRONMENT=1" >> /etc/environment

# Create systemd service for ev0x
cat > /etc/systemd/system/ev0x.service << 'EOT'
[Unit]
Description=Ev0x Evolutionary Consensus System
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ev0x
Environment="TEE_ENVIRONMENT=1"
EnvironmentFile=/opt/ev0x/.env
ExecStart=/opt/ev0x/.venv/bin/python /opt/ev0x/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOT

# Enable and start the service
systemctl daemon-reload
systemctl enable ev0x
systemctl start ev0x

echo "Ev0x application deployed and started!"
EOF

# Make the startup script executable
chmod +x $DEPLOY_DIR/startup.sh

# Copy the .env file with the OpenRouter API key
cp .env $DEPLOY_DIR/

# Upload files to the VM
echo -e "${YELLOW}Uploading application files to the VM...${NC}"
gcloud compute scp --recurse $DEPLOY_DIR/* $INSTANCE_NAME:/tmp/deploy --zone=$ZONE

# Execute the startup script on the VM
echo -e "${YELLOW}Executing setup script on the VM...${NC}"
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="sudo bash /tmp/deploy/startup.sh"

# Clean up temporary files
rm -rf $DEPLOY_DIR

# Get the external IP address of the VM
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}    Deployment Completed Successfully!            ${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "Ev0x is now running on a GCP Confidential Space with Sensitive Information Access (SIA) support."
echo -e "VM Instance Name: ${BLUE}$INSTANCE_NAME${NC}"
echo -e "External IP: ${BLUE}$EXTERNAL_IP${NC}"
echo -e "API Endpoint: ${BLUE}http://$EXTERNAL_IP:5000${NC}"
echo -e "\n"
echo -e "To check the status of the application:"
echo -e "${YELLOW}gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command=\"sudo systemctl status ev0x\"${NC}"
echo -e "\n"
echo -e "To view logs:"
echo -e "${YELLOW}gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command=\"sudo journalctl -u ev0x -f\"${NC}"
echo -e "\n"
echo -e "To SSH into the VM:"
echo -e "${YELLOW}gcloud compute ssh $INSTANCE_NAME --zone=$ZONE${NC}"
echo -e "\n"
echo -e "${RED}Note: This VM will incur charges as long as it is running. To delete it:${NC}"
echo -e "${YELLOW}gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE${NC}"

