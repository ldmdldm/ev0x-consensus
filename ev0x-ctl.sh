#!/bin/bash
#
# ev0x-ctl.sh - Comprehensive management script for the ev0x system
#
# This script provides commands for deploying, monitoring, configuring, updating,
# and diagnosing the ev0x system across different environments.

set -e  # Exit immediately if a command exits with a non-zero status

# Constants
VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
CONFIG_DIR="${SCRIPT_DIR}/config"
DEPLOYMENT_DIR="${SCRIPT_DIR}/deployment"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Log file path with timestamp
LOG_FILE="${LOG_DIR}/ev0x-ctl-$(date +%Y%m%d-%H%M%S).log"

# Color definitions for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log messages to both console and log file
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Format based on log level
    case "${level}" in
        "INFO")
            echo -e "${GREEN}[${timestamp}] [INFO] ${message}${NC}" >&2
            ;;
        "WARN")
            echo -e "${YELLOW}[${timestamp}] [WARN] ${message}${NC}" >&2
            ;;
        "ERROR")
            echo -e "${RED}[${timestamp}] [ERROR] ${message}${NC}" >&2
            ;;
        "DEBUG")
            if [[ "${DEBUG}" == "true" ]]; then
                echo -e "${BLUE}[${timestamp}] [DEBUG] ${message}${NC}" >&2
            fi
            ;;
        *)
            echo -e "[${timestamp}] [${level}] ${message}" >&2
            ;;
    esac
    
    # Also log to file
    echo "[${timestamp}] [${level}] ${message}" >> "${LOG_FILE}"
}

# Function to check if a command exists
command_exists() {
    type "$1" &> /dev/null
}

# Function to check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    local missing_prereqs=false
    
    # Check for required commands
    for cmd in docker kubectl gcloud python3 pip3; do
        if ! command_exists "${cmd}"; then
            log "ERROR" "Required command '${cmd}' not found"
            missing_prereqs=true
        else
            log "DEBUG" "Found required command: ${cmd}"
        fi
    done
    
    # Check for configuration directory
    if [[ ! -d "${CONFIG_DIR}" ]]; then
        log "ERROR" "Configuration directory not found: ${CONFIG_DIR}"
        missing_prereqs=true
    fi
    
    if [[ "${missing_prereqs}" == "true" ]]; then
        log "ERROR" "Prerequisites check failed. Please install missing dependencies."
        return 1
    fi
    
    log "INFO" "All prerequisites satisfied."
    return 0
}

# Function to validate environment name
validate_environment() {
    local env="$1"
    
    case "${env}" in
        "dev"|"staging"|"production")
            return 0
            ;;
        *)
            log "ERROR" "Invalid environment: ${env}. Must be one of: dev, staging, production"
            return 1
            ;;
    esac
}

# Function to deploy the system to a specified environment
deploy() {
    local env="$1"
    
    if ! validate_environment "${env}"; then
        return 1
    fi
    
    log "INFO" "Deploying ev0x system to ${env} environment..."
    
    # Load environment-specific configuration
    local config_file="${CONFIG_DIR}/${env}.json"
    if [[ ! -f "${config_file}" ]]; then
        log "ERROR" "Configuration file not found: ${config_file}"
        return 1
    fi
    
    log "INFO" "Using configuration file: ${config_file}"
    
    # Build the Docker image
    log "INFO" "Building Docker image for ${env}..."
    docker build -t "ev0x:${env}-$(date +%Y%m%d)" -f Dockerfile --build-arg ENV="${env}" . || {
        log "ERROR" "Failed to build Docker image"
        return 1
    }
    
    # Deploy to the appropriate environment
    case "${env}" in
        "dev")
            log "INFO" "Deploying to development environment..."
            docker-compose -f "${DEPLOYMENT_DIR}/docker-compose.${env}.yml" up -d || {
                log "ERROR" "Failed to deploy to development environment"
                return 1
            }
            ;;
        "staging"|"production")
            log "INFO" "Deploying to ${env} environment using Kubernetes..."
            
            # Apply Kubernetes configurations
            kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/configmap.yaml" --namespace "${env}" || {
                log "ERROR" "Failed to apply ConfigMap"
                return 1
            }
            
            # Apply environment-specific secrets
            kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/secrets.${env}.yaml" --namespace "${env}" || {
                log "ERROR" "Failed to apply Secrets"
                return 1
            }
            
            # Apply the deployment
            kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/deployment.yaml" --namespace "${env}" || {
                log "ERROR" "Failed to apply Deployment"
                return 1
            }
            
            # Apply the service
            kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/service.yaml" --namespace "${env}" || {
                log "ERROR" "Failed to apply Service"
                return 1
            }
            
            # If production, apply ingress rules
            if [[ "${env}" == "production" ]]; then
                kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/ingress.yaml" --namespace "${env}" || {
                    log "ERROR" "Failed to apply Ingress"
                    return 1
                }
            fi
            ;;
    esac
    
    log "INFO" "Deployment to ${env} completed successfully"
    return 0
}

# Function to monitor the system's status and health
monitor() {
    local env="$1"
    
    if ! validate_environment "${env}"; then
        return 1
    fi
    
    log "INFO" "Monitoring ev0x system in ${env} environment..."
    
    case "${env}" in
        "dev")
            # For dev environment, use Docker commands to check status
            log "INFO" "Checking container status..."
            docker-compose -f "${DEPLOYMENT_DIR}/docker-compose.${env}.yml" ps || {
                log "ERROR" "Failed to get container status"
                return 1
            }
            
            log "INFO" "Checking container logs..."
            docker-compose -f "${DEPLOYMENT_DIR}/docker-compose.${env}.yml" logs --tail=100 || {
                log "ERROR" "Failed to get container logs"
                return 1
            }
            ;;
        "staging"|"production")
            # For staging/production environments, use kubectl
            log "INFO" "Checking pod status..."
            kubectl get pods --namespace "${env}" -l app=ev0x || {
                log "ERROR" "Failed to get pod status"
                return 1
            }
            
            log "INFO" "Checking service status..."
            kubectl get svc --namespace "${env}" -l app=ev0x || {
                log "ERROR" "Failed to get service status"
                return 1
            }
            
            # Get the name of the first pod
            local pod_name=$(kubectl get pods --namespace "${env}" -l app=ev0x -o jsonpath="{.items[0].metadata.name}")
            
            if [[ -n "${pod_name}" ]]; then
                log "INFO" "Checking logs for pod: ${pod_name}"
                kubectl logs "${pod_name}" --namespace "${env}" --tail=100 || {
                    log "ERROR" "Failed to get pod logs"
                    return 1
                }
            else
                log "WARN" "No pods found for the application"
            fi
            ;;
    esac
    
    # If metrics server is available, fetch health metrics
    if command_exists "curl"; then
        log "INFO" "Fetching health metrics..."
        
        local metrics_url
        
        case "${env}" in
            "dev")
                metrics_url="http://localhost:8000/metrics"
                ;;
            "staging")
                metrics_url="http://ev0x-metrics.staging:8000/metrics"
                ;;
            "production")
                metrics_url="http://ev0x-metrics.production:8000/metrics"
                ;;
        esac
        
        curl -s "${metrics_url}" || {
            log "WARN" "Failed to fetch metrics from ${metrics_url}"
        }
    fi
    
    log "INFO" "Monitoring completed"
    return 0
}

# Function to manage model configurations
manage_models() {
    local action="$1"
    local model_name="$2"
    
    log "INFO" "Managing model configurations: ${action} ${model_name}"
    
    case "${action}" in
        "list")
            # List all available models
            log "INFO" "Listing available models..."
            python3 -m src.config.models list || {
                log "ERROR" "Failed to list models"
                return 1
            }
            ;;
        "enable")
            # Enable a specific model
            if [[ -z "${model_name}" ]]; then
                log "ERROR" "Model name is required for 'enable' action"
                return 1
            fi
            
            log "INFO" "Enabling model: ${model_name}"
            python3 -m src.config.models enable "${model_name}" || {
                log "ERROR" "Failed to enable model: ${model_name}"
                return 1
            }
            ;;
        "disable")
            # Disable a specific model
            if [[ -z "${model_name}" ]]; then
                log "ERROR" "Model name is required for 'disable' action"
                return 1
            }
            
            log "INFO" "Disabling model: ${model_name}"
            python3 -m src.config.models disable "${model_name}" || {
                log "ERROR" "Failed to disable model: ${model_name}"
                return 1
            }
            ;;
        "update")
            # Update model configurations
            log "INFO" "Updating model configurations..."
            python3 -m src.config.models update || {
                log "ERROR" "Failed to update model configurations"
                return 1
            }
            ;;
        *)
            log "ERROR" "Invalid model management action: ${action}. Must be one of: list, enable, disable, update"
            return 1
            ;;
    esac
    
    log "INFO" "Model management completed successfully"
    return 0
}

# Function to perform system updates and maintenance
update() {
    local component="$1"
    
    log "INFO" "Performing system updates for component: ${component}"
    
    case "${component}" in
        "all")
            # Update all components
            log "INFO" "Updating all components..."
            
            # Pull latest code
            log "INFO" "Pulling latest code from repository..."
            git pull || {
                log "ERROR" "Failed to pull latest code"
                return 1
            }
            
            # Update dependencies
            log "INFO" "Updating Python dependencies..."
            pip3 install -r requirements.txt --upgrade || {
                log "ERROR" "Failed to update Python dependencies"
                return 1
            }
            
            # Rebuild images
            log "INFO" "Rebuilding Docker images..."
            docker build -t "ev0x:latest" . || {
                log "ERROR" "Failed to rebuild Docker images"
                return 1
            }
            
            log "INFO" "All components updated successfully"
            ;;
        "code")
            # Update code only
            log "INFO" "Updating code only..."
            git pull || {
                log "ERROR" "Failed to pull latest code"
                return 1
            }
            ;;
        "dependencies")
            # Update dependencies only
            log "INFO" "Updating dependencies only..."
            pip3 install -r requirements.txt --upgrade || {
                log "ERROR" "Failed to update dependencies"
                return 1
            }
            ;;
        "images")
            # Update Docker images only
            log "INFO" "Updating Docker images only..."
            docker build -t "ev0x:latest" . || {
                log "ERROR" "Failed to rebuild Docker images"
                return 1
            }
            ;;
        *)
            log "ERROR" "Invalid update component: ${component}. Must be one of: all, code, dependencies, images"
            return 1
            ;;
    esac
    
    log "INFO" "Update completed successfully"
    return 0
}

# Function to run diagnostics and generate reports
diagnose() {
    local report_type="$1"
    local output_file="$2"
    
    # If output file not specified, create one with timestamp
    if [[ -z "${output_file}" ]]; then
        output_file="${SCRIPT_DIR}/reports/diagnostic-$(date +%Y%m%d-%H%M%S).html"
        mkdir -p "${SCRIPT_DIR}/reports"
    fi
    
    log "INFO" "Running diagnostics: ${report_type} > ${output_file}"
    
    case "${report_type}" in
        "health")
            # Run health checks
            log "INFO" "Running health checks..."
            python3 -m src.testing.diagnostics --type health --output "${output_file}" || {
                log "ERROR" "Failed to run health diagnostics"
                return 1
            }
            ;;
        "performance")
            # Run performance tests
            log "INFO" "Running performance tests..."
            python3 -m src.testing.diagnostics --type performance --output "${output_file}" || {
                log "ERROR" "Failed to run performance diagnostics"
                return 1
            }
            ;;
        "security")
            # Run security checks
            log "INFO" "Running security checks..."
            python3 -m src.testing.diagnostics --type security --output "${output_file}" || {
                log "ERROR" "Failed to run security diagnostics"
                return 1
            }
            ;;
        "full")
            # Run comprehensive diagnostics

