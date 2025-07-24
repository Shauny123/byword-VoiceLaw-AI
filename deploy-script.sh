#!/bin/bash
# deploy.sh
# Complete deployment script for Legal AI Intake System
# Supports development, staging, and production environments

set -euo pipefail

# ===================================================================
# CONFIGURATION & VARIABLES
# ===================================================================

# Script metadata
SCRIPT_VERSION="1.0.0"
SCRIPT_NAME="Legal AI Intake Deployment"
PROJECT_NAME="legal-ai-intake"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Environment configuration
ENVIRONMENT="${ENVIRONMENT:-development}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${PROJECT_ROOT}/deploy.log"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"

# Default configuration
DEFAULT_API_PORT=8000
DEFAULT_NGINX_PORT=80
DEFAULT_NGINX_SSL_PORT=443
DEFAULT_POSTGRES_PORT=5432
DEFAULT_REDIS_PORT=6379
DEFAULT_GRAFANA_PORT=3001
DEFAULT_PROMETHEUS_PORT=9090

# ===================================================================
# UTILITY FUNCTIONS
# ===================================================================

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

# Progress bar function
show_progress() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    local temp
    
    while ps -p $pid > /dev/null 2>&1; do
        temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}

# Wait for service to be ready
wait_for_service() {
    local url=$1
    local timeout=${2:-60}
    local interval=${3:-2}
    local elapsed=0
    
    info "Waiting for service at $url to be ready..."
    
    while [ $elapsed -lt $timeout ]; do
        if curl -sSf "$url" >/dev/null 2>&1; then
            success "Service at $url is ready!"
            return 0
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
        printf "."
    done
    
    error "Service at $url failed to start within $timeout seconds"
}

# ===================================================================
# BANNER & HELP
# ===================================================================

show_banner() {
    cat << "EOF"
 ╔═══════════════════════════════════════════════════════════════════╗
 ║                                                                   ║
 ║    🏛️  LEGAL AI INTAKE SYSTEM DEPLOYMENT  ⚖️                     ║
 ║                                                                   ║
 ║    🤖 Hollywood-Quality Lip Sync + PodGPT RAG Integration        ║
 ║    🌍 30+ Languages • 5.5B+ People Served                       ║
 ║    🚀 Production-Ready • Infinitely Scalable                     ║
 ║                                                                   ║
 ╚═══════════════════════════════════════════════════════════════════╝
EOF
}

show_help() {
    cat << EOF

Usage: $0 [OPTIONS] [COMMAND]

COMMANDS:
    dev         Start development environment
    staging     Deploy to staging environment  
    production  Deploy to production environment
    scale       Scale services up
    down        Stop all services
    logs        View service logs
    status      Check service status
    backup      Create database backup
    restore     Restore from backup
    test        Run test suite
    clean       Clean up containers and volumes
    help        Show this help message

OPTIONS:
    -e, --env ENV       Environment (development|staging|production)
    -p, --profile PROF  Docker compose profile
    -s, --scale NUM     Number of API instances to scale to
    -f, --force         Force rebuild containers
    -v, --verbose       Verbose output
    -q, --quiet         Quiet mode
    --no-ssl            Disable SSL in production
    --skip-deps         Skip dependency checks
    --backup-dir DIR    Custom backup directory

EXAMPLES:
    $0 dev                          # Start development environment
    $0 production --scale 5         # Deploy production with 5 API instances
    $0 staging --force              # Force rebuild staging environment
    $0 logs api                     # View API service logs
    $0 backup --backup-dir ./backups # Create backup in custom directory

ENVIRONMENT VARIABLES:
    ENVIRONMENT         Target environment (development|staging|production)
    API_PORT           API service port (default: 8000)
    NGINX_PORT         Nginx HTTP port (default: 80)
    NGINX_SSL_PORT     Nginx HTTPS port (default: 443)
    POSTGRES_PASSWORD  Database password
    OPENAI_API_KEY     OpenAI API key for Whisper
    NVIDIA_API_KEY     NVIDIA API key for Flamingo 3

EOF
}

# ===================================================================
# PRE-FLIGHT CHECKS
# ===================================================================

check_prerequisites() {
    info "Checking prerequisites..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command_exists "$cmd"; then
            error "$cmd is required but not installed. Please install $cmd first."
        fi
    done
    
    # Check Docker is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running. Please start Docker first."
    fi
    
    # Check Docker Compose version
    local compose_version=$(docker-compose --version | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n1)
    if [[ $(echo "$compose_version" | cut -d. -f1) -lt 2 ]]; then
        warn "Docker Compose version $compose_version detected. Version 2.0+ is recommended."
    fi
    
    # Check available disk space (minimum 10GB)
    local available_space=$(df . | tail -1 | awk '{print $4}')
    local required_space=10485760  # 10GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        warn "Less than 10GB disk space available. Deployment may fail."
    fi
    
    # Check available memory (minimum 4GB)
    local available_memory=$(free -m | grep '^Mem:' | awk '{print $7}')
    if [ "$available_memory" -lt 4096 ]; then
        warn "Less than 4GB RAM available. Performance may be degraded."
    fi
    
    success "Prerequisites check completed"
}

check_ports() {
    info "Checking port availability..."
    
    local ports=(
        "${API_PORT:-$DEFAULT_API_PORT}"
        "${NGINX_PORT:-$DEFAULT_NGINX_PORT}"
        "${POSTGRES_PORT:-$DEFAULT_POSTGRES_PORT}"
        "${REDIS_PORT:-$DEFAULT_REDIS_PORT}"
    )
    
    for port in "${ports[@]}"; do
        if ! check_port "$port"; then
            error "Port $port is already in use. Please free the port or change configuration."
        fi
    done
    
    success "All required ports are available"
}

# ===================================================================
# ENVIRONMENT SETUP
# ===================================================================

setup_environment() {
    local env=$1
    
    info "Setting up $env environment..."
    
    # Create necessary directories
    mkdir -p logs backups data/podgpt data/legal_rag monitoring/prometheus monitoring/grafana/dashboards monitoring/grafana/datasources nginx/ssl
    
    # Set permissions
    chmod 755 logs backups data
    chmod 600 nginx/ssl/* 2>/dev/null || true
    
    # Generate SSL certificates for development
    if [[ "$env" == "development" ]] && [[ ! -f "nginx/ssl/nginx-selfsigned.crt" ]]; then
        info "Generating self-signed SSL certificates for development..."
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/nginx-selfsigned.key \
            -out nginx/ssl/nginx-selfsigned.crt \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" 2>/dev/null || warn "Failed to generate SSL certificates"
    fi
    
    # Create environment file if it doesn't exist
    if [[ ! -f ".env" ]]; then
        info "Creating environment configuration..."
        cp .env.example .env 2>/dev/null || {
            cat > .env << EOF
# Legal AI Intake System Configuration
ENVIRONMENT=$env
API_PORT=${API_PORT:-$DEFAULT_API_PORT}
NGINX_PORT=${NGINX_PORT:-$DEFAULT_NGINX_PORT}
NGINX_SSL_PORT=${NGINX_SSL_PORT:-$DEFAULT_NGINX_SSL_PORT}
POSTGRES_PORT=${POSTGRES_PORT:-$DEFAULT_POSTGRES_PORT}
REDIS_PORT=${REDIS_PORT:-$DEFAULT_REDIS_PORT}
GRAFANA_PORT=${GRAFANA_PORT:-$DEFAULT_GRAFANA_PORT}
PROMETHEUS_PORT=${PROMETHEUS_PORT:-$DEFAULT_PROMETHEUS_PORT}

# Database
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-$(openssl rand -hex 16)}

# API Keys (set these in production)
OPENAI_API_KEY=${OPENAI_API_KEY:-}
NVIDIA_API_KEY=${NVIDIA_API_KEY:-}

# Monitoring
GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-$(openssl rand -hex 8)}

EOF
        }
    fi
    
    # Load environment variables
    source .env
    
    success "Environment setup completed for $env"
}

create_monitoring_config() {
    info "Creating monitoring configuration..."
    
    # Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'legal-ai-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:8081']
    metrics_path: '/health'
    scrape_interval: 30s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 30s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s

EOF

    # Grafana datasource configuration
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true

EOF

    success "Monitoring configuration created"
}

# ===================================================================
# DEPLOYMENT FUNCTIONS
# ===================================================================

deploy_development() {
    info "Deploying development environment..."
    
    setup_environment "development"
    create_monitoring_config
    
    # Build and start development services
    info "Building development containers..."
    docker-compose build --no-cache api-dev || error "Failed to build development containers"
    
    info "Starting development services..."
    docker-compose --profile dev up -d
    
    # Wait for services to be ready
    wait_for_service "http://localhost:${API_PORT:-$DEFAULT_API_PORT}/api/health" 120
    
    success "Development environment deployed successfully!"
    
    # Show access information
    cat << EOF

🎉 Development Environment Ready!

📍 Access URLs:
   • Frontend:    http://localhost:3000
   • API Docs:    http://localhost:${API_PORT:-$DEFAULT_API_PORT}/docs
   • Health:      http://localhost:${API_PORT:-$DEFAULT_API_PORT}/api/health
   • Monitoring:  http://localhost:${GRAFANA_PORT:-$DEFAULT_GRAFANA_PORT}

🔧 Development Commands:
   • View logs:   docker-compose logs -f api-dev
   • Shell:       docker-compose exec api-dev bash
   • Tests:       $0 test
   • Stop:        $0 down

EOF
}

deploy_staging() {
    info "Deploying staging environment..."
    
    setup_environment "staging"
    create_monitoring_config
    
    # Build production containers
    info "Building production containers..."
    docker-compose build --no-cache || error "Failed to build containers"
    
    # Start staging services
    info "Starting staging services..."
    docker-compose up -d
    
    # Run database migrations
    info "Running database migrations..."
    docker-compose --profile migrate up --abort-on-container-exit
    
    # Wait for services
    wait_for_service "http://localhost:${API_PORT:-$DEFAULT_API_PORT}/api/health" 180
    
    success "Staging environment deployed successfully!"
    
    show_deployment_info "staging"
}

deploy_production() {
    info "Deploying production environment..."
    
    # Additional production checks
    check_production_requirements
    
    setup_environment "production"
    create_monitoring_config
    
    # Build production containers with optimizations
    info "Building optimized production containers..."
    docker-compose build --no-cache --compress || error "Failed to build production containers"
    
    # Start production services
    info "Starting production services..."
    docker-compose up -d
    
    # Run database migrations
    info "Running database migrations..."
    docker-compose --profile migrate up --abort-on-container-exit
    
    # Scale services if specified
    if [[ -n "${SCALE_COUNT:-}" ]]; then
        scale_services "$SCALE_COUNT"
    fi
    
    # Wait for all services
    wait_for_service "http://localhost:${API_PORT:-$DEFAULT_API_PORT}/api/health" 300
    wait_for_service "http://localhost:${NGINX_PORT:-$DEFAULT_NGINX_PORT}/health" 60
    
    # Run health checks
    run_health_checks
    
    success "Production environment deployed successfully!"
    
    show_deployment_info "production"
}

check_production_requirements() {
    info "Checking production requirements..."
    
    # Check environment variables
    local required_vars=("POSTGRES_PASSWORD" "OPENAI_API_KEY")
    for var in
