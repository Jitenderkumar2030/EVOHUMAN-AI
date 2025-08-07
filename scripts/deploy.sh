#!/bin/bash

# EvoHuman.AI Deployment Script
# Automated deployment for development, staging, and production environments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-development}"
COMPOSE_FILE=""
ENV_FILE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
EvoHuman.AI Deployment Script

Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENTS:
    development     Deploy for local development (default)
    staging         Deploy for staging environment
    production      Deploy for production environment

OPTIONS:
    --build         Force rebuild of all images
    --no-cache      Build without using cache
    --scale         Scale services (e.g., --scale aice-service=3)
    --health-check  Run health checks after deployment
    --backup        Create backup before deployment (production only)
    --rollback      Rollback to previous version
    --help          Show this help message

EXAMPLES:
    $0 development
    $0 production --build --health-check
    $0 staging --scale aice-service=2 --scale proteus-service=2
    $0 production --backup --health-check

EOF
}

# Parse command line arguments
parse_arguments() {
    BUILD_FLAG=""
    SCALE_ARGS=""
    HEALTH_CHECK=false
    BACKUP=false
    ROLLBACK=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)
                BUILD_FLAG="--build"
                shift
                ;;
            --no-cache)
                BUILD_FLAG="--build --no-cache"
                shift
                ;;
            --scale)
                SCALE_ARGS="$SCALE_ARGS --scale $2"
                shift 2
                ;;
            --health-check)
                HEALTH_CHECK=true
                shift
                ;;
            --backup)
                BACKUP=true
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Set environment-specific configuration
setup_environment() {
    case $ENVIRONMENT in
        development)
            COMPOSE_FILE="docker-compose.yml"
            ENV_FILE=".env"
            ;;
        staging)
            COMPOSE_FILE="docker-compose.staging.yml"
            ENV_FILE=".env.staging"
            ;;
        production)
            COMPOSE_FILE="deployment/docker-compose.prod.yml"
            ENV_FILE=".env.production"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_info "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
    
    log_info "Deploying to $ENVIRONMENT environment"
    log_info "Using compose file: $COMPOSE_FILE"
    log_info "Using environment file: $ENV_FILE"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check environment file
    if [[ ! -f "$PROJECT_ROOT/$ENV_FILE" ]]; then
        log_error "Environment file not found: $ENV_FILE"
        log_info "Please copy and configure the environment file:"
        log_info "cp .env.example $ENV_FILE"
        exit 1
    fi
    
    # Check compose file
    if [[ ! -f "$PROJECT_ROOT/$COMPOSE_FILE" ]]; then
        log_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create backup (production only)
create_backup() {
    if [[ "$ENVIRONMENT" == "production" && "$BACKUP" == true ]]; then
        log_info "Creating backup..."
        
        BACKUP_DIR="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$BACKUP_DIR"
        
        # Backup database
        if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "Up"; then
            log_info "Backing up PostgreSQL database..."
            docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dumpall -U postgres > "$BACKUP_DIR/postgres_backup.sql"
        fi
        
        # Backup Redis data
        if docker-compose -f "$COMPOSE_FILE" ps redis | grep -q "Up"; then
            log_info "Backing up Redis data..."
            docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli BGSAVE
            docker cp "$(docker-compose -f "$COMPOSE_FILE" ps -q redis):/data/dump.rdb" "$BACKUP_DIR/redis_backup.rdb"
        fi
        
        # Backup configuration files
        log_info "Backing up configuration files..."
        cp "$PROJECT_ROOT/$ENV_FILE" "$BACKUP_DIR/"
        cp "$PROJECT_ROOT/$COMPOSE_FILE" "$BACKUP_DIR/"
        
        log_success "Backup created: $BACKUP_DIR"
    fi
}

# Rollback to previous version
rollback_deployment() {
    if [[ "$ROLLBACK" == true ]]; then
        log_info "Rolling back to previous version..."
        
        # Find latest backup
        LATEST_BACKUP=$(find "$PROJECT_ROOT/backups" -type d -name "*" | sort -r | head -n 1)
        
        if [[ -z "$LATEST_BACKUP" ]]; then
            log_error "No backup found for rollback"
            exit 1
        fi
        
        log_info "Rolling back using backup: $LATEST_BACKUP"
        
        # Stop current services
        docker-compose -f "$COMPOSE_FILE" down
        
        # Restore configuration
        cp "$LATEST_BACKUP/$ENV_FILE" "$PROJECT_ROOT/"
        
        # Restore database
        if [[ -f "$LATEST_BACKUP/postgres_backup.sql" ]]; then
            log_info "Restoring PostgreSQL database..."
            docker-compose -f "$COMPOSE_FILE" up -d postgres
            sleep 10
            docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U postgres < "$LATEST_BACKUP/postgres_backup.sql"
        fi
        
        # Restore Redis data
        if [[ -f "$LATEST_BACKUP/redis_backup.rdb" ]]; then
            log_info "Restoring Redis data..."
            docker cp "$LATEST_BACKUP/redis_backup.rdb" "$(docker-compose -f "$COMPOSE_FILE" ps -q redis):/data/dump.rdb"
            docker-compose -f "$COMPOSE_FILE" restart redis
        fi
        
        log_success "Rollback completed"
        exit 0
    fi
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    cd "$PROJECT_ROOT"
    
    # Pull latest images (if not building)
    if [[ -z "$BUILD_FLAG" ]]; then
        log_info "Pulling latest images..."
        docker-compose -f "$COMPOSE_FILE" pull
    fi
    
    # Deploy services
    log_info "Starting services..."
    docker-compose -f "$COMPOSE_FILE" up -d $BUILD_FLAG $SCALE_ARGS
    
    log_success "Services deployed successfully"
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    local services=(
        "http://localhost:3000:Frontend"
        "http://localhost:8000/health:API Gateway"
        "http://localhost:8001/health:AiCE Service"
        "http://localhost:8002/health:Proteus Service"
        "http://localhost:8003/health:ESM3 Service"
        "http://localhost:8004/health:SymbioticAIS Service"
        "http://localhost:8005/health:Bio-Twin Service"
        "http://localhost:8006/health:ExoStack Service"
    )
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r url name <<< "$service_info"
        
        log_info "Checking $name..."
        
        local retries=30
        while [[ $retries -gt 0 ]]; do
            if curl -f -s "$url" > /dev/null 2>&1; then
                log_success "$name is ready"
                break
            fi
            
            retries=$((retries - 1))
            if [[ $retries -gt 0 ]]; then
                sleep 2
            else
                log_warning "$name is not responding"
            fi
        done
    done
}

# Run health checks
run_health_checks() {
    if [[ "$HEALTH_CHECK" == true ]]; then
        log_info "Running health checks..."
        
        # Run integration tests
        if [[ -f "$PROJECT_ROOT/tests/integration/test_service_integration.py" ]]; then
            log_info "Running integration tests..."
            cd "$PROJECT_ROOT"
            python -m pytest tests/integration/test_service_integration.py -v --tb=short
        fi
        
        # Check service metrics
        log_info "Checking service metrics..."
        local services=("8001" "8002" "8003" "8004" "8005" "8006")
        
        for port in "${services[@]}"; do
            if curl -f -s "http://localhost:$port/metrics" | grep -q "http_requests_total"; then
                log_success "Service on port $port has metrics"
            else
                log_warning "Service on port $port missing metrics"
            fi
        done
        
        # Check monitoring stack
        if [[ "$ENVIRONMENT" == "production" ]]; then
            log_info "Checking monitoring stack..."
            
            if curl -f -s "http://localhost:9090/-/healthy" > /dev/null; then
                log_success "Prometheus is healthy"
            else
                log_warning "Prometheus is not healthy"
            fi
            
            if curl -f -s "http://localhost:3001/api/health" > /dev/null; then
                log_success "Grafana is healthy"
            else
                log_warning "Grafana is not healthy"
            fi
        fi
        
        log_success "Health checks completed"
    fi
}

# Show deployment summary
show_summary() {
    log_info "Deployment Summary"
    echo "===================="
    echo "Environment: $ENVIRONMENT"
    echo "Compose File: $COMPOSE_FILE"
    echo "Environment File: $ENV_FILE"
    echo ""
    
    log_info "Service Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    log_info "Access URLs:"
    echo "Frontend: http://localhost:3000"
    echo "API Gateway: http://localhost:8000"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo "Prometheus: http://localhost:9090"
        echo "Grafana: http://localhost:3001"
        echo "Kibana: http://localhost:5601"
    fi
    
    echo ""
    log_success "Deployment completed successfully!"
}

# Cleanup on exit
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
        
        # Show logs for debugging
        log_info "Recent logs:"
        docker-compose -f "$COMPOSE_FILE" logs --tail=50
    fi
    exit $exit_code
}

# Main deployment function
main() {
    # Set up error handling
    trap cleanup EXIT
    
    # Parse arguments (skip environment argument)
    shift
    parse_arguments "$@"
    
    # Setup environment configuration
    setup_environment
    
    # Check prerequisites
    check_prerequisites
    
    # Handle rollback
    rollback_deployment
    
    # Create backup if requested
    create_backup
    
    # Deploy services
    deploy_services
    
    # Wait for services
    wait_for_services
    
    # Run health checks
    run_health_checks
    
    # Show summary
    show_summary
}

# Run main function
main "$@"
