#!/bin/bash

# EvoHuman.AI Platform Setup Script
# Initializes the development environment and dependencies

set -e

echo "🧬 Setting up EvoHuman.AI Platform..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    print_status "Checking operating system..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_success "Linux detected"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_success "macOS detected"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check required dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker found"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose found"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    print_success "Python 3 found"
    
    # Check Node.js (for UI development)
    if ! command -v node &> /dev/null; then
        print_warning "Node.js not found. UI development will not be available."
    else
        print_success "Node.js found"
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
    print_success "Git found"
}

# Create necessary directories
create_directories() {
    print_status "Creating directory structure..."
    
    mkdir -p data/{users,models,logs,backups}
    mkdir -p models/{esm3,proteus,aice,symbiotic}
    mkdir -p logs/{services,jobs,errors}
    mkdir -p configs/{environments,secrets}
    
    print_success "Directory structure created"
}

# Setup Python virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
}

# Setup environment variables
setup_environment() {
    print_status "Setting up environment variables..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# EvoHuman.AI Environment Configuration

# Database
DATABASE_URL=postgresql://evohuman:evohuman@localhost:5432/evohuman
REDIS_URL=redis://localhost:6379

# Security
JWT_SECRET=$(openssl rand -hex 32)
ENCRYPTION_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# Services
GATEWAY_URL=http://localhost:8000
BIO_TWIN_URL=http://localhost:8001
ESM3_SERVICE_URL=http://localhost:8002
PROTEUS_SERVICE_URL=http://localhost:8003
AICE_SERVICE_URL=http://localhost:8004
SYMBIOTIC_SERVICE_URL=http://localhost:8005
EXOSTACK_SERVICE_URL=http://localhost:8006

# UI
REACT_APP_API_URL=http://localhost:8000

# Development
DEBUG=true
LOG_LEVEL=INFO
ENVIRONMENT=development

# Model Paths
ESM3_MODEL_PATH=./models/esm3
PROTEUS_MODEL_PATH=./models/proteus
AICE_MODEL_PATH=./models/aice
SYMBIOTIC_MODEL_PATH=./models/symbiotic

# ExoStack Configuration
EXOSTACK_HUB_URL=http://localhost:8006
EXOSTACK_NODE_REGISTRY=./data/nodes
EXOSTACK_JOB_QUEUE=redis://localhost:6379/1

# Privacy and Security
DATA_ENCRYPTION_ENABLED=true
PRIVACY_MODE=high
CONSENT_REQUIRED=true
DATA_RETENTION_DAYS=90
EOF
        print_success "Environment file created"
    else
        print_warning "Environment file already exists"
    fi
}

# Initialize database
init_database() {
    print_status "Initializing database..."
    
    # Create init SQL script
    cat > scripts/init-db.sql << EOF
-- EvoHuman.AI Database Initialization

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS evohuman;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS jobs;

-- Set default schema
SET search_path TO evohuman, public;

-- Create initial admin user (password: admin123)
-- This should be changed in production
INSERT INTO users (id, email, username, full_name, role, password_hash, created_at, updated_at)
VALUES (
    uuid_generate_v4()::text,
    'admin@evohuman.ai',
    'admin',
    'System Administrator',
    'admin',
    crypt('admin123', gen_salt('bf')),
    NOW(),
    NOW()
) ON CONFLICT (email) DO NOTHING;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_bio_twin_snapshots_user_id ON bio_twin_snapshots(user_id);
CREATE INDEX IF NOT EXISTS idx_bio_twin_snapshots_created_at ON bio_twin_snapshots(created_at);
CREATE INDEX IF NOT EXISTS idx_evolution_goals_user_id ON evolution_goals(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_service_requests_user_id ON ai_service_requests(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_service_requests_status ON ai_service_requests(status);
CREATE INDEX IF NOT EXISTS idx_feedback_loops_user_id ON feedback_loops(user_id);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA evohuman TO evohuman;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA evohuman TO evohuman;
EOF
    
    print_success "Database initialization script created"
}

# Download model placeholders
setup_models() {
    print_status "Setting up AI model placeholders..."
    
    # Create model info files
    cat > models/esm3/model_info.json << EOF
{
    "name": "ESM3 Protein Language Model",
    "version": "sm_open_v1",
    "description": "Evolutionary Scale Modeling for protein structure prediction",
    "source": "https://github.com/facebookresearch/esm",
    "requirements": {
        "gpu_memory": "8GB",
        "cpu_cores": 4,
        "ram": "16GB"
    },
    "download_instructions": "Run: python scripts/download_models.py --model esm3"
}
EOF

    cat > models/proteus/model_info.json << EOF
{
    "name": "Proteus Biological Framework",
    "version": "1.0.0",
    "description": "Biological AI framework for cellular simulations",
    "source": "Custom EvoHuman.AI model",
    "requirements": {
        "gpu_memory": "6GB",
        "cpu_cores": 8,
        "ram": "12GB"
    },
    "download_instructions": "Model will be trained during platform initialization"
}
EOF

    cat > models/aice/model_info.json << EOF
{
    "name": "AiCE Cognitive Enhancer",
    "version": "1.0.0",
    "description": "AI Cognitive Enhancement for brain development",
    "source": "Custom EvoHuman.AI model",
    "requirements": {
        "gpu_memory": "4GB",
        "cpu_cores": 6,
        "ram": "8GB"
    },
    "download_instructions": "Model will be trained during platform initialization"
}
EOF

    cat > models/symbiotic/model_info.json << EOF
{
    "name": "SymbioticAIS Evolution Engine",
    "version": "1.0.0",
    "description": "Human-AI symbiotic evolution system",
    "source": "https://github.com/Rqcker/SymbioticAIS.git",
    "requirements": {
        "gpu_memory": "6GB",
        "cpu_cores": 4,
        "ram": "10GB"
    },
    "download_instructions": "Run: git clone https://github.com/Rqcker/SymbioticAIS.git models/symbiotic/source"
}
EOF
    
    print_success "Model placeholders created"
}

# Create development scripts
create_dev_scripts() {
    print_status "Creating development scripts..."
    
    # Start script
    cat > scripts/start.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting EvoHuman.AI Platform..."
docker-compose up -d
echo "✅ Platform started!"
echo "🌐 Dashboard: http://localhost:3000"
echo "📡 API Gateway: http://localhost:8000"
echo "📊 Health Check: http://localhost:8000/health"
EOF
    
    # Stop script
    cat > scripts/stop.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping EvoHuman.AI Platform..."
docker-compose down
echo "✅ Platform stopped!"
EOF
    
    # Logs script
    cat > scripts/logs.sh << 'EOF'
#!/bin/bash
if [ -z "$1" ]; then
    echo "📋 Showing all service logs..."
    docker-compose logs -f
else
    echo "📋 Showing logs for service: $1"
    docker-compose logs -f "$1"
fi
EOF
    
    # Make scripts executable
    chmod +x scripts/*.sh
    
    print_success "Development scripts created"
}

# Main setup function
main() {
    echo "🧬 EvoHuman.AI Platform Setup"
    echo "================================"
    
    check_os
    check_dependencies
    create_directories
    setup_python_env
    setup_environment
    init_database
    setup_models
    create_dev_scripts
    
    echo ""
    print_success "🎉 EvoHuman.AI Platform setup completed!"
    echo ""
    echo "Next steps:"
    echo "1. Review and update .env file with your configurations"
    echo "2. Download AI models: python scripts/download_models.py"
    echo "3. Start the platform: ./scripts/start.sh"
    echo "4. Access the dashboard: http://localhost:3000"
    echo ""
    echo "For development:"
    echo "- View logs: ./scripts/logs.sh [service_name]"
    echo "- Stop platform: ./scripts/stop.sh"
    echo "- Rebuild services: docker-compose build"
    echo ""
}

# Run main function
main "$@"
