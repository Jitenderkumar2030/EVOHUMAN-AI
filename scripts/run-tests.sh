#!/bin/bash

# EvoHuman.AI Test Runner Script
# Comprehensive test execution for all test types

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_TYPE="${1:-all}"

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
EvoHuman.AI Test Runner

Usage: $0 [TEST_TYPE] [OPTIONS]

TEST TYPES:
    all             Run all tests (default)
    unit            Run unit tests only
    integration     Run integration tests only
    e2e             Run end-to-end tests only
    performance     Run performance benchmarks only
    security        Run security tests only
    frontend        Run frontend tests only
    backend         Run backend tests only

OPTIONS:
    --coverage      Generate code coverage report
    --parallel      Run tests in parallel
    --verbose       Verbose output
    --fast          Skip slow tests
    --report        Generate detailed test report
    --help          Show this help message

EXAMPLES:
    $0 all --coverage --report
    $0 unit --parallel --verbose
    $0 e2e --fast
    $0 performance

EOF
}

# Parse command line arguments
parse_arguments() {
    COVERAGE=false
    PARALLEL=false
    VERBOSE=false
    FAST=false
    REPORT=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --coverage)
                COVERAGE=true
                shift
                ;;
            --parallel)
                PARALLEL=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --fast)
                FAST=true
                shift
                ;;
            --report)
                REPORT=true
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

# Setup test environment
setup_test_environment() {
    log_info "Setting up test environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create test results directory
    mkdir -p test-results
    
    # Setup Python virtual environment if needed
    if [[ ! -d "venv" ]]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install test dependencies
    log_info "Installing test dependencies..."
    pip install -q pytest pytest-asyncio pytest-cov pytest-html pytest-xdist aiohttp structlog
    
    # Install Node.js dependencies for frontend tests
    if [[ -f "ui/package.json" ]]; then
        log_info "Installing frontend test dependencies..."
        cd ui
        npm install --silent
        cd ..
    fi
    
    # Install Playwright for E2E tests
    if [[ "$TEST_TYPE" == "all" || "$TEST_TYPE" == "e2e" ]]; then
        log_info "Installing Playwright..."
        pip install -q playwright
        playwright install --with-deps chromium firefox webkit
    fi
    
    log_success "Test environment setup complete"
}

# Run unit tests
run_unit_tests() {
    log_info "Running unit tests..."
    
    local pytest_args="-v"
    
    if [[ "$COVERAGE" == true ]]; then
        pytest_args="$pytest_args --cov=services --cov=shared --cov-report=html --cov-report=xml"
    fi
    
    if [[ "$PARALLEL" == true ]]; then
        pytest_args="$pytest_args -n auto"
    fi
    
    if [[ "$FAST" == true ]]; then
        pytest_args="$pytest_args -m 'not slow'"
    fi
    
    if [[ "$REPORT" == true ]]; then
        pytest_args="$pytest_args --html=test-results/unit-tests.html --self-contained-html"
    fi
    
    # Run unit tests for each service
    local services=("aice-service" "proteus-service" "esm3-service" "symbiotic-service" "bio-twin-service" "exostack-service")
    
    for service in "${services[@]}"; do
        if [[ -d "services/$service/tests" ]]; then
            log_info "Running unit tests for $service..."
            python -m pytest "services/$service/tests" $pytest_args --junit-xml="test-results/unit-$service.xml"
        fi
    done
    
    # Run shared module tests
    if [[ -d "shared/tests" ]]; then
        log_info "Running shared module tests..."
        python -m pytest "shared/tests" $pytest_args --junit-xml="test-results/unit-shared.xml"
    fi
    
    log_success "Unit tests completed"
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    # Start test services
    log_info "Starting test services..."
    docker-compose -f docker-compose.test.yml up -d
    
    # Wait for services to be ready
    sleep 30
    
    local pytest_args="-v"
    
    if [[ "$VERBOSE" == true ]]; then
        pytest_args="$pytest_args -s"
    fi
    
    if [[ "$REPORT" == true ]]; then
        pytest_args="$pytest_args --html=test-results/integration-tests.html --self-contained-html"
    fi
    
    # Run integration tests
    python -m pytest tests/integration $pytest_args --junit-xml="test-results/integration.xml"
    
    # Stop test services
    docker-compose -f docker-compose.test.yml down
    
    log_success "Integration tests completed"
}

# Run E2E tests
run_e2e_tests() {
    log_info "Running end-to-end tests..."
    
    # Start full application stack
    log_info "Starting application stack..."
    docker-compose up -d
    
    # Wait for services to be ready
    sleep 60
    
    # Run Playwright tests
    cd tests/e2e
    
    local playwright_args=""
    
    if [[ "$PARALLEL" == true ]]; then
        playwright_args="$playwright_args --workers=4"
    fi
    
    if [[ "$VERBOSE" == true ]]; then
        playwright_args="$playwright_args --headed"
    fi
    
    if [[ "$FAST" == true ]]; then
        playwright_args="$playwright_args --grep-invert=@slow"
    fi
    
    npx playwright test $playwright_args
    
    cd "$PROJECT_ROOT"
    
    # Stop application stack
    docker-compose down
    
    log_success "End-to-end tests completed"
}

# Run performance tests
run_performance_tests() {
    log_info "Running performance benchmarks..."
    
    # Start application stack
    log_info "Starting application stack for performance testing..."
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for services to be ready
    sleep 60
    
    # Run performance benchmarks
    python tests/performance/benchmark_suite.py
    
    # Generate performance report
    if [[ "$REPORT" == true ]]; then
        log_info "Generating performance report..."
        python -c "
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load benchmark results
with open('benchmark_report.json', 'r') as f:
    data = json.load(f)

# Create performance charts
df = pd.DataFrame(data['summaries'])
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Response time chart
axes[0,0].bar(df['test_name'], df['avg_response_time'])
axes[0,0].set_title('Average Response Time')
axes[0,0].set_ylabel('Time (ms)')
axes[0,0].tick_params(axis='x', rotation=45)

# Success rate chart
axes[0,1].bar(df['test_name'], df['success_rate'])
axes[0,1].set_title('Success Rate')
axes[0,1].set_ylabel('Success Rate')
axes[0,1].tick_params(axis='x', rotation=45)

# Requests per second chart
axes[1,0].bar(df['test_name'], df['requests_per_second'])
axes[1,0].set_title('Requests per Second')
axes[1,0].set_ylabel('RPS')
axes[1,0].tick_params(axis='x', rotation=45)

# P95 response time chart
axes[1,1].bar(df['test_name'], df['p95_response_time'])
axes[1,1].set_title('P95 Response Time')
axes[1,1].set_ylabel('Time (ms)')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('test-results/performance-report.png', dpi=300, bbox_inches='tight')
plt.close()

print('Performance report generated: test-results/performance-report.png')
"
    fi
    
    # Stop application stack
    docker-compose -f docker-compose.prod.yml down
    
    log_success "Performance tests completed"
}

# Run security tests
run_security_tests() {
    log_info "Running security tests..."
    
    # Start application stack
    log_info "Starting application stack for security testing..."
    docker-compose up -d
    
    # Wait for services to be ready
    sleep 30
    
    # Run security scans
    log_info "Running container security scans..."
    
    local services=("aice-service" "proteus-service" "esm3-service" "symbiotic-service" "bio-twin-service" "exostack-service")
    
    for service in "${services[@]}"; do
        log_info "Scanning $service for vulnerabilities..."
        
        # Use Trivy for container scanning
        if command -v trivy &> /dev/null; then
            trivy image "evohuman-ai/$service:latest" --format json --output "test-results/security-$service.json"
        else
            log_warning "Trivy not installed, skipping container security scan"
        fi
    done
    
    # Run OWASP ZAP security tests
    if command -v zap-baseline.py &> /dev/null; then
        log_info "Running OWASP ZAP baseline scan..."
        zap-baseline.py -t http://localhost:3000 -J test-results/zap-report.json
    else
        log_warning "OWASP ZAP not installed, skipping web security scan"
    fi
    
    # Run custom security tests
    if [[ -f "tests/security/test_security.py" ]]; then
        log_info "Running custom security tests..."
        python -m pytest tests/security/test_security.py -v --junit-xml="test-results/security.xml"
    fi
    
    # Stop application stack
    docker-compose down
    
    log_success "Security tests completed"
}

# Run frontend tests
run_frontend_tests() {
    log_info "Running frontend tests..."
    
    cd ui
    
    # Run Jest unit tests
    log_info "Running Jest unit tests..."
    npm test -- --coverage --watchAll=false --testResultsProcessor=jest-junit
    
    # Run ESLint
    log_info "Running ESLint..."
    npm run lint -- --format junit --output-file ../test-results/eslint.xml
    
    # Run TypeScript type checking
    log_info "Running TypeScript type checking..."
    npm run type-check
    
    # Build frontend to check for build errors
    log_info "Testing frontend build..."
    npm run build
    
    cd "$PROJECT_ROOT"
    
    log_success "Frontend tests completed"
}

# Run backend tests
run_backend_tests() {
    log_info "Running backend tests..."
    
    # Run unit tests for all services
    run_unit_tests
    
    # Run API tests
    if [[ -f "tests/api/test_api.py" ]]; then
        log_info "Running API tests..."
        python -m pytest tests/api/test_api.py -v --junit-xml="test-results/api.xml"
    fi
    
    log_success "Backend tests completed"
}

# Generate test report
generate_test_report() {
    if [[ "$REPORT" == true ]]; then
        log_info "Generating comprehensive test report..."
        
        # Create HTML report
        cat > test-results/test-report.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>EvoHuman.AI Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .success { color: green; }
        .warning { color: orange; }
        .error { color: red; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>EvoHuman.AI Test Report</h1>
        <p>Generated: $(date)</p>
        <p>Test Type: $TEST_TYPE</p>
    </div>
    
    <div class="section">
        <h2>Test Summary</h2>
        <table>
            <tr><th>Test Type</th><th>Status</th><th>Results</th></tr>
EOF

        # Add test results to report
        if [[ -f "test-results/unit-tests.html" ]]; then
            echo "<tr><td>Unit Tests</td><td class='success'>✓ Passed</td><td><a href='unit-tests.html'>View Results</a></td></tr>" >> test-results/test-report.html
        fi
        
        if [[ -f "test-results/integration.xml" ]]; then
            echo "<tr><td>Integration Tests</td><td class='success'>✓ Passed</td><td><a href='integration-tests.html'>View Results</a></td></tr>" >> test-results/test-report.html
        fi
        
        if [[ -f "tests/e2e/test-results/results.json" ]]; then
            echo "<tr><td>E2E Tests</td><td class='success'>✓ Passed</td><td><a href='../e2e/test-results/index.html'>View Results</a></td></tr>" >> test-results/test-report.html
        fi
        
        if [[ -f "benchmark_report.json" ]]; then
            echo "<tr><td>Performance Tests</td><td class='success'>✓ Passed</td><td><a href='performance-report.png'>View Results</a></td></tr>" >> test-results/test-report.html
        fi
        
        cat >> test-results/test-report.html << EOF
        </table>
    </div>
    
    <div class="section">
        <h2>Coverage Report</h2>
EOF

        if [[ -f "htmlcov/index.html" ]]; then
            echo "<p><a href='../htmlcov/index.html'>View Code Coverage Report</a></p>" >> test-results/test-report.html
        else
            echo "<p>No coverage report available</p>" >> test-results/test-report.html
        fi
        
        cat >> test-results/test-report.html << EOF
    </div>
</body>
</html>
EOF
        
        log_success "Test report generated: test-results/test-report.html"
    fi
}

# Main test runner function
main() {
    # Parse arguments (skip test type argument)
    shift
    parse_arguments "$@"
    
    # Setup test environment
    setup_test_environment
    
    # Run tests based on type
    case $TEST_TYPE in
        all)
            run_unit_tests
            run_integration_tests
            run_e2e_tests
            run_performance_tests
            run_security_tests
            ;;
        unit)
            run_unit_tests
            ;;
        integration)
            run_integration_tests
            ;;
        e2e)
            run_e2e_tests
            ;;
        performance)
            run_performance_tests
            ;;
        security)
            run_security_tests
            ;;
        frontend)
            run_frontend_tests
            ;;
        backend)
            run_backend_tests
            ;;
        *)
            log_error "Invalid test type: $TEST_TYPE"
            show_help
            exit 1
            ;;
    esac
    
    # Generate test report
    generate_test_report
    
    log_success "All tests completed successfully!"
}

# Run main function
main "$@"
