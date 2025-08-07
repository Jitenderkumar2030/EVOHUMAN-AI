# 🚀 EvoHuman.AI Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the EvoHuman.AI platform in various environments, from local development to production-scale deployments.

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores (8 recommended)
- RAM: 16GB (32GB recommended)
- Storage: 100GB SSD (500GB recommended)
- Network: 1Gbps connection

**Software Requirements:**
- Docker 24.0+ and Docker Compose 2.20+
- Python 3.11+
- Node.js 18+ and npm 9+
- Redis 7.0+
- PostgreSQL 15+ (optional, for persistent data)

### Development Tools
- Git 2.40+
- Make (for build automation)
- curl (for health checks)

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Frontend UI   │
│    (Nginx)      │────│   (FastAPI)     │────│   (React)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Microservices │    │   Data Storage  │
│  (Prometheus)   │    │   - AiCE        │    │   - Redis       │
│  (Grafana)      │    │   - Proteus     │    │   - PostgreSQL  │
│                 │    │   - ESM3        │    │   - File System │
└─────────────────┘    │   - SymbioticAIS│    └─────────────────┘
                       │   - Bio-Twin    │
                       │   - ExoStack    │
                       └─────────────────┘
```

## 🐳 Docker Deployment

### Quick Start (Development)

1. **Clone Repository**
```bash
git clone https://github.com/your-org/evohuman-ai.git
cd evohuman-ai
```

2. **Environment Setup**
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

3. **Start Services**
```bash
# Build and start all services
docker-compose up -d

# Check service health
docker-compose ps
```

4. **Verify Deployment**
```bash
# Check frontend
curl http://localhost:3000

# Check API health
curl http://localhost:8000/health
```

### Production Deployment

1. **Production Environment File**
```bash
# Create production environment
cp .env.production.example .env.production

# Configure production settings
nano .env.production
```

2. **Deploy with Production Compose**
```bash
# Deploy production stack
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale aice-service=3
```

3. **SSL/TLS Setup**
```bash
# Generate SSL certificates (Let's Encrypt)
./scripts/setup-ssl.sh your-domain.com

# Update nginx configuration
./scripts/update-nginx-ssl.sh
```

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes cluster 1.28+
- kubectl configured
- Helm 3.12+

### Deploy to Kubernetes

1. **Create Namespace**
```bash
kubectl create namespace evohuman-ai
```

2. **Deploy with Helm**
```bash
# Add Helm repository
helm repo add evohuman-ai ./helm/evohuman-ai

# Install release
helm install evohuman-ai evohuman-ai/evohuman-ai \
  --namespace evohuman-ai \
  --values values.production.yaml
```

3. **Configure Ingress**
```bash
# Apply ingress configuration
kubectl apply -f k8s/ingress.yaml
```

4. **Monitor Deployment**
```bash
# Check pod status
kubectl get pods -n evohuman-ai

# Check services
kubectl get services -n evohuman-ai

# View logs
kubectl logs -f deployment/aice-service -n evohuman-ai
```

## 🌩️ Cloud Deployment

### AWS Deployment

#### Using ECS (Elastic Container Service)

1. **Setup AWS CLI**
```bash
aws configure
```

2. **Create ECS Cluster**
```bash
# Create cluster
aws ecs create-cluster --cluster-name evohuman-ai-cluster

# Register task definitions
aws ecs register-task-definition --cli-input-json file://aws/task-definitions/aice-service.json
```

3. **Deploy Services**
```bash
# Deploy using CloudFormation
aws cloudformation deploy \
  --template-file aws/cloudformation/evohuman-ai-stack.yaml \
  --stack-name evohuman-ai \
  --capabilities CAPABILITY_IAM
```

#### Using EKS (Elastic Kubernetes Service)

1. **Create EKS Cluster**
```bash
# Create cluster
eksctl create cluster --name evohuman-ai --region us-west-2

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name evohuman-ai
```

2. **Deploy Application**
```bash
# Deploy using Helm
helm install evohuman-ai ./helm/evohuman-ai \
  --namespace evohuman-ai \
  --create-namespace \
  --values values.aws.yaml
```

### Google Cloud Platform (GCP)

#### Using Cloud Run

1. **Build and Push Images**
```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build and push images
./scripts/build-and-push-gcp.sh
```

2. **Deploy Services**
```bash
# Deploy each service
gcloud run deploy aice-service \
  --image gcr.io/your-project/aice-service:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Using GKE (Google Kubernetes Engine)

1. **Create GKE Cluster**
```bash
# Create cluster
gcloud container clusters create evohuman-ai \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4
```

2. **Deploy Application**
```bash
# Get credentials
gcloud container clusters get-credentials evohuman-ai --zone us-central1-a

# Deploy with Helm
helm install evohuman-ai ./helm/evohuman-ai \
  --namespace evohuman-ai \
  --create-namespace \
  --values values.gcp.yaml
```

### Microsoft Azure

#### Using Container Instances

1. **Create Resource Group**
```bash
az group create --name evohuman-ai-rg --location eastus
```

2. **Deploy Container Group**
```bash
az container create \
  --resource-group evohuman-ai-rg \
  --file azure/container-group.yaml
```

#### Using AKS (Azure Kubernetes Service)

1. **Create AKS Cluster**
```bash
az aks create \
  --resource-group evohuman-ai-rg \
  --name evohuman-ai-aks \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys
```

2. **Deploy Application**
```bash
# Get credentials
az aks get-credentials --resource-group evohuman-ai-rg --name evohuman-ai-aks

# Deploy with Helm
helm install evohuman-ai ./helm/evohuman-ai \
  --namespace evohuman-ai \
  --create-namespace \
  --values values.azure.yaml
```

## 🔧 Configuration

### Environment Variables

**Core Configuration:**
```bash
# Application
APP_ENV=production
APP_DEBUG=false
APP_SECRET_KEY=your-secret-key-here

# Database
REDIS_URL=redis://redis:6379
POSTGRES_URL=postgresql://user:pass@postgres:5432/evohuman

# Security
JWT_SECRET=your-jwt-secret-here
BCRYPT_ROUNDS=12
RATE_LIMIT_REQUESTS=100

# Services
AICE_SERVICE_URL=http://aice-service:8001
PROTEUS_SERVICE_URL=http://proteus-service:8002
ESM3_SERVICE_URL=http://esm3-service:8003
```

**Monitoring Configuration:**
```bash
# Prometheus
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Performance
WORKER_PROCESSES=4
MAX_CONNECTIONS=1000
```

### Service Configuration

Each service can be configured via YAML files in the `config/` directory:

- `config/aice-service.yaml` - AiCE service configuration
- `config/proteus-service.yaml` - Proteus service configuration
- `config/esm3-service.yaml` - ESM3 service configuration
- `config/symbiotic-service.yaml` - SymbioticAIS configuration

## 📊 Monitoring and Observability

### Metrics Collection

**Prometheus Configuration:**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'evohuman-ai'
    static_configs:
      - targets: ['aice-service:8001', 'proteus-service:8002']
```

**Grafana Dashboards:**
- System metrics (CPU, Memory, Disk)
- Application metrics (Request rate, Response time, Error rate)
- Business metrics (User activity, Analysis requests)

### Logging

**Centralized Logging with ELK Stack:**
```bash
# Deploy ELK stack
docker-compose -f docker-compose.elk.yml up -d

# Configure log shipping
./scripts/configure-logging.sh
```

### Health Checks

**Service Health Endpoints:**
- Frontend: `http://localhost:3000/health`
- AiCE Service: `http://localhost:8001/health`
- Proteus Service: `http://localhost:8002/health`
- ESM3 Service: `http://localhost:8003/health`

**Automated Health Monitoring:**
```bash
# Run health check script
./scripts/health-check.sh

# Setup monitoring cron job
crontab -e
# Add: */5 * * * * /path/to/evohuman-ai/scripts/health-check.sh
```

## 🔒 Security

### SSL/TLS Configuration

**Let's Encrypt Setup:**
```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Generate certificates
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Firewall Configuration

**UFW (Ubuntu Firewall):**
```bash
# Enable firewall
sudo ufw enable

# Allow SSH
sudo ufw allow ssh

# Allow HTTP/HTTPS
sudo ufw allow 80
sudo ufw allow 443

# Allow specific services (if needed)
sudo ufw allow 8000:8010/tcp
```

### Security Scanning

**Container Security:**
```bash
# Scan images for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image evohuman-ai/aice-service:latest
```

## 🚨 Troubleshooting

### Common Issues

**Service Won't Start:**
```bash
# Check logs
docker-compose logs service-name

# Check resource usage
docker stats

# Restart service
docker-compose restart service-name
```

**Database Connection Issues:**
```bash
# Test Redis connection
redis-cli -h redis-host ping

# Test PostgreSQL connection
psql -h postgres-host -U username -d database
```

**Performance Issues:**
```bash
# Check system resources
htop
iotop
nethogs

# Check service metrics
curl http://localhost:8001/metrics
```

### Log Analysis

**Common Log Locations:**
- Application logs: `/var/log/evohuman-ai/`
- Docker logs: `docker-compose logs`
- System logs: `/var/log/syslog`

**Log Analysis Commands:**
```bash
# Search for errors
grep -i error /var/log/evohuman-ai/*.log

# Monitor real-time logs
tail -f /var/log/evohuman-ai/aice-service.log

# Analyze performance
grep "response_time" /var/log/evohuman-ai/*.log | awk '{print $NF}' | sort -n
```

## 📞 Support

### Getting Help

- **Documentation**: [docs.evohuman.ai](https://docs.evohuman.ai)
- **Issues**: [GitHub Issues](https://github.com/your-org/evohuman-ai/issues)
- **Community**: [Discord Server](https://discord.gg/evohuman-ai)
- **Email**: support@evohuman.ai

### Reporting Issues

When reporting issues, please include:
1. Environment details (OS, Docker version, etc.)
2. Service logs
3. Steps to reproduce
4. Expected vs actual behavior

---

**Last Updated**: 2025-08-07  
**Version**: 1.0.0
