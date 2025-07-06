# Deployment Guide

Complete guide for deploying the Reflective Sequential Thinking MCP Tool in various environments.

## Table of Contents

1. [Quick Setup](#quick-setup)
2. [Development Environment](#development-environment)
3. [Production Deployment](#production-deployment)
4. [Configuration](#configuration)
5. [Cloud Deployments](#cloud-deployments)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Quick Setup

### Prerequisites

**System Requirements:**
- Python 3.10 or higher
- 4GB+ RAM (8GB+ recommended for production)
- 1GB+ disk space
- Internet connection for LLM provider access

**Required Accounts:**
- LLM Provider account (OpenRouter recommended)
- Optional: Redis instance for production deployments

### 5-Minute Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/reflective-sequential-thinking-mcp
cd reflective-sequential-thinking-mcp

# 2. Install dependencies
pip install uv  # If not already installed
uv pip install -e ".[dev]"

# 3. Set environment variables
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=your_api_key_here

# 4. Test installation
uv run pytest tests/ -v

# 5. Start server
uv run python main_refactored.py
```

---

## Development Environment

### Local Development Setup

#### 1. Environment Preparation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install UV for faster package management
pip install uv

# Install project with development dependencies
uv pip install -e ".[dev]"
```

#### 2. Environment Variables

Create a `.env` file in the project root:

```bash
# LLM Provider Configuration
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_TEAM_MODEL_ID=openai/gpt-4-turbo
OPENROUTER_AGENT_MODEL_ID=anthropic/claude-3-opus

# Alternative: OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_key
# OPENAI_TEAM_MODEL_ID=gpt-4-turbo
# OPENAI_AGENT_MODEL_ID=gpt-4-turbo

# Alternative: Gemini
# LLM_PROVIDER=gemini
# GOOGLE_API_KEY=your_google_key
# GEMINI_TEAM_MODEL_ID=gemini-2.0-flash
# GEMINI_AGENT_MODEL_ID=gemini-2.5-pro-preview

# Development Settings
CONTEXT_BACKEND=memory
ENABLE_REFLECTION=true
REFLECTION_DELAY_MS=500
DEBUG_MODE=true
```

#### 3. Claude Code Integration

Install Claude Code hooks for automated development workflow:

```bash
# Install hooks
./.claude/install-hooks.sh

# Test hooks functionality
./.claude/test-hooks.sh --verbose

# Check status
./.claude/manage-hooks.sh status
```

#### 4. Development Commands

```bash
# Code quality checks
ruff check . --fix
ruff format .
pyright .

# Run tests
pytest tests/ -v
pytest tests/test_integration.py -v --tb=short

# Run tests with coverage
pytest --cov=. --cov-report=html --cov-report=term

# Start development server
uv run python main_refactored.py

# Run in debug mode
DEBUG=true uv run python main_refactored.py
```

---

## Production Deployment

### Docker Deployment

#### 1. Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Copy requirements and install Python dependencies
COPY pyproject.toml .
COPY README.md .
RUN uv pip install --system -e "."

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "main_refactored.py"]
```

#### 2. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LLM_PROVIDER=${LLM_PROVIDER}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - CONTEXT_BACKEND=redis
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - mcp-server
    restart: unless-stopped

volumes:
  redis_data:
```

#### 3. Build and Deploy

```bash
# Build image
docker build -t reflective-thinking-mcp:latest .

# Deploy with compose
docker-compose up -d

# Check logs
docker-compose logs -f mcp-server

# Scale instances
docker-compose up -d --scale mcp-server=3
```

### Traditional Server Deployment

#### 1. System Setup (Ubuntu/Debian)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install system dependencies
sudo apt install git curl nginx redis-server supervisor

# Create application user
sudo useradd -m -s /bin/bash mcp-app
sudo mkdir -p /opt/mcp-server
sudo chown mcp-app:mcp-app /opt/mcp-server
```

#### 2. Application Deployment

```bash
# Switch to app user
sudo -u mcp-app -i

# Deploy application
cd /opt/mcp-server
git clone https://github.com/your-org/reflective-sequential-thinking-mcp .
python3.11 -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install -e "."

# Create environment file
cat > .env << EOF
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_api_key
CONTEXT_BACKEND=redis
REDIS_URL=redis://localhost:6379
ENABLE_REFLECTION=true
LOG_LEVEL=INFO
EOF
```

#### 3. Service Configuration

Create supervisor configuration:

```ini
# /etc/supervisor/conf.d/mcp-server.conf
[program:mcp-server]
command=/opt/mcp-server/.venv/bin/python main_refactored.py
directory=/opt/mcp-server
user=mcp-app
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/mcp-server.log
environment=HOME="/home/mcp-app",USER="mcp-app"
```

#### 4. Nginx Configuration

```nginx
# /etc/nginx/sites-available/mcp-server
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
```

#### 5. Start Services

```bash
# Enable and start Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Enable and start Supervisor
sudo systemctl enable supervisor
sudo systemctl start supervisor

# Reload Supervisor configuration
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start mcp-server

# Enable and configure Nginx
sudo ln -s /etc/nginx/sites-available/mcp-server /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx
```

---

## Configuration

### Environment Variables Reference

#### Core Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | Yes | `openrouter` | LLM provider (openrouter, openai, gemini) |
| `CONTEXT_BACKEND` | No | `memory` | Context storage (memory, redis) |
| `ENABLE_REFLECTION` | No | `true` | Enable reflection team |
| `REFLECTION_DELAY_MS` | No | `500` | Delay before reflection team starts |

#### Provider-Specific

**OpenRouter:**
```bash
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_TEAM_MODEL_ID=openai/gpt-4-turbo
OPENROUTER_AGENT_MODEL_ID=anthropic/claude-3-opus
```

**OpenAI:**
```bash
OPENAI_API_KEY=sk-...
OPENAI_TEAM_MODEL_ID=gpt-4-turbo
OPENAI_AGENT_MODEL_ID=gpt-4-turbo
```

**Gemini:**
```bash
GOOGLE_API_KEY=AI...
GEMINI_TEAM_MODEL_ID=gemini-2.0-flash
GEMINI_AGENT_MODEL_ID=gemini-2.5-pro-preview
```

#### Redis Configuration

```bash
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your_password  # If auth enabled
REDIS_DB=0
REDIS_TIMEOUT=30
```

### Performance Tuning

#### Memory Settings

```bash
# For high-volume deployments
MAX_CONTEXT_SIZE=100000
CONTEXT_TTL_HOURS=24
MAX_CONCURRENT_SESSIONS=50
```

#### Timeout Configuration

```bash
# LLM request timeouts
LLM_TIMEOUT_SECONDS=60
TEAM_COORDINATION_TIMEOUT=120
REFLECTION_TIMEOUT=45
```

### Security Configuration

```bash
# Enable security features
VALIDATE_INPUTS=true
SANITIZE_OUTPUTS=true
ENABLE_RATE_LIMITING=true
MAX_REQUESTS_PER_MINUTE=30

# Session security
SESSION_TIMEOUT_MINUTES=60
SECURE_CONTEXT_ISOLATION=true
```

---

## Cloud Deployments

### AWS Deployment

#### 1. ECS with Fargate

```yaml
# ecs-task-definition.json
{
  "family": "mcp-server",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "mcp-server",
      "image": "your-account.dkr.ecr.region.amazonaws.com/mcp-server:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LLM_PROVIDER",
          "value": "openrouter"
        }
      ],
      "secrets": [
        {
          "name": "OPENROUTER_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:mcp-secrets"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/mcp-server",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### 2. Application Load Balancer

```yaml
# alb-config.yml
Resources:
  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Scheme: internet-facing
      SecurityGroups: [!Ref ALBSecurityGroup]
      Subnets: [!Ref PublicSubnet1, !Ref PublicSubnet2]
      
  TargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Port: 8000
      Protocol: HTTP
      TargetType: ip
      VpcId: !Ref VPC
      HealthCheckPath: /health
      HealthCheckProtocol: HTTP
```

#### 3. ElastiCache Redis

```yaml
RedisCluster:
  Type: AWS::ElastiCache::ReplicationGroup
  Properties:
    ReplicationGroupDescription: MCP Server Redis
    Engine: redis
    CacheNodeType: cache.t3.micro
    NumCacheClusters: 2
    SecurityGroupIds: [!Ref RedisSecurityGroup]
    SubnetGroupName: !Ref RedisSubnetGroup
```

### Google Cloud Platform

#### 1. Cloud Run Deployment

```yaml
# cloudrun-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: mcp-server
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/execution-environment: gen2
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 10
      containers:
      - image: gcr.io/PROJECT_ID/mcp-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: LLM_PROVIDER
          value: "openrouter"
        - name: OPENROUTER_API_KEY
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: openrouter_api_key
        resources:
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

#### 2. Memorystore Redis

```bash
# Create Redis instance
gcloud redis instances create mcp-redis \
    --size=1 \
    --region=us-central1 \
    --redis-version=redis_7_0

# Get connection info
gcloud redis instances describe mcp-redis --region=us-central1
```

### Azure Deployment

#### 1. Container Instances

```yaml
# azure-container-group.yaml
apiVersion: 2019-12-01
location: eastus
name: mcp-server-group
properties:
  containers:
  - name: mcp-server
    properties:
      image: your-registry.azurecr.io/mcp-server:latest
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: LLM_PROVIDER
        value: openrouter
      - name: OPENROUTER_API_KEY
        secureValue: your_secret_key
      resources:
        requests:
          memoryInGB: 1
          cpu: 0.5
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
```

---

## Monitoring and Maintenance

### Health Checks

#### Built-in Health Endpoint

```bash
# Check service health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "checks": {
    "llm_provider": "ok",
    "context_backend": "ok",
    "memory_usage": "67%"
  }
}
```

#### Custom Health Checks

```python
# health_check.py
import asyncio
import aiohttp
import sys

async def check_health():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/health', timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('status') == 'healthy':
                        print("✅ Service healthy")
                        return 0
                    else:
                        print("❌ Service unhealthy")
                        return 1
                else:
                    print(f"❌ Health check failed: {resp.status}")
                    return 1
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(check_health())
    sys.exit(exit_code)
```

### Monitoring Setup

#### Prometheus Metrics

Add to `main_refactored.py`:

```python
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
request_count = Counter('mcp_requests_total', 'Total requests', ['tool', 'status'])
request_duration = Histogram('mcp_request_duration_seconds', 'Request duration')

# Start metrics server
start_http_server(9090)
```

#### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "MCP Server Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(mcp_requests_total[5m])",
            "legendFormat": "{{tool}} - {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, mcp_request_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### Log Management

#### Structured Logging

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "thought_processed",
    thought_number=1,
    session_id="abc123",
    execution_time_ms=1500,
    reflection_applied=True
)
```

#### Log Aggregation (ELK Stack)

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/mcp-server.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "mcp-server-%{+yyyy.MM.dd}"
```

### Backup and Recovery

#### Context Backup

```python
# backup_context.py
import asyncio
import json
from datetime import datetime
from src.context.shared_context import SharedContext

async def backup_context():
    context = SharedContext(backend="redis")
    state = await context.export_state()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"context_backup_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"Context backed up to {filename}")

if __name__ == "__main__":
    asyncio.run(backup_context())
```

#### Automated Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/opt/backups/mcp-server"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup context
cd /opt/mcp-server
python backup_context.py

# Move backup to backup directory
mv context_backup_*.json "$BACKUP_DIR/"

# Backup application logs
cp /var/log/mcp-server.log "$BACKUP_DIR/logs_$DATE.log"

# Clean old backups (keep last 30 days)
find "$BACKUP_DIR" -name "context_backup_*.json" -mtime +30 -delete
find "$BACKUP_DIR" -name "logs_*.log" -mtime +30 -delete

echo "Backup completed: $DATE"
```

#### Cron Setup

```bash
# Add to crontab
0 2 * * * /opt/mcp-server/backup.sh >> /var/log/mcp-backup.log 2>&1
```

### Performance Monitoring

#### Key Metrics to Track

1. **Request Metrics:**
   - Requests per second
   - Response times (p50, p95, p99)
   - Error rates by tool

2. **Resource Metrics:**
   - CPU usage
   - Memory consumption
   - Disk I/O
   - Network traffic

3. **Application Metrics:**
   - Active sessions
   - Context size
   - Reflection team usage
   - LLM provider latency

#### Alerting Rules

```yaml
# prometheus-alerts.yml
groups:
- name: mcp-server
  rules:
  - alert: HighErrorRate
    expr: rate(mcp_requests_total{status!="success"}[5m]) > 0.1
    for: 2m
    annotations:
      summary: "High error rate in MCP server"
      
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, mcp_request_duration_seconds_bucket) > 10
    for: 5m
    annotations:
      summary: "High response time in MCP server"
```

This deployment guide provides comprehensive instructions for setting up the Reflective Sequential Thinking MCP Tool in various environments, from local development to production cloud deployments.