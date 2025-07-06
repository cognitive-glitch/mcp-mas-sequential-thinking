# Troubleshooting Guide

Common issues and solutions for the Reflective Sequential Thinking MCP Tool.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Configuration Problems](#configuration-problems)
3. [Runtime Errors](#runtime-errors)
4. [Performance Issues](#performance-issues)
5. [API and Provider Issues](#api-and-provider-issues)
6. [Claude Code Integration](#claude-code-integration)
7. [Debugging Tools](#debugging-tools)

## Installation Issues

### Python Version Compatibility

**Problem**: `SyntaxError` or `ModuleNotFoundError` during installation
```
SyntaxError: invalid syntax (main_refactored.py, line 45)
```

**Solution**: Ensure Python 3.10+ is installed
```bash
# Check Python version
python --version
python3 --version

# Install Python 3.11 on Ubuntu
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Use specific Python version
python3.11 -m venv .venv
source .venv/bin/activate
```

### UV Installation Issues

**Problem**: `uv` command not found
```bash
uv: command not found
```

**Solution**: Install UV package manager
```bash
# Option 1: pip install
pip install uv

# Option 2: curl install (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Option 3: conda install
conda install -c conda-forge uv
```

### Dependency Installation Failures

**Problem**: Package installation fails with build errors
```
error: Microsoft Visual C++ 14.0 is required
```

**Solution**: Platform-specific fixes
```bash
# Windows: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# macOS: Install Xcode command line tools
xcode-select --install

# Linux: Install build essentials
sudo apt install build-essential python3-dev

# Alternative: Use conda environment
conda create -n mcp-env python=3.11
conda activate mcp-env
conda install pip
pip install -e ".[dev]"
```

---

## Configuration Problems

### Environment Variable Issues

**Problem**: `LLM_PROVIDER` not recognized
```
ValueError: Unsupported provider: None
```

**Solution**: Set environment variables correctly
```bash
# Check current variables
echo $LLM_PROVIDER
echo $OPENROUTER_API_KEY

# Set for current session
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=sk-or-v1-...

# Set permanently (Linux/macOS)
echo 'export LLM_PROVIDER=openrouter' >> ~/.bashrc
echo 'export OPENROUTER_API_KEY=sk-or-v1-...' >> ~/.bashrc
source ~/.bashrc

# Windows PowerShell
$env:LLM_PROVIDER="openrouter"
$env:OPENROUTER_API_KEY="sk-or-v1-..."

# Windows Command Prompt
set LLM_PROVIDER=openrouter
set OPENROUTER_API_KEY=sk-or-v1-...
```

### API Key Validation

**Problem**: Invalid API key format
```
ValueError: Missing required API key: OPENROUTER_API_KEY
```

**Solution**: Verify API key format and permissions
```bash
# OpenRouter keys start with: sk-or-v1-
# OpenAI keys start with: sk-
# Google keys start with: AIza

# Test API key (OpenRouter example)
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models

# Test API key (OpenAI example)
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

### Model Configuration Errors

**Problem**: Specified model not available
```
Model 'invalid-model' not found
```

**Solution**: Use valid model identifiers
```bash
# Check available models for your provider
# OpenRouter models: https://openrouter.ai/models
# OpenAI models: https://platform.openai.com/docs/models

# Example valid configurations:
export OPENROUTER_TEAM_MODEL_ID=openai/gpt-4-turbo
export OPENROUTER_AGENT_MODEL_ID=anthropic/claude-3-opus

export OPENAI_TEAM_MODEL_ID=gpt-4-turbo
export OPENAI_AGENT_MODEL_ID=gpt-4-turbo

export GEMINI_TEAM_MODEL_ID=gemini-2.0-flash
export GEMINI_AGENT_MODEL_ID=gemini-2.5-pro-preview
```

---

## Runtime Errors

### Zero Token API Calls

**Problem**: Empty or zero-token requests to LLM providers
```
Error: Request contains no tokens
```

**Solution**: This was fixed in `main_refactored.py`. If still occurring:
```python
# Check if using main.py instead of main_refactored.py
# Use the refactored version:
uv run python main_refactored.py

# Verify team response validation
if not team_response or not hasattr(team_response, 'content') or not team_response.content:
    logger.error("Empty response from team")
    return "Error: Team returned empty response. Please retry with a clearer thought."
```

### Team Initialization Failures

**Problem**: Teams not initializing properly
```
AttributeError: 'NoneType' object has no attribute 'arun'
```

**Solution**: Check team initialization
```python
# Debug team initialization
from main_refactored import app_context

# Check if teams are initialized
print(f"Primary team: {app_context.primary_team}")
print(f"Reflection team: {app_context.reflection_team}")

# Manually initialize if needed
await app_context.initialize_models()
await app_context.initialize_teams()
```

### Pydantic Validation Errors

**Problem**: Input validation failures
```
ValidationError: thoughtNumber must be >= 1
```

**Solution**: Check input parameters
```python
# Common validation issues and fixes:

# ❌ Invalid thought number
thoughtNumber=0  # Must be >= 1

# ✅ Correct thought number
thoughtNumber=1

# ❌ Revision without target
isRevision=True
# Missing: revisesThought=1

# ✅ Correct revision
isRevision=True
revisesThought=1

# ❌ Branch without ID
branchFromThought=2
# Missing: branchId="branch-name"

# ✅ Correct branch
branchFromThought=2
branchId="alternative-approach"
```

### Context Update Failures

**Problem**: SharedContext operations failing
```
TypeError: Object of type 'ThoughtData' is not JSON serializable
```

**Solution**: Check context backend and serialization
```bash
# Use memory backend for development
export CONTEXT_BACKEND=memory

# For Redis backend, ensure Redis is running
redis-cli ping  # Should return "PONG"

# Check Redis configuration
export REDIS_URL=redis://localhost:6379
export REDIS_DB=0
```

---

## Performance Issues

### Slow Response Times

**Problem**: Requests taking longer than expected (>30 seconds)
```
TimeoutError: Request timeout after 60 seconds
```

**Solution**: Optimize configuration and check resources
```bash
# Check provider latency
time curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"openai/gpt-4-turbo","messages":[{"role":"user","content":"Hi"}]}' \
     https://openrouter.ai/api/v1/chat/completions

# Use faster models for development
export OPENROUTER_TEAM_MODEL_ID=openai/gpt-3.5-turbo
export OPENROUTER_AGENT_MODEL_ID=openai/gpt-3.5-turbo

# Disable reflection for faster responses
export ENABLE_REFLECTION=false

# Reduce reflection delay
export REFLECTION_DELAY_MS=100
```

### Memory Usage Issues

**Problem**: High memory consumption or out-of-memory errors
```
MemoryError: Unable to allocate memory
```

**Solution**: Optimize memory usage
```bash
# Monitor memory usage
top -p $(pgrep -f main_refactored.py)

# Reduce context size
export MAX_CONTEXT_SIZE=50000
export CONTEXT_TTL_HOURS=12

# Use Redis backend to offload memory
export CONTEXT_BACKEND=redis
export REDIS_URL=redis://localhost:6379

# Clear context periodically
python -c "
from src.context.shared_context import SharedContext
import asyncio
async def clear(): 
    ctx = SharedContext()
    await ctx.clear_expired_contexts()
asyncio.run(clear())
"
```

### High CPU Usage

**Problem**: Excessive CPU utilization
```bash
# CPU usage consistently >80%
```

**Solution**: Optimize processing and concurrency
```bash
# Reduce concurrent processing
export MAX_CONCURRENT_SESSIONS=10

# Use lighter models
export OPENROUTER_TEAM_MODEL_ID=openai/gpt-3.5-turbo

# Enable model response caching
export ENABLE_RESPONSE_CACHE=true
export CACHE_TTL_MINUTES=60
```

---

## API and Provider Issues

### Rate Limiting

**Problem**: API rate limit exceeded
```
RateLimitError: Rate limit exceeded. Please try again later.
```

**Solution**: Implement backoff and check quotas
```python
# Check rate limits in code
import time
import random

async def api_call_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await func()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            await asyncio.sleep(delay)
```

```bash
# Check API quotas
# OpenRouter: Check dashboard at https://openrouter.ai/
# OpenAI: Check usage at https://platform.openai.com/usage

# Use different providers as fallback
export LLM_PROVIDER=gemini  # Switch to Gemini if OpenRouter limited
```

### Network Connectivity Issues

**Problem**: Connection errors to LLM providers
```
ConnectionError: Failed to connect to api.openai.com
```

**Solution**: Check network and proxy settings
```bash
# Test connectivity
ping api.openai.com
ping openrouter.ai
ping generativelanguage.googleapis.com

# Test with curl
curl -I https://api.openai.com/v1/models
curl -I https://openrouter.ai/api/v1/models

# Configure proxy if needed
export https_proxy=http://proxy.company.com:8080
export http_proxy=http://proxy.company.com:8080

# Or disable SSL verification (NOT recommended for production)
export PYTHONHTTPSVERIFY=0
```

### Authentication Errors

**Problem**: API authentication failures
```
AuthenticationError: Invalid API key provided
```

**Solution**: Verify credentials and permissions
```bash
# Check API key format
echo "Key length: ${#OPENROUTER_API_KEY}"  # Should be ~70 characters
echo "Key prefix: ${OPENROUTER_API_KEY:0:10}"  # Should start with sk-or-v1-

# Test authentication
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/auth/key

# Check account status and billing
# Visit provider dashboard to verify account status
```

---

## Claude Code Integration

### MCP Tools Not Recognized

**Problem**: Claude Code doesn't see the tools
```
Tool 'sequentialthinking' not found
```

**Solution**: Verify MCP server registration
```bash
# Check if server is running
netstat -tulpn | grep :8000
lsof -i :8000

# Test MCP endpoints
curl http://localhost:8000/health

# Check Claude Code MCP configuration
cat ~/.claude/mcp_servers.json

# Restart Claude Code CLI
# Tools should show: sequentialthinking, sequentialreview
```

### Hook Installation Issues

**Problem**: Claude Code hooks not working
```bash
./.claude/install-hooks.sh
# Permission denied or hooks not triggering
```

**Solution**: Fix hook configuration
```bash
# Make scripts executable
chmod +x ./.claude/*.sh

# Check jq is installed (required for hooks)
command -v jq || sudo apt install jq

# Test hook configuration
./.claude/test-hooks.sh --verbose

# Manually copy hooks if automatic install fails
cp .claude/settings.json ~/.claude/settings.json

# Check hook status
./.claude/manage-hooks.sh status
```

### Hook Execution Failures

**Problem**: Hooks fail during execution
```
Hook execution failed: ruff: command not found
```

**Solution**: Ensure tools are available in PATH
```bash
# Check tool availability
which ruff pyright pytest

# Install missing tools
uv add ruff pyright pytest

# Update hook commands with full paths
# Edit ~/.claude/settings.json:
"command": "/full/path/to/uv run ruff check --fix $CLAUDE_FILE_PATHS"

# Test hook execution
echo "# test" >> test.py
# Should trigger ruff automatically
```

---

## Debugging Tools

### Enable Debug Logging

```python
# Add to main_refactored.py or as environment variable
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use environment variable
export DEBUG=true
export LOG_LEVEL=DEBUG
```

### Context Inspection

```python
# Inspect shared context state
from main_refactored import app_context

# Check context contents
print(f"Total thoughts: {app_context.total_thoughts}")
print(f"Session ID: {app_context.session_id}")

# Examine thought graph
if app_context.shared_context:
    graph = app_context.shared_context.thought_graph
    print(f"Graph nodes: {list(graph.nodes())}")
    print(f"Graph edges: {list(graph.edges())}")
```

### Performance Profiling

```python
# Profile thought processing
import time
import cProfile

def profile_thought_processing():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your sequentialthinking call here
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')

# Memory profiling
from memory_profiler import profile

@profile
def process_thought_with_memory_tracking():
    # Your code here
    pass
```

### Health Check Script

```python
# health_check.py - Comprehensive system check
import asyncio
import os
from src.providers.base import LLMProviderFactory
from src.context.shared_context import SharedContext

async def system_health_check():
    checks = {}
    
    # Check environment variables
    checks['env_vars'] = {
        'LLM_PROVIDER': os.getenv('LLM_PROVIDER'),
        'API_KEY_SET': bool(os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY'))
    }
    
    # Check provider initialization
    try:
        team_model, agent_model, config = LLMProviderFactory.create_models()
        checks['provider'] = 'OK'
    except Exception as e:
        checks['provider'] = f'FAILED: {e}'
    
    # Check context backend
    try:
        context = SharedContext()
        await context.update_context('test', 'value')
        test_value = await context.get_context('test')
        checks['context'] = 'OK' if test_value == 'value' else 'FAILED'
    except Exception as e:
        checks['context'] = f'FAILED: {e}'
    
    return checks

# Run health check
if __name__ == "__main__":
    results = asyncio.run(system_health_check())
    for check, status in results.items():
        print(f"{check}: {status}")
```

### Common Log Patterns

**Successful Processing:**
```
INFO: Enhanced App Context initialized
INFO: Teams initialized successfully
INFO: Processing thought 1/3: "Analyze performance..."
DEBUG: Primary team processing: [detailed prompt]
DEBUG: Reflection team processing: [reflection input]
INFO: Thought processed successfully in 2.5s
```

**Error Patterns to Look For:**
```
ERROR: Empty response from team
ERROR: Team response timeout
WARNING: Reflection team failed, continuing with primary only
ERROR: Context update failed: [serialization error]
ERROR: Provider authentication failed
```

## Getting Help

If you're still experiencing issues:

1. **Check logs** for specific error messages
2. **Run health check** script to identify problems
3. **Review configuration** against working examples
4. **Test with minimal setup** to isolate issues
5. **Check GitHub issues** for similar problems
6. **Create issue** with detailed error logs and configuration

**When creating issues, include:**
- Python version (`python --version`)
- Operating system and version
- Environment variables (without API keys)
- Complete error traceback
- Steps to reproduce the issue
- Expected vs actual behavior

This troubleshooting guide covers the most common issues users encounter. Most problems can be resolved by carefully checking configuration and ensuring all dependencies are properly installed.