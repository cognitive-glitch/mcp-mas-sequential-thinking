# Suggested Development Commands

## Essential Commands

### Installation & Setup
```bash
# Install dependencies (recommended)
uv pip install -e ".[dev]"

# Install uv if not available
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Running the Server
```bash
# Run MCP server (stdio mode)
uv run python src/main.py

# The server communicates via standard input/output as required by MCP protocol
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_thought_models.py -v

# Run comprehensive test suite
./run_comprehensive_tests.sh
```

### Code Quality & Linting
```bash
# Run all checks (recommended before commits)
ruff check . --fix && ruff format . && pyright .

# Individual commands
ruff check . --fix          # Linter with auto-fix
ruff format .               # Code formatter
pyright . --pythonversion 3.10  # Type checker (strict mode)
```

### Environment Setup
```bash
# Create .env file for development
cp .env.example .env

# Edit .env with your API keys:
# REFLECTIVE_LLM_PROVIDER=openai
# OPENAI_API_KEY=your_key_here
```

## Git Workflow
```bash
# Standard development workflow
git status
git add .
git commit -m "descriptive message"
git push origin main
```

## Common Development Tasks
```bash
# Check project structure
find src -name "*.py" | head -20

# Run specific test categories
uv run pytest tests/test_error_handling.py -v
uv run pytest tests/test_mcp_tools.py -v

# Check logs (if server running)
tail -f ~/.reflective_thinking/logs/reflective_thinking.log
```