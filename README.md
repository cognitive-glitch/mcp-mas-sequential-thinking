# Reflective Sequential Thinking MCP Server

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Model_Context_Protocol-green)](https://modelcontextprotocol.io)

An advanced Model Context Protocol (MCP) server that implements reflective sequential thinking through a dual-team multi-agent system. This tool enables AI models to engage in sophisticated, structured thinking with self-reflection and intelligent tool selection.

## 🌟 Key Features

- **Dual-Team Architecture**: Primary thinking team + reflection team for meta-analysis
- **Async-Native Design**: Custom team coordination without asyncio.run() conflicts
- **Intelligent Tool Selection**: Built-in tool recommendation engine with confidence scoring
- **Reflective Reasoning**: Meta-analysis of thinking processes with quality assessment
- **Branching & Revision Support**: Non-linear thinking with thought revision capabilities
- **Circuit Breaker Pattern**: Fault tolerance with graceful degradation
- **In-Memory Context**: Lightweight shared context (no persistence by design)

## 🏗️ Architecture Overview

### Dual-Team System

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP Client (LLM)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastMCP Server                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Reflective Thinking Tool                │    │
│  │  ┌─────────────────┐    ┌─────────────────┐       │    │
│  │  │ Primary Team     │    │ Reflection Team │       │    │
│  │  │ • Planner       │    │ • MetaAnalyzer  │       │    │
│  │  │ • Researcher    │    │ • PatternRecog  │       │    │
│  │  │ • Analyzer      │    │ • QualityAssess │       │    │
│  │  │ • Critic        │    │ • DecisionCrit  │       │    │
│  │  │ • Synthesizer   │    │                 │       │    │
│  │  └─────────────────┘    └─────────────────┘       │    │
│  │           │                      │                  │    │
│  │           └──────────┬───────────┘                 │    │
│  │                      ▼                             │    │
│  │              Shared Context                        │    │
│  │         (In-Memory State Management)               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```


## 🚀 Installation

### Prerequisites

- Python 3.13 or higher
- An API key for one of the supported LLM providers:
  - OpenAI (recommended)
  - OpenRouter
  - Google Gemini
  - Groq

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/reflective-sequential-thinking-mcp.git
   cd reflective-sequential-thinking-mcp
   ```

2. **Install dependencies with uv (recommended):**
   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install project dependencies
   uv pip install -e ".[dev]"
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file
   cat > .env << EOF
   # LLM Provider Configuration
   REFLECTIVE_LLM_PROVIDER=openai  # or: openrouter, gemini, groq
   OPENAI_API_KEY=your_openai_api_key
   
   # Model IDs (optional - defaults provided)
   OPENAI_TEAM_MODEL_ID=gpt-4-turbo
   OPENAI_AGENT_MODEL_ID=gpt-4-mini
   EOF
   ```

4. **Run the MCP server:**
   ```bash
   uv run python src/main.py
   ```
   
   The server runs in **stdio mode** by default, communicating via standard input/output as required by the MCP protocol.

## 📚 Available MCP Tools

### 1. `reflectivethinking`
Main tool for processing thoughts through the dual-team system.

```python
@mcp.tool()
async def reflectivethinking(thought_data: ThoughtData) -> str:
    """Process a thought through primary and reflection teams."""
```

**Parameters:**
- `thought`: The thought content (min 10 characters)
- `thoughtNumber`: Current thought number (≥1)
- `totalThoughts`: Total estimated thoughts (≥5)
- `nextThoughtNeeded`: Whether more thoughts are needed
- `domain`: Domain type (general, technical, creative, analytical, strategic)
- Additional optional parameters for revision, branching, etc.

### 2. `toolselectthinking`
Intelligent tool selection based on thought content and context.

```python
@mcp.tool()
async def toolselectthinking(
    thought: str,
    available_tools: Optional[List[str]] = None,
    domain: str = "general",
    context: Optional[Dict[str, Any]] = None
) -> str:
    """Get tool recommendations for a given thought."""
```

### 3. `reflectivereview`
Review and analyze a sequence of thoughts from the current session.

```python
@mcp.tool()
async def reflectivereview(
    session_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    min_quality_threshold: float = 0.0
) -> str:
    """Review thought sequence with quality analysis."""
```

## 🎯 MCP Prompts

The server provides four pre-configured prompts:

1. **`sequential-thinking`**: Starter prompt for sequential thinking
2. **`tool-selection`**: Guide for intelligent tool selection
3. **`thought-review`**: Template for reviewing thought sequences
4. **`complex-problem`**: Advanced prompt for complex problem-solving

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Provider Selection
REFLECTIVE_LLM_PROVIDER=openai  # Options: openai, openrouter, gemini, groq

# Provider-specific API Keys
OPENAI_API_KEY=your_key
OPENROUTER_API_KEY=your_key
GOOGLE_API_KEY=your_key
GROQ_API_KEY=your_key

# Model Configuration (optional)
OPENAI_TEAM_MODEL_ID=gpt-4-turbo      # For team coordination
OPENAI_AGENT_MODEL_ID=gpt-4-mini      # For individual agents

# Advanced Settings
ENABLE_REFLECTION=true                 # Enable/disable reflection team
REFLECTION_DELAY_MS=500               # Delay before reflection starts
MAX_CONTEXT_ITEMS=100                 # Maximum items in shared context
```

### MCP Client Configuration

The server operates in **stdio mode** (standard input/output) which is the default for MCP servers.

For Claude Desktop or other MCP clients:

```json
{
  "mcpServers": {
    "reflective-thinking": {
      "command": "uv",
      "args": ["--directory", "/path/to/project", "run", "python", "src/main.py"],
      "env": {
        "REFLECTIVE_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "your_api_key"
      }
    }
  }
}
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_thought_models.py -v
```

### Code Quality

```bash
# Run all checks (recommended before commits)
uv run ruff check . --fix && uv run ruff format . && uv run pyright .
```

### Current Test Status

- ✅ 88 tests passing
- 🔧 35 tests need updates for AsyncTeam migration
- 📊 Core functionality fully operational

## Troubleshooting

### Common Issues

1. **"Empty response from team"**
   - **Cause**: Model API issues or timeout
   - **Solution**: Check API keys and network connectivity

2. **Validation errors on thoughts**
   - **Cause**: Thoughts too short or invalid parameters
   - **Solution**: Ensure thoughts are ≥10 characters, totalThoughts ≥5

3. **High token usage**
   - **Cause**: Dual-team architecture processes each thought multiple times
   - **Solution**: This is by design for quality; adjust models if needed

## 📄 License

This project is licensed under the MIT License.