# Technology Stack

## Core Technologies
- **Python 3.13+**: Required Python version
- **FastMCP**: Model Context Protocol server framework with async event loop
- **Custom AsyncTeam**: Async-compatible team coordination (replaces Agno Team)
- **Agno Agent**: Individual agent framework (still used for agents)
- **Pydantic v2**: Data validation with strict typing and advanced features

## LLM Providers Supported
- **OpenAI** (recommended): GPT-4 Turbo, GPT-4 Mini
- **OpenRouter**: Multiple model access
- **Google Gemini**: Gemini 2.0 Flash, Gemini 2.5 Pro Preview
- **Groq**: High-speed inference

## Development Tools
- **uv**: Fast Python package installer and resolver (recommended)
- **ruff**: Fast Python linter and formatter
- **pyright**: Static type checker (strict mode)
- **pytest**: Testing framework with async support
- **coverage.py**: Code coverage reporting

## Architecture Patterns
- **Async/Await**: Core to the entire architecture
- **Circuit Breaker Pattern**: Fault tolerance
- **Dependency Injection**: For FastMCP instance management
- **Dual-Team System**: Primary Thinking + Reflection teams
- **In-Memory State Management**: No persistence by design

## Key Design Decisions
- **No Persistent Storage**: In-memory context only for simplicity
- **AsyncTeam over Agno Team**: Solves asyncio.run() conflicts with FastMCP
- **Strict Type Checking**: Pydantic v2 with comprehensive validation
- **TDD Approach**: Test-driven development for reliability