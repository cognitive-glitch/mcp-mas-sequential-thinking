# Code Style and Conventions

## Type Hints and Validation
- **ALWAYS** use strict type hints throughout the codebase
- Use Pydantic v2 models for data validation with advanced features
- Prefer `TypeVar`, `Protocol`, `Literal`, `TypeAlias` for complex typing
- Use `typing_extensions` for advanced type features
- Replace all 'Any' type hints with proper interfaces where possible

## Code Quality Standards
- **Strict type checking**: pyright in strict mode
- **Auto-formatting**: ruff format for consistent styling
- **Linting**: ruff check with auto-fix enabled
- **No comments unless explicitly requested**: Clean, self-documenting code preferred
- **Comprehensive error handling**: All exceptions should be caught and handled gracefully

## Naming Conventions
- **Classes**: PascalCase (e.g., `EnhancedAppContext`, `CircuitBreaker`)
- **Functions/Methods**: snake_case (e.g., `process_thought`, `initialize_teams`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MIN_THOUGHT_LENGTH`, `DEFAULT_TIMEOUT`)
- **Private methods**: Leading underscore (e.g., `_assess_severity`)
- **Async functions**: Prefix or clear indication of async nature

## Documentation Standards
- **Docstrings**: Clear, concise descriptions for all public functions and classes
- **Type hints**: Comprehensive typing is the primary documentation
- **README maintenance**: Keep README.md updated with current capabilities
- **CLAUDE.md**: Project-specific guidance for AI assistants

## Async/Await Patterns
- **Async-first design**: All I/O operations should be async
- **AsyncTeam pattern**: Use custom AsyncTeam for agent coordination
- **No asyncio.run()**: Avoid in favor of asyncio.gather() for concurrency
- **Error handling**: Proper async exception handling with context managers

## Testing Conventions
- **TDD approach**: Write tests FIRST, then implementation
- **Comprehensive coverage**: Aim for >80% test coverage
- **Async testing**: Use pytest-asyncio for async test support
- **Mock patterns**: Use unittest.mock for external dependencies
- **Test organization**: Group related tests in classes

## Architecture Patterns
- **Dependency Injection**: For FastMCP instance and context management
- **Circuit Breaker**: For fault tolerance in team processing
- **In-Memory State**: No persistent storage, simple context management
- **Dual-Team System**: Separate primary thinking and reflection teams
- **Error Recovery**: Graceful degradation with meaningful error messages

## File Organization
```
src/
├── models/          # Pydantic models and data structures
├── handlers/        # Business logic and processing
├── tools/           # MCP tools and utilities
├── providers/       # LLM provider implementations
├── context/         # Shared context management
├── team/            # AsyncTeam and coordination
├── exceptions.py    # Custom exception classes
├── config.py        # Configuration and constants
└── main.py         # MCP server entry point
```