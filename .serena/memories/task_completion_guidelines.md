# Task Completion Guidelines

## Before Committing Code

### 1. Run All Quality Checks
```bash
# Essential command to run before any commit
ruff check . --fix && ruff format . && pyright .
```

### 2. Test Execution
```bash
# Run full test suite
uv run pytest

# Run with coverage if making significant changes
uv run pytest --cov=src --cov-report=html
```

### 3. Specific Test Categories
```bash
# Run relevant tests for your changes
uv run pytest tests/test_error_handling.py -v     # For error handling changes
uv run pytest tests/test_mcp_tools.py -v         # For MCP tool changes
uv run pytest tests/test_thought_models.py -v    # For model changes
```

## TDD Development Process

### Phase 1: Error Handling (Current Focus)
1. **Write tests FIRST** for CircuitBreaker, EnhancedErrorHandler
2. **Extract classes** from main.py AFTER tests pass
3. **Verify extraction** doesn't break existing functionality

### Phase 2: Context Management
1. **Write tests FIRST** for EnhancedAppContext
2. **Extract to separate module** after tests pass
3. **Update imports** and ensure compatibility

### Phase 3: Team Coordination
1. **Write tests FIRST** for ThoughtProcessor, AsyncTeam, TeamCoordinator
2. **Refactor for modularity** while maintaining async compatibility
3. **Validate all integrations** work correctly

## Code Quality Checklist

- [ ] All type hints are specific (no 'Any' types)
- [ ] All functions have proper error handling
- [ ] Async patterns are used consistently
- [ ] No hardcoded values (use config.py)
- [ ] Tests cover edge cases and error conditions
- [ ] Documentation is updated if public API changed

## Architecture Validation

- [ ] main.py remains under 200 lines after extractions
- [ ] No circular imports introduced
- [ ] AsyncTeam compatibility maintained
- [ ] FastMCP integration works correctly
- [ ] Circuit breaker patterns function properly

## Performance Considerations

- [ ] No blocking operations in async code
- [ ] Memory usage stays within reasonable bounds
- [ ] Context management doesn't leak memory
- [ ] Agent coordination doesn't create deadlocks

## Final Verification

```bash
# Comprehensive verification
uv run python src/main.py &
sleep 2
kill %1  # Basic server startup test

# Integration test
python test_basic_mcp.py
```

## Git Workflow
1. **Create feature branch** for significant changes
2. **Commit frequently** with descriptive messages
3. **Test thoroughly** before pushing
4. **Clean up commits** if needed before merge

## When Task is Complete
- All tests pass
- Code quality checks pass
- Architecture goals achieved
- Documentation updated
- Performance verified