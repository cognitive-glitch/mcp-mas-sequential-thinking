# Claude Code Hooks Integration

This directory contains Claude Code hooks configuration for automated development workflow enhancement.

## Overview

Claude Code hooks allow you to execute shell commands automatically at specific lifecycle events, providing:

- **Automatic code quality checks** after file modifications
- **Pre-execution validation** for potentially destructive commands  
- **Development workflow automation** with linting, testing, and git operations
- **Session activity logging** for debugging and audit trails

## Setup Instructions

### 1. Copy Hooks Configuration

Copy the hooks configuration to your global Claude settings:

```bash
# Backup existing settings (if any)
cp ~/.claude/settings.json ~/.claude/settings.json.backup 2>/dev/null || true

# Copy project hooks configuration
cp .claude/settings.json ~/.claude/settings.json

# Or merge with existing settings manually
```

### 2. Verify Environment

Ensure required tools are available:

```bash
# Check UV (Python package manager)
uv --version

# Check development tools
uv run ruff --version
uv run pyright --version  
uv run pytest --version

# Check git
git --version
```

### 3. Test Hooks

Test hooks by editing a Python file:

```bash
# Edit any Python file to trigger PostToolUse hooks
echo "# Test comment" >> src/models/__init__.py
```

Expected behavior:
- Automatic ruff linting and fixing
- Pyright type checking
- Results displayed in Claude output

## Hook Categories

### PostToolUse Hooks
Execute after successful tool completion:

| **File Pattern** | **Action** | **Description** |
|------------------|------------|-----------------|
| `*.py` | `ruff check --fix && pyright` | Auto-lint and type-check Python files |
| `tests/*.py` | `pytest $FILE -v` | Run specific tests after modification |
| `pyproject.toml` | Dependency notification | Alert about dependency changes |
| Core files | Integration test reminder | Suggest running integration tests |

### PreToolUse Hooks  
Execute before tool execution (can block):

| **Command Pattern** | **Action** | **Description** |
|---------------------|------------|-----------------|
| `rm -rf`, `sudo` | User confirmation | Confirm destructive operations |
| `git push`, `git merge` | Git status display | Show git state before operations |

### Notification & Stop Hooks
- **Notification**: Log all Claude notifications to `~/.claude_activity.log`
- **Stop**: Display project status when session ends

## Environment Variables

Hooks have access to these environment variables:

- `$CLAUDE_FILE_PATHS`: Space-separated file paths for file operations
- `$CLAUDE_COMMAND`: The command being executed (Bash tool)  
- `$CLAUDE_NOTIFICATION`: Notification content
- `$CLAUDE_TOOL_OUTPUT`: Tool execution output

## Configuration Options

### Hook Settings
```json
{
  "hookSettings": {
    "enableHooks": true,           // Master enable/disable
    "defaultTimeout": 30,          // Default timeout in seconds
    "continueOnHookFailure": true, // Continue if hooks fail
    "logHookExecution": true,      // Log hook execution
    "maxConcurrentHooks": 3        // Max parallel hooks
  }
}
```

### Matcher Patterns
```json
{
  "matcher": {
    "tool_name": "Edit|Write|MultiEdit",     // Tool name regex
    "file_paths": ["*.py", "src/**/*.py"],   // File glob patterns  
    "command_patterns": ["git push"]         // Command regex patterns
  }
}
```

## Hook Control Flow

### PreToolUse Return Codes
PreToolUse hooks can control execution:

```json
{
  "decision": "approve|block",    // Allow or block execution
  "reason": "Security check",     // Reason for decision
  "suppressOutput": true          // Hide hook output
}
```

### Error Handling
- Hooks timeout after specified duration
- Failed hooks log errors but don't stop execution (if `continueOnHookFailure: true`)
- Hook output is displayed in Claude interface

## Customization

### Adding Custom Hooks

1. **Performance Monitoring**:
```json
{
  "matcher": {"tool_name": "Bash", "command_patterns": ["pytest"]},
  "hooks": [{
    "type": "command", 
    "command": "time $CLAUDE_COMMAND",
    "description": "Time test execution"
  }]
}
```

2. **Code Coverage**:
```json
{
  "matcher": {"tool_name": "Edit", "file_paths": ["*.py"]},
  "hooks": [{
    "type": "command",
    "command": "uv run pytest --cov=. --cov-report=term-missing",
    "description": "Check code coverage after changes"
  }]
}
```

3. **Security Scanning**:
```json
{
  "matcher": {"file_paths": ["requirements*.txt", "pyproject.toml"]},
  "hooks": [{
    "type": "command",
    "command": "uv run safety check",
    "description": "Scan dependencies for vulnerabilities"  
  }]
}
```

### Project-Specific Hooks

For different projects, create separate `.claude/settings.json` files:

```bash
# Project A hooks
cp .claude/settings-project-a.json ~/.claude/settings.json

# Project B hooks  
cp .claude/settings-project-b.json ~/.claude/settings.json
```

## Troubleshooting

### Common Issues

1. **Hooks not executing**:
   - Check `enableHooks: true` in settings
   - Verify file paths match patterns
   - Check tool names are correct

2. **Permission errors**:
   - Ensure scripts are executable: `chmod +x script.sh`
   - Check file permissions for hook commands

3. **Timeout issues**:
   - Increase timeout values for slow operations
   - Use background processes for long-running tasks

4. **Path issues**:
   - Use absolute paths in hook commands
   - Set working directory with `cd` in commands

### Debug Mode

Enable verbose logging:
```json
{
  "hookSettings": {
    "logHookExecution": true,
    "debugMode": true
  }
}
```

### Disable Hooks Temporarily
```bash
# Quick disable
echo '{"hooks": {}, "hookSettings": {"enableHooks": false}}' > ~/.claude/settings.json

# Re-enable 
cp .claude/settings.json ~/.claude/settings.json
```

## Security Considerations

⚠️ **Security Warning**: Hooks execute with full user permissions.

### Best Practices:
1. **Validate hook commands** before deploying
2. **Use specific matchers** to avoid unexpected execution
3. **Test hooks thoroughly** in safe environments
4. **Avoid hardcoded credentials** in hook commands
5. **Use timeout limits** to prevent hanging
6. **Review hook logs** regularly for suspicious activity

### Example Safe Hook:
```json
{
  "type": "command",
  "command": "cd /safe/project/path && timeout 30 ruff check --fix *.py",
  "timeout": 35,
  "description": "Safe linting with timeout and path restriction"
}
```

## Integration with Development Tools

### VS Code Integration
Use hooks to trigger VS Code extensions:
```bash
code --command "python.linting.lint" $CLAUDE_FILE_PATHS
```

### CI/CD Integration  
Hooks can prepare for CI/CD:
```bash
# Pre-commit style checks
pre-commit run --files $CLAUDE_FILE_PATHS

# Update CI cache
docker build --cache-from myproject:cache .
```

### Monitoring Integration
Send metrics to monitoring systems:
```bash
curl -X POST "http://metrics-server/hooks" -d "{\"event\": \"file_modified\", \"files\": \"$CLAUDE_FILE_PATHS\"}"
```

This hooks system provides powerful automation while maintaining safety through configurable controls and clear documentation.