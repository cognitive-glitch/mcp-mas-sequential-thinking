# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Reflective Sequential Thinking MCP Tool** that implements an advanced multi-agent system for complex problem-solving with reflective reasoning capabilities. The system uses a custom AsyncTeam implementation (replacing Agno Team for async compatibility) and FastMCP for serving tools via the Model Context Protocol.

**Key Technologies:**
- **AsyncTeam**: Custom async-compatible team coordination (replaces Agno Team)
- **Agno Agent**: Individual agent framework (still used for agents)
- **FastMCP**: Model Context Protocol server
- **Pydantic v2**: Data validation with strict typing and advanced features
- **Python 3.13+**: Required Python version
- **Async/Await**: Core to the entire architecture

**Architecture Highlights:**
- Dual-team system (Primary Thinking + Reflection teams)
- In-memory shared context (no persistence by design)
- Tool selection intelligence with confidence scoring
- Circuit breaker pattern for fault tolerance
- Comprehensive error handling with graceful degradation

## Development Commands

### Build & Run
```bash
# Install dependencies (using uv - recommended)
uv pip install -e ".[dev]"

# Run the MCP server
uv run python src/main.py
```

### Code Quality & Linting
```bash
# Run ruff linter (auto-fix enabled)
ruff check . --fix
ruff format .

# Run pyright type checker (strict mode)
pyright . --pythonversion 3.10

# Run both checks (recommended before commits)
ruff check . --fix && ruff format . && pyright .
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_thought_models.py -v
```

### Type Checking Configuration
The project uses strict type checking. Key pyright settings in `pyproject.toml`:
- `typeCheckingMode = "strict"`
- `reportMissingTypeStubs = false` (for third-party libs)
- `reportUnknownMemberType = false` (for dynamic Agno attributes)

## High-Level Architecture

### Current Architecture (Implemented)

#### AsyncTeam Implementation
The project uses a custom `AsyncTeam` class that replaces Agno's Team to solve async compatibility issues:

```python
class AsyncTeam:
    """Simple async-compatible team replacement for Agno Team."""
    
    async def arun(self, input_prompt: str) -> Any:
        """Async run method that coordinates team members without asyncio.run()."""
        # Runs agents concurrently using asyncio.gather()
        # Returns MockResponse object compatible with original Team interface
```

**Why AsyncTeam?**
- Agno Team internally calls `asyncio.run()` which fails in MCP's async context
- FastMCP tools already run in an event loop
- AsyncTeam uses `asyncio.gather()` for concurrent agent execution
- Maintains compatibility with existing code expecting Team responses

#### Dual-Team System (Active)
```python
# Primary Thinking Team - Handles main reasoning flow
primary_team = AsyncTeam(
    name="PrimaryThinkingTeam",
    members=[planner, researcher, analyzer, critic, synthesizer],
    instructions=await self._generate_adaptive_coordinator_instructions(),
    model=team_model
)

# Reflection Team - Provides meta-analysis and feedback
reflection_team = AsyncTeam(
    name="ReflectionTeam", 
    members=[meta_analyzer, pattern_recognizer, quality_assessor, decision_critic],
    instructions=[...],  # Reflection-specific instructions
    model=team_model
)
```


#### 2. **Enhanced ThoughtData Model**
```python
class ThoughtData(BaseModel):
    # Existing fields...
    
    # New fields for reflection
    context_snapshot: Dict[str, Any] = Field(default_factory=dict)
    tool_decisions: List[ToolDecision] = Field(default_factory=list)
    reflection_feedback: Optional[ReflectionFeedback] = None
    confidence_score: float = Field(0.5, ge=0.0, le=1.0)
    thought_relationships: List[ThoughtRelation] = Field(default_factory=list)
    
class ToolDecision(BaseModel):
    tool_name: str
    rationale: str
    alternatives_considered: List[str]
    confidence: float
    outcome: Optional[str] = None
    
class ReflectionFeedback(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    patterns_detected: List[str]
    overall_quality: float
```

#### 3. **Shared Context System (Simplified)**
```python
class SharedContext:
    """Simple in-memory shared context for multi-agent coordination.
    Maintains state only for the current execution - no persistence, no sessions."""
    
    def __init__(self):
        self.memory_store: Dict[str, Any] = {}
        self.tool_usage_history: List[ToolDecision] = []
        self.thought_graph: nx.DiGraph = nx.DiGraph()
        self.key_insights: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
    async def update_context(self, key: str, value: Any) -> None:
        """Update context with automatic memory management"""
        # Implements FIFO eviction when memory exceeds 100 items
        
    async def get_relevant_context(self, thought_data: ThoughtData) -> Dict[str, Any]:
        """Retrieve context relevant to current thought"""
        # Returns filtered context based on thought domain and keywords
```

**Design Decision**: No persistent storage per user feedback - "persistent memory is not a good design"

#### 4. **Reflective Processing Flow**
```python
async def process_thought_with_reflection(thought_data: ThoughtData) -> ProcessedThought:
    # 1. Update shared context
    await shared_context.update_from_thought(thought_data)
    
    # 2. Primary thinking team processes
    thinking_response = await thinking_team.arun(thought_data)
    
    # 3. Reflection team analyzes the thinking process
    reflection_input = ReflectionInput(
        thought=thought_data,
        thinking_response=thinking_response,
        context=await shared_context.get_relevant_context(thought_data.thought)
    )
    reflection_response = await reflection_team.arun(reflection_input)
    
    # 4. Integrate feedback
    final_response = integrate_reflection_feedback(
        thinking_response, 
        reflection_response
    )
    
    # 5. Update context with outcomes
    await shared_context.update_outcomes(final_response)
    
    return final_response
```

#### 5. **MCP Tools and Prompts**

**Available MCP Tools:**
```python
@mcp.tool()
async def reflectivethinking(thought_data: ThoughtData) -> str:
    """Main reflective thinking tool with dual-team processing"""

@mcp.tool()
async def toolselectthinking(
    thought: str,
    available_tools: Optional[List[str]] = None,
    domain: str = "general",
    context: Optional[Dict[str, Any]] = None
) -> str:
    """Intelligent tool selection for a given thought or task"""

@mcp.tool()
async def reflectivereview(
    session_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    min_quality_threshold: float = 0.0
) -> str:
    """Review and analyze a sequence of thoughts from current session"""
```

**Available MCP Prompts:**
```python
@mcp.prompt("sequential-thinking")
@mcp.prompt("tool-selection") 
@mcp.prompt("thought-review")
@mcp.prompt("complex-problem")
```

## Key Implementation Guidelines

### 1. **Type Safety First**
```python
# Always use strict typing
from typing import TypeVar, Protocol, Literal, TypeAlias
from typing_extensions import TypedDict, NotRequired

# Define protocols for agent interfaces
class ReflectiveAgent(Protocol):
    async def reflect(self, thought: ThoughtData, context: SharedContext) -> ReflectionResult:
        ...
```

### 2. **Async Coordination**
```python
# Use asyncio for efficient multi-agent coordination
async def coordinate_teams(
    thinking_team: Team,
    reflection_team: Team,
    thought: ThoughtData
) -> Tuple[ThinkingResult, ReflectionResult]:
    # Run both teams concurrently when possible
    thinking_task = asyncio.create_task(thinking_team.arun(thought))
    
    # Reflection can start with partial results
    await asyncio.sleep(0.5)  # Small delay
    reflection_task = asyncio.create_task(
        reflection_team.arun({"thought": thought, "partial": True})
    )
    
    return await asyncio.gather(thinking_task, reflection_task)
```

### 3. **Error Handling & Validation**
```python
# Comprehensive error handling
class ThoughtValidationError(ValueError):
    """Raised when thought data fails validation"""
    def __init__(self, field: str, value: Any, reason: str):
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(f"Validation failed for {field}: {reason}")

# Use Pydantic validators extensively
class ThoughtData(BaseModel):
    @field_validator("thought_relationships")
    @classmethod
    def validate_relationships(cls, v: List[ThoughtRelation]) -> List[ThoughtRelation]:
        # Ensure no circular dependencies
        # Validate relationship types
        return v
```

### 4. **Logging & Observability**
```python
# Structured logging for better debugging
logger = structlog.get_logger()

async def log_thought_processing(thought_data: ThoughtData) -> None:
    logger.info(
        "processing_thought",
        thought_number=thought_data.thoughtNumber,
        branch_id=thought_data.branchId,
        has_reflection=thought_data.reflection_feedback is not None,
        confidence=thought_data.confidence_score,
        tool_count=len(thought_data.tool_decisions)
    )
```

## Critical Implementation Details

### AsyncTeam vs Agno Team
The original Agno Team class is **incompatible** with FastMCP's async context:

```python
# ❌ WRONG - This causes "asyncio.run() cannot be called from a running event loop"
from agno.team.team import Team
team = Team(...)
await team.arun(...)  # Internally calls asyncio.run() 

# ✅ CORRECT - Use AsyncTeam
team = AsyncTeam(...)
await team.arun(...)  # Uses asyncio.gather() instead
```

### Agent Model Invocation
Agno models have different async methods:

```python
# In AsyncTeam._run_agent_safe()
if hasattr(agent.model, 'aresponse'):
    response = await agent.model.aresponse(prompt)  # OpenAI models
elif hasattr(agent.model, 'ainvoke'):
    response = await agent.model.ainvoke(prompt)    # Other models
```

## Environment Variables

```bash
# LLM Configuration (DEFAULT: OpenRouter)
REFLECTIVE_LLM_PROVIDER=openrouter  # DEFAULT. Alternatives: openai, gemini, groq
OPENROUTER_API_KEY=your_key
OPENROUTER_TEAM_MODEL_ID=openai/gpt-4-turbo
OPENROUTER_REFLECTION_MODEL_ID=anthropic/claude-3-opus  # NEW

# OpenAI Configuration (Alternative)
OPENAI_API_KEY=your_openai_key
OPENAI_TEAM_MODEL_ID=gpt-4-turbo
OPENAI_REFLECTION_MODEL_ID=gpt-4-turbo  # or gpt-4o

# Google Gemini Configuration (Alternative)
GOOGLE_API_KEY=your_google_key  # For Gemini via Google AI Studio
GEMINI_TEAM_MODEL_ID=gemini-2.0-flash
GEMINI_REFLECTION_MODEL_ID=gemini-2.5-pro-preview

# Context Backend
CONTEXT_BACKEND=memory  # or redis
REDIS_URL=redis://localhost:6379  # if using Redis

# Feature Flags
ENABLE_REFLECTION=true
ENABLE_SHARED_CONTEXT=true
REFLECTION_DELAY_MS=500
```




## Claude Code Hooks Integration

Claude Code supports hooks that execute shell commands at specific lifecycle events. Configure in `~/.claude/settings.json`:

### Hook Events
- **PreToolUse**: Before tool execution (can block/approve)
- **PostToolUse**: After successful tool completion
- **Notification**: When Claude sends notifications
- **Stop**: When main agent finishes responding
- **SubagentStop**: When subagent completes

### Configuration Example
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": {
          "tool_name": "Edit|Write",
          "file_paths": ["*.py"]
        },
        "hooks": [{
          "type": "command",
          "command": "ruff check --fix $CLAUDE_FILE_PATHS && pyright $CLAUDE_FILE_PATHS",
          "timeout": 30
        }]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{
          "type": "command",
          "command": "echo '🚀 Running: $CLAUDE_COMMAND' >> ~/.claude_commands.log"
        }]
      }
    ]
  }
}
```

### Environment Variables in Hooks
- `$CLAUDE_FILE_PATHS`: Space-separated file paths
- `$CLAUDE_NOTIFICATION`: Notification content
- `$CLAUDE_TOOL_OUTPUT`: Tool execution output
- `$CLAUDE_COMMAND`: Command being executed (Bash tool)

### Hook Control Flow
PreToolUse hooks can return JSON to control execution:
```json
{
  "decision": "approve" | "block",
  "reason": "Explanation",
  "suppressOutput": true
}
```

### Security Warning
⚠️ Hooks execute with full user permissions. Validate all commands carefully.

## Known Issues and Solutions

### 1. ✅ FIXED: Team Processing Error
**Original Problem**: `asyncio.run() cannot be called from a running event loop`
**Root Cause**: Agno Team internally uses `asyncio.run()` which conflicts with FastMCP's event loop
**Solution**: Implemented custom `AsyncTeam` class that uses `asyncio.gather()` instead

### 2. Agent Model Method Compatibility
**Issue**: Different model providers have different async method names
**Solution**: AsyncTeam checks for multiple method names (`aresponse`, `ainvoke`)

### 3. Validation Constraints
**ThoughtData Validation Rules:**
- Minimum thought length: 10 characters
- Minimum total thoughts: 5 (enforced by MIN_TOTAL_THOUGHTS)
- Cannot end sequence before thoughtNumber >= totalThoughts
- Keywords are case-preserved (not forced to lowercase)

### 4. Memory Management
**Design Decision**: No persistent storage
- All context is in-memory only
- Automatic FIFO eviction at 100 items
- No session persistence between runs

## Claude Code Hooks Integration (Optional)

**Note**: Claude Code hooks are completely optional. The MCP server runs in stdio mode by default and works perfectly without any hooks configuration. The hooks are only for enhancing the development workflow when using Claude Code.

This project includes optional Claude Code hooks for automated development workflow enhancement.

### Quick Setup

1. **Install hooks configuration**:
```bash
./.claude/install-hooks.sh
```

2. **Test hooks functionality**:
```bash
./.claude/test-hooks.sh
```

3. **Manage hooks status**:
```bash
./.claude/manage-hooks.sh status
```

### Available Hook Scripts

| **Script**         | **Purpose**                        | **Usage**                                             |
| ------------------ | ---------------------------------- | ----------------------------------------------------- |
| `install-hooks.sh` | Install/update hooks configuration | `./.claude/install-hooks.sh [--force] [--test]`       |
| `test-hooks.sh`    | Test hooks functionality           | `./.claude/test-hooks.sh [--simulate] [--verbose]`    |
| `manage-hooks.sh`  | Enable/disable/manage hooks        | `./.claude/manage-hooks.sh [enable\|disable\|status]` |

### Automated Workflows

When properly configured, Claude Code hooks provide:

#### PostToolUse Automation
- **Python files (`*.py`)**: Automatic `ruff check --fix` and `pyright` type checking
- **Test files (`tests/*.py`)**: Automatic test execution with `pytest`
- **Dependency files**: Notifications about dependency updates
- **Core files**: Suggestions to run integration tests

#### PreToolUse Safety
- **Destructive commands**: User confirmation for `rm -rf`, `sudo` operations
- **Git operations**: Status display before `git push`, `git merge`

#### Session Management
- **Activity logging**: All Claude notifications logged to `~/.claude_activity.log`
- **Session summaries**: Project status display when sessions end

### Configuration Details

The hooks configuration is stored in `~/.claude/settings.json` and includes:

```json
{
  "hooks": {
    "PostToolUse": [...],
    "PreToolUse": [...],
    "Notification": [...],
    "Stop": [...]
  },
  "hookSettings": {
    "enableHooks": true,
    "defaultTimeout": 30,
    "continueOnHookFailure": true,
    "logHookExecution": true,
    "maxConcurrentHooks": 3
  }
}
```

### Environment Variables in Hooks

Hooks have access to these environment variables:
- `$CLAUDE_FILE_PATHS`: Space-separated file paths for file operations
- `$CLAUDE_COMMAND`: The command being executed (Bash tool)
- `$CLAUDE_NOTIFICATION`: Notification content
- `$CLAUDE_TOOL_OUTPUT`: Tool execution output

### Security Considerations

⚠️ **Important**: Hooks execute with full user permissions. The provided configuration:
- Uses safe, non-destructive commands by default
- Includes timeout limits to prevent hanging
- Validates inputs and uses absolute paths
- Provides user confirmation for potentially dangerous operations

### Troubleshooting

**Common Issues:**
1. **Hooks not executing**: Check `enableHooks: true` in settings
2. **Permission errors**: Ensure scripts are executable (`chmod +x`)
3. **Tool not found**: Verify `uv`, `ruff`, `pyright` are available
4. **Timeout issues**: Increase timeout values for slow operations

**Debug Commands:**
```bash
# Check hooks status
./.claude/manage-hooks.sh status

# Test specific functionality
./.claude/test-hooks.sh --verbose

# View recent activity
tail ~/.claude_activity.log

# Temporarily disable hooks
./.claude/manage-hooks.sh disable
```

For complete documentation, see [`.claude/README.md`](.claude/README.md).