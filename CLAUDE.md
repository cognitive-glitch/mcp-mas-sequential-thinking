# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Reflective Sequential Thinking MCP Tool** that implements an advanced multi-agent system for complex problem-solving with reflective reasoning capabilities. The system uses Agno framework for multi-agent coordination and FastMCP for serving tools via the Model Context Protocol.

**Key Technologies:**
- **Agno**: Multi-agent system framework
- **FastMCP**: Model Context Protocol server
- **Pydantic**: Data validation with strict typing
- **Python 3.13+**: Required Python version
- **Async/Await**: For efficient multi-agent coordination

## Development Commands

### Build & Run
```bash
# Install dependencies (using uv - recommended)
uv pip install -e ".[dev]"

# Run the MCP server
uv run python main.py

# Or run directly
python main.py
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
pytest

# Run specific test file
pytest tests/test_thought_data.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run single test
pytest tests/test_thought_data.py::test_validation -v
```

### Type Checking Configuration
The project uses strict type checking. Key pyright settings in `pyproject.toml`:
- `typeCheckingMode = "strict"`
- `reportMissingTypeStubs = false` (for third-party libs)
- `reportUnknownMemberType = false` (for dynamic Agno attributes)

## High-Level Architecture

### Current Architecture (To Be Overhauled)
- **Single Team Coordinator**: Uses Agno Team in coordinate mode
- **5 Specialist Agents**: Planner, Researcher, Analyzer, Critic, Synthesizer
- **Linear Processing**: Each thought processed independently
- **Limited Context**: No persistent memory between thoughts
- **One-way Communication**: Coordinator → Specialists → Coordinator

### Proposed Reflective Architecture

#### 1. **Dual-Team System**
```python
# Primary Thinking Team - Handles main reasoning flow
thinking_team = Team(
    name="ThinkingTeam",
    mode="coordinate",
    members=[planner, researcher, analyzer, implementer],
    shared_context=SharedContext()  # NEW: Persistent context
)

# Reflection Team - Provides meta-analysis and feedback
reflection_team = Team(
    name="ReflectionTeam", 
    mode="coordinate",
    members=[meta_analyzer, pattern_recognizer, quality_assessor, decision_critic],
    shared_context=SharedContext()  # Same context instance
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

#### 3. **Shared Context System**
```python
class SharedContext:
    """Persistent context shared across all agents and thoughts"""
    def __init__(self, backend: Literal["redis", "memory"] = "memory"):
        self.memory_store: Dict[str, Any] = {}
        self.tool_usage_history: List[ToolDecision] = []
        self.thought_graph: nx.DiGraph = nx.DiGraph()  # Track relationships
        self.key_insights: List[Insight] = []
        
    async def update_context(self, key: str, value: Any) -> None:
        """Thread-safe context update"""
        
    async def get_relevant_context(self, thought: str) -> Dict[str, Any]:
        """Retrieve context relevant to current thought"""
```

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

#### 5. **New Agent Roles**

**Reflection Team Agents:**
- **MetaAnalyzer**: Analyzes the thinking process itself, identifies cognitive biases
- **PatternRecognizer**: Detects recurring patterns, successful strategies
- **QualityAssessor**: Evaluates response quality, completeness, accuracy
- **DecisionCritic**: Reviews tool selection decisions and their outcomes

**Enhanced Thinking Team:**
- **Implementer**: New agent focused on code generation with tool usage
- **ContextManager**: Manages and retrieves relevant context

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

## Migration Path

1. **Phase 1**: Add SharedContext to existing system
2. **Phase 2**: Implement reflection team alongside existing team
3. **Phase 3**: Enhance ThoughtData model incrementally
4. **Phase 4**: Full bidirectional feedback implementation

## Environment Variables

```bash
# LLM Configuration (DEFAULT: OpenRouter)
LLM_PROVIDER=openrouter  # DEFAULT. Alternatives: openai, gemini, groq
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

## Testing Strategy

### Unit Tests
- Test each agent in isolation with mocked dependencies
- Test ThoughtData validation comprehensively
- Test SharedContext operations with different backends

### Integration Tests
- Test team coordination with real agents
- Test reflection feedback integration
- Test context persistence across thoughts

### End-to-End Tests
- Test complete thought processing flows
- Test revision and branching with reflection
- Test error recovery and resilience

## Performance Considerations

1. **Token Usage**: Reflection team adds ~2-3x token overhead
2. **Latency**: Parallel processing keeps latency increase minimal
3. **Memory**: SharedContext requires careful memory management
4. **Concurrency**: Use connection pooling for Redis backend

## Security Considerations

1. **Input Validation**: Strict Pydantic validation on all inputs
2. **Context Isolation**: Separate contexts for different sessions
3. **Rate Limiting**: Implement per-session rate limits
4. **Sanitization**: Clean context data before storage

## Common Pitfalls to Avoid

1. **Circular Dependencies**: In thought relationships
2. **Context Bloat**: Unbounded context growth
3. **Over-Reflection**: Too many reflection cycles
4. **Type Mismatches**: Between teams and agents
5. **Blocking Operations**: In async code paths

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

## Critical Bug Fixes Needed

### 1. Zero Token API Calls
**Problem**: The tool sometimes sends empty content to the API, causing errors.
**Location**: `main.py` lines ~740-745
**Fix**:
```python
# Replace this problematic code:
coordinator_response_content = team_response.content if hasattr(team_response, 'content') else None
coordinator_response = str(coordinator_response_content) if coordinator_response_content is not None else ""

# With proper error handling:
if not team_response or not hasattr(team_response, 'content') or not team_response.content:
    logger.error(f"Empty response from team for thought #{final_thought_data.thoughtNumber}")
    return "Error: Team coordinator returned empty response. Please retry with a clearer thought."

coordinator_response = str(team_response.content)
```

### 2. Repetitive Guidance Text
**Problem**: Hardcoded guidance text appears even when coordinator provides no meaningful response.
**Fix**:
```python
# Only add guidance if coordinator response has substantial content
if len(coordinator_response.strip()) > 50:  # Minimum meaningful response
    additional_guidance = build_contextual_guidance(coordinator_response)
else:
    additional_guidance = "\n\nError: Insufficient response from coordinator. Please reformulate your thought."
```

### 3. Model Configuration Validation
**Problem**: Missing model configurations can cause silent failures.
**Fix**: Add model validation in `get_model_config()`:
```python
def get_model_config() -> tuple[Type[Model], str, str]:
    provider = os.environ.get("THINKING_LLM_PROVIDER", "openrouter").lower()
    
    # Validate provider is supported
    if provider not in ["openrouter", "openai", "gemini", "groq"]:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Ensure API keys exist
    api_key_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY", 
        "gemini": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY"
    }
    
    if api_key_map[provider] not in os.environ:
        raise ValueError(f"Missing required API key: {api_key_map[provider]}")
```

### 4. Async Team Response Handling
**Problem**: Team responses may timeout or fail silently.
**Fix**: Add timeout and retry logic:
```python
async def get_team_response_with_retry(team, input_prompt, max_retries=2):
    for attempt in range(max_retries):
        try:
            response = await asyncio.wait_for(
                team.arun(input_prompt),
                timeout=30.0  # 30 second timeout
            )
            if response and hasattr(response, 'content') and response.content:
                return response
        except asyncio.TimeoutError:
            logger.warning(f"Team response timeout, attempt {attempt + 1}/{max_retries}")
        except Exception as e:
            logger.error(f"Team response error: {e}")
    
    return None
```

## Claude Code Hooks Integration

This project includes comprehensive Claude Code hooks integration for automated development workflow enhancement.

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

| **Script** | **Purpose** | **Usage** |
|------------|-------------|-----------|
| `install-hooks.sh` | Install/update hooks configuration | `./.claude/install-hooks.sh [--force] [--test]` |
| `test-hooks.sh` | Test hooks functionality | `./.claude/test-hooks.sh [--simulate] [--verbose]` |
| `manage-hooks.sh` | Enable/disable/manage hooks | `./.claude/manage-hooks.sh [enable\|disable\|status]` |

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