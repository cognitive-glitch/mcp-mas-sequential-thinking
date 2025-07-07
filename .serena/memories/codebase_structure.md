# Codebase Structure and Design Patterns

## High-Level Architecture

### Current Architecture (Implemented)
```
FastMCP Server
├── MCP Tools (reflectivethinking, reflectivereview)
├── Dual-Team System
│   ├── Primary Team (Planner, Researcher, Analyzer, Critic, Synthesizer)
│   └── Reflection Team (MetaAnalyzer, PatternRecognizer, QualityAssessor, DecisionCritic)
├── Shared Context (In-memory state management)
├── Error Handling (Circuit breakers, recovery strategies)
└── LLM Provider Abstraction (OpenAI, OpenRouter, Gemini, Groq)
```

## Key Components

### src/main.py (Current: 1104 lines - needs reduction)
- **EnhancedAppContext**: Application state and team management
- **CircuitBreaker**: Fault tolerance pattern
- **EnhancedErrorHandler**: Comprehensive error management
- **FastMCP server setup**: Tool registration and lifecycle

### src/models/thought_models.py (908 lines)
- **ThoughtData**: Core thought representation with Pydantic validation
- **ProcessedThought**: Result of dual-team processing
- **ToolRecommendation**: Intelligent tool selection data
- **ReflectionFeedback**: Meta-analysis results

### src/team/async_team.py
- **AsyncTeam**: Custom async-compatible team coordination
- **MockResponse**: Team response abstraction
- Replaces Agno Team to avoid asyncio.run() conflicts

### src/handlers/
- **ThoughtProcessor**: Business logic for thought processing
- **TeamCoordinator**: Team initialization and management

### src/tools/mcp_tools.py (481 lines)
- MCP tool implementations
- Input validation
- Tool registration with dependency injection

## Critical Design Patterns

### 1. AsyncTeam Pattern (Solves FastMCP Compatibility)
```python
# Problem: Agno Team uses asyncio.run() which conflicts with FastMCP
# Solution: Custom AsyncTeam with asyncio.gather()

class AsyncTeam:
    async def arun(self, input_prompt: str) -> Any:
        # Uses asyncio.gather() instead of asyncio.run()
        return await self._coordinate_agents(input_prompt)
```

### 2. Circuit Breaker Pattern (Fault Tolerance)
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=60):
        self.is_open = False
        self.failure_count = 0
    
    def can_proceed(self) -> bool:
        # Implements open/closed/half-open states
```

### 3. Dependency Injection (MCP Tools)
```python
# Tools don't directly reference FastMCP instance
mcp = None  # Injected at runtime

def set_mcp_instance(mcp_instance):
    global mcp
    mcp = mcp_instance
    _register_mcp_tools()  # Register after injection
```

### 4. Dual-Team Processing
```python
# Primary team handles main analysis
primary_response = await primary_team.arun(thought_input)

# Reflection team provides meta-analysis
reflection_response = await reflection_team.arun(reflection_input)

# Integration of both perspectives
integrated_response = combine_responses(primary, reflection)
```

## Current Refactoring Status (TDD Phase 1)

### Completed
- ✅ MCP tools extracted from main.py
- ✅ Dependency injection implemented
- ✅ AsyncTeam compatibility maintained
- ✅ Test framework established

### In Progress (TDD Phase 1)
- 🔄 Error handling tests written FIRST
- 🔄 CircuitBreaker extraction pending
- 🔄 EnhancedErrorHandler extraction pending

### Planned (TDD Phases 2-3)
- ⏳ EnhancedAppContext extraction
- ⏳ ThoughtProcessor modularity
- ⏳ Team coordination improvements
- ⏳ main.py size reduction to <200 lines

## Important Constraints

1. **No Persistent Storage**: Design choice for simplicity
2. **Async-First**: All I/O must be async-compatible
3. **No asyncio.run()**: Causes conflicts with FastMCP
4. **Strict Typing**: Pydantic v2 validation required
5. **TDD Approach**: Tests must be written before refactoring