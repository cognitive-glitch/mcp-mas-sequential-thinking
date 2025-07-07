# API Reference

Complete API documentation for the Reflective Sequential Thinking MCP Tool.

## Table of Contents

1. [MCP Tools](#mcp-tools)
2. [Core Models](#core-models)
3. [Agent Architecture](#agent-architecture)
4. [Context Management](#context-management)
5. [Provider Configuration](#provider-configuration)

## MCP Tools

### sequentialthinking

The primary tool for processing individual thoughts in a reflective sequential thinking workflow.

#### Signature
```python
async def sequentialthinking(
    thought: str,
    thoughtNumber: int,
    totalThoughts: int,
    nextThoughtNeeded: bool,
    isRevision: bool = False,
    revisesThought: Optional[int] = None,
    branchFromThought: Optional[int] = None,
    branchId: Optional[str] = None,
    topic: Optional[str] = None,
    subject: Optional[str] = None,
    domain: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    timestamp_ms: Optional[int] = None
) -> str
```

#### Parameters

| Parameter           | Type        | Required | Description                                 |
| ------------------- | ----------- | -------- | ------------------------------------------- |
| `thought`           | `str`       | ✅        | The thought content to process              |
| `thoughtNumber`     | `int`       | ✅        | Current thought number (1-based)            |
| `totalThoughts`     | `int`       | ✅        | Total expected thoughts in sequence         |
| `nextThoughtNeeded` | `bool`      | ✅        | Whether another thought follows             |
| `isRevision`        | `bool`      | ❌        | Whether this revises a previous thought     |
| `revisesThought`    | `int`       | ❌        | Thought number being revised                |
| `branchFromThought` | `int`       | ❌        | Thought to branch from                      |
| `branchId`          | `str`       | ❌        | Unique identifier for branch                |
| `topic`             | `str`       | ❌        | Main topic or subject matter                |
| `subject`           | `str`       | ❌        | Specific subject area                       |
| `domain`            | `str`       | ❌        | Problem domain (technical, creative, etc.)  |
| `keywords`          | `List[str]` | ❌        | Relevant keywords for context               |
| `timestamp_ms`      | `int`       | ❌        | Custom timestamp (defaults to current time) |

#### Returns
- **Type**: `str`
- **Content**: Integrated analysis from primary and reflection teams with next step guidance

#### Example Usage
```python
# Basic thought processing
result = await sequentialthinking(
    thought="Analyze the performance bottlenecks in our API",
    thoughtNumber=1,
    totalThoughts=3,
    nextThoughtNeeded=True,
    topic="Performance Analysis",
    domain="technical",
    keywords=["performance", "API", "bottlenecks"]
)

# Revision example
revision_result = await sequentialthinking(
    thought="Revised analysis with additional metrics",
    thoughtNumber=2,
    totalThoughts=3,
    nextThoughtNeeded=True,
    isRevision=True,
    revisesThought=1,
    topic="Performance Analysis"
)

# Branching example
branch_result = await sequentialthinking(
    thought="Alternative approach using caching",
    thoughtNumber=3,
    totalThoughts=3,
    nextThoughtNeeded=False,
    branchFromThought=1,
    branchId="caching-approach",
    topic="Performance Analysis"
)
```

#### Error Handling
- **ValidationError**: Invalid parameters (e.g., thoughtNumber <= 0)
- **ProcessingError**: Team coordination failures
- **TimeoutError**: Processing timeout (default: 60 seconds)

---

### sequentialreview

Generates a comprehensive review of the thought sequence and provides insights.

#### Signature
```python
async def sequentialreview() -> str
```

#### Parameters
None - analyzes the current session context automatically.

#### Returns
- **Type**: `str`
- **Content**: Formatted review including session overview, branch analysis, insights, and recommendations

#### Example Usage
```python
# Generate review after processing multiple thoughts
review = await sequentialreview()
print(review)
```

#### Sample Output
```
# Sequential Thinking Review

## Session Overview
- **Session ID**: 550e8400-e29b-41d4-a716-446655440000
- **Total Thoughts**: 5
- **Active Branches**: 2
- **Overall Quality**: 0.85/1.0
- **Topic Alignment**: 0.90/1.0

## Branch Analysis
### Main Branch (5 thoughts)
- **Quality Score**: 0.87/1.0
- **Key Insights**: Performance optimization, API restructuring
- **Status**: Completed

### Alternative Branch: caching-approach (2 thoughts)  
- **Quality Score**: 0.82/1.0
- **Key Insights**: Redis integration, cache invalidation
- **Status**: Ongoing

## Recommendations
1. Continue development of caching approach
2. Integrate findings from both branches
3. Focus on API restructuring for next iteration
```

---

## Core Models

### ThoughtData

Primary data model for representing a thought in the system.

#### Definition
```python
class ThoughtData(BaseModel):
    # Core fields
    thought: str
    thoughtNumber: int = Field(..., ge=1)
    totalThoughts: int = Field(..., ge=1) 
    nextThoughtNeeded: bool
    
    # Revision and branching
    isRevision: bool = False
    revisesThought: Optional[int] = None
    branchFromThought: Optional[int] = None
    branchId: Optional[str] = None
    
    # Topic alignment
    topic: Optional[str] = None
    subject: Optional[str] = None
    domain: DomainType = DomainType.GENERAL
    keywords: List[str] = Field(default_factory=list)
    
    # Tool recommendations
    current_step: Optional[StepRecommendation] = None
    previous_steps: List[StepRecommendation] = Field(default_factory=list)
    
    # Context and metadata
    session_context: SessionContext
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    confidence_score: float = Field(0.5, ge=0.0, le=1.0)
    
    # Reflection data
    tool_decisions: List[ToolDecision] = Field(default_factory=list)
    thought_relationships: List[ThoughtRelation] = Field(default_factory=list)
    reflection_feedback: Optional[ReflectionFeedback] = None
```

#### Methods

##### `to_log_format() -> str`
Returns a formatted string representation for logging.

```python
thought = ThoughtData(...)
log_output = thought.to_log_format()
# Output: "Thought 2/5 | Topic: Performance | Domain: technical | ..."
```

##### `get_tool_summary() -> Dict[str, Any]`
Returns summary of tool recommendations.

```python
summary = thought.get_tool_summary()
# Returns: {"step_description": "...", "tools": [...], "expected_outcome": "..."}
```

##### `get_alignment_summary() -> Dict[str, Any]`
Returns topic/subject alignment information.

```python
alignment = thought.get_alignment_summary()  
# Returns: {"topic": "...", "subject": "...", "domain": "...", "keywords": "..."}
```

#### Validation Rules
- `thoughtNumber` must be ≥ 1
- `totalThoughts` must be ≥ `thoughtNumber`
- If `isRevision=True`, `revisesThought` must be < `thoughtNumber`
- If `branchFromThought` is set, `branchId` must be provided
- Keywords are automatically cleaned and validated

---

### ToolRecommendation

Represents a recommendation for tool usage.

#### Definition
```python
class ToolRecommendation(BaseModel):
    tool_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    priority: int = Field(..., ge=1)
    expected_outcome: str
    alternatives: List[str] = Field(default_factory=list)
```

#### Example
```python
tool_rec = ToolRecommendation(
    tool_name="performance_profiler",
    confidence=0.9,
    rationale="System shows latency issues that need profiling",
    priority=1,
    expected_outcome="Identify bottleneck functions and slow queries",
    alternatives=["benchmark_suite", "load_tester"]
)
```

---

### ProcessedThought

Result object from thought processing pipeline.

#### Definition
```python
class ProcessedThought(BaseModel):
    thought_data: ThoughtData
    coordinator_response: str
    reflection_response: Optional[str] = None
    integrated_response: str
    next_step_guidance: str
    
    # Performance metrics
    execution_time_ms: int
    token_usage: Dict[str, int] = Field(default_factory=dict)
    
    # Status flags
    success: bool = True
    error: Optional[str] = None
    tool_recommendations_generated: bool = False
    reflection_applied: bool = False
    context_updated: bool = False
```

#### Example
```python
result = await process_thought_with_dual_teams(thought_data)
if result.success:
    print(f"Processing time: {result.execution_time_ms}ms")
    print(f"Reflection applied: {result.reflection_applied}")
    print(result.integrated_response)
```

---

## Agent Architecture

### Team Structure

#### Primary Team
Handles core thought processing and analysis.

**Members:**
- **Planner**: Strategic planning and approach design
- **Researcher**: Information gathering and context analysis  
- **Analyzer**: Core analytical processing
- **Critic**: Quality assessment and validation
- **Synthesizer**: Integration and conclusion generation

#### Reflection Team  
Provides meta-analysis and feedback on thinking process.

**Members:**
- **MetaAnalyzer**: Analyzes thinking patterns and cognitive processes
- **PatternRecognizer**: Identifies recurring patterns and biases
- **QualityAssessor**: Evaluates response quality and completeness  
- **DecisionCritic**: Reviews tool selection and decision processes

### Team Coordination

```python
# Team initialization
primary_team = Team(
    name="PrimaryTeam",
    mode="coordinate", 
    members=[planner, researcher, analyzer, critic, synthesizer],
    shared_context=shared_context
)

reflection_team = Team(
    name="ReflectionTeam",
    mode="coordinate",
    members=[meta_analyzer, pattern_recognizer, quality_assessor, decision_critic], 
    shared_context=shared_context
)

# Dual-team processing
async def process_thought_with_dual_teams(thought_data: ThoughtData) -> ProcessedThought:
    # Primary team processes thought
    primary_response = await primary_team.arun(input_prompt)
    
    # Reflection team analyzes if response is substantial
    if len(primary_response.content.strip()) > 50:
        reflection_response = await reflection_team.arun(reflection_input)
        
    # Integrate responses
    return integrate_team_responses(primary_response, reflection_response)
```

---

## Context Management

### SharedContext

Thread-safe shared memory system for maintaining context across thoughts and agents.

#### Definition
```python
class SharedContext:
    def __init__(self, backend: Literal["memory", "redis"] = "memory"):
        self.backend = backend
        self.thought_graph = nx.DiGraph()
        self.key_insights: List[Insight] = []
        self.tool_usage_history: List[ToolDecision] = []
        self.performance_metrics: Dict[str, List[float]] = {}
```

#### Key Methods

##### `update_from_thought(thought: ThoughtData) -> None`
Adds thought to context and updates relationship graph.

```python
await shared_context.update_from_thought(thought_data)
```

##### `get_relevant_context(query: str, max_items: int = 10) -> Dict[str, Any]`
Retrieves context relevant to a query using semantic similarity.

```python
context = await shared_context.get_relevant_context(
    "database performance optimization", 
    max_items=5
)
# Returns: {"recent_thoughts": [...], "keywords": [...], "tool_patterns": [...]}
```

##### `add_insight(content: str, source_thought: int, confidence: float, category: str) -> None`
Adds a key insight to the knowledge base.

```python
await shared_context.add_insight(
    "Database queries are the primary bottleneck",
    source_thought=1,
    confidence=0.9,
    category="performance"
)
```

##### `export_state() -> Dict[str, Any]`
Exports complete context state for persistence.

```python
state = await shared_context.export_state()
# Save to file or database
```

##### `import_state(state: Dict[str, Any]) -> None`
Imports previously exported context state.

```python
await shared_context.import_state(saved_state)
```

---

## Provider Configuration

### LLMProviderFactory

Factory for creating LLM model instances across different providers.

#### Supported Providers
- **OpenRouter** (default): Multi-model access
- **OpenAI**: GPT-4 and other OpenAI models  
- **Gemini**: Google's Gemini models
- **Groq**: Fast inference models

#### Usage
```python
from src.providers.base import LLMProviderFactory

# Create models for current provider
team_model, agent_model, config = LLMProviderFactory.create_models()

# Team model for coordination
team_response = await team_model.arun(prompt)

# Agent model for individual agents  
agent_response = await agent_model.arun(prompt)
```

#### Environment Configuration
```bash
# OpenRouter (default)
REFLECTIVE_LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key
OPENROUTER_TEAM_MODEL_ID=openai/gpt-4-turbo
OPENROUTER_AGENT_MODEL_ID=anthropic/claude-3-opus

# OpenAI
REFLECTIVE_LLM_PROVIDER=openai
OPENAI_API_KEY=your_key  
OPENAI_TEAM_MODEL_ID=gpt-4-turbo
OPENAI_AGENT_MODEL_ID=gpt-4-turbo

# Gemini
REFLECTIVE_LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_key
GEMINI_TEAM_MODEL_ID=gemini-2.0-flash
GEMINI_AGENT_MODEL_ID=gemini-2.5-pro-preview
```

---

## Error Handling

### Common Exceptions

#### `ValidationError`
Raised when input data fails Pydantic validation.

```python
try:
    thought = ThoughtData(thoughtNumber=0, ...)  # Invalid
except ValidationError as e:
    print(f"Validation failed: {e}")
```

#### `ProcessingError`  
Raised when thought processing fails.

```python
try:
    result = await sequentialthinking(...)
except ProcessingError as e:
    print(f"Processing failed: {e}")
```

#### `TimeoutError`
Raised when operations exceed timeout limits.

```python
try:
    result = await sequentialthinking(...)
except TimeoutError:
    print("Processing timed out")
```

### Error Recovery

The system includes automatic error recovery mechanisms:
- **Retry logic** for transient failures
- **Graceful degradation** when reflection team fails
- **Partial results** when possible
- **Detailed error logging** for debugging

---

## Type Safety

All public APIs include complete type annotations compatible with:
- **Python 3.10+**: Full type support
- **mypy**: Static type checking
- **pyright**: Microsoft type checker  
- **IDE support**: VS Code, PyCharm, etc.

Example type-safe usage:
```python
from typing import List, Optional
from src.models.thought_models import ThoughtData, ProcessedThought

async def process_thoughts(thoughts: List[ThoughtData]) -> List[ProcessedThought]:
    results: List[ProcessedThought] = []
    for thought in thoughts:
        result = await process_thought_with_dual_teams(thought)
        results.append(result)
    return results
```