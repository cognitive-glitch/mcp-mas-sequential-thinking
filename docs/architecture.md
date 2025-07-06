# Architecture Guide

Comprehensive architectural overview of the Reflective Sequential Thinking MCP Tool.

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Dual-Team Design](#dual-team-design)
4. [Data Flow](#data-flow)
5. [Memory Management](#memory-management)
6. [Scalability Patterns](#scalability-patterns)
7. [Design Principles](#design-principles)

## System Overview

The Reflective Sequential Thinking MCP Tool implements a sophisticated multi-agent system that combines sequential reasoning with reflective analysis. The architecture is designed around the principle of **dual-perspective processing**, where every thought undergoes both primary analysis and meta-analytical reflection.

### High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        CLI[Claude Code CLI]
        API[External APIs]
    end
    
    subgraph "MCP Server Layer"
        FastMCP[FastMCP Server]
        Tools[MCP Tools]
    end
    
    subgraph "Core Processing Layer"
        AppContext[Enhanced App Context]
        PrimaryTeam[Primary Team]
        ReflectionTeam[Reflection Team]
    end
    
    subgraph "Data Layer"
        SharedContext[Shared Context]
        ThoughtGraph[Thought Graph]
        SessionData[Session Data]
    end
    
    subgraph "Provider Layer"
        OpenRouter[OpenRouter]
        OpenAI[OpenAI]
        Gemini[Gemini]
    end
    
    CLI --> FastMCP
    API --> FastMCP
    FastMCP --> Tools
    Tools --> AppContext
    AppContext --> PrimaryTeam
    AppContext --> ReflectionTeam
    PrimaryTeam --> SharedContext
    ReflectionTeam --> SharedContext
    SharedContext --> ThoughtGraph
    SharedContext --> SessionData
    AppContext --> OpenRouter
    AppContext --> OpenAI
    AppContext --> Gemini
```

### Key Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **FastMCP Server** | MCP protocol implementation | FastMCP framework |
| **Dual Teams** | Primary + reflection processing | Agno multi-agent system |
| **Shared Context** | Persistent memory management | NetworkX + async patterns |
| **Thought Models** | Type-safe data structures | Pydantic v2 |
| **Provider Factory** | LLM abstraction layer | Factory pattern |

---

## Core Architecture

### Layered Architecture Pattern

The system follows a **layered architecture** with clear separation of concerns:

```mermaid
graph TD
    subgraph "Presentation Layer"
        A1[MCP Tools Interface]
        A2[Claude Code Integration]
    end
    
    subgraph "Application Layer"
        B1[Sequential Thinking Logic]
        B2[Review Generation]
        B3[Context Management]
    end
    
    subgraph "Domain Layer"
        C1[Thought Models]
        C2[Agent Coordination]
        C3[Reflection Engine]
    end
    
    subgraph "Infrastructure Layer"
        D1[LLM Providers]
        D2[Memory Backend]
        D3[Graph Storage]
    end
    
    A1 --> B1
    A2 --> B1
    B1 --> C1
    B2 --> C2
    B3 --> C3
    C1 --> D1
    C2 --> D2
    C3 --> D3
```

#### Layer Responsibilities

**Presentation Layer:**
- MCP tool interfaces (`sequentialthinking`, `sequentialreview`)
- Input validation and output formatting
- Error handling and user feedback

**Application Layer:**
- Core business logic for thought processing
- Session management and state coordination
- Inter-team communication orchestration

**Domain Layer:**
- Rich domain models with validation
- Agent behavior and team coordination
- Reflection algorithms and patterns

**Infrastructure Layer:**
- LLM provider abstractions
- Persistence mechanisms
- External service integrations

---

## Dual-Team Design

### Conceptual Foundation

The dual-team architecture implements **System 1 vs System 2 thinking**:

- **Primary Team (System 1)**: Fast, intuitive, direct analysis
- **Reflection Team (System 2)**: Slow, deliberate, meta-analytical

```mermaid
sequenceDiagram
    participant User
    participant MCP as MCP Server
    participant Primary as Primary Team
    participant Reflection as Reflection Team
    participant Context as Shared Context
    
    User->>MCP: sequentialthinking(thought)
    MCP->>Context: Update context with thought
    MCP->>Primary: Process thought
    Primary->>Primary: Collaborate internally
    Primary->>MCP: Return analysis
    
    alt Response substantial (>50 chars)
        MCP->>Reflection: Analyze primary response
        Reflection->>Reflection: Meta-analysis
        Reflection->>MCP: Return feedback
        MCP->>MCP: Integrate responses
    end
    
    MCP->>Context: Update with results
    MCP->>User: Return integrated response
```

### Primary Team Composition

```mermaid
graph LR
    subgraph "Primary Team"
        Planner[Strategic Planner]
        Researcher[Information Gatherer]
        Analyzer[Core Analyst]
        Critic[Quality Controller]
        Synthesizer[Integration Specialist]
    end
    
    Planner --> Researcher
    Researcher --> Analyzer
    Analyzer --> Critic
    Critic --> Synthesizer
    Synthesizer --> Output[Coordinated Response]
```

**Role Descriptions:**

- **Strategic Planner**: Breaks down problems into manageable components
- **Information Gatherer**: Collects relevant context and background
- **Core Analyst**: Performs deep analytical processing
- **Quality Controller**: Validates reasoning and identifies gaps
- **Integration Specialist**: Synthesizes findings into coherent output

### Reflection Team Composition

```mermaid
graph LR
    subgraph "Reflection Team"
        MetaAnalyzer[Thinking Pattern Analyst]
        PatternRecognizer[Bias Detection Specialist]
        QualityAssessor[Quality Evaluator]
        DecisionCritic[Decision Process Analyst]
    end
    
    MetaAnalyzer --> PatternRecognizer
    PatternRecognizer --> QualityAssessor
    QualityAssessor --> DecisionCritic
    DecisionCritic --> Feedback[Reflection Feedback]
```

**Role Descriptions:**

- **Thinking Pattern Analyst**: Examines cognitive processes and reasoning chains
- **Bias Detection Specialist**: Identifies cognitive biases and blind spots
- **Quality Evaluator**: Assesses completeness, accuracy, and depth
- **Decision Process Analyst**: Reviews tool choices and decision rationale

---

## Data Flow

### Thought Processing Pipeline

```mermaid
flowchart TD
    Start([User Input]) --> Validate{Validate Input}
    Validate -->|Valid| Create[Create ThoughtData]
    Validate -->|Invalid| Error1[Return Validation Error]
    
    Create --> UpdateContext[Update Shared Context]
    UpdateContext --> BuildPrompt[Build Team Prompt]
    BuildPrompt --> PrimaryProcess[Primary Team Processing]
    
    PrimaryProcess --> CheckResponse{Response Substantial?}
    CheckResponse -->|Yes| ReflectionProcess[Reflection Team Processing]
    CheckResponse -->|No| SkipReflection[Skip Reflection]
    
    ReflectionProcess --> Integrate[Integrate Responses]
    SkipReflection --> Integrate
    
    Integrate --> BuildGuidance[Build Next Step Guidance]
    BuildGuidance --> UpdateResults[Update Context with Results]
    UpdateResults --> Return([Return Integrated Response])
    
    PrimaryProcess -->|Error| Error2[Handle Processing Error]
    ReflectionProcess -->|Error| Error3[Handle Reflection Error]
    Error2 --> Return
    Error3 --> Integrate
```

### Context Update Flow

```mermaid
flowchart LR
    subgraph "Input Processing"
        A1[New Thought] --> A2[Validate Data]
        A2 --> A3[Extract Keywords]
        A3 --> A4[Determine Domain]
    end
    
    subgraph "Graph Operations"
        B1[Add Node] --> B2[Create Relationships]
        B2 --> B3[Update Paths]
        B3 --> B4[Check Cycles]
    end
    
    subgraph "Context Enrichment"
        C1[Semantic Analysis] --> C2[Pattern Detection]
        C2 --> C3[Insight Generation]
        C3 --> C4[Performance Tracking]
    end
    
    A4 --> B1
    B4 --> C1
    C4 --> D1[Updated Context]
```

---

## Memory Management

### Shared Context Architecture

The shared context system implements a **graph-based memory model** with semantic retrieval:

```mermaid
graph TB
    subgraph "Memory Store"
        MemStore[Key-Value Store]
        Insights[Insight Database]
        Metrics[Performance Metrics]
    end
    
    subgraph "Graph Layer"
        Nodes[Thought Nodes]
        Edges[Relationships]
        Paths[Semantic Paths]
    end
    
    subgraph "Retrieval Layer"
        Semantic[Semantic Search]
        Temporal[Temporal Queries]
        Pattern[Pattern Matching]
    end
    
    MemStore --> Nodes
    Insights --> Nodes
    Metrics --> Nodes
    Nodes --> Edges
    Edges --> Paths
    Paths --> Semantic
    Paths --> Temporal
    Paths --> Pattern
```

### Context Persistence Strategies

#### Memory Backend (Default)
```python
class MemoryBackend:
    """In-memory storage with thread-safe operations."""
    
    async def store(self, key: str, value: Any) -> None:
        async with self._lock:
            self._storage[key] = value
    
    async def retrieve(self, key: str) -> Optional[Any]:
        async with self._lock:
            return self._storage.get(key)
```

#### Redis Backend (Optional)
```python
class RedisBackend:
    """Redis-backed storage for production deployments."""
    
    async def store(self, key: str, value: Any) -> None:
        serialized = pickle.dumps(value)
        await self._redis.set(key, serialized)
    
    async def retrieve(self, key: str) -> Optional[Any]:
        data = await self._redis.get(key)
        return pickle.loads(data) if data else None
```

### Graph-Based Relationships

The system maintains a **directed graph** of thought relationships:

```mermaid
graph TD
    T1[Thought 1: Problem Analysis]
    T2[Thought 2: Solution Design]
    T3[Thought 3: Implementation Plan]
    T4[Thought 4: Alternative Approach]
    T5[Thought 5: Revised Analysis]
    
    T1 -->|Sequential| T2
    T2 -->|Sequential| T3
    T2 -->|Branch| T4
    T5 -->|Revision| T1
    T4 -->|Merge| T3
    
    style T1 fill:#e1f5fe
    style T2 fill:#e8f5e8
    style T3 fill:#fff3e0
    style T4 fill:#fce4ec
    style T5 fill:#f3e5f5
```

**Relationship Types:**
- **Sequential**: Natural progression (T1 → T2)
- **Revision**: Improvement of previous thought (T5 → T1)
- **Branch**: Alternative exploration (T2 → T4)
- **Merge**: Integration of branches (T4 → T3)

---

## Scalability Patterns

### Horizontal Scaling

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Load Balancer]
    end
    
    subgraph "MCP Server Cluster"
        S1[Server Instance 1]
        S2[Server Instance 2]
        S3[Server Instance 3]
    end
    
    subgraph "Shared Storage"
        Redis[Redis Cluster]
        Graph[Graph Database]
    end
    
    LB --> S1
    LB --> S2
    LB --> S3
    S1 --> Redis
    S2 --> Redis
    S3 --> Redis
    S1 --> Graph
    S2 --> Graph
    S3 --> Graph
```

#### Session Affinity Strategy
```python
class SessionManager:
    """Manages session routing for stateful operations."""
    
    def get_server_for_session(self, session_id: str) -> str:
        # Consistent hashing for session affinity
        hash_value = hashlib.md5(session_id.encode()).hexdigest()
        server_index = int(hash_value[:8], 16) % len(self.servers)
        return self.servers[server_index]
```

### Vertical Scaling

#### Resource Optimization
- **Connection Pooling**: Reuse LLM provider connections
- **Async Processing**: Non-blocking I/O operations
- **Memory Efficiency**: Streaming for large contexts
- **CPU Optimization**: Parallel team processing

```python
# Example: Parallel team processing
async def process_teams_parallel(thought_data: ThoughtData) -> ProcessedThought:
    # Start both teams concurrently
    primary_task = asyncio.create_task(primary_team.arun(primary_prompt))
    
    # Reflection starts with slight delay for better coordination
    await asyncio.sleep(0.5)
    reflection_task = asyncio.create_task(reflection_team.arun(reflection_prompt))
    
    # Wait for both to complete
    primary_result, reflection_result = await asyncio.gather(
        primary_task, reflection_task, return_exceptions=True
    )
    
    return integrate_results(primary_result, reflection_result)
```

---

## Design Principles

### 1. Separation of Concerns

Each component has a **single, well-defined responsibility**:
- **Models**: Data structure and validation
- **Agents**: Specialized reasoning capabilities
- **Teams**: Coordination and orchestration
- **Context**: Memory and state management
- **Providers**: External service abstraction

### 2. Fail-Safe Degradation

The system continues operating even when components fail:
- **Reflection Optional**: Primary team can work alone
- **Provider Fallback**: Multiple LLM providers supported
- **Partial Results**: Return available analysis if complete processing fails
- **Error Recovery**: Automatic retry with exponential backoff

### 3. Extensibility

New capabilities can be added without modifying core components:
- **Plugin Architecture**: New agents can be added to teams
- **Provider Plugins**: New LLM providers via factory pattern
- **Tool Extensions**: Additional MCP tools through registration
- **Context Backends**: Alternative storage mechanisms

### 4. Type Safety

Complete type coverage ensures reliability:
- **Pydantic Models**: Runtime validation with type hints
- **Generic Types**: Parameterized types for flexibility
- **Protocol Definitions**: Interface contracts
- **Async Type Support**: Proper async/await typing

### 5. Observability

Comprehensive monitoring and debugging capabilities:
- **Structured Logging**: Machine-readable log formats
- **Performance Metrics**: Execution time and resource usage
- **Error Tracking**: Detailed error context and stack traces
- **Context Inspection**: Full context state export/import

### 6. Security

Defense-in-depth security approach:
- **Input Validation**: Strict input sanitization
- **Context Isolation**: Session-based context separation
- **Provider Security**: Secure credential management
- **Audit Logging**: Complete operation history

---

## Performance Characteristics

### Latency Profile

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| Single Thought | 2-5 seconds | Depends on LLM provider |
| With Reflection | 4-8 seconds | Includes meta-analysis |
| Context Update | <100ms | In-memory operations |
| Sequential Review | 1-3 seconds | Graph analysis + LLM |

### Memory Usage

| Component | Memory Usage | Scaling Factor |
|-----------|-------------|----------------|
| Thought Graph | ~1KB per thought | Linear with thought count |
| Shared Context | ~10-50MB | Based on session size |
| Team Models | ~500MB-2GB | Depends on LLM provider |
| Total System | ~1-3GB | Production deployment |

### Optimization Strategies

1. **Context Pruning**: Remove old contexts based on TTL
2. **Response Caching**: Cache similar prompts and responses
3. **Model Optimization**: Use smaller models for simple tasks
4. **Async Coordination**: Overlap processing where possible
5. **Resource Pooling**: Share connections and models

This architecture enables the system to handle complex reasoning tasks while maintaining reliability, scalability, and extensibility.