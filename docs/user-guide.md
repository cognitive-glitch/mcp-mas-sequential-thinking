# User Guide

Complete guide to using the Reflective Sequential Thinking MCP Tool effectively.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Best Practices](#best-practices)
5. [Common Workflows](#common-workflows)
6. [Tips and Tricks](#tips-and-tricks)

## Getting Started

### Prerequisites

Before using the tool, ensure you have:

1. **Python 3.10+** installed
2. **UV package manager** (recommended) or pip
3. **LLM API key** (OpenRouter, OpenAI, or Gemini)
4. **Claude Code CLI** installed and configured

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/reflective-sequential-thinking-mcp
cd reflective-sequential-thinking-mcp

# Install dependencies
uv pip install -e ".[dev]"

# Set up environment variables
export REFLECTIVE_LLM_PROVIDER=openrouter  # or openai, gemini
export OPENROUTER_API_KEY=your_api_key_here
```

### First Run

Start the MCP server:

```bash
uv run python main_refactored.py
```

The server will start and register the following tools:
- `sequentialthinking` - Process individual thoughts
- `sequentialreview` - Generate session reviews

---

## Basic Usage

### Processing Your First Thought

The `sequentialthinking` tool is your primary interface for analytical reasoning:

```python
# Basic thought processing
result = await sequentialthinking(
    thought="What are the key challenges in implementing microservices architecture?",
    thoughtNumber=1,
    totalThoughts=3,
    nextThoughtNeeded=True
)
```

**Required Parameters:**
- `thought`: Your question or analysis topic
- `thoughtNumber`: Current thought number (starts at 1)
- `totalThoughts`: How many thoughts you plan to process
- `nextThoughtNeeded`: Whether you'll continue with more thoughts

### Example: Simple Analysis Sequence

```python
# Thought 1: Problem identification
await sequentialthinking(
    thought="Identify the main performance bottlenecks in our e-commerce platform",
    thoughtNumber=1,
    totalThoughts=3,
    nextThoughtNeeded=True,
    topic="Performance Analysis",
    domain="technical"
)

# Thought 2: Root cause analysis
await sequentialthinking(
    thought="Analyze the database query patterns causing the bottlenecks",
    thoughtNumber=2,
    totalThoughts=3,
    nextThoughtNeeded=True,
    topic="Performance Analysis",
    domain="technical"
)

# Thought 3: Solution design
await sequentialthinking(
    thought="Design an optimization strategy with caching and query improvements",
    thoughtNumber=3,
    totalThoughts=3,
    nextThoughtNeeded=False,
    topic="Performance Analysis",
    domain="technical"
)

# Generate comprehensive review
review = await sequentialreview()
```

### Understanding the Output

Each `sequentialthinking` call returns:

```
Primary Team Analysis:
[Detailed analysis from the primary team]

Reflection Team Feedback:
Strengths: [What the primary team did well]
Weaknesses: [Areas that could be improved]
Suggestions: [Specific recommendations]

Integrated Analysis:
[Combined insights from both teams]

Next Step Guidance:
[Suggestions for your next thought or action]
```

---

## Advanced Features

### Revision and Branching

#### Revising Previous Thoughts

Sometimes you need to improve or correct earlier analysis:

```python
# Original analysis
await sequentialthinking(
    thought="Our API performance issues are caused by database overload",
    thoughtNumber=1,
    totalThoughts=4,
    nextThoughtNeeded=True
)

# Later, you realize you need to revise
await sequentialthinking(
    thought="Actually, the API performance issues are multifaceted: database queries, network latency, and inefficient algorithms",
    thoughtNumber=2,
    totalThoughts=4,
    nextThoughtNeeded=True,
    isRevision=True,
    revisesThought=1  # Revising thought #1
)
```

#### Exploring Alternative Approaches

Branch from any thought to explore different paths:

```python
# Main analysis path
await sequentialthinking(
    thought="Evaluate microservices architecture for our monolithic app",
    thoughtNumber=1,
    totalThoughts=4,
    nextThoughtNeeded=True,
    topic="Architecture Decision"
)

await sequentialthinking(
    thought="Analyze the benefits of microservices decomposition",
    thoughtNumber=2,
    totalThoughts=4,
    nextThoughtNeeded=True,
    topic="Architecture Decision"
)

# Branch to explore alternative
await sequentialthinking(
    thought="Alternatively, consider modular monolith with domain boundaries",
    thoughtNumber=3,
    totalThoughts=4,
    nextThoughtNeeded=True,
    branchFromThought=1,  # Branch from thought #1
    branchId="modular-monolith",
    topic="Architecture Decision"
)

# Continue main path
await sequentialthinking(
    thought="Compare microservices vs modular monolith approaches",
    thoughtNumber=4,
    totalThoughts=4,
    nextThoughtNeeded=False,
    topic="Architecture Decision"
)
```

### Topic and Domain Alignment

Use topic and domain parameters for better context:

#### Domain Types
- `technical`: Software engineering, architecture, performance
- `creative`: Design, brainstorming, innovation
- `analytical`: Data analysis, research, investigation
- `strategic`: Business planning, decision making
- `research`: Information gathering, literature review
- `planning`: Project management, roadmapping

```python
await sequentialthinking(
    thought="How can we improve user experience on our mobile app?",
    thoughtNumber=1,
    totalThoughts=3,
    nextThoughtNeeded=True,
    topic="Mobile UX Improvement",
    subject="User Interface Design",
    domain="creative",
    keywords=["mobile", "UX", "interface", "usability"]
)
```

### Keyword-Driven Context

Keywords help the system understand and connect related thoughts:

```python
await sequentialthinking(
    thought="Implement real-time notifications for user engagement",
    thoughtNumber=1,
    totalThoughts=2,
    nextThoughtNeeded=True,
    keywords=["realtime", "notifications", "engagement", "websockets", "push"]
)

# Later thought with related keywords
await sequentialthinking(
    thought="Design the notification delivery system architecture",
    thoughtNumber=2,
    totalThoughts=2,
    nextThoughtNeeded=False,
    keywords=["architecture", "delivery", "system", "notifications", "scalability"]
)
```

---

## Best Practices

### 1. Structure Your Thinking Process

**Start Broad, Then Narrow:**
```python
# Thought 1: High-level exploration
"What are all the factors affecting our customer retention?"

# Thought 2: Focus on specific area
"Analyze the impact of onboarding experience on retention rates"

# Thought 3: Dive deep
"Design specific improvements to the first-week user journey"
```

**Use the Rule of 3-7 Thoughts:**
- 3 thoughts: Simple problems (identify → analyze → solve)
- 5 thoughts: Complex problems (explore → analyze → design → validate → implement)
- 7+ thoughts: Very complex or multi-faceted problems

### 2. Leverage Reflection Feedback

The reflection team provides valuable meta-analysis:

**Pay Attention To:**
- **Strengths**: What approaches are working well
- **Weaknesses**: Where your analysis might be incomplete
- **Suggestions**: Specific next steps or considerations

**Act On Feedback:**
```python
# If reflection suggests "Consider security implications"
await sequentialthinking(
    thought="Evaluate the security aspects of the proposed API design",
    thoughtNumber=4,
    totalThoughts=5,
    nextThoughtNeeded=True,
    topic="API Security Review"
)
```

### 3. Use Meaningful Topics and Keywords

**Good Topic Examples:**
- "Database Performance Optimization"
- "User Authentication System Design"
- "Mobile App Feature Planning"

**Effective Keywords:**
- **Technical terms**: "authentication", "performance", "scalability"
- **Business terms**: "revenue", "engagement", "conversion"
- **Domain-specific**: "microservices", "machine learning", "responsive design"

### 4. Strategic Branching

**When to Branch:**
- Exploring fundamentally different approaches
- Considering alternative technologies or methodologies  
- Investigating different stakeholder perspectives

**When to Revise:**
- You discovered new information that changes your analysis
- Initial assumptions were incorrect
- Need to incorporate additional considerations

---

## Common Workflows

### 1. Problem Analysis Workflow

```python
# Phase 1: Problem Definition
await sequentialthinking(
    thought="Define the core problem: slow page load times affecting user experience",
    thoughtNumber=1,
    totalThoughts=6,
    nextThoughtNeeded=True,
    topic="Website Performance Issue",
    domain="technical"
)

# Phase 2: Data Gathering
await sequentialthinking(
    thought="Analyze current performance metrics: average load time 8.5s, 35% bounce rate",
    thoughtNumber=2,
    totalThoughts=6,
    nextThoughtNeeded=True,
    topic="Website Performance Issue",
    keywords=["metrics", "performance", "data"]
)

# Phase 3: Root Cause Analysis
await sequentialthinking(
    thought="Identify root causes: large image files, inefficient database queries, lack of CDN",
    thoughtNumber=3,
    totalThoughts=6,
    nextThoughtNeeded=True,
    topic="Website Performance Issue",
    keywords=["root-cause", "images", "database", "CDN"]
)

# Phase 4: Solution Design
await sequentialthinking(
    thought="Design optimization strategy: image compression, query optimization, CDN implementation",
    thoughtNumber=4,
    totalThoughts=6,
    nextThoughtNeeded=True,
    topic="Website Performance Issue",
    keywords=["optimization", "strategy", "implementation"]
)

# Phase 5: Implementation Planning
await sequentialthinking(
    thought="Create implementation roadmap with priorities and timelines",
    thoughtNumber=5,
    totalThoughts=6,
    nextThoughtNeeded=True,
    topic="Website Performance Issue",
    domain="planning"
)

# Phase 6: Success Metrics
await sequentialthinking(
    thought="Define success metrics and monitoring approach for the optimization",
    thoughtNumber=6,
    totalThoughts=6,
    nextThoughtNeeded=False,
    topic="Website Performance Issue",
    keywords=["metrics", "monitoring", "success"]
)

# Generate comprehensive analysis
review = await sequentialreview()
```

### 2. Architecture Decision Workflow

```python
# Initial exploration
await sequentialthinking(
    thought="Evaluate architecture options for our new microservices platform",
    thoughtNumber=1,
    totalThoughts=5,
    nextThoughtNeeded=True,
    topic="Microservices Architecture",
    domain="strategic"
)

# Option A: Container orchestration
await sequentialthinking(
    thought="Analyze Kubernetes-based deployment with service mesh",
    thoughtNumber=2,
    totalThoughts=5,
    nextThoughtNeeded=True,
    branchFromThought=1,
    branchId="kubernetes-approach",
    topic="Microservices Architecture",
    keywords=["kubernetes", "service-mesh", "containers"]
)

# Option B: Serverless approach  
await sequentialthinking(
    thought="Evaluate serverless architecture with AWS Lambda and API Gateway",
    thoughtNumber=3,
    totalThoughts=5,
    nextThoughtNeeded=True,
    branchFromThought=1,
    branchId="serverless-approach",
    topic="Microservices Architecture", 
    keywords=["serverless", "lambda", "api-gateway"]
)

# Comparative analysis
await sequentialthinking(
    thought="Compare operational complexity, costs, and scalability of both approaches",
    thoughtNumber=4,
    totalThoughts=5,
    nextThoughtNeeded=True,
    topic="Microservices Architecture",
    domain="analytical"
)

# Final recommendation
await sequentialthinking(
    thought="Make final architecture recommendation based on our specific requirements",
    thoughtNumber=5,
    totalThoughts=5,
    nextThoughtNeeded=False,
    topic="Microservices Architecture",
    domain="strategic"
)
```

### 3. Research and Investigation Workflow

```python
# Research question formulation
await sequentialthinking(
    thought="How do successful SaaS companies approach customer onboarding?",
    thoughtNumber=1,
    totalThoughts=4,
    nextThoughtNeeded=True,
    topic="SaaS Onboarding Research",
    domain="research"
)

# Industry analysis
await sequentialthinking(
    thought="Analyze onboarding patterns from Slack, Notion, and Figma",
    thoughtNumber=2,
    totalThoughts=4,
    nextThoughtNeeded=True,
    topic="SaaS Onboarding Research",
    keywords=["slack", "notion", "figma", "patterns", "best-practices"]
)

# Pattern identification
await sequentialthinking(
    thought="Identify common success patterns: progressive disclosure, early wins, social proof",
    thoughtNumber=3,
    totalThoughts=4,
    nextThoughtNeeded=True,
    topic="SaaS Onboarding Research",
    keywords=["patterns", "progressive-disclosure", "social-proof"]
)

# Application to our context
await sequentialthinking(
    thought="Adapt successful patterns to our project management SaaS platform",
    thoughtNumber=4,
    totalThoughts=4,
    nextThoughtNeeded=False,
    topic="SaaS Onboarding Research",
    domain="strategic"
)
```

---

## Tips and Tricks

### 1. Effective Prompt Crafting

**Be Specific:**
```python
# Vague (less effective)
"Improve our website"

# Specific (more effective)  
"Reduce our e-commerce website's checkout abandonment rate from 69% to under 50%"
```

**Include Context:**
```python
await sequentialthinking(
    thought="Given our startup's limited resources and 6-month runway, prioritize mobile app features that maximize user retention",
    thoughtNumber=1,
    totalThoughts=3,
    nextThoughtNeeded=True,
    keywords=["startup", "resources", "mobile", "retention", "prioritization"]
)
```

### 2. Managing Complex Analyses

**Use Thought Numbers Strategically:**
- Reserve final thoughts (last 1-2) for synthesis and recommendations
- Use middle thoughts for deep dives and exploration
- Start with broad exploration in early thoughts

**Track Your Branches:**
```python
# Main analysis
thoughtNumber=1  # Problem exploration
thoughtNumber=2  # Initial solution direction

# Alternative branches
branchFromThought=2, branchId="approach-a"  # Alternative A
branchFromThought=2, branchId="approach-b"  # Alternative B

# Synthesis
thoughtNumber=5  # Compare approaches
thoughtNumber=6  # Final recommendation
```

### 3. Maximizing Reflection Value

**Review Reflection Patterns:**
After several thoughts, look for patterns in reflection feedback:
- Are you consistently missing certain types of analysis?
- Does the reflection team often suggest similar improvements?
- Are there recurring blind spots in your thinking?

**Incorporate Reflection Proactively:**
```python
# If reflection often suggests "consider implementation challenges"
await sequentialthinking(
    thought="Analyze potential implementation challenges and mitigation strategies for the proposed solution",
    thoughtNumber=4,
    totalThoughts=5,
    nextThoughtNeeded=True,
    keywords=["implementation", "challenges", "risks", "mitigation"]
)
```

### 4. Session Review Best Practices

**Generate Reviews Strategically:**
- After completing a major analysis sequence
- Before making important decisions  
- When switching between different topics
- At the end of long thinking sessions

**Act on Review Insights:**
The review provides:
- **Quality metrics**: Which thoughts were most/least effective
- **Pattern identification**: Recurring themes and approaches
- **Branch analysis**: Which exploration paths were most valuable
- **Next steps**: Concrete recommendations for follow-up

### 5. Error Recovery

**If Processing Fails:**
```python
# Simplify your thought
await sequentialthinking(
    thought="Break down the API design problem into smaller components",
    thoughtNumber=2,
    totalThoughts=4,
    nextThoughtNeeded=True
)
```

**If Context Seems Lost:**
- Use more specific keywords to rebuild context
- Reference previous thoughts explicitly
- Consider starting a new topic if the context is too fragmented

### 6. Collaboration Patterns

**Preparing for Team Discussions:**
```python
# Use the tool to prepare comprehensive analysis
await sequentialthinking(
    thought="Prepare arguments for tomorrow's architecture review meeting",
    thoughtNumber=1,
    totalThoughts=3,
    nextThoughtNeeded=True,
    topic="Architecture Review Prep",
    keywords=["meeting", "arguments", "architecture", "review"]
)

# Generate review for sharing
review = await sequentialreview()
# Share the review output with your team
```

**Following Up on Meetings:**
```python
await sequentialthinking(
    thought="Analyze the feedback from today's design review and plan revisions",
    thoughtNumber=1,
    totalThoughts=2,
    nextThoughtNeeded=True,
    topic="Post-Review Analysis",
    keywords=["feedback", "revisions", "design-review"]
)
```

Remember: The tool is most effective when you approach it as a thinking partner rather than just a question-answering system. Engage with the reflection feedback, explore different perspectives through branching, and use the session reviews to understand your thinking patterns and improve over time.