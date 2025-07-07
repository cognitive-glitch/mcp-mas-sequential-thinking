# Quick Start Guide

Get up and running with the Reflective Sequential Thinking MCP Tool in 5 minutes.

## Prerequisites

- Python 3.10+
- OpenRouter API key (recommended) or OpenAI/Gemini key
- Claude Code CLI installed

## Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/your-org/reflective-sequential-thinking-mcp
cd reflective-sequential-thinking-mcp

# Install dependencies using UV (recommended)
pip install uv
uv pip install -e ".[dev]"
```

## Step 2: Configuration

Set your API key:

```bash
# Option 1: OpenRouter (Recommended - access to multiple models)
export REFLECTIVE_LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=your_openrouter_key

# Option 2: OpenAI
export REFLECTIVE_LLM_PROVIDER=openai  
export OPENAI_API_KEY=your_openai_key

# Option 3: Google Gemini
export REFLECTIVE_LLM_PROVIDER=gemini
export GOOGLE_API_KEY=your_google_key
```

## Step 3: Start the Server

```bash
uv run python main_refactored.py
```

You should see:
```
✅ Enhanced App Context initialized
✅ Teams initialized successfully  
🚀 FastMCP server starting on port 8000
📋 Available tools: sequentialthinking, sequentialreview
```

## Step 4: Your First Analysis

Open Claude Code and use the tools:

### Basic Sequential Thinking

```python
# Start with a clear problem statement
result1 = await sequentialthinking(
    thought="How can we improve the loading speed of our e-commerce website?",
    thoughtNumber=1,
    totalThoughts=3,
    nextThoughtNeeded=True,
    topic="Website Performance",
    domain="technical"
)
```

### Continue the Analysis

```python
# Analyze specific aspects
result2 = await sequentialthinking(
    thought="Identify the main performance bottlenecks: large images, database queries, and third-party scripts",
    thoughtNumber=2,
    totalThoughts=3,
    nextThoughtNeeded=True,
    topic="Website Performance",
    keywords=["images", "database", "scripts", "bottlenecks"]
)
```

### Conclude with Solutions

```python
# Provide concrete solutions
result3 = await sequentialthinking(
    thought="Design optimization strategy: implement image compression, query optimization, and async script loading",
    thoughtNumber=3,
    totalThoughts=3,
    nextThoughtNeeded=False,
    topic="Website Performance",
    keywords=["optimization", "compression", "async", "strategy"]
)
```

### Generate Review

```python
# Get comprehensive analysis summary
review = await sequentialreview()
print(review)
```

## Expected Output

Each `sequentialthinking` call returns structured analysis:

```
Primary Team Analysis:
The e-commerce website performance issue requires a systematic approach...

Reflection Team Feedback:
Strengths: Clear problem identification and systematic breakdown
Weaknesses: Could include specific metrics and benchmarks
Suggestions: Consider mobile performance impact and user experience metrics

Integrated Analysis:
Based on dual-team analysis, the optimization strategy should focus on...

Next Step Guidance:
For your next thought, consider diving deeper into implementation priorities...
```

The `sequentialreview` provides session overview:

```
# Sequential Thinking Review

## Session Overview
- **Total Thoughts**: 3
- **Topic**: Website Performance
- **Overall Quality**: 0.87/1.0

## Key Insights
1. Performance bottlenecks identified in three main areas
2. Optimization strategy provides actionable solutions
3. Implementation approach balances impact and effort

## Recommendations
1. Start with image optimization for quick wins
2. Implement database query improvements
3. Consider Progressive Web App features for mobile performance
```

## Advanced Example: Branching Analysis

```python
# Main analysis path
await sequentialthinking(
    thought="Evaluate cloud architecture options for our microservices platform",
    thoughtNumber=1,
    totalThoughts=4,
    nextThoughtNeeded=True,
    topic="Cloud Architecture"
)

# Explore Container approach
await sequentialthinking(
    thought="Analyze Kubernetes deployment with container orchestration benefits",
    thoughtNumber=2,
    totalThoughts=4,
    nextThoughtNeeded=True,
    branchFromThought=1,
    branchId="kubernetes-path",
    topic="Cloud Architecture",
    keywords=["kubernetes", "containers", "orchestration"]
)

# Explore Serverless approach  
await sequentialthinking(
    thought="Evaluate serverless architecture with AWS Lambda and managed services",
    thoughtNumber=3,
    totalThoughts=4,
    nextThoughtNeeded=True,
    branchFromThought=1,
    branchId="serverless-path", 
    topic="Cloud Architecture",
    keywords=["serverless", "lambda", "managed-services"]
)

# Compare and decide
await sequentialthinking(
    thought="Compare both approaches and recommend the best fit for our requirements",
    thoughtNumber=4,
    totalThoughts=4,
    nextThoughtNeeded=False,
    topic="Cloud Architecture",
    domain="strategic"
)

# Generate comprehensive review
review = await sequentialreview()
```

## Key Features Demonstrated

✅ **Dual-Team Processing**: Primary analysis + reflection feedback  
✅ **Context Persistence**: Thoughts build on previous analysis  
✅ **Topic Alignment**: Related thoughts grouped by topic  
✅ **Branching**: Explore multiple approaches simultaneously  
✅ **Domain Awareness**: Technical, creative, strategic, analytical domains  
✅ **Comprehensive Reviews**: Session summaries with insights  

## Next Steps

- Read the [User Guide](../user-guide.md) for detailed usage patterns
- Explore [Advanced Examples](advanced.md) for complex scenarios
- Set up [Claude Code Hooks](../hooks.md) for automated workflow
- Check [Troubleshooting](../troubleshooting.md) if you encounter issues

## Common Quick Fixes

**Server won't start?**
```bash
# Check your API key is set
echo $OPENROUTER_API_KEY

# Verify dependencies
uv pip install -e ".[dev]"
```

**Empty responses?**
```bash
# Check your provider configuration
python -c "from src.providers.base import LLMProviderFactory; print(LLMProviderFactory.create_models())"
```

**Need help?**
- Check the [troubleshooting guide](../troubleshooting.md)
- Review example outputs in [use cases](use-cases.md)
- Join our community discussions

You're now ready to use the Reflective Sequential Thinking MCP Tool for sophisticated analytical reasoning!