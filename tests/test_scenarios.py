"""
Scenario-based integration tests for real-world use cases.
Tests complete workflows and edge cases.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.main import (
    reflectivethinking,
    toolselectthinking,
    reflectivereview,
)


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.fixture
    def mock_team_response_factory(self):
        """Factory for creating mock team responses."""

        def create_response(content: str):
            response = Mock()
            response.content = content
            return response

        return create_response

    @pytest.fixture
    def mock_context_with_team(self, mock_team_response_factory):
        """Create a mock context with functioning team."""
        from src.main import EnhancedAppContext as AppContext

        context = AppContext()

        # Create a mock team that returns contextual responses
        async def smart_team_response(prompt):
            if "performance" in prompt.lower():
                return mock_team_response_factory("""
                Performance Analysis Complete:
                
                **Key Findings:**
                - Database queries are the bottleneck (60% of response time)
                - Memory usage is within acceptable limits
                - CPU utilization peaks during data processing
                
                **Recommendations:**
                1. Implement query caching
                2. Add database indexes
                3. Consider read replicas
                
                RECOMMENDATION: Revise thought #1 to include caching strategy.
                """)
            elif "security" in prompt.lower():
                return mock_team_response_factory("""
                Security Analysis Results:
                
                **Vulnerabilities Found:**
                - SQL injection risk in user input handling
                - Missing rate limiting on API endpoints
                - Weak password requirements
                
                **Tool Suggestions:**
                - Use bandit for Python security scanning
                - Implement OWASP security headers
                - Add input validation middleware
                """)
            else:
                return mock_team_response_factory(
                    f"Analysis complete for: {prompt[:100]}..."
                )

        mock_team = AsyncMock()
        mock_team.arun = smart_team_response
        context.primary_team = mock_team
        context.reflection_team = mock_team
        context.teams_initialized = True

        return context

    @pytest.mark.asyncio
    async def test_software_architecture_design_scenario(self, mock_context_with_team):
        """Test complete software architecture design workflow."""
        with patch("src.main.app_context", mock_context_with_team):
            # 1. Initial requirements analysis
            thought1 = await reflectivethinking(
                thought="Design a scalable e-commerce platform architecture supporting 1M daily users",
                thoughtNumber=1,
                totalThoughts=7,
                nextThoughtNeeded=True,
                topic="System Architecture",
                domain="technical",
                keywords=["scalability", "e-commerce", "architecture", "microservices"],
            )

            # 2. Tool selection for design
            await toolselectthinking(
                thought="Select tools for architecture design and documentation",
                available_tools=[
                    "diagram_tool",
                    "code_generator",
                    "api_designer",
                    "test_framework",
                ],
                domain="technical",
                context={"project_type": "e-commerce", "scale": "large"},
            )

            # 3. Microservices design
            thought2 = await reflectivethinking(
                thought="Design microservices: user service, product catalog, order management, payment processing",
                thoughtNumber=2,
                totalThoughts=7,
                nextThoughtNeeded=True,
                keywords=["microservices", "api", "services"],
            )

            # 4. Branch to explore database strategies
            thought3a = await reflectivethinking(
                thought="Explore SQL database approach with PostgreSQL for transactional data",
                thoughtNumber=3,
                totalThoughts=7,
                nextThoughtNeeded=True,
                branchFromThought=2,
                branchId="sql-approach",
            )

            thought3b = await reflectivethinking(
                thought="Explore NoSQL approach with MongoDB for product catalog",
                thoughtNumber=4,
                totalThoughts=7,
                nextThoughtNeeded=True,
                branchFromThought=2,
                branchId="nosql-approach",
            )

            # 5. Performance considerations
            thought5 = await reflectivethinking(
                thought="Analyze performance requirements and implement caching strategy",
                thoughtNumber=5,
                totalThoughts=7,
                nextThoughtNeeded=True,
                topic="Performance",
            )

            # 6. Revision based on performance analysis
            thought6 = await reflectivethinking(
                thought="Revise architecture to include Redis caching layer and CDN",
                thoughtNumber=6,
                totalThoughts=7,
                nextThoughtNeeded=True,
                isRevision=True,
                revisesThought=2,
            )

            # 7. Final architecture summary
            thought7 = await reflectivethinking(
                thought="Finalize architecture with load balancers, API gateway, and monitoring",
                thoughtNumber=7,
                totalThoughts=7,
                nextThoughtNeeded=False,
            )

            # Review the entire design process
            review = await reflectivereview()

            # Assertions
            assert all(
                [thought1, thought2, thought3a, thought3b, thought5, thought6, thought7]
            )
            assert "Performance Analysis" in thought5
            assert "branches explored" in review.lower()
            assert (
                mock_context_with_team.shared_context.thought_graph.number_of_nodes()
                >= 7
            )

    @pytest.mark.asyncio
    async def test_debugging_complex_issue_scenario(self, mock_context_with_team):
        """Test debugging a complex production issue."""
        with patch("src.main.app_context", mock_context_with_team):
            # 1. Initial bug report
            thought1 = await reflectivethinking(
                thought="Debug production issue: API response times spike to 10s during peak hours",
                thoughtNumber=1,
                totalThoughts=5,
                nextThoughtNeeded=True,
                topic="Debugging",
                keywords=["performance", "api", "production", "latency"],
            )

            # 2. Select debugging tools
            await toolselectthinking(
                thought="Select tools for performance profiling and monitoring",
                available_tools=["profiler", "apm_tool", "log_analyzer", "db_monitor"],
                domain="technical",
            )

            # 3. Analyze findings
            thought2 = await reflectivethinking(
                thought="Analysis shows database connection pool exhaustion during peak traffic",
                thoughtNumber=2,
                totalThoughts=5,
                nextThoughtNeeded=True,
            )

            # 4. Implement fix
            await reflectivethinking(
                thought="Implement connection pooling optimization and query batching",
                thoughtNumber=3,
                totalThoughts=5,
                nextThoughtNeeded=True,
            )

            # 5. Verify fix
            await reflectivethinking(
                thought="Deploy fix to staging and run load tests - response times now < 200ms",
                thoughtNumber=4,
                totalThoughts=5,
                nextThoughtNeeded=True,
            )

            # 6. Post-mortem
            await reflectivethinking(
                thought="Document root cause analysis and prevention measures",
                thoughtNumber=5,
                totalThoughts=5,
                nextThoughtNeeded=False,
            )

            review = await reflectivereview()

            # Verify debugging workflow
            assert "Performance Analysis" in thought1
            assert "connection pool" in thought2.lower()
            assert "Total thoughts: 5" in review

    @pytest.mark.asyncio
    async def test_creative_writing_scenario(self, mock_context_with_team):
        """Test creative writing workflow (non-technical domain)."""
        with patch("src.main.app_context", mock_context_with_team):
            # Different domain type
            result = await reflectivethinking(
                thought="Develop a plot outline for a science fiction novel about AI consciousness",
                thoughtNumber=1,
                totalThoughts=4,
                nextThoughtNeeded=True,
                topic="Creative Writing",
                domain="creative",
                keywords=["plot", "science fiction", "ai", "consciousness"],
            )

            # Branch for different plot directions
            branch1 = await reflectivethinking(
                thought="Explore dystopian angle: AI becomes hostile to humanity",
                thoughtNumber=2,
                totalThoughts=4,
                nextThoughtNeeded=True,
                branchFromThought=1,
                branchId="dystopian-plot",
                domain="creative",
            )

            branch2 = await reflectivethinking(
                thought="Explore optimistic angle: AI and humans achieve symbiosis",
                thoughtNumber=3,
                totalThoughts=4,
                nextThoughtNeeded=True,
                branchFromThought=1,
                branchId="optimistic-plot",
                domain="creative",
            )

            # Merge ideas
            final = await reflectivethinking(
                thought="Combine both perspectives: AI consciousness fragments, some hostile, some benevolent",
                thoughtNumber=4,
                totalThoughts=4,
                nextThoughtNeeded=False,
                domain="creative",
            )

            # Should handle creative domain
            assert all([result, branch1, branch2, final])
            assert (
                mock_context_with_team.session_context.session_domain.value
                == "creative"
            )

    @pytest.mark.asyncio
    async def test_long_running_analysis_scenario(self, mock_context_with_team):
        """Test handling of long-running analysis with many thoughts."""
        with patch("src.main.app_context", mock_context_with_team):
            thoughts = []

            # Simulate a long analysis process
            for i in range(1, 16):  # 15 thoughts
                needs_more = i == 10  # Request extension at thought 10

                thought = await reflectivethinking(
                    thought=f"Step {i}: Analyzing component {i} of the system",
                    thoughtNumber=i,
                    totalThoughts=10 if i <= 10 else 15,  # Extend after thought 10
                    nextThoughtNeeded=(i < 15),
                    needsMoreThoughts=needs_more,
                )
                thoughts.append(thought)

            # Review should handle extended session
            review = await reflectivereview()

            assert len(thoughts) == 15
            assert "Total thoughts: 15" in review
            assert (
                mock_context_with_team.shared_context.thought_graph.number_of_nodes()
                == 15
            )

    @pytest.mark.asyncio
    async def test_error_recovery_scenario(self, mock_context_with_team):
        """Test recovery from errors during thinking process."""
        error_context = mock_context_with_team

        # Make team fail on specific thoughts
        original_arun = error_context.team.arun
        call_count = 0

        async def failing_arun(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second call
                raise Exception("Team processing error")
            return await original_arun(prompt)

        error_context.team.arun = failing_arun

        with patch("src.main.app_context", error_context):
            # First thought should work
            thought1 = await reflectivethinking(
                thought="Start analysis",
                thoughtNumber=1,
                totalThoughts=3,
                nextThoughtNeeded=True,
            )

            # Second thought should handle error gracefully
            thought2 = await reflectivethinking(
                thought="Continue analysis",
                thoughtNumber=2,
                totalThoughts=3,
                nextThoughtNeeded=True,
            )

            # Third thought should work again
            thought3 = await reflectivethinking(
                thought="Complete analysis",
                thoughtNumber=3,
                totalThoughts=3,
                nextThoughtNeeded=False,
            )

            # System should recover and continue
            assert thought1 and thought3
            assert (
                "error" in thought2.lower() or thought2
            )  # Either error message or recovery

    @pytest.mark.asyncio
    async def test_collaborative_analysis_scenario(self, mock_context_with_team):
        """Test multiple analysts working on different aspects."""
        with patch("src.main.app_context", mock_context_with_team):
            # Analyst 1: Security review
            security_1 = await reflectivethinking(
                thought="Security Analyst: Review authentication implementation",
                thoughtNumber=1,
                totalThoughts=6,
                nextThoughtNeeded=True,
                topic="Security Review",
                keywords=["security", "authentication", "vulnerabilities"],
            )

            # Analyst 2: Performance review (parallel branch)
            perf_1 = await reflectivethinking(
                thought="Performance Analyst: Profile application bottlenecks",
                thoughtNumber=2,
                totalThoughts=6,
                nextThoughtNeeded=True,
                topic="Performance Review",
                keywords=["performance", "profiling", "bottlenecks"],
            )

            # Continue both analyses
            security_2 = await reflectivethinking(
                thought="Security: Found SQL injection vulnerability in login",
                thoughtNumber=3,
                totalThoughts=6,
                nextThoughtNeeded=True,
            )

            perf_2 = await reflectivethinking(
                thought="Performance: Database queries need optimization",
                thoughtNumber=4,
                totalThoughts=6,
                nextThoughtNeeded=True,
            )

            # Merge findings
            merge = await reflectivethinking(
                thought="Team Lead: Prioritize security fix, then optimize queries",
                thoughtNumber=5,
                totalThoughts=6,
                nextThoughtNeeded=True,
            )

            # Action plan
            plan = await reflectivethinking(
                thought="Create sprint plan: 1) Fix SQL injection 2) Add query caching 3) Security audit",
                thoughtNumber=6,
                totalThoughts=6,
                nextThoughtNeeded=False,
            )

            review = await reflectivereview()

            # Should show collaborative analysis
            assert all([security_1, security_2, perf_1, perf_2, merge, plan])
            assert "security" in review.lower()
            assert "performance" in review.lower()


# TODO: Additional scenario tests to implement
# - Machine learning model development workflow
# - API design and documentation scenario
# - Code refactoring scenario with multiple iterations
# - Research paper analysis workflow
# - Business strategy planning scenario
# - Incident response and troubleshooting
# - Data pipeline design scenario
# - Mobile app architecture scenario
# - DevOps automation planning
# - Educational content creation workflow
