"""
Tests for split thought models following TDD principles.
Tests written FIRST before refactoring thought_models.py.
"""

import pytest
from typing import Any, Dict
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestCoreModelsModule:
    """Test that core models are properly extracted to core_models.py."""
    
    def test_import_thought_data_from_core_models(self):
        """Test ThoughtData can be imported from core_models module."""
        from models.core_models import ThoughtData, DomainType
        
        # Should be able to create ThoughtData
        thought = ThoughtData(
            thought="Test thought with sufficient content",
            thoughtNumber=1,
            totalThoughts=5,
            nextThoughtNeeded=True
        )
        
        assert thought.thoughtNumber == 1
        assert thought.domain == DomainType.GENERAL  # Default
    
    def test_import_domain_type_enum(self):
        """Test DomainType enum is in core_models."""
        from models.core_models import DomainType
        
        assert DomainType.GENERAL.value == "general"
        assert DomainType.TECHNICAL.value == "technical"
        assert DomainType.CREATIVE.value == "creative"
        assert DomainType.ANALYTICAL.value == "analytical"
        assert DomainType.STRATEGIC.value == "strategic"
    
    def test_import_thought_relation(self):
        """Test ThoughtRelation is in core_models."""
        from models.core_models import ThoughtRelation
        
        relation = ThoughtRelation(
            from_thought=1,
            to_thought=2,
            relation_type="leads_to",
            strength=0.8
        )
        
        assert relation.from_thought == 1
        assert relation.to_thought == 2
    
    def test_import_processed_thought(self):
        """Test ProcessedThought is in core_models."""
        from models.core_models import ProcessedThought, ThoughtData
        
        # Create sample thought data
        thought_data = ThoughtData(
            thought="Sample thought for processing",
            thoughtNumber=1,
            totalThoughts=5,
            nextThoughtNeeded=True
        )
        
        processed = ProcessedThought(
            thought_data=thought_data,
            coordinator_response="Primary team response",
            integrated_response="Integrated analysis",
            next_step_guidance="Continue to next step",
            execution_time_ms=100
        )
        
        assert processed.thought_data.thoughtNumber == 1
        assert processed.execution_time_ms == 100


class TestAnalysisModelsModule:
    """Test that analysis models are properly extracted to analysis_models.py."""
    
    def test_import_reflection_feedback(self):
        """Test ReflectionFeedback is in analysis_models."""
        from models.analysis_models import ReflectionFeedback
        
        feedback = ReflectionFeedback(
            strengths=["Clear thinking"],
            weaknesses=["Needs more depth"],
            suggestions=["Explore alternatives"],
            patterns_detected=["Sequential"],
            overall_quality=0.75
        )
        
        assert len(feedback.strengths) == 1
        assert feedback.overall_quality == 0.75
    
    def test_import_branch_analysis(self):
        """Test BranchAnalysis is in analysis_models."""
        from models.analysis_models import BranchAnalysis
        
        analysis = BranchAnalysis(
            branchId="branch1",
            thoughtCount=5,
            avgConfidence=0.8,
            keyThemes=["optimization"],
            divergencePoint=2,
            convergencePoints=[4]
        )
        
        assert analysis.branchId == "branch1"
        assert analysis.avgConfidence == 0.8
    
    def test_import_thought_sequence_review(self):
        """Test ThoughtSequenceReview is in analysis_models."""
        from models.analysis_models import ThoughtSequenceReview
        
        review = ThoughtSequenceReview(
            totalThoughts=10,
            branches=["main", "branch1"],
            summary="Good progress",
            keyInsights=["insight1", "insight2"],
            strengthsIdentified=["Clear logic"],
            areasForImprovement=["More exploration"],
            overallCoherence=0.85,
            recommendedNextSteps=["Continue analysis"]
        )
        
        assert review.totalThoughts == 10
        assert len(review.branches) == 2
    
    def test_import_quality_indicators(self):
        """Test QualityIndicators is in analysis_models."""
        from models.analysis_models import QualityIndicators
        
        indicators = QualityIndicators(
            clarity_score=0.9,
            depth_score=0.7,
            coherence_score=0.8,
            relevance_score=0.95,
            innovation_score=0.6,
            completeness_score=0.8
        )
        
        assert indicators.clarity_score == 0.9
        # progress_percentage is computed from overall_quality_estimate
        assert indicators.progress_percentage > 80.0  # Should be around 82


class TestToolModelsModule:
    """Test that tool-related models are properly extracted to tool_models.py."""
    
    def test_import_tool_recommendation(self):
        """Test ToolRecommendation is in tool_models."""
        from models.tool_models import ToolRecommendation
        
        tool_rec = ToolRecommendation(
            tool_name="ThinkingTools",
            confidence=0.9,
            rationale="Deep analysis needed",
            priority=1,
            alternatives=["ExaTools"],
            suggested_inputs={"depth": "comprehensive"}
        )
        
        assert tool_rec.tool_name == "ThinkingTools"
        assert tool_rec.confidence == 0.9
    
    def test_import_step_recommendation(self):
        """Test StepRecommendation is in tool_models."""
        from models.tool_models import StepRecommendation
        
        step = StepRecommendation(
            step_description="Analyze the data",
            recommended_tools=[],
            expected_outcome="Clear insights",
            next_step_conditions=["If successful, proceed"]
        )
        
        assert step.step_description == "Analyze the data"
        assert len(step.next_step_conditions) == 1
    
    def test_import_tool_decision(self):
        """Test ToolDecision is in tool_models."""
        from models.tool_models import ToolDecision
        
        decision = ToolDecision(
            tool_name="ExaTools",
            rationale="Research needed",
            alternatives_considered=["ThinkingTools"],
            confidence=0.8,
            outcome="Success"
        )
        
        assert decision.tool_name == "ExaTools"
        assert decision.outcome == "Success"
    
    def test_import_tool_selection_result(self):
        """Test ToolSelectionResult is in tool_models."""
        from models.tool_models import ToolSelectionResult
        
        # First create a tool recommendation
        from models.tool_models import ToolRecommendation
        
        tool_rec = ToolRecommendation(
            tool_name="TestTool",
            confidence=0.85,
            rationale="Testing",
            priority=1
        )
        
        result = ToolSelectionResult(
            recommended_tool=tool_rec,
            reasoning="Based on analysis",
            confidence_score=0.85,
            alternative_tools=["OtherTool"],
            context_factors={"domain": "test"}
        )
        
        assert result.recommended_tool.tool_name == "TestTool"
        assert result.confidence_score == 0.85


class TestBackwardCompatibility:
    """Test that imports from thought_models still work for backward compatibility."""
    
    def test_import_all_from_thought_models(self):
        """Test that all models can still be imported from thought_models."""
        # Core models
        from models.thought_models import ThoughtData, DomainType, ThoughtRelation, ProcessedThought
        
        # Analysis models
        from models.thought_models import ReflectionFeedback, BranchAnalysis, ThoughtSequenceReview
        
        # Tool models
        from models.thought_models import ToolRecommendation, StepRecommendation, ToolDecision
        
        # Should all import successfully
        assert ThoughtData is not None
        assert ReflectionFeedback is not None
        assert ToolRecommendation is not None
    
    def test_models_are_same_objects(self):
        """Test that models imported from different modules are the same objects."""
        from models.thought_models import ThoughtData as ThoughtData1
        from models.core_models import ThoughtData as ThoughtData2
        
        # Should be the exact same class
        assert ThoughtData1 is ThoughtData2
    
    def test_cross_module_relationships(self):
        """Test that models from different modules can work together."""
        from models.core_models import ThoughtData
        from models.analysis_models import ReflectionFeedback
        from models.tool_models import StepRecommendation, ToolRecommendation
        
        # Create a thought with tool recommendations
        tool_rec = ToolRecommendation(
            tool_name="TestTool",
            confidence=0.9,
            rationale="Testing",
            priority=1
        )
        
        step = StepRecommendation(
            step_description="Test step",
            recommended_tools=[tool_rec],
            expected_outcome="Success"
        )
        
        thought = ThoughtData(
            thought="Test thought with tool recommendations",
            thoughtNumber=1,
            totalThoughts=5,
            nextThoughtNeeded=True,
            current_step=step
        )
        
        # Add reflection feedback
        feedback = ReflectionFeedback(
            strengths=["Good"],
            weaknesses=["None"],
            suggestions=["Continue"],
            patterns_detected=["Linear"],
            overall_quality=0.9
        )
        
        thought.reflection_feedback = feedback
        
        # Should all work together
        assert thought.current_step.recommended_tools[0].tool_name == "TestTool"
        assert thought.reflection_feedback.overall_quality == 0.9


class TestModuleStructure:
    """Test the structure and organization of split modules."""
    
    def test_no_circular_imports(self):
        """Test that there are no circular import issues."""
        # Import in different orders
        import models.core_models
        import models.analysis_models
        import models.tool_models
        import models.thought_models
        
        # Reverse order
        import models.thought_models
        import models.tool_models
        import models.analysis_models
        import models.core_models
        
        # Should not raise any import errors
        assert True
    
    def test_module_has_all_exports(self):
        """Test that __all__ is properly defined in each module."""
        import models.core_models
        import models.analysis_models
        import models.tool_models
        
        # Check __all__ exists
        assert hasattr(models.core_models, '__all__')
        assert hasattr(models.analysis_models, '__all__')
        assert hasattr(models.tool_models, '__all__')
        
        # Check key exports
        assert 'ThoughtData' in models.core_models.__all__
        assert 'ReflectionFeedback' in models.analysis_models.__all__
        assert 'ToolRecommendation' in models.tool_models.__all__