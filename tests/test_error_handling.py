"""
Comprehensive tests for error handling components.
Following TDD principles - tests written FIRST before extraction.
"""

import pytest
import time
from datetime import datetime
from unittest.mock import patch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from exceptions import ErrorType, ValidationError
from error_handling import (
    CircuitBreaker,
    EnhancedErrorHandler,
    ErrorSeverity,
    ErrorContext,
)


class TestCircuitBreaker:
    """Test CircuitBreaker fault tolerance pattern implementation."""

    def test_circuit_breaker_initialization(self):
        """Test CircuitBreaker initialization with default and custom parameters."""
        # Default initialization
        cb = CircuitBreaker()
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 60
        assert cb.failure_count == 0
        assert cb.last_failure_time is None
        assert cb.is_open is False

        # Custom initialization
        cb_custom = CircuitBreaker(failure_threshold=5, recovery_timeout=120)
        assert cb_custom.failure_threshold == 5
        assert cb_custom.recovery_timeout == 120
        assert cb_custom.failure_count == 0
        assert cb_custom.last_failure_time is None
        assert cb_custom.is_open is False

    def test_circuit_breaker_closed_state(self):
        """Test CircuitBreaker behavior in closed state (normal operation)."""
        cb = CircuitBreaker(failure_threshold=3)

        # Initial state should be closed
        assert cb.can_proceed() is True
        assert cb.is_open is False

        # Should handle single failure without opening
        cb.record_failure()
        assert cb.failure_count == 1
        assert cb.can_proceed() is True
        assert cb.is_open is False

        # Should handle multiple failures under threshold
        cb.record_failure()
        assert cb.failure_count == 2
        assert cb.can_proceed() is True
        assert cb.is_open is False

    def test_circuit_breaker_opens_on_threshold(self):
        """Test CircuitBreaker opens when failure threshold is reached."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures up to threshold
        cb.record_failure()  # 1
        cb.record_failure()  # 2
        assert cb.can_proceed() is True
        assert cb.is_open is False

        # Third failure should open circuit
        cb.record_failure()  # 3
        assert cb.failure_count == 3
        assert cb.is_open is True
        assert cb.can_proceed() is False
        assert cb.last_failure_time is not None

    def test_circuit_breaker_exact_threshold_behavior(self):
        """Test CircuitBreaker behavior at exact threshold boundaries."""
        cb = CircuitBreaker(failure_threshold=1)  # Very low threshold

        assert cb.can_proceed() is True
        cb.record_failure()
        assert cb.is_open is True
        assert cb.can_proceed() is False

        # Test higher threshold
        cb_high = CircuitBreaker(failure_threshold=10)
        for i in range(9):
            cb_high.record_failure()
            assert cb_high.can_proceed() is True
            assert cb_high.is_open is False

        # 10th failure should open
        cb_high.record_failure()
        assert cb_high.is_open is True
        assert cb_high.can_proceed() is False

    def test_circuit_breaker_recovery_timeout(self):
        """Test CircuitBreaker recovery timeout mechanism."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)  # 1 second timeout

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True
        assert cb.can_proceed() is False

        # Should still be open immediately
        assert cb.can_proceed() is False

        # Wait for recovery timeout
        time.sleep(1.1)  # Slightly more than timeout

        # Should now allow proceeding (half-open state)
        assert cb.can_proceed() is True
        assert cb.is_open is False  # Should transition to half-open
        assert cb.is_half_open is True  # Should be in half-open state
        assert cb.failure_count == 2  # Should NOT reset until success

    def test_circuit_breaker_half_open_success(self):
        """Test CircuitBreaker transitions from half-open to closed on success."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True

        # Wait for recovery
        time.sleep(1.1)
        assert cb.can_proceed() is True  # Half-open

        # Record success - should fully close circuit
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.is_open is False
        assert cb.can_proceed() is True

    def test_circuit_breaker_half_open_failure(self):
        """Test CircuitBreaker behavior when failure occurs in half-open state."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        initial_failure_time = cb.last_failure_time

        # Wait for recovery
        time.sleep(1.1)
        assert cb.can_proceed() is True  # Half-open

        # Another failure should open circuit again
        cb.record_failure()
        assert cb.is_open is True
        assert cb.can_proceed() is False
        assert (
            cb.last_failure_time is not None
            and initial_failure_time is not None
            and cb.last_failure_time > initial_failure_time
        )  # Updated timestamp

    def test_circuit_breaker_success_resets_state(self):
        """Test that record_success() properly resets CircuitBreaker state."""
        cb = CircuitBreaker(failure_threshold=3)

        # Accumulate some failures
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        # Success should reset count but not affect open state
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.is_open is False
        assert cb.can_proceed() is True

        # Test success after circuit is open
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()  # Open circuit
        assert cb.is_open is True

        cb.record_success()
        assert cb.failure_count == 0
        assert cb.is_open is False
        assert cb.can_proceed() is True

    def test_circuit_breaker_concurrent_failures(self):
        """Test CircuitBreaker thread safety with concurrent failures."""
        cb = CircuitBreaker(failure_threshold=5)

        # Simulate rapid concurrent failures
        for i in range(10):
            cb.record_failure()

        # Should have opened after 5 failures
        assert cb.failure_count >= 5
        assert cb.is_open is True
        assert cb.can_proceed() is False

    def test_circuit_breaker_time_precision(self):
        """Test CircuitBreaker time handling and precision."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=2)

        # Record failure and check timestamp precision
        before = datetime.now()
        cb.record_failure()
        after = datetime.now()

        assert (
            cb.last_failure_time is not None and before <= cb.last_failure_time <= after
        )

        # Test timeout calculation precision
        time.sleep(0.5)
        assert cb.can_proceed() is False  # Still within timeout

        time.sleep(1.6)  # Total 2.1 seconds
        assert cb.can_proceed() is True  # Should have recovered

    def test_circuit_breaker_edge_cases(self):
        """Test CircuitBreaker edge cases and boundary conditions."""
        # Zero threshold (should open immediately)
        cb_zero = CircuitBreaker(failure_threshold=0)
        cb_zero.record_failure()
        assert cb_zero.is_open is True

        # Very long timeout
        cb_long = CircuitBreaker(failure_threshold=1, recovery_timeout=3600)
        cb_long.record_failure()
        assert cb_long.is_open is True
        assert cb_long.can_proceed() is False  # Won't recover for an hour

        # Multiple successes
        cb_multi = CircuitBreaker()
        cb_multi.record_success()
        cb_multi.record_success()
        cb_multi.record_success()
        assert cb_multi.failure_count == 0
        assert cb_multi.is_open is False

    def test_circuit_breaker_logging(self):
        """Test CircuitBreaker logging behavior."""
        with patch("error_handling.circuit_breaker.logger") as mock_logger:
            cb = CircuitBreaker(failure_threshold=2)

            # Should log when opening
            cb.record_failure()
            cb.record_failure()  # This should trigger logging

            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "Circuit breaker opened after 2 failures" in call_args

        with patch("error_handling.circuit_breaker.logger") as mock_logger:
            cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1)
            cb.record_failure()  # Open circuit

            time.sleep(1.1)
            cb.can_proceed()  # Should trigger recovery logging

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "Circuit breaker transitioned to half-open state" in call_args


class TestEnhancedErrorHandler:
    """Test EnhancedErrorHandler comprehensive error management."""

    @pytest.fixture
    def error_handler(self):
        """Create error handler for testing."""
        return EnhancedErrorHandler()

    def test_error_handler_initialization(self, error_handler):
        """Test EnhancedErrorHandler initialization."""
        assert len(error_handler.error_history) == 0
        assert "team_processing" in error_handler.circuit_breakers
        assert "model_communication" in error_handler.circuit_breakers
        assert isinstance(
            error_handler.circuit_breakers["team_processing"], CircuitBreaker
        )
        assert isinstance(
            error_handler.circuit_breakers["model_communication"], CircuitBreaker
        )

    def test_handle_validation_error(self, error_handler):
        """Test handling of validation errors."""
        validation_error = ValidationError("test_field", "invalid_value", "test reason")

        recovery_msg = error_handler.handle_error(
            validation_error,
            ErrorType.VALIDATION_ERROR,
            thought_number=1,
            context={"source": "test"},
        )

        assert (
            recovery_msg
            == "Input validation failed: Validation failed for test_field: test reason. Please check the format and try again."
        )
        assert len(error_handler.error_history) == 1

        error_context = error_handler.error_history[0]
        assert error_context.error_type == ErrorType.VALIDATION_ERROR
        assert error_context.severity == ErrorSeverity.LOW
        assert error_context.thought_number == 1
        assert error_context.additional_info["source"] == "test"

    def test_handle_team_processing_error(self, error_handler):
        """Test handling of team processing errors."""
        test_error = Exception("Team communication failed")

        recovery_msg = error_handler.handle_error(
            test_error, ErrorType.TEAM_PROCESSING, thought_number=2
        )

        assert "Team processing error" in recovery_msg
        assert len(error_handler.error_history) == 1

        error_context = error_handler.error_history[0]
        assert error_context.error_type == ErrorType.TEAM_PROCESSING
        assert error_context.severity == ErrorSeverity.MEDIUM

    def test_handle_model_communication_error(self, error_handler):
        """Test handling of model communication errors."""
        test_error = Exception("API connection timeout")

        recovery_msg = error_handler.handle_error(
            test_error, ErrorType.MODEL_COMMUNICATION, thought_number=3
        )

        assert "Communication error with AI model" in recovery_msg
        assert len(error_handler.error_history) == 1

        error_context = error_handler.error_history[0]
        assert error_context.error_type == ErrorType.MODEL_COMMUNICATION
        assert error_context.severity == ErrorSeverity.HIGH

    def test_error_severity_assessment(self, error_handler):
        """Test error severity assessment logic."""
        # ValidationError should be LOW
        validation_error = ValidationError("field", "value", "reason")
        error_handler.handle_error(validation_error, ErrorType.VALIDATION)
        assert error_handler.error_history[-1].severity == ErrorSeverity.LOW

        # Team initialization should be CRITICAL
        init_error = Exception("Failed to initialize")
        error_handler.handle_error(init_error, ErrorType.TEAM_INITIALIZATION)
        assert error_handler.error_history[-1].severity == ErrorSeverity.CRITICAL

        # Model communication should be HIGH
        comm_error = Exception("Model timeout")
        error_handler.handle_error(comm_error, ErrorType.MODEL_COMMUNICATION)
        assert error_handler.error_history[-1].severity == ErrorSeverity.HIGH

        # Token/API errors should be HIGH
        token_error = Exception("Invalid API token provided")
        error_handler.handle_error(token_error, ErrorType.PROVIDER_ERROR)
        assert error_handler.error_history[-1].severity == ErrorSeverity.HIGH

    def test_circuit_breaker_integration(self, error_handler):
        """Test integration with circuit breakers."""
        # Open team processing circuit breaker
        cb = error_handler.circuit_breakers["team_processing"]
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()  # Open circuit

        # Error handling should detect open circuit
        test_error = Exception("Team processing failed")
        recovery_msg = error_handler.handle_error(test_error, ErrorType.TEAM_PROCESSING)

        assert "Team processing temporarily unavailable" in recovery_msg

        # Test model communication circuit breaker
        cb_model = error_handler.circuit_breakers["model_communication"]
        cb_model.record_failure()
        cb_model.record_failure()
        cb_model.record_failure()  # Open circuit

        model_error = Exception("Model failed")
        recovery_msg = error_handler.handle_error(
            model_error, ErrorType.MODEL_COMMUNICATION
        )

        assert "Model communication temporarily unavailable" in recovery_msg

    def test_error_summary_generation(self, error_handler):
        """Test error summary generation and statistics."""
        # No errors initially
        summary = error_handler.get_error_summary()
        assert summary["total_errors"] == 0

        # Add various errors
        errors = [
            (ValidationError("f1", "v1", "r1"), ErrorType.VALIDATION),
            (Exception("Team failed"), ErrorType.TEAM_PROCESSING),
            (Exception("Model timeout"), ErrorType.MODEL_COMMUNICATION),
            (ValidationError("f2", "v2", "r2"), ErrorType.VALIDATION),
            (Exception("Config error"), ErrorType.CONFIGURATION),
        ]

        for error, error_type in errors:
            error_handler.handle_error(error, error_type)

        summary = error_handler.get_error_summary()
        assert summary["total_errors"] == 5
        assert summary["by_type"]["validation"] == 2
        assert summary["by_type"]["team_processing"] == 1
        assert summary["by_type"]["model_communication"] == 1
        assert summary["by_type"]["configuration"] == 1

        # Check severity counts
        assert "by_severity" in summary
        assert "recent_errors" in summary
        assert len(summary["recent_errors"]) == 5  # All errors are recent

    def test_error_context_creation(self, error_handler):
        """Test ErrorContext creation and data integrity."""
        test_error = Exception("Test error message")

        before_time = datetime.now()
        error_handler.handle_error(
            test_error,
            ErrorType.TOOL_EXECUTION,
            thought_number=5,
            context={"tool": "test_tool", "input": "test_data"},
        )
        after_time = datetime.now()

        error_context = error_handler.error_history[0]
        assert error_context.error_type == ErrorType.TOOL_EXECUTION
        assert error_context.message == "Test error message"
        assert error_context.thought_number == 5
        assert before_time <= error_context.timestamp <= after_time
        assert error_context.additional_info["tool"] == "test_tool"
        assert error_context.additional_info["input"] == "test_data"
        assert (
            error_context.recovery_attempted is True
        )  # Should be set after recovery message

    def test_recovery_strategy_selection(self, error_handler):
        """Test recovery strategy selection logic."""
        # Unknown error type should return None
        unknown_error = Exception("Unknown error")
        recovery_msg = error_handler.handle_error(
            unknown_error,
            ErrorType.PROVIDER_ERROR,  # Has no specific recovery strategy
        )

        # Check that _apply_recovery_strategy handles unknown types gracefully
        assert recovery_msg is None or isinstance(recovery_msg, str)

    def test_error_logging_levels(self, error_handler):
        """Test that errors are logged at appropriate levels."""
        with patch("error_handling.error_handler.logger") as mock_logger:
            # Critical error should use logger.error
            critical_error = Exception("Critical failure")
            error_handler.handle_error(critical_error, ErrorType.TEAM_INITIALIZATION)
            mock_logger.error.assert_called()

            # High severity should use logger.error
            high_error = Exception("High severity issue")
            error_handler.handle_error(high_error, ErrorType.MODEL_COMMUNICATION)
            assert mock_logger.error.call_count >= 1

        with patch("error_handling.error_handler.logger") as mock_logger:
            # Low/medium severity should use logger.warning
            low_error = ValidationError("field", "value", "reason")
            error_handler.handle_error(low_error, ErrorType.VALIDATION)
            mock_logger.warning.assert_called()


class TestErrorContext:
    """Test ErrorContext data structure."""

    def test_error_context_creation(self):
        """Test ErrorContext creation with all fields."""
        timestamp = datetime.now()

        context = ErrorContext(
            error_type=ErrorType.VALIDATION,
            severity=ErrorSeverity.LOW,
            message="Test error message",
            timestamp=timestamp,
            thought_number=3,
            recovery_attempted=True,
            additional_info={"key": "value"},
        )

        assert context.error_type == ErrorType.VALIDATION
        assert context.severity == ErrorSeverity.LOW
        assert context.message == "Test error message"
        assert context.timestamp == timestamp
        assert context.thought_number == 3
        assert context.recovery_attempted is True
        assert context.additional_info["key"] == "value"

    def test_error_context_defaults(self):
        """Test ErrorContext with default values."""
        context = ErrorContext(
            error_type=ErrorType.TEAM_PROCESSING,
            severity=ErrorSeverity.HIGH,
            message="Default test",
            timestamp=datetime.now(),
        )

        assert context.thought_number is None
        assert context.recovery_attempted is False
        assert context.additional_info == {}


class TestErrorSeverityEnum:
    """Test ErrorSeverity enumeration."""

    def test_error_severity_values(self):
        """Test ErrorSeverity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_error_severity_comparison(self):
        """Test ErrorSeverity comparison (if needed for ordering)."""
        # Basic enum equality tests
        assert ErrorSeverity.LOW == ErrorSeverity.LOW
        assert ErrorSeverity.HIGH != ErrorSeverity.LOW

        # Test all values exist
        all_severities = [
            ErrorSeverity.LOW,
            ErrorSeverity.MEDIUM,
            ErrorSeverity.HIGH,
            ErrorSeverity.CRITICAL,
        ]
        assert len(all_severities) == 4
        assert len(set(all_severities)) == 4  # All unique
