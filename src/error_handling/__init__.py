"""
Error handling module for comprehensive error management.
Provides circuit breaker patterns, error categorization, and recovery strategies.
"""

from .circuit_breaker import CircuitBreaker
from .error_context import ErrorContext, ErrorSeverity
from .error_handler import EnhancedErrorHandler

__all__ = [
    "CircuitBreaker",
    "ErrorContext",
    "ErrorSeverity",
    "EnhancedErrorHandler",
]
