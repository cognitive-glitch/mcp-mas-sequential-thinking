"""
Circuit breaker pattern implementation for fault tolerance.
Prevents cascade failures by temporarily blocking operations after repeated failures.
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascade failures."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.is_open = False
        self.is_half_open = False

    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.is_open = False
        self.is_half_open = False

    def record_failure(self):
        """Record failed operation and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        # In half-open state, any failure immediately opens the circuit
        if self.is_half_open:
            self.is_open = True
            self.is_half_open = False
            logger.warning("Circuit breaker opened after failure in half-open state")
        elif self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def can_proceed(self) -> bool:
        """Check if operation can proceed."""
        if not self.is_open and not self.is_half_open:
            return True
        
        if self.is_half_open:
            return True  # Allow requests in half-open state

        # Check if recovery timeout has passed (transition from open to half-open)
        if self.last_failure_time:
            time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
            if time_since_failure > self.recovery_timeout:
                self.is_half_open = True
                self.is_open = False
                logger.info("Circuit breaker transitioned to half-open state")
                return True

        return False