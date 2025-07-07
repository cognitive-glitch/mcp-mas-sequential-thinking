"""
Error context and severity definitions for comprehensive error categorization.
"""

from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from exceptions import ErrorType


class ErrorSeverity(Enum):
    """Error severity levels for prioritization."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""

    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    timestamp: datetime
    thought_number: Optional[int] = None
    recovery_attempted: bool = False
    additional_info: Dict[str, Any] = field(default_factory=dict)