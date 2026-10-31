"""Domain-specific exceptions for FinSight Enterprise System."""

class FinSightException(Exception):
    """Base exception for all FinSight system errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class GuardrailViolationException(FinSightException):
    """Raised when an input or output fails security/compliance guardrails."""
    def __init__(self, violation_type: str, details: dict = None):
        super().__init__(f"Guardrail check failed: {violation_type}", details)
        self.violation_type = violation_type


class RetrievalException(FinSightException):
    """Raised when context retrieval encounters an error."""
    pass


class LLMGenerationException(FinSightException):
    """Raised when LLM synthesis or schema extraction fails."""
    pass


class ContextWindowExceededException(FinSightException):
    """Raised when conversation or document context exceeds prompt capacity."""
    pass
