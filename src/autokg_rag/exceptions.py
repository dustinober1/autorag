"""Domain exceptions for pipeline operations."""


class AutoRAGError(Exception):
    """Base domain exception."""


class IngestError(AutoRAGError):
    """Raised when ingest input is invalid or unreadable."""


class SchemaError(AutoRAGError):
    """Raised when artifact schema validation fails."""


class RetrievalError(AutoRAGError):
    """Raised when retrieval cannot produce a result."""
