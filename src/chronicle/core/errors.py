class ChronicleError(RuntimeError):
    """Base Chronicle error."""


class IndexBuildError(ChronicleError):
    """Raised when Chronicle cannot build an index."""


class ValidationError(ChronicleError):
    """Raised when grounded validation fails."""
