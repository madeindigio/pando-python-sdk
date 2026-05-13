"""Exception hierarchy for the Pando SDK."""

from __future__ import annotations


class PandoError(Exception):
    """Base exception for all Pando SDK errors."""


class PandoBinaryNotFoundError(PandoError):
    """Raised when the pando binary cannot be located.

    This typically means pando is not installed or not on the system PATH.
    Set the ``PANDO_PATH`` environment variable or pass ``pando_path`` to point
    to the binary explicitly.
    """


class PandoConnectionError(PandoError):
    """Raised when the connection to the pando process is lost or cannot be established.

    This can happen if the pando subprocess exits unexpectedly while processing
    a request, or if the process cannot be started at all.
    """


class PandoSessionError(PandoError):
    """Raised for session-level errors such as unknown session IDs or duplicate sessions."""


class PandoTimeoutError(PandoError):
    """Raised when an operation does not complete within the specified timeout."""


class PandoRPCError(PandoError):
    """Raised when the JSON-RPC server returns an error response.

    Attributes:
        code: The JSON-RPC error code (negative integer).
        message: The human-readable error message from the server.
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"RPC error {code}: {message}")
        self.code = code
        self.message = message
