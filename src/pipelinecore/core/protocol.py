"""
Protocol definitions for pipeline logging and timing.

Following Linus: "Good programmers worry about data structures."
Protocols define the contract without coupling to implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LoggerLike(Protocol):
    """Protocol for logger-like objects used in pipelines.

    Compatible with:
    - logging.Logger (standard library)
    - TimedLogger (with timing collection)
    - Any object with info/warning/error/debug/exception methods

    Example:
        def process(logger: LoggerLike) -> None:
            logger.info("Starting process...")
            # Works with both logging.Logger and TimedLogger
    """

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        ...

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        ...

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        ...

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        ...

    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        ...


class NullLogger:
    """A logger that does nothing.

    Useful as a default when no logging is needed.
    Implements LoggerLike protocol.
    """

    def info(self, msg: str, *args, **kwargs) -> None:
        """No-op info."""
        pass

    def warning(self, msg: str, *args, **kwargs) -> None:
        """No-op warning."""
        pass

    def error(self, msg: str, *args, **kwargs) -> None:
        """No-op error."""
        pass

    def debug(self, msg: str, *args, **kwargs) -> None:
        """No-op debug."""
        pass

    def exception(self, msg: str, *args, **kwargs) -> None:
        """No-op exception."""
        pass


# Singleton null logger instance
null_logger = NullLogger()
