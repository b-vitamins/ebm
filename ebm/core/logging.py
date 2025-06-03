"""Structured logging configuration for the EBM library.

This module sets up structured logging using structlog, providing:
- Consistent log formatting
- Context preservation
- Performance metrics
- Integration with training loops
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

import structlog
from structlog.types import EventDict, WrappedLogger

# Try to import rich for pretty console output
try:
    from rich.console import Console
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def add_timestamp(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add timestamp to log entries."""
    event_dict["timestamp"] = time.time()
    return event_dict


def add_logger_name(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add logger name to log entries."""
    if logger_name := event_dict.get("logger_name"):
        event_dict["logger"] = logger_name
        event_dict.pop("logger_name", None)
    return event_dict


class LogConfig:
    """Configuration for logging system."""

    def __init__(
        self,
        level: str | int = "INFO",
        console: bool = True,
        file: str | Path | None = None,
        structured: bool = True,
        colors: bool = True,
        metrics: bool = True,
    ):
        self.level = (
            level if isinstance(level, int) else getattr(logging, level.upper())
        )
        self.console = console
        self.file = Path(file) if file else None
        self.structured = structured
        self.colors = colors and RICH_AVAILABLE
        self.metrics = metrics

    def setup(self) -> structlog.BoundLogger:
        """Configure and return logger instance."""
        # Configure structlog processors
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            add_timestamp,
            add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]

        # Add metric processor if enabled
        if self.metrics:
            processors.append(MetricProcessor())

        # Configure output format
        if self.structured:
            processors.append(structlog.processors.JSONRenderer())
        elif self.colors and RICH_AVAILABLE:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=False))

        # Configure structlog
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Configure standard logging
        handlers = []

        if self.console:
            if self.colors and RICH_AVAILABLE:
                console_handler = RichHandler(
                    console=Console(stderr=True),
                    show_time=False,  # We add our own timestamp
                    show_level=False,  # We add our own level
                )
            else:
                console_handler = logging.StreamHandler(sys.stderr)
            handlers.append(console_handler)

        if self.file:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.file)
            handlers.append(file_handler)

        logging.basicConfig(
            level=self.level,
            handlers=handlers,
            format="%(message)s"
            if self.structured
            else "%(asctime)s [%(levelname)s] %(message)s",
        )

        return structlog.get_logger()


class MetricProcessor:
    """Processor that extracts and formats metrics from log events."""

    def __call__(
        self, logger: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        """Extract metrics from event dict."""
        metrics = {}

        # Look for common metric patterns
        for key, value in list(event_dict.items()):
            if key.endswith(("_loss", "_error", "_accuracy", "_score")):
                if isinstance(value, int | float):
                    metrics[key] = value
                    event_dict.pop(key)
            elif key in {"epoch", "step", "iteration", "batch"}:
                metrics[key] = value

        if metrics:
            event_dict["metrics"] = metrics

        return event_dict


class LoggerMixin:
    """Mixin class that provides logging functionality."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._logger = None

    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger instance for this class."""
        if self._logger is None:
            self._logger = structlog.get_logger(self.__class__.__name__)
        return self._logger

    def log_debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, **kwargs)

    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(message, **kwargs)

    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, **kwargs)

    def log_error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(message, **kwargs)


@contextmanager
def log_context(**kwargs: Any) -> Iterator[None]:
    """Context manager that adds context to all logs within the block.

    Example:
        with log_context(epoch=1, phase='training'):
            logger.info('Starting batch', batch=0)
            # Logs: {"event": "Starting batch", "epoch": 1, "phase": "training", "batch": 0}
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**kwargs)
    try:
        yield
    finally:
        structlog.contextvars.clear_contextvars()


@contextmanager
def log_duration(
    logger: structlog.BoundLogger, message: str, **kwargs: Any
) -> Iterator[None]:
    """Context manager that logs the duration of a block.

    Example:
        with log_duration(logger, 'Training epoch'):
            train_one_epoch()
            # Logs: {"event": "Training epoch", "duration": 123.45}
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        logger.info(message, duration=duration, **kwargs)


P = ParamSpec("P")
R = TypeVar("R")


def log_function_call(logger: structlog.BoundLogger | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Log function calls with arguments and return values.

    Args:
        logger: Logger instance to use. If None, creates one based on function module.

    Example:
        @log_function_call()
        def train_model(epochs: int, lr: float) -> float:
            ...
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        nonlocal logger
        if logger is None:
            logger = structlog.get_logger(func.__module__)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            logger.debug(f"Calling {func.__name__}", args=args, kwargs=kwargs)
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                logger.debug(
                    f"Completed {func.__name__}",
                    duration=duration,
                    result_type=type(result).__name__,
                )
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.error(
                    f"Failed {func.__name__}",
                    duration=duration,
                    error=str(e),
                    exc_info=True,
                )
                raise

        return cast(Callable[P, R], wrapper)

    return decorator


def setup_logging(
    level: str | int = "INFO",
    console: bool = True,
    file: str | Path | None = None,
    structured: bool = False,
    colors: bool = True,
    metrics: bool = True,
) -> structlog.BoundLogger:
    """Configure logging for the application.

    Args:
        level: Logging level
        console: Whether to log to console
        file: File path for logging (if any)
        structured: Whether to use structured (JSON) logging
        colors: Whether to use colored output (requires rich)
        metrics: Whether to enable metric extraction

    Returns
    -------
        Configured logger instance
    """
    config = LogConfig(
        level=level,
        console=console,
        file=file,
        structured=structured,
        colors=colors,
        metrics=metrics,
    )
    return config.setup()


# Create default logger
logger = structlog.get_logger("ebm")


# Convenience functions
def debug(message: str, **kwargs: Any) -> None:
    """Log debug message."""
    logger.debug(message, **kwargs)


def info(message: str, **kwargs: Any) -> None:
    """Log info message."""
    logger.info(message, **kwargs)


def warning(message: str, **kwargs: Any) -> None:
    """Log warning message."""
    logger.warning(message, **kwargs)


def error(message: str, **kwargs: Any) -> None:
    """Log error message."""
    logger.error(message, **kwargs)


def metrics(message: str, **metric_values: Any) -> None:
    """Log metrics with automatic formatting.

    Example:
        metrics("Training", epoch=1, loss=0.123, accuracy=0.95)
    """
    logger.info(message, **metric_values)
