"""
Timing utilities for pipeline execution tracking.

Provides TimingResult dataclass and timed_execution wrapper
for measuring and recording pipeline step durations.

Following Linus: "Make the common case easy."
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .protocol import LoggerLike

T = TypeVar("T")


@dataclass
class TimingResult:
    """Result of a timed execution step.

    Attributes:
        step_name: Name of the pipeline step (e.g., "prepare", "run_inference")
        elapsed_seconds: Execution time in seconds
        success: Whether the step completed without exception
        message: Optional message (error details on failure, notes on success)
    """

    step_name: str
    elapsed_seconds: float
    success: bool
    message: str = ""

    @property
    def elapsed_minutes(self) -> float:
        """Elapsed time in minutes."""
        return self.elapsed_seconds / 60

    @property
    def elapsed_formatted(self) -> str:
        """Human-readable elapsed time (e.g., '1m 23.4s' or '45.2s')."""
        if self.elapsed_seconds >= 60:
            minutes = int(self.elapsed_seconds // 60)
            seconds = self.elapsed_seconds % 60
            return f"{minutes}m {seconds:.1f}s"
        return f"{self.elapsed_seconds:.2f}s"

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return f"[{status}] {self.step_name}: {self.elapsed_formatted}"


def timed_execution(
    func: Callable[..., T],
    step_name: str,
    *args: Any,
    logger: LoggerLike | logging.Logger | None = None,
    **kwargs: Any,
) -> tuple[T | None, TimingResult]:
    """
    Execute a function with timing and return result + TimingResult.

    Args:
        func: Function to execute
        step_name: Name for this step (used in TimingResult and logging)
        *args: Positional arguments passed to func
        logger: Optional logger for progress messages (LoggerLike or logging.Logger)
        **kwargs: Keyword arguments passed to func

    Returns:
        Tuple of (function_result, TimingResult)
        If func raises exception: (None, TimingResult with success=False)

    Example:
        result, timing = timed_execution(
            model.predict, "inference",
            input_data,
            logger=self.context.logger
        )
        if timing.success:
            print(f"Completed in {timing.elapsed_formatted}")
    """
    if logger:
        logger.info(f"[開始] {step_name}")

    start_time = time.perf_counter()

    try:
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        timing = TimingResult(
            step_name=step_name,
            elapsed_seconds=elapsed,
            success=True,
            message=f"{step_name} 完成",
        )
        if logger:
            logger.info(f"[完成] {step_name} | 耗時: {elapsed:.2f} 秒")
        return result, timing

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        timing = TimingResult(
            step_name=step_name,
            elapsed_seconds=elapsed,
            success=False,
            message=f"{step_name} 失敗: {str(e)}",
        )
        if logger:
            logger.error(f"[失敗] {step_name} | 耗時: {elapsed:.2f} 秒 | 錯誤: {str(e)}")
        return None, timing


@dataclass
class TimingCollector:
    """Collector for aggregating TimingResults across pipeline execution.

    Provides summary statistics and formatted reports.

    Example:
        collector = TimingCollector()
        result, timing = timed_execution(func, "step1", logger=logger)
        collector.add(timing)
        print(collector.summary())
    """

    results: list[TimingResult] = field(default_factory=list)

    def add(self, timing: TimingResult) -> None:
        """Add a timing result to the collection."""
        self.results.append(timing)

    def clear(self) -> None:
        """Clear all collected results."""
        self.results.clear()

    @property
    def total_seconds(self) -> float:
        """Total elapsed time across all steps."""
        return sum(r.elapsed_seconds for r in self.results)

    @property
    def total_minutes(self) -> float:
        """Total elapsed time in minutes."""
        return self.total_seconds / 60

    @property
    def all_success(self) -> bool:
        """Whether all steps succeeded."""
        return all(r.success for r in self.results)

    @property
    def failed_steps(self) -> list[TimingResult]:
        """List of failed steps."""
        return [r for r in self.results if not r.success]

    @property
    def successful_steps(self) -> list[TimingResult]:
        """List of successful steps."""
        return [r for r in self.results if r.success]

    def summary(self) -> str:
        """Generate summary report."""
        lines = ["Pipeline Timing Summary", "=" * 50]
        for r in self.results:
            status = "OK" if r.success else "FAIL"
            lines.append(f"  [{status}] {r.step_name}: {r.elapsed_formatted}")
        lines.append("-" * 50)

        total_formatted = f"{self.total_minutes:.2f}m" if self.total_seconds >= 60 else f"{self.total_seconds:.2f}s"
        lines.append(f"  Total: {self.total_seconds:.2f}s ({total_formatted})")

        if not self.all_success:
            lines.append(f"  Failed: {len(self.failed_steps)} step(s)")
        return "\n".join(lines)


class TimedLogger:
    """Logger wrapper that records timing information.

    Wraps an existing logger and collects TimingResult objects.
    Implements LoggerLike protocol for compatibility.

    Example:
        base_logger = LogManager(log_dir, "pipeline").create_logger()
        logger = TimedLogger(base_logger)

        # Use as regular logger
        logger.info("Starting...")

        # Use timing features
        result, timing = logger.timed_call(func, "step_name", arg1, arg2)

        # Get timing summary
        print(logger.collector.summary())
    """

    def __init__(
        self,
        logger: logging.Logger,
        collector: TimingCollector | None = None,
    ) -> None:
        self._logger = logger
        self._collector = collector or TimingCollector()

    @property
    def collector(self) -> TimingCollector:
        """Access the timing collector."""
        return self._collector

    def record_timing(self, timing: TimingResult) -> None:
        """Record a timing result and log it."""
        self._collector.add(timing)
        self._logger.info(str(timing))

    def timed_call(
        self,
        func: Callable[..., T],
        step_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[T | None, TimingResult]:
        """Execute function with timing and record result.

        Args:
            func: Function to execute
            step_name: Name for this step
            *args: Positional arguments passed to func
            **kwargs: Keyword arguments passed to func

        Returns:
            Tuple of (function_result, TimingResult)
        """
        result, timing = timed_execution(func, step_name, *args, logger=self._logger, **kwargs)
        self._collector.add(timing)
        return result, timing

    # LoggerLike protocol methods (delegate to wrapped logger)
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(msg, *args, **kwargs)
