"""Unit tests for logging functionality."""

import json
from pathlib import Path
from typing import Never
from unittest.mock import MagicMock, patch

import pytest

from ebm.core.logging import (
    LogConfig,
    LoggerMixin,
    MetricProcessor,
    add_logger_name,
    add_timestamp,
    debug,
    error,
    info,
    log_context,
    log_duration,
    log_function_call,
    metrics,
    setup_logging,
    warning,
)


class TestLogConfig:
    """Test LogConfig class."""

    def test_default_config(self) -> None:
        """Test default log configuration."""
        config = LogConfig()

        assert config.level == 20  # INFO level
        assert config.console is True
        assert config.file is None
        assert config.structured is True
        assert config.metrics is True

    def test_level_conversion(self) -> None:
        """Test string to int level conversion."""
        config = LogConfig(level="DEBUG")
        assert config.level == 10

        config = LogConfig(level="WARNING")
        assert config.level == 30

        config = LogConfig(level=40)  # ERROR as int
        assert config.level == 40

    def test_file_path_conversion(self) -> None:
        """Test file path handling."""
        config = LogConfig(file="logs/test.log")
        assert isinstance(config.file, Path)
        assert config.file == Path("logs/test.log")

    @patch('structlog.configure')
    def test_setup(self, mock_configure) -> None:
        """Test logger setup."""
        config = LogConfig(
            level="INFO",
            console=True,
            structured=False,
            metrics=True
        )

        logger = config.setup()

        # Should configure structlog
        mock_configure.assert_called_once()

        # Should return a logger
        assert logger is not None


class TestLoggerMixin:
    """Test LoggerMixin functionality."""

    def test_logger_property(self) -> None:
        """Test logger property creation."""
        class TestClass(LoggerMixin):
            pass

        obj = TestClass()

        # First access creates logger
        logger1 = obj.logger
        assert logger1 is not None

        # Subsequent access returns same logger
        logger2 = obj.logger
        assert logger1 is logger2

    def test_log_methods(self) -> None:
        """Test convenience log methods."""
        class TestClass(LoggerMixin):
            pass

        obj = TestClass()

        # Mock the logger
        mock_logger = MagicMock()
        obj._logger = mock_logger

        # Test each log level
        obj.log_debug("Debug message", extra="data")
        mock_logger.debug.assert_called_with("Debug message", extra="data")

        obj.log_info("Info message", value=42)
        mock_logger.info.assert_called_with("Info message", value=42)

        obj.log_warning("Warning message")
        mock_logger.warning.assert_called_with("Warning message")

        obj.log_error("Error message", error="details")
        mock_logger.error.assert_called_with("Error message", error="details")


class TestMetricProcessor:
    """Test MetricProcessor."""

    def test_metric_extraction(self) -> None:
        """Test extracting metrics from event dict."""
        processor = MetricProcessor()

        event_dict = {
            "event": "Training step",
            "loss": 0.123,
            "accuracy": 0.95,
            "learning_rate": 0.001,
            "batch_size": 32,
            "grad_norm": 1.23
        }

        result = processor(None, None, event_dict.copy())

        # Should extract metric-like keys
        assert "metrics" in result
        assert "loss" not in result  # Moved to metrics
        assert result["metrics"]["loss"] == 0.123
        assert result["event"] == "Training step"  # Non-metric preserved

    def test_special_metrics(self) -> None:
        """Test extraction of special metric keys."""
        processor = MetricProcessor()

        event_dict = {
            "event": "Epoch complete",
            "epoch": 10,
            "step": 1000,
            "iteration": 500,
            "batch": 32
        }

        result = processor(None, None, event_dict)

        assert "metrics" in result
        assert result["metrics"]["epoch"] == 10
        assert result["metrics"]["step"] == 1000

    def test_non_numeric_ignored(self) -> None:
        """Test that non-numeric values are ignored."""
        processor = MetricProcessor()

        event_dict = {
            "event": "Test",
            "loss": 0.123,
            "model_name": "RBM",  # String, not metric
            "error_details": {"type": "ValueError"}  # Dict, not metric
        }

        result = processor(None, None, event_dict)

        assert "loss" not in result
        assert result["metrics"]["loss"] == 0.123
        assert "model_name" in result  # Preserved
        assert "model_name" not in result["metrics"]


class TestProcessorFunctions:
    """Test custom processor functions."""

    def test_add_timestamp(self) -> None:
        """Test timestamp processor."""
        event_dict = {"event": "test"}

        # Mock time
        with patch('time.time', return_value=1234567890.123):
            result = add_timestamp(None, None, event_dict)

        assert "timestamp" in result
        assert result["timestamp"] == 1234567890.123

    def test_add_logger_name(self) -> None:
        """Test logger name processor."""
        event_dict = {
            "event": "test",
            "logger_name": "ebm.models.rbm"
        }

        result = add_logger_name(None, None, event_dict)

        assert "logger" in result
        assert result["logger"] == "ebm.models.rbm"
        assert "logger_name" not in result  # Removed


class TestContextManagers:
    """Test logging context managers."""

    def test_log_context(self) -> None:
        """Test log context manager."""
        with patch('structlog.contextvars.bind_contextvars') as mock_bind:
            with patch('structlog.contextvars.clear_contextvars') as mock_clear:
                with log_context(epoch=1, phase="training"):
                    mock_bind.assert_called_once_with(epoch=1, phase="training")

                # Context should be cleared on exit
                assert mock_clear.call_count == 2  # Once on enter, once on exit

    def test_log_duration(self) -> None:
        """Test duration logging context manager."""
        mock_logger = MagicMock()

        with patch('time.perf_counter', side_effect=[1.0, 3.5]):
            with log_duration(mock_logger, "Operation", extra="data"):
                # Simulate some work
                pass

        mock_logger.info.assert_called_once_with(
            "Operation",
            duration=2.5,
            extra="data"
        )

    def test_log_duration_with_exception(self) -> Never:
        """Test duration logging with exception."""
        mock_logger = MagicMock()

        with patch('time.perf_counter', side_effect=[1.0, 2.0]):
            with pytest.raises(ValueError):
                with log_duration(mock_logger, "Failing operation"):
                    raise ValueError("Test error")

        # Should still log duration
        mock_logger.info.assert_called_once_with(
            "Failing operation",
            duration=1.0
        )


class TestFunctionDecorator:
    """Test function call logging decorator."""

    def test_log_function_call_success(self) -> None:
        """Test logging successful function calls."""
        mock_logger = MagicMock()

        @log_function_call(mock_logger)
        def test_func(a: int, b: int = 2) -> int:
            return a + b

        with patch('time.perf_counter', side_effect=[1.0, 1.1]):
            result = test_func(1, b=3)

        assert result == 4

        # Check debug calls
        debug_calls = mock_logger.debug.call_args_list
        assert len(debug_calls) == 2

        # First call - function entry
        assert "Calling test_func" in debug_calls[0][0][0]
        assert debug_calls[0][1]["args"] == (1,)
        assert debug_calls[0][1]["kwargs"] == {"b": 3}

        # Second call - function exit
        assert "Completed test_func" in debug_calls[1][0][0]
        assert debug_calls[1][1]["duration"] == pytest.approx(0.1)
        assert debug_calls[1][1]["result_type"] == "int"

    def test_log_function_call_exception(self) -> None:
        """Test logging function calls that raise exceptions."""
        mock_logger = MagicMock()

        @log_function_call(mock_logger)
        def failing_func() -> Never:
            raise ValueError("Test error")

        with patch('time.perf_counter', side_effect=[1.0, 1.5]):
            with pytest.raises(ValueError):
                failing_func()

        # Should log the error
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args

        assert "Failed failing_func" in error_call[0][0]
        assert error_call[1]["duration"] == pytest.approx(0.5)
        assert error_call[1]["error"] == "Test error"
        assert error_call[1]["exc_info"] is True

    def test_log_function_call_auto_logger(self) -> None:
        """Test decorator with automatic logger creation."""
        @log_function_call()  # No logger provided
        def auto_func() -> str:
            return "result"

        # Should not raise error
        result = auto_func()
        assert result == "result"


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    @patch('ebm.core.logging.logger')
    def test_debug(self, mock_logger) -> None:
        """Test debug convenience function."""
        debug("Debug message", value=123)
        mock_logger.debug.assert_called_once_with("Debug message", value=123)

    @patch('ebm.core.logging.logger')
    def test_info(self, mock_logger) -> None:
        """Test info convenience function."""
        info("Info message", status="ok")
        mock_logger.info.assert_called_once_with("Info message", status="ok")

    @patch('ebm.core.logging.logger')
    def test_warning(self, mock_logger) -> None:
        """Test warning convenience function."""
        warning("Warning message")
        mock_logger.warning.assert_called_once_with("Warning message")

    @patch('ebm.core.logging.logger')
    def test_error(self, mock_logger) -> None:
        """Test error convenience function."""
        error("Error message", code=500)
        mock_logger.error.assert_called_once_with("Error message", code=500)

    @patch('ebm.core.logging.logger')
    def test_metrics(self, mock_logger) -> None:
        """Test metrics convenience function."""
        metrics("Training metrics", loss=0.123, accuracy=0.95)
        mock_logger.info.assert_called_once_with(
            "Training metrics",
            loss=0.123,
            accuracy=0.95
        )


class TestSetupLogging:
    """Test setup_logging function."""

    @patch('ebm.core.logging.LogConfig')
    def test_setup_logging_defaults(self, mock_config_class) -> None:
        """Test setup with default parameters."""
        mock_config = MagicMock()
        mock_config.setup.return_value = MagicMock()
        mock_config_class.return_value = mock_config

        setup_logging()

        mock_config_class.assert_called_once_with(
            level="INFO",
            console=True,
            file=None,
            structured=False,
            colors=True,
            metrics=True
        )
        mock_config.setup.assert_called_once()

    @patch('ebm.core.logging.LogConfig')
    def test_setup_logging_custom(self, mock_config_class) -> None:
        """Test setup with custom parameters."""
        mock_config = MagicMock()
        mock_config.setup.return_value = MagicMock()
        mock_config_class.return_value = mock_config

        setup_logging(
            level="DEBUG",
            console=False,
            file="test.log",
            structured=True,
            colors=False,
            metrics=False
        )

        mock_config_class.assert_called_once_with(
            level="DEBUG",
            console=False,
            file="test.log",
            structured=True,
            colors=False,
            metrics=False
        )


class TestIntegration:
    """Integration tests for logging system."""

    def test_full_logging_flow(self, tmp_path) -> None:
        """Test complete logging workflow."""
        log_file = tmp_path / "test.log"

        # Setup logging
        test_logger = setup_logging(
            level="DEBUG",
            file=str(log_file),
            structured=False,
            colors=False
        )

        # Log some messages
        test_logger.debug("Debug message", debug_data=123)
        test_logger.info("Info message", status="running")
        test_logger.warning("Warning message")
        test_logger.error("Error message", error_code=500)

        # Check that log file was created
        assert log_file.exists()

        # Read and verify content
        content = log_file.read_text()
        assert "Debug message" in content
        assert "Info message" in content
        assert "Warning message" in content
        assert "Error message" in content

    def test_structured_logging(self, tmp_path) -> None:
        """Test structured JSON logging."""
        log_file = tmp_path / "structured.log"

        # Setup structured logging
        test_logger = setup_logging(
            file=str(log_file),
            structured=True,
            console=False
        )

        # Log structured data
        test_logger.info(
            "Training step",
            epoch=1,
            batch=10,
            loss=0.123,
            accuracy=0.95
        )

        # Read and parse JSON
        content = log_file.read_text().strip()
        data = json.loads(content)

        assert data["event"] == "Training step"
        assert data["epoch"] == 1
        assert data["batch"] == 10
        assert data["loss"] == 0.123
        assert data["accuracy"] == 0.95
        assert "timestamp" in data
