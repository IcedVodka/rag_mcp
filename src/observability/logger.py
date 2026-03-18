#!/usr/bin/env python3
"""
Logger Module - Structured Logging Support

Provides logging utilities with JSON formatting support for
observability and tracing.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Union


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs log records as JSON lines.
    
    Useful for structured logging and log aggregation systems.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'getMessage',
                'asctime'
            ):
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Ensure handler is set up
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        logger.addHandler(handler)
    
    return logger


def setup_json_logging(
    logger: logging.Logger,
    output_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Setup JSON formatting for a logger.
    
    Args:
        logger: Logger to configure
        output_path: Optional file path to write logs to
    """
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = JSONFormatter()
    
    # Add file handler if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Always add stderr handler
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def get_trace_logger(output_path: Union[str, Path]) -> logging.Logger:
    """
    Get a logger configured for trace output.
    
    Args:
        output_path: Path to write trace logs to
        
    Returns:
        Configured trace logger
    """
    logger = logging.getLogger("trace")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Add file handler
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    handler = logging.FileHandler(output_path, encoding='utf-8')
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger


def write_trace(trace_dict: dict, output_path: Union[str, Path]) -> None:
    """
    Write a trace dictionary to the trace log file.
    
    Args:
        trace_dict: Trace data to write
        output_path: Path to trace log file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(trace_dict, ensure_ascii=False) + '\n')
