"""
Turtle Trading Bot - A Python implementation of the Turtle Trading system

This package provides a modular, class-based implementation of the Turtle Trading strategy
for algorithmic trading on the Binance exchange.
"""

import os
import logging
from logging.handlers import RotatingFileHandler

from bot.core import TurtleTradingBot


def setup_logger():
    """Set up the logger for the application."""
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Configure logger
    logger = logging.getLogger("turtle_trading_bot")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create file handler that logs all messages
    file_handler = RotatingFileHandler(
        "logs/turtle_trading_bot.log", maxBytes=10485760, backupCount=5  # 10MB
    )
    file_handler.setLevel(logging.INFO)

    # Create formatters and add them to handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


__version__ = "2.0.0"
__all__ = ["TurtleTradingBot", "setup_logger"]
