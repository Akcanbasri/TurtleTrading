"""
Utility functions for the Turtle Trading Bot
"""

import logging
import math
import time
import random
import json
from pathlib import Path
from decimal import Decimal
from typing import Optional, Union
from bot.models import PositionState
from datetime import datetime


def setup_logging(name: str = "turtle_trading_bot") -> logging.Logger:
    """
    Set up logging with console and file handlers

    Parameters
    ----------
    name : str
        Logger name

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Create file handler
    file_handler = logging.FileHandler(
        log_dir / f'turtle_bot_{time.strftime("%Y%m%d")}.log'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def round_step_size(quantity: Union[Decimal, float], step_size: Decimal) -> float:
    """
    Round quantity to the nearest step size

    Parameters
    ----------
    quantity : Union[Decimal, float]
        Quantity to round
    step_size : Decimal
        Step size from exchange rules

    Returns
    -------
    float
        Rounded quantity
    """
    if step_size == 0:
        return float(quantity)

    precision = int(round(-math.log10(float(step_size))))
    quantity_decimal = (
        Decimal(str(quantity)) if not isinstance(quantity, Decimal) else quantity
    )
    result = float(
        Decimal(int(quantity_decimal * Decimal(10**precision)) / Decimal(10**precision))
    )
    return result


def format_price(price: Union[Decimal, float], precision: Optional[int] = None) -> str:
    """
    Format price with the correct number of decimal places

    Parameters
    ----------
    price : Union[Decimal, float]
        Price to format
    precision : Optional[int]
        Number of decimal places (if None, use price_precision from symbol info)

    Returns
    -------
    str
        Formatted price string
    """
    if precision is None:
        precision = 8  # Default precision if not specified

    format_str = f"{{:.{precision}f}}"
    return format_str.format(float(price))


def format_quantity(
    quantity: Union[Decimal, float],
    precision: Optional[int] = None,
    step_size: Optional[Decimal] = None,
) -> str:
    """
    Format quantity with the correct number of decimal places and step size

    Parameters
    ----------
    quantity : Union[Decimal, float]
        Quantity to format
    precision : Optional[int]
        Number of decimal places (if None, use quantity_precision from symbol info)
    step_size : Optional[Decimal]
        Step size from exchange rules for rounding

    Returns
    -------
    str
        Formatted quantity string
    """
    # First round to step size if provided
    if step_size is not None:
        quantity = round_step_size(quantity, step_size)

    if precision is None:
        precision = 8  # Default precision if not specified

    format_str = f"{{:.{precision}f}}"
    return format_str.format(float(quantity))


def get_sleep_time(timeframe: str) -> int:
    """
    Calculate the time to sleep until the next candle close

    Parameters
    ----------
    timeframe : str
        The timeframe of the candlesticks (e.g., '1h', '4h', '1d')

    Returns
    -------
    int
        Seconds to sleep until the next candle close
    """
    now = time.time()
    current_time = time.localtime(now)

    # Initialize seconds to next candle close
    seconds_to_next_close = 60  # Default 1 minute

    if timeframe == "1m":
        # Next minute
        seconds_to_next_close = 60 - current_time.tm_sec

    elif timeframe == "3m":
        # Next 3-minute mark
        current_minute = current_time.tm_min
        next_3min = 3 * (current_minute // 3 + 1)
        seconds_to_next_close = (next_3min - current_minute) * 60 - current_time.tm_sec

    elif timeframe == "5m":
        # Next 5-minute mark
        current_minute = current_time.tm_min
        next_5min = 5 * (current_minute // 5 + 1)
        seconds_to_next_close = (next_5min - current_minute) * 60 - current_time.tm_sec

    elif timeframe == "15m":
        # Next 15-minute mark
        current_minute = current_time.tm_min
        next_15min = 15 * (current_minute // 15 + 1)
        seconds_to_next_close = (next_15min - current_minute) * 60 - current_time.tm_sec

    elif timeframe == "30m":
        # Next 30-minute mark
        current_minute = current_time.tm_min
        next_30min = 30 * (current_minute // 30 + 1)
        seconds_to_next_close = (next_30min - current_minute) * 60 - current_time.tm_sec

    elif timeframe == "1h":
        # Next hour
        seconds_to_next_close = (60 - current_time.tm_min) * 60 - current_time.tm_sec

    elif timeframe == "2h":
        # Next 2-hour mark
        current_hour = current_time.tm_hour
        next_2hour = 2 * (current_hour // 2 + 1)
        hours_remaining = next_2hour - current_hour - 1
        seconds_to_next_close = (
            hours_remaining * 3600
            + (60 - current_time.tm_min) * 60
            - current_time.tm_sec
        )

    elif timeframe == "4h":
        # Next 4-hour mark
        current_hour = current_time.tm_hour
        next_4hour = 4 * (current_hour // 4 + 1)
        hours_remaining = next_4hour - current_hour - 1
        seconds_to_next_close = (
            hours_remaining * 3600
            + (60 - current_time.tm_min) * 60
            - current_time.tm_sec
        )

    elif timeframe in ["1d", "1D"]:
        # Next day at 00:00 UTC
        seconds_to_midnight = (
            (24 - current_time.tm_hour - 1) * 3600
            + (60 - current_time.tm_min) * 60
            - current_time.tm_sec
        )
        seconds_to_next_close = seconds_to_midnight

    else:
        # Default: check every 5 minutes
        logger = logging.getLogger("turtle_trading_bot")
        logger.warning(
            f"Unrecognized timeframe {timeframe}, defaulting to 5-minute checks"
        )
        seconds_to_next_close = 300

    # Add a small buffer to ensure the candle has closed (15 seconds)
    seconds_to_next_close += 15

    # Never sleep less than 30 seconds
    if seconds_to_next_close < 30:
        seconds_to_next_close = 30

    # Add a bit of randomness to avoid API rate limits when many bots run simultaneously
    seconds_to_next_close += random.randint(1, 10)

    return seconds_to_next_close


def save_position_state(
    position: PositionState,
    symbol: str,
    state_file: Path = Path("config/bot_state.json"),
) -> None:
    """
    Save the current position state to a JSON file

    Parameters
    ----------
    position : PositionState
        Current position state
    symbol : str
        Trading symbol
    state_file : Path
        Path to the state file
    """
    logger = logging.getLogger("turtle_trading_bot")

    try:
        # Convert position to dictionary - use all attributes from the object
        state = {}
        # Get all attributes that don't start with underscore
        for attr_name in dir(position):
            if not attr_name.startswith("_") and not callable(
                getattr(position, attr_name)
            ):
                attr_value = getattr(position, attr_name)
                # Convert special types to string
                if isinstance(attr_value, Decimal):
                    state[attr_name] = str(attr_value)
                elif isinstance(attr_value, (datetime, time.struct_time)):
                    state[attr_name] = str(attr_value)
                else:
                    state[attr_name] = attr_value

        # Add symbol and timestamp
        state["symbol"] = symbol
        state["last_update"] = int(time.time() * 1000)

        # Create config directory if it doesn't exist
        state_file.parent.mkdir(exist_ok=True)

        with open(state_file, "w") as f:
            json.dump(state, f, indent=4)

        logger.info(f"Bot state saved to {state_file}")
    except Exception as e:
        logger.error(f"Error saving bot state: {e}")


def load_position_state(
    symbol: str, state_file: Path = Path("config/bot_state.json")
) -> Optional[PositionState]:
    """
    Load the bot state from the state file if it exists

    Parameters
    ----------
    symbol : str
        Trading symbol
    state_file : Path
        Path to the state file

    Returns
    -------
    Optional[PositionState]
        Loaded position state or None if file doesn't exist or is invalid
    """
    logger = logging.getLogger("turtle_trading_bot")

    if not state_file.exists():
        logger.info(f"No state file found at {state_file}. Starting with clean state.")
        return None

    try:
        with open(state_file, "r") as f:
            state = json.load(f)

        # Verify the state is for the same symbol
        if state.get("symbol") != symbol:
            logger.warning(
                f"State file is for a different symbol ({state.get('symbol')}, current: {symbol}). Not loading state."
            )
            return None

        # Create a new position state
        position = PositionState()

        # Update attributes from the loaded state
        for key, value in state.items():
            if hasattr(position, key) and key not in ["symbol", "last_update"]:
                # Convert string representation of Decimal back to Decimal
                if key in [
                    "entry_price",
                    "quantity",
                    "stop_loss_price",
                    "take_profit_price",
                    "trailing_stop_price",
                    "entry_atr",
                    "partial_exit_price",
                    "first_target_price",
                    "second_target_price",
                    "max_price",
                    "min_price",
                    "max_running_profit",
                    "max_running_loss",
                    "pnl_amount",
                    "pnl_percentage",
                ]:
                    if value is not None and value != "None":
                        setattr(position, key, float(value))
                # Handle timestamps
                elif key in ["entry_time", "exit_time", "partial_exit_time"]:
                    if value is not None and value != "None":
                        # If it's a string representation of datetime, convert appropriately
                        # For simplicity, we're keeping it as string here
                        setattr(position, key, value)
                else:
                    setattr(position, key, value)

        logger.info(f"Bot state loaded from {state_file}")
        return position
    except Exception as e:
        logger.error(f"Error loading bot state: {e}")
        return None
