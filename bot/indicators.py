"""
Technical indicators calculation for the Turtle Trading Bot
"""

import logging
from typing import Optional, Union, Tuple
import pandas as pd


def calculate_indicators(
    df: pd.DataFrame,
    dc_enter: int,
    dc_exit: int,
    atr_len: int,
    atr_smooth: Union[int, str],
) -> Optional[pd.DataFrame]:
    """
    Calculate Donchian Channels and ATR indicators on a DataFrame

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
    dc_enter : int
        Period for the entry Donchian Channel
    dc_exit : int
        Period for the exit Donchian Channel
    atr_len : int
        Period for ATR calculation
    atr_smooth : int or str
        Smoothing factor for ATR (integer for EMA period or 'RMA'/'SMA'/'WMA' etc.)

    Returns
    -------
    Optional[pandas.DataFrame]
        DataFrame with added indicator columns, NaN rows removed,
        or None if calculation fails
    """
    logger = logging.getLogger("turtle_trading_bot")

    try:
        # Make a copy to avoid modifying the original dataframe
        result = df.copy()

        # Calculate entry Donchian Channel with shift(1) to look at previous completed bars
        result["dc_upper_entry"] = (
            result["high"].rolling(window=dc_enter).max().shift(1)
        )
        result["dc_lower_entry"] = result["low"].rolling(window=dc_enter).min().shift(1)
        result["dc_middle_entry"] = (
            result["dc_upper_entry"] + result["dc_lower_entry"]
        ) / 2

        # Calculate exit Donchian Channel with shift(1)
        result["dc_upper_exit"] = result["high"].rolling(window=dc_exit).max().shift(1)
        result["dc_lower_exit"] = result["low"].rolling(window=dc_exit).min().shift(1)
        result["dc_middle_exit"] = (
            result["dc_upper_exit"] + result["dc_lower_exit"]
        ) / 2

        # Calculate ATR using pandas_ta
        if isinstance(atr_smooth, int):
            # When atr_smooth is an integer, use EMA smoothing
            smoothing = f"ema_{atr_smooth}"
        else:
            # When atr_smooth is a string like 'RMA', 'SMA', use that directly
            smoothing = atr_smooth

        # Calculate ATR using pandas_ta
        result["atr"] = result.ta.atr(length=atr_len, mamode=smoothing)

        # Drop rows with NaN values that result from the rolling calculations
        result = result.dropna()

        logger.info(
            f"Calculated indicators with parameters: dc_enter={dc_enter}, dc_exit={dc_exit}, atr_len={atr_len}, atr_smooth={atr_smooth}"
        )
        logger.info(f"DataFrame shape after indicator calculation: {result.shape}")

        return result

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None


def check_entry_signal(row: pd.Series, side: str = "BOTH") -> bool:
    """
    Check if entry signal is triggered based on latest data

    Parameters
    ----------
    row : pd.Series
        Latest data row with indicator values
    side : str
        Direction to check ('BUY', 'SELL', or 'BOTH')

    Returns
    -------
    bool
        True if entry signal is triggered, False otherwise
    """
    if side in ["BUY", "BOTH"]:
        # Check for long entry signal - price breaks above upper Donchian Channel
        if row["close"] > row["dc_upper_entry"]:
            return True

    if side in ["SELL", "BOTH"]:
        # Check for short entry signal - price breaks below lower Donchian Channel
        if row["close"] < row["dc_lower_entry"]:
            return True

    return False


def check_exit_signal(row: pd.Series, position_side: str) -> bool:
    """
    Check if exit signal is triggered based on latest data

    Parameters
    ----------
    row : pd.Series
        Latest data row with indicator values
    position_side : str
        Current position side ('BUY' or 'SELL')

    Returns
    -------
    bool
        True if exit signal is triggered, False otherwise
    """
    if position_side == "BUY":
        # Check for long exit signal - price breaks below lower exit Donchian Channel
        return row["close"] < row["dc_lower_exit"]
    elif position_side == "SELL":
        # Check for short exit signal - price breaks above upper exit Donchian Channel
        return row["close"] > row["dc_upper_exit"]
    else:
        return False


def check_stop_loss(
    current_price: float, stop_loss_price: float, position_side: str
) -> bool:
    """
    Check if stop loss is triggered

    Parameters
    ----------
    current_price : float
        Current market price
    stop_loss_price : float
        Stop loss price level
    position_side : str
        Current position side ('BUY' or 'SELL')

    Returns
    -------
    bool
        True if stop loss is triggered, False otherwise
    """
    if position_side == "BUY":
        return current_price <= stop_loss_price
    elif position_side == "SELL":
        return current_price >= stop_loss_price
    else:
        return False


def calculate_stop_loss_take_profit(
    entry_price: float, atr_value: float, atr_multiple: float, position_side: str
) -> Tuple[float, float]:
    """
    Calculate stop loss and take profit levels based on ATR

    Parameters
    ----------
    entry_price : float
        Position entry price
    atr_value : float
        Current ATR value
    atr_multiple : float
        ATR multiple for stop loss distance
    position_side : str
        Position side ('BUY' or 'SELL')

    Returns
    -------
    Tuple[float, float]
        (stop_loss_price, take_profit_price)
    """
    stop_distance = atr_multiple * atr_value

    if position_side == "BUY":
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + (stop_distance * 2)  # 2:1 reward-to-risk
    else:  # SELL position
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - (stop_distance * 2)  # 2:1 reward-to-risk

    return stop_loss, take_profit
