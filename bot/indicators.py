"""
Technical indicators for the Turtle Trading Bot
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pandas import DataFrame


def calculate_donchian_channel(
    df: pd.DataFrame, n: int, suffix: str = ""
) -> pd.DataFrame:
    """
    Calculate Donchian Channel for a given period

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with 'high' and 'low' columns
    n : int
        Donchian Channel period
    suffix : str, optional
        Suffix for column names

    Returns
    -------
    pd.DataFrame
        DataFrame with added Donchian Channel columns
    """
    if df.empty or n <= 0:
        return df

    df_result = df.copy()

    # Calculate upper and lower bands
    df_result[f"dc_upper{suffix}"] = df_result["high"].rolling(window=n).max()
    df_result[f"dc_lower{suffix}"] = df_result["low"].rolling(window=n).min()
    df_result[f"dc_middle{suffix}"] = (
        df_result[f"dc_upper{suffix}"] + df_result[f"dc_lower{suffix}"]
    ) / 2

    return df_result


def calculate_atr(df: pd.DataFrame, n: int, smooth_type: str = "RMA") -> pd.DataFrame:
    """
    Calculate Average True Range (ATR)

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with 'high', 'low', and 'close' columns
    n : int
        ATR period
    smooth_type : str, optional
        Smoothing type ('RMA', 'SMA', 'EMA')

    Returns
    -------
    pd.DataFrame
        DataFrame with added ATR column
    """
    if df.empty or n <= 0:
        return df

    df_result = df.copy()

    # Calculate true range
    high_low = df_result["high"] - df_result["low"]
    high_close_prev = abs(df_result["high"] - df_result["close"].shift(1))
    low_close_prev = abs(df_result["low"] - df_result["close"].shift(1))

    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(
        axis=1
    )

    # Calculate ATR based on smoothing type
    if smooth_type.upper() == "RMA":
        # RMA (Wilder's Smoothing)
        atr = true_range.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    elif smooth_type.upper() == "EMA":
        # Exponential Moving Average
        atr = true_range.ewm(span=n, min_periods=n).mean()
    else:
        # Simple Moving Average
        atr = true_range.rolling(window=n).mean()

    df_result["atr"] = atr

    return df_result


def calculate_ma(df: pd.DataFrame, period: int, type: str = "SMA") -> pd.DataFrame:
    """
    Calculate Moving Average

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with 'close' column
    period : int
        MA period
    type : str
        Type of MA ('SMA', 'EMA')

    Returns
    -------
    pd.DataFrame
        DataFrame with added MA column
    """
    if df.empty or period <= 0:
        return df

    df_result = df.copy()

    if type.upper() == "EMA":
        df_result["ma"] = df_result["close"].ewm(span=period, min_periods=period).mean()
    else:  # SMA
        df_result["ma"] = df_result["close"].rolling(window=period).mean()

    return df_result


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with 'high', 'low', and 'close' columns
    period : int
        ADX period

    Returns
    -------
    pd.DataFrame
        DataFrame with added ADX columns
    """
    if df.empty or period <= 0:
        return df

    df_result = df.copy()

    # Calculate +DM, -DM
    df_result["high_diff"] = df_result["high"].diff()
    df_result["low_diff"] = -df_result["low"].diff()

    df_result["plus_dm"] = (
        (df_result["high_diff"] > df_result["low_diff"]) & (df_result["high_diff"] > 0)
    ) * df_result["high_diff"]
    df_result["minus_dm"] = (
        (df_result["low_diff"] > df_result["high_diff"]) & (df_result["low_diff"] > 0)
    ) * df_result["low_diff"]

    # Calculate TR
    df_result["tr"] = df_result.apply(
        lambda x: (
            max(
                [
                    x["high"] - x["low"],
                    abs(x["high"] - x["close"].shift(1)),
                    abs(x["low"] - x["close"].shift(1)),
                ]
            )
            if not pd.isna(x["close"].shift(1))
            else x["high"] - x["low"]
        ),
        axis=1,
    )

    # Calculate smoothed values
    df_result["tr_" + str(period)] = df_result["tr"].rolling(window=period).sum()
    df_result["plus_dm_" + str(period)] = (
        df_result["plus_dm"].rolling(window=period).sum()
    )
    df_result["minus_dm_" + str(period)] = (
        df_result["minus_dm"].rolling(window=period).sum()
    )

    # Calculate +DI, -DI
    df_result["plus_di_" + str(period)] = (
        100 * df_result["plus_dm_" + str(period)] / df_result["tr_" + str(period)]
    )
    df_result["minus_di_" + str(period)] = (
        100 * df_result["minus_dm_" + str(period)] / df_result["tr_" + str(period)]
    )

    # Calculate DX and ADX
    df_result["dx_" + str(period)] = (
        100
        * abs(
            df_result["plus_di_" + str(period)] - df_result["minus_di_" + str(period)]
        )
        / (df_result["plus_di_" + str(period)] + df_result["minus_di_" + str(period)])
    )
    df_result["adx_" + str(period)] = (
        df_result["dx_" + str(period)].rolling(window=period).mean()
    )

    # Simplify column names
    df_result["plus_di"] = df_result["plus_di_" + str(period)]
    df_result["minus_di"] = df_result["minus_di_" + str(period)]
    df_result["adx"] = df_result["adx_" + str(period)]

    # Clean up dataframe by dropping temporary columns
    columns_to_drop = [
        "high_diff",
        "low_diff",
        "plus_dm",
        "minus_dm",
        "tr",
        "tr_" + str(period),
        "plus_dm_" + str(period),
        "minus_dm_" + str(period),
        "plus_di_" + str(period),
        "minus_di_" + str(period),
        "dx_" + str(period),
    ]
    df_result = df_result.drop(columns=columns_to_drop, errors="ignore")

    return df_result


def calculate_indicators(
    df: pd.DataFrame,
    dc_enter: int,
    dc_exit: int,
    atr_len: int,
    atr_smooth: str = "RMA",
    ma_period: int = 200,
    adx_period: int = 14,
) -> pd.DataFrame:
    """
    Calculate all indicators needed for the Turtle Trading strategy

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with 'open', 'high', 'low', 'close', and 'volume' columns
    dc_enter : int
        Donchian Channel period for entry signals
    dc_exit : int
        Donchian Channel period for exit signals
    atr_len : int
        ATR period
    atr_smooth : str, optional
        ATR smoothing type ('RMA', 'SMA', 'EMA')
    ma_period : int, optional
        Moving Average period for trend filter
    adx_period : int, optional
        ADX period for trend strength

    Returns
    -------
    pd.DataFrame
        DataFrame with all indicators calculated
    """
    if df.empty:
        return df

    # Make a copy to avoid modifying the original
    df_result = df.copy()

    # Calculate Donchian Channels for Entry
    df_result = calculate_donchian_channel(df_result, dc_enter, suffix="_entry")

    # Calculate Donchian Channels for Exit
    df_result = calculate_donchian_channel(df_result, dc_exit, suffix="_exit")

    # Calculate ATR
    df_result = calculate_atr(df_result, atr_len, atr_smooth)

    # Calculate MA for trend filter
    df_result = calculate_ma(df_result, ma_period)

    # Calculate ADX
    df_result = calculate_adx(df_result, adx_period)

    return df_result


def check_entry_signal(row: pd.Series, side: str) -> bool:
    """
    Check if there's an entry signal based on Donchian Channel breakout

    Parameters
    ----------
    row : pd.Series
        DataFrame row with indicator values
    side : str
        Trade side ('BUY' or 'SELL')

    Returns
    -------
    bool
        True if entry signal, False otherwise
    """
    if side == "BUY":
        # For long entries, price breaks above upper band
        return row["close"] > row["dc_upper_entry"]
    else:
        # For short entries, price breaks below lower band
        return row["close"] < row["dc_lower_entry"]


def check_adx_filter(row: pd.Series, threshold: int = 25) -> bool:
    """
    Check ADX filter for trend strength

    Parameters
    ----------
    row : pd.Series
        DataFrame row with indicator values
    threshold : int
        ADX threshold value

    Returns
    -------
    bool
        True if ADX is above threshold, False otherwise
    """
    return row.get("adx", 0) >= threshold


def check_ma_filter(row: pd.Series, side: str) -> bool:
    """
    Check MA filter for trend direction

    Parameters
    ----------
    row : pd.Series
        DataFrame row with indicator values
    side : str
        Trade side ('BUY' or 'SELL')

    Returns
    -------
    bool
        True if price is in the correct side of MA, False otherwise
    """
    if "ma" not in row:
        return True  # No MA filter if MA is not calculated

    if side == "BUY":
        # For long entries, price should be above MA
        return row["close"] > row["ma"]
    else:
        # For short entries, price should be below MA
        return row["close"] < row["ma"]


def check_exit_signal(row: pd.Series, position_side: str) -> bool:
    """
    Check if there's an exit signal based on Donchian Channel breakout

    Parameters
    ----------
    row : pd.Series
        DataFrame row with indicator values
    position_side : str
        Current position side ('BUY' or 'SELL')

    Returns
    -------
    bool
        True if exit signal, False otherwise
    """
    if position_side == "BUY":
        # For long positions, exit when price breaks below lower band
        return row["close"] < row["dc_lower_exit"]
    else:
        # For short positions, exit when price breaks above upper band
        return row["close"] > row["dc_upper_exit"]


def check_partial_exit(
    current_price: float,
    entry_price: float,
    atr_value: float,
    position_side: str,
    target_multiple: float,
) -> bool:
    """
    Check if partial take profit target is reached

    Parameters
    ----------
    current_price : float
        Current market price
    entry_price : float
        Position entry price
    atr_value : float
        ATR value at entry
    position_side : str
        Position side ('BUY' or 'SELL')
    target_multiple : float
        Target as multiple of ATR

    Returns
    -------
    bool
        True if target reached, False otherwise
    """
    target_distance = atr_value * target_multiple

    if position_side == "BUY":
        target_price = entry_price + target_distance
        return current_price >= target_price
    else:
        target_price = entry_price - target_distance
        return current_price <= target_price


def check_stop_loss(
    current_price: float, stop_price: float, position_side: str
) -> bool:
    """
    Check if stop loss is triggered

    Parameters
    ----------
    current_price : float
        Current market price
    stop_price : float
        Stop loss price
    position_side : str
        Position side ('BUY' or 'SELL')

    Returns
    -------
    bool
        True if stop loss triggered, False otherwise
    """
    if position_side == "BUY":
        # For long positions, stop is below entry
        return current_price <= stop_price
    else:
        # For short positions, stop is above entry
        return current_price >= stop_price


def calculate_stop_loss_take_profit(
    entry_price: float, atr_value: float, atr_multiple: float, position_side: str
) -> Tuple[float, float]:
    """
    Calculate Stop Loss and Take Profit levels based on ATR

    Parameters
    ----------
    entry_price : float
        Position entry price
    atr_value : float
        ATR value at entry
    atr_multiple : float
        Multiple of ATR for stop loss distance
    position_side : str
        Position side ('BUY' or 'SELL')

    Returns
    -------
    Tuple[float, float]
        (stop_loss_price, take_profit_price)
    """
    stop_distance = atr_value * atr_multiple

    if position_side == "BUY":
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + (stop_distance * 2)  # 2:1 reward-to-risk ratio
    else:
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - (stop_distance * 2)  # 2:1 reward-to-risk ratio

    return stop_loss, take_profit


def update_trailing_stop(
    current_price: float,
    entry_price: float,
    current_trailing_stop: float,
    atr_value: float,
    position_side: str,
    min_profit_multiple: float = 1.0,
) -> float:
    """
    Update trailing stop price once in profit

    Parameters
    ----------
    current_price : float
        Current market price
    entry_price : float
        Position entry price
    current_trailing_stop : float
        Current trailing stop price
    atr_value : float
        ATR value at entry
    position_side : str
        Position side ('BUY' or 'SELL')
    min_profit_multiple : float
        Minimum profit as multiple of ATR to activate trailing stop

    Returns
    -------
    float
        Updated trailing stop price
    """
    min_profit_distance = atr_value * min_profit_multiple

    if position_side == "BUY":
        # For long positions
        min_profit_price = entry_price + min_profit_distance

        # Only start trailing if we're in sufficient profit
        if current_price <= min_profit_price:
            return current_trailing_stop

        # Calculate new potential trailing stop (5-day low)
        new_stop = current_price - (atr_value * 1.0)  # 1 ATR below current price

        # Only move stop up, never down
        if new_stop > current_trailing_stop:
            return new_stop
    else:
        # For short positions
        min_profit_price = entry_price - min_profit_distance

        # Only start trailing if we're in sufficient profit
        if current_price >= min_profit_price:
            return current_trailing_stop

        # Calculate new potential trailing stop (5-day high)
        new_stop = current_price + (atr_value * 1.0)  # 1 ATR above current price

        # Only move stop down, never up
        if new_stop < current_trailing_stop or current_trailing_stop == 0:
            return new_stop

    return current_trailing_stop
