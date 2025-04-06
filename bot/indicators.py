"""
Technical indicators for the Turtle Trading Bot
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from pandas import DataFrame
import logging


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

    try:
        # Ensure we have a proper DataFrame with index
        if not isinstance(df_result, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Calculate true range more robustly
        high_low = df_result["high"] - df_result["low"]

        # Initialize series for prev close comparisons
        high_close_prev = pd.Series(0, index=df_result.index)
        low_close_prev = pd.Series(0, index=df_result.index)

        # Only calculate with shift for rows beyond the first one
        if len(df_result) > 1:
            close_prev = df_result["close"].shift(1)
            # Fill first row with a reasonable default
            close_prev.iloc[0] = df_result["close"].iloc[0]

            high_close_prev = (df_result["high"] - close_prev).abs()
            low_close_prev = (df_result["low"] - close_prev).abs()

        # Combine the three series and get the maximum at each position
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

        # Add ATR to result DataFrame
        df_result["atr"] = atr

        return df_result

    except Exception as e:
        # In case of any error, return DataFrame with default ATR
        logger = logging.getLogger("turtle_trading_bot")
        logger.error(f"Error calculating ATR: {e}. Using default value.")

        # Set a default ATR value based on average price
        if "close" in df_result.columns:
            avg_price = df_result["close"].mean()
            default_atr = avg_price * 0.02  # Use 2% of average price as default ATR
        else:
            default_atr = 1000  # Fallback default for BTC

        df_result["atr"] = default_atr
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

    # Make a copy to avoid modifying the original
    df_result = df.copy()

    try:
        # Ensure we have a proper DataFrame with index
        if not isinstance(df_result, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Calculate +DM, -DM more reliably
        high_diff = df_result["high"].diff().fillna(0)
        low_diff = -df_result["low"].diff().fillna(0)

        # Calculate +DM (directional movement up)
        plus_dm = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
        plus_dm = plus_dm.fillna(0)

        # Calculate -DM (directional movement down)
        minus_dm = ((low_diff > high_diff) & (low_diff > 0)) * low_diff
        minus_dm = minus_dm.fillna(0)

        # Calculate true range without using apply
        high_low = df_result["high"] - df_result["low"]
        high_close_prev = pd.Series(0, index=df_result.index)
        low_close_prev = pd.Series(0, index=df_result.index)

        # Only calculate with shift for rows beyond the first one
        if len(df_result) > 1:
            close_prev = df_result["close"].shift(1)
            high_close_prev = (df_result["high"] - close_prev).abs()
            low_close_prev = (df_result["low"] - close_prev).abs()

        # True range is the greatest of the three
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        # Calculate smoothed values
        # Instead of rolling sum, we'll calculate the exponential moving average (Wilder's smoothing)
        smooth_period = period

        # Initialize smoothed series
        smoothed_tr = tr.ewm(
            alpha=1 / smooth_period, min_periods=period, adjust=False
        ).mean()
        smoothed_plus_dm = plus_dm.ewm(
            alpha=1 / smooth_period, min_periods=period, adjust=False
        ).mean()
        smoothed_minus_dm = minus_dm.ewm(
            alpha=1 / smooth_period, min_periods=period, adjust=False
        ).mean()

        # Calculate +DI and -DI
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr

        # Calculate DX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).fillna(0)
        # Replace infinity values with 0
        dx = dx.replace([np.inf, -np.inf], 0)

        # Calculate ADX
        adx = dx.ewm(alpha=1 / smooth_period, min_periods=period, adjust=False).mean()

        # Add columns to result DataFrame
        df_result["plus_di"] = plus_di
        df_result["minus_di"] = minus_di
        df_result["adx"] = adx

        return df_result

    except Exception as e:
        # In case of any error, return DataFrame with default ADX value
        logger = logging.getLogger("turtle_trading_bot")
        logger.error(f"Error calculating ADX: {e}. Using default values.")

        # Add default ADX columns
        df_result["plus_di"] = 25.0
        df_result["minus_di"] = 25.0
        df_result["adx"] = 25.0

        return df_result


def calculate_bollinger_bands(
    df: pd.DataFrame, period: int = 20, stdev_multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands for a given period

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with 'close' column
    period : int
        Bollinger Bands period (default: 20)
    stdev_multiplier : float
        Standard deviation multiplier for band width (default: 2.0)

    Returns
    -------
    pd.DataFrame
        DataFrame with added Bollinger Bands columns
    """
    if df.empty or period <= 0:
        return df

    df_result = df.copy()

    # Calculate middle band (SMA)
    df_result["bb_middle"] = df_result["close"].rolling(window=period).mean()

    # Calculate band width (standard deviation)
    rolling_std = df_result["close"].rolling(window=period).std()

    # Calculate upper and lower bands
    df_result["bb_upper"] = df_result["bb_middle"] + (rolling_std * stdev_multiplier)
    df_result["bb_lower"] = df_result["bb_middle"] - (rolling_std * stdev_multiplier)

    # Calculate bandwidth (useful for volatility)
    df_result["bb_bandwidth"] = (
        df_result["bb_upper"] - df_result["bb_lower"]
    ) / df_result["bb_middle"]

    # Calculate percent B (where price is relative to bands)
    df_result["bb_percent_b"] = (df_result["close"] - df_result["bb_lower"]) / (
        df_result["bb_upper"] - df_result["bb_lower"]
    )

    return df_result


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI)

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with 'close' column
    period : int
        RSI period (default: 14)

    Returns
    -------
    pd.DataFrame
        DataFrame with added RSI column
    """
    if df.empty or period <= 0:
        return df

    df_result = df.copy()

    try:
        # Calculate price changes
        delta = df_result["close"].diff()

        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)  # Convert losses to positive values

        # First average gains and losses over period
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Calculate RS (Relative Strength) and RSI
        rs = avg_gain / avg_loss
        df_result["rsi"] = 100 - (100 / (1 + rs))

        # Check for NaN values in RSI
        if df_result["rsi"].isna().any():
            # Use EMA (Exponential Moving Average) for gains and losses if we get NaNs
            # This is more robust for crypto data
            avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
            avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

            rs = avg_gain / avg_loss
            df_result["rsi"] = 100 - (100 / (1 + rs))

        # Handle remaining NaN values
        df_result["rsi"].fillna(50, inplace=True)

        # Handle infinity values
        df_result["rsi"].replace([np.inf, -np.inf], 50, inplace=True)

        return df_result

    except Exception as e:
        # In case of any error, return DataFrame with default RSI value of 50
        logger = logging.getLogger("turtle_trading_bot")
        logger.error(f"Error calculating RSI: {e}. Using default value.")

        df_result["rsi"] = 50
        return df_result


def calculate_macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Calculate Moving Average Convergence Divergence (MACD)

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with 'close' column
    fast_period : int
        Fast EMA period (default: 12)
    slow_period : int
        Slow EMA period (default: 26)
    signal_period : int
        Signal EMA period (default: 9)

    Returns
    -------
    pd.DataFrame
        DataFrame with added MACD columns
    """
    if df.empty:
        return df

    df_result = df.copy()

    try:
        # Calculate fast and slow EMAs
        ema_fast = df_result["close"].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df_result["close"].ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line
        df_result["macd"] = ema_fast - ema_slow

        # Calculate signal line
        df_result["macd_signal"] = (
            df_result["macd"].ewm(span=signal_period, adjust=False).mean()
        )

        # Calculate histogram
        df_result["macd_hist"] = df_result["macd"] - df_result["macd_signal"]

        return df_result

    except Exception as e:
        # In case of any error, return DataFrame with default MACD values
        logger = logging.getLogger("turtle_trading_bot")
        logger.error(f"Error calculating MACD: {e}. Using default values.")

        df_result["macd"] = 0.0
        df_result["macd_signal"] = 0.0
        df_result["macd_hist"] = 0.0

        return df_result


def calculate_indicators(
    df: pd.DataFrame,
    dc_enter: int,
    dc_exit: int,
    atr_len: int,
    atr_smooth: str = "RMA",
    ma_period: int = 200,
    adx_period: int = 14,
    bb_period: int = 20,
    rsi_period: int = 14,
    include_additional: bool = True,
    position_side: str = None,  # 'BUY' veya 'SELL' - Short pozisyonlar için özel hesaplamalar
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
    bb_period : int, optional
        Bollinger Bands period
    rsi_period : int, optional
        RSI period
    include_additional : bool, optional
        Whether to include additional indicators (BB, RSI)
    position_side : str, optional
        Position side to optimize parameters for ('BUY' or 'SELL')

    Returns
    -------
    pd.DataFrame
        DataFrame with all indicators calculated
    """
    if df.empty:
        return df

    # Make a copy to avoid modifying the original
    df_result = df.copy()

    # Adjust parameters for short positions if needed
    if position_side == "SELL":
        # Short pozisyonlar için farklı ATR çarpanı (kısa pozisyonlar için daha büyük ATR)
        atr_len = int(atr_len * 1.2)  # %20 daha uzun ATR periyodu

    # Calculate Donchian Channels for Entry
    df_result = calculate_donchian_channel(df_result, dc_enter, suffix="_entry")

    # Calculate Donchian Channels for Exit
    df_result = calculate_donchian_channel(df_result, dc_exit, suffix="_exit")

    # Calculate ATR
    df_result = calculate_atr(df_result, atr_len, atr_smooth)

    # Calculate MA for trend filter
    if ma_period > 0:
        try:
            df_result["ma"] = df_result["close"].rolling(window=ma_period).mean()
        except Exception as e:
            # Hata yerine NaN değerler koyarak devam et
            df_result["ma"] = float("nan")
            logger = logging.getLogger("turtle_trading_bot")
            logger.warning(
                f"MA hesaplaması yapılamadı: {e}, varsayılan değerler kullanılıyor"
            )

    # Calculate ADX
    df_result = calculate_adx(df_result, adx_period)

    # Calculate additional indicators if requested
    if include_additional:
        # Calculate Bollinger Bands
        df_result = calculate_bollinger_bands(df_result, bb_period)

        # Calculate RSI
        df_result = calculate_rsi(df_result, rsi_period)

        # Calculate MACD
        df_result = calculate_macd(df_result)

    return df_result


def check_entry_signal(row, side):
    """
    Check if there is an entry signal based on the Donchian Channel breakout.

    Args:
        row: DataFrame row with indicators
        side: Position side ("BUY" or "SELL")

    Returns:
        bool: True if there's an entry signal, False otherwise
    """
    try:
        if side == "BUY":
            # Long entry when price breaks above upper Donchian Channel
            return row["close"] > row["dc_upper"]
        elif side == "SELL":
            # Short entry when price breaks below lower Donchian Channel
            return row["close"] < row["dc_lower"]
        else:
            return False
    except Exception as e:
        logging.error(f"Error in check_entry_signal: {e}")
        return False


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


def check_ma_filter(row, signal_direction):
    """
    Check if the price position relative to MA confirms the trend direction.

    Args:
        row: DataFrame row with indicators
        signal_direction: "long" or "short" signal direction

    Returns:
        bool: True if MA filter confirms trend, False otherwise
    """
    try:
        if signal_direction == "long":
            # For long signals, we want price to be above MA
            return row["close"] > row["ma"]
        elif signal_direction == "short":
            # For short signals, we want price to be below MA
            # For short trades, we want price to be at least 1% below MA for stronger confirmation
            return row["close"] < row["ma"] * 0.99
        return False
    except Exception as e:
        logging.error(f"Error in check_ma_filter: {e}")
        return False


def check_exit_signal(row, side):
    """
    Check if there is an exit signal based on the Donchian Channel breakout in the opposite direction.

    Args:
        row: DataFrame row with indicators
        side: Position side ("BUY" or "SELL")

    Returns:
        bool: True if there's an exit signal, False otherwise
    """
    try:
        if side == "BUY":
            # Exit long position when price breaks below lower Donchian Channel
            return row["close"] < row["dc_lower"]
        elif side == "SELL":
            # Exit short position when price breaks above upper Donchian Channel
            return row["close"] > row["dc_upper"]
        else:
            return False
    except Exception as e:
        logging.error(f"Error in check_exit_signal: {e}")
        return False


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


def check_bb_squeeze(row: pd.Series) -> bool:
    """
    Check for Bollinger Bands squeeze (low volatility)

    Parameters
    ----------
    row : pd.Series
        DataFrame row with indicator values

    Returns
    -------
    bool
        True if BB squeeze is detected, False otherwise
    """
    if "bb_bandwidth" not in row:
        return False

    # BB bandwidth threshold for squeeze detection
    # Lower values indicate tighter bands (less volatility)
    return row["bb_bandwidth"] < 0.1


def check_rsi_conditions(row: pd.Series, side: str) -> bool:
    """
    Check RSI conditions for entry confirmation

    Parameters
    ----------
    row : pd.Series
        DataFrame row with indicator values
    side : str
        Trade side ('BUY' or 'SELL')

    Returns
    -------
    bool
        True if RSI conditions confirm entry, False otherwise
    """
    if "rsi" not in row:
        return True  # No RSI filter if RSI is not calculated

    if side == "BUY":
        # For long entries, RSI should be above 40 (not oversold)
        # but below 70 (not overbought)
        return 40 <= row["rsi"] <= 70
    else:
        # For short entries, RSI should be below 60 (not overbought)
        # but above 30 (not oversold)
        return 30 <= row["rsi"] <= 60


def check_macd_confirmation(row: pd.Series, side: str) -> bool:
    """
    Check MACD confirmation for entry signal

    Parameters
    ----------
    row : pd.Series
        DataFrame row with indicator values
    side : str
        Trade side ('BUY' or 'SELL')

    Returns
    -------
    bool
        True if MACD confirms the entry signal, False otherwise
    """
    if "macd" not in row or "macd_signal" not in row:
        return True  # No MACD filter if MACD is not calculated

    if side == "BUY":
        # For long entries: MACD > Signal Line (bullish momentum)
        return row["macd"] > row["macd_signal"] and row["macd_hist"] > 0
    else:
        # For short entries: MACD < Signal Line (bearish momentum)
        return row["macd"] < row["macd_signal"] and row["macd_hist"] < 0


def calculate_indicators_incremental(
    current_df: pd.DataFrame,
    new_row: pd.Series,
    dc_enter: int,
    dc_exit: int,
    atr_len: int,
    position_side: str = None,
    ma_period: int = 200,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    adx_period: int = 14,
    bb_period: int = 20,
    include_additional: bool = True,
) -> pd.DataFrame:
    """
    Incrementally calculate indicators when new data arrives
    This is much more efficient than recalculating all indicators from scratch

    Parameters
    ----------
    current_df : pd.DataFrame
        Current DataFrame with existing indicators
    new_row : pd.Series
        New price data row to append
    dc_enter : int
        Donchian Channel entry period
    dc_exit : int
        Donchian Channel exit period
    atr_len : int
        ATR period
    position_side : str, optional
        Current position side for specialized calculations ('BUY' or 'SELL')
    ma_period : int, optional
        Moving Average period
    rsi_period : int, optional
        RSI period
    macd_fast : int, optional
        MACD fast period
    macd_slow : int, optional
        MACD slow period
    macd_signal : int, optional
        MACD signal period
    adx_period : int, optional
        ADX period
    bb_period : int, optional
        Bollinger Bands period
    include_additional : bool, optional
        Whether to include additional indicators

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with new indicators
    """
    # Create a copy of the current DataFrame
    df = current_df.copy()

    # Ensure the new row has a valid index
    if isinstance(new_row, dict):
        new_row = pd.Series(new_row)

    # If the index is not a datetime, convert it
    if not isinstance(new_row.name, pd.Timestamp):
        if "timestamp" in new_row:
            new_row.name = pd.to_datetime(new_row["timestamp"], unit="ms")
        else:
            new_row.name = pd.Timestamp.now()

    # Append the new row
    df_new = pd.concat([df, new_row.to_frame().T])

    # Ensure required columns exist
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df_new.columns:
            if col in new_row:
                # New row has the column but DataFrame doesn't
                df_new[col] = np.nan
                df_new.loc[new_row.name, col] = new_row[col]
            else:
                # Neither has it, create with default values
                df_new[col] = np.nan

    # Get last windows of data needed for calculations
    max_lookback = (
        max(
            dc_enter,
            dc_exit,
            atr_len,
            ma_period,
            rsi_period,
            macd_slow + macd_signal,
            adx_period,
            bb_period,
        )
        + 10
    )  # Add buffer

    # Get the most recent data needed for calculations
    # This prevents processing the entire DataFrame
    df_window = df_new.iloc[-max_lookback:] if len(df_new) > max_lookback else df_new

    # Incrementally update indicators

    # 1. Donchian Channels
    # Entry channel
    df_new.loc[new_row.name, "dc_upper"] = (
        df_window["high"].rolling(window=dc_enter).max().iloc[-1]
    )
    df_new.loc[new_row.name, "dc_lower"] = (
        df_window["low"].rolling(window=dc_enter).min().iloc[-1]
    )
    df_new.loc[new_row.name, "dc_middle"] = (
        df_new.loc[new_row.name, "dc_upper"] + df_new.loc[new_row.name, "dc_lower"]
    ) / 2

    # Exit channel
    df_new.loc[new_row.name, "dc_upper_exit"] = (
        df_window["high"].rolling(window=dc_exit).max().iloc[-1]
    )
    df_new.loc[new_row.name, "dc_lower_exit"] = (
        df_window["low"].rolling(window=dc_exit).min().iloc[-1]
    )
    df_new.loc[new_row.name, "dc_middle_exit"] = (
        df_new.loc[new_row.name, "dc_upper_exit"]
        + df_new.loc[new_row.name, "dc_lower_exit"]
    ) / 2

    # 2. ATR - We need to calculate TR first
    # Get previous close
    prev_close = (
        df_window["close"].iloc[-2]
        if len(df_window) > 1
        else df_window["open"].iloc[-1]
    )

    # Calculate true range components
    high_low = df_window["high"].iloc[-1] - df_window["low"].iloc[-1]
    high_close_prev = abs(df_window["high"].iloc[-1] - prev_close)
    low_close_prev = abs(df_window["low"].iloc[-1] - prev_close)

    # Get the maximum as true range
    true_range = max(high_low, high_close_prev, low_close_prev)

    # Check if ATR already exists
    if (
        "atr" in df_new.columns
        and not pd.isna(df_new["atr"].iloc[-2])
        and len(df_new) > 1
    ):
        # Use Wilder's smoothing formula for incremental update
        prev_atr = df_new["atr"].iloc[-2]
        current_atr = ((prev_atr * (atr_len - 1)) + true_range) / atr_len
        df_new.loc[new_row.name, "atr"] = current_atr
    else:
        # Initial ATR calculation if not enough history
        if len(df_window) >= atr_len:
            # Calculate ATR using the window
            true_ranges = []
            for i in range(1, len(df_window)):
                hl = df_window["high"].iloc[i] - df_window["low"].iloc[i]
                hcp = abs(df_window["high"].iloc[i] - df_window["close"].iloc[i - 1])
                lcp = abs(df_window["low"].iloc[i] - df_window["close"].iloc[i - 1])
                true_ranges.append(max(hl, hcp, lcp))

            if len(true_ranges) >= atr_len:
                # Simple average of first ATR_LENGTH true ranges
                first_atr = sum(true_ranges[:atr_len]) / atr_len
                # Smooth subsequent values
                current_atr = first_atr
                for tr in true_ranges[atr_len:]:
                    current_atr = ((current_atr * (atr_len - 1)) + tr) / atr_len
                df_new.loc[new_row.name, "atr"] = current_atr
            else:
                # Not enough data, use a simple average
                df_new.loc[new_row.name, "atr"] = (
                    sum(true_ranges) / len(true_ranges) if true_ranges else np.nan
                )
        else:
            # Not enough data points, set to NaN or estimate
            df_new.loc[new_row.name, "atr"] = np.nan

    # 3. Moving Average (SMA)
    if include_additional and ma_period > 0:
        if len(df_window) >= ma_period:
            df_new.loc[new_row.name, "ma"] = (
                df_window["close"].rolling(window=ma_period).mean().iloc[-1]
            )
        else:
            df_new.loc[new_row.name, "ma"] = df_window[
                "close"
            ].mean()  # Use available data

    # 4. RSI - more complex incremental calculation
    if include_additional and rsi_period > 0:
        # Get the lookback window we need
        rsi_window = df_window["close"].iloc[-(rsi_period + 1) :].values

        if len(rsi_window) > 1:
            # Calculate price changes
            deltas = np.diff(rsi_window)

            # Get gains and losses
            gains = deltas.copy()
            losses = deltas.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)

            # Check if we have enough data for the full calculation
            if len(gains) >= rsi_period:
                # If we have previous average gain/loss values
                if (
                    "rsi" in df_new.columns
                    and not pd.isna(df_new["rsi"].iloc[-2])
                    and len(df_new) > rsi_period + 1
                ):
                    # Get previous values (these should be stored for truly incremental)
                    # For this example, we'll recalculate from the window
                    avg_gain = np.mean(gains[:rsi_period])
                    avg_loss = np.mean(losses[:rsi_period])

                    # Smooth with new values
                    for i in range(rsi_period, len(gains)):
                        avg_gain = (
                            (avg_gain * (rsi_period - 1)) + gains[i]
                        ) / rsi_period
                        avg_loss = (
                            (avg_loss * (rsi_period - 1)) + losses[i]
                        ) / rsi_period
                else:
                    # Calculate first average gain/loss
                    avg_gain = np.mean(gains[:rsi_period])
                    avg_loss = np.mean(losses[:rsi_period])

                    # Smooth subsequent values
                    for i in range(rsi_period, len(gains)):
                        avg_gain = (
                            (avg_gain * (rsi_period - 1)) + gains[i]
                        ) / rsi_period
                        avg_loss = (
                            (avg_loss * (rsi_period - 1)) + losses[i]
                        ) / rsi_period

                # Calculate RS and RSI
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    # No losses, RSI is 100
                    rsi = 100

                df_new.loc[new_row.name, "rsi"] = rsi
            else:
                # Not enough data
                df_new.loc[new_row.name, "rsi"] = 50  # Neutral value

    # 5. MACD - incremental implementation
    if include_additional:
        # Calculate fast EMA
        if (
            "ema_fast" in df_new.columns
            and not pd.isna(df_new["ema_fast"].iloc[-2])
            and len(df_new) > 1
        ):
            # Update fast EMA
            alpha_fast = 2 / (macd_fast + 1)
            prev_ema_fast = df_new["ema_fast"].iloc[-2]
            current_ema_fast = (
                df_window["close"].iloc[-1] - prev_ema_fast
            ) * alpha_fast + prev_ema_fast
            df_new.loc[new_row.name, "ema_fast"] = current_ema_fast
        else:
            # Calculate from scratch if needed
            if len(df_window) >= macd_fast:
                df_new.loc[new_row.name, "ema_fast"] = (
                    df_window["close"].ewm(span=macd_fast, adjust=False).mean().iloc[-1]
                )
            else:
                df_new.loc[new_row.name, "ema_fast"] = df_window["close"].mean()

        # Calculate slow EMA
        if (
            "ema_slow" in df_new.columns
            and not pd.isna(df_new["ema_slow"].iloc[-2])
            and len(df_new) > 1
        ):
            # Update slow EMA
            alpha_slow = 2 / (macd_slow + 1)
            prev_ema_slow = df_new["ema_slow"].iloc[-2]
            current_ema_slow = (
                df_window["close"].iloc[-1] - prev_ema_slow
            ) * alpha_slow + prev_ema_slow
            df_new.loc[new_row.name, "ema_slow"] = current_ema_slow
        else:
            # Calculate from scratch if needed
            if len(df_window) >= macd_slow:
                df_new.loc[new_row.name, "ema_slow"] = (
                    df_window["close"].ewm(span=macd_slow, adjust=False).mean().iloc[-1]
                )
            else:
                df_new.loc[new_row.name, "ema_slow"] = df_window["close"].mean()

        # Calculate MACD line
        df_new.loc[new_row.name, "macd"] = (
            df_new.loc[new_row.name, "ema_fast"] - df_new.loc[new_row.name, "ema_slow"]
        )

        # Calculate signal line
        if (
            "macd_signal" in df_new.columns
            and not pd.isna(df_new["macd_signal"].iloc[-2])
            and len(df_new) > 1
        ):
            # Update signal EMA
            alpha_signal = 2 / (macd_signal + 1)
            prev_signal = df_new["macd_signal"].iloc[-2]
            current_signal = (
                df_new.loc[new_row.name, "macd"] - prev_signal
            ) * alpha_signal + prev_signal
            df_new.loc[new_row.name, "macd_signal"] = current_signal
        else:
            # Calculate from scratch if needed
            if "macd" in df_new.columns:
                macd_series = df_new["macd"].dropna()
                if len(macd_series) >= macd_signal:
                    df_new.loc[new_row.name, "macd_signal"] = (
                        macd_series.ewm(span=macd_signal, adjust=False).mean().iloc[-1]
                    )
                else:
                    df_new.loc[new_row.name, "macd_signal"] = (
                        macd_series.mean() if not macd_series.empty else 0
                    )
            else:
                df_new.loc[new_row.name, "macd_signal"] = 0

        # Calculate histogram
        df_new.loc[new_row.name, "macd_hist"] = (
            df_new.loc[new_row.name, "macd"] - df_new.loc[new_row.name, "macd_signal"]
        )

    # 6. Add two-way price action detection
    if include_additional:
        lookback = 5  # Short lookback for quick detection
        if len(df_window) >= lookback:
            # Get recent price data
            recent_window = df_window.iloc[-lookback:]

            # Calculate price movements
            price_movements = np.diff(recent_window["close"].values)

            # Count positive and negative movements
            pos_moves = sum(1 for x in price_movements if x > 0)
            neg_moves = sum(1 for x in price_movements if x < 0)

            # Check if we have significant movements in both directions
            has_two_way_action = pos_moves >= 2 and neg_moves >= 2

            # Calculate the std dev of the movements
            price_volatility = np.std(recent_window["close"]) / np.mean(
                recent_window["close"]
            )

            df_new.loc[new_row.name, "two_way_action"] = has_two_way_action
            df_new.loc[new_row.name, "price_volatility"] = price_volatility

    # 7. Check for volatility squeeze (Bollinger Bands and Keltner Channels convergence)
    if include_additional and bb_period > 0:
        if len(df_window) >= bb_period:
            # Calculate rolling standard deviation
            rolling_std = df_window["close"].rolling(window=bb_period).std().iloc[-1]

            # Calculate Bollinger Bands
            middle_band = df_window["close"].rolling(window=bb_period).mean().iloc[-1]
            upper_band = middle_band + (rolling_std * 2)
            lower_band = middle_band - (rolling_std * 2)

            # Store BB values
            df_new.loc[new_row.name, "bb_middle"] = middle_band
            df_new.loc[new_row.name, "bb_upper"] = upper_band
            df_new.loc[new_row.name, "bb_lower"] = lower_band

            # Calculate Keltner Channels if ATR is available
            if "atr" in df_new.columns and not pd.isna(df_new.loc[new_row.name, "atr"]):
                kc_middle = middle_band  # Same as BB middle
                kc_upper = kc_middle + (df_new.loc[new_row.name, "atr"] * 1.5)
                kc_lower = kc_middle - (df_new.loc[new_row.name, "atr"] * 1.5)

                # Store KC values
                df_new.loc[new_row.name, "kc_middle"] = kc_middle
                df_new.loc[new_row.name, "kc_upper"] = kc_upper
                df_new.loc[new_row.name, "kc_lower"] = kc_lower

                # Check for squeeze (BB inside KC)
                is_squeeze = upper_band < kc_upper and lower_band > kc_lower
                df_new.loc[new_row.name, "squeeze"] = is_squeeze

    return df_new


def calculate_wicks_ratio(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Mumların fitil/gövde oranını hesaplar - bu, kripto piyasasındaki
    manipülatif hareketleri filtrelemede yardımcı olur

    Parameters
    ----------
    df : pd.DataFrame
        Fiyat DataFrame'i
    period : int
        Hesaplama periyodu

    Returns
    -------
    pd.DataFrame
        Fitil/gövde oranı eklenmiş DataFrame
    """
    df_result = df.copy()

    # Mum gövdesi boyutu
    df_result["body_size"] = abs(df_result["close"] - df_result["open"])

    # Üst fitil boyutu
    df_result["upper_wick"] = df_result.apply(
        lambda x: x["high"] - max(x["open"], x["close"]), axis=1
    )

    # Alt fitil boyutu
    df_result["lower_wick"] = df_result.apply(
        lambda x: min(x["open"], x["close"]) - x["low"], axis=1
    )

    # Toplam fitil boyutu
    df_result["total_wick"] = df_result["upper_wick"] + df_result["lower_wick"]

    # Fitil/gövde oranı (0'a bölünme durumlarını ele al)
    df_result["wick_body_ratio"] = df_result.apply(
        lambda x: x["total_wick"] / x["body_size"] if x["body_size"] > 0 else 0, axis=1
    )

    # Ortalama oran
    df_result["avg_wick_body_ratio"] = (
        df_result["wick_body_ratio"].rolling(window=period).mean()
    )

    return df_result


def calculate_volatility_ratio(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Volatilite rasyosunu hesaplar - kısa vadeli volatilitenin
    uzun vadeli volatiliteye oranı

    Parameters
    ----------
    df : pd.DataFrame
        Fiyat DataFrame'i
    period : int
        Hesaplama periyodu

    Returns
    -------
    pd.DataFrame
        Volatilite rasyosu eklenmiş DataFrame
    """
    df_result = df.copy()

    # Kısa vadeli volatilite (5 gün)
    df_result["short_vol"] = df_result["close"].pct_change().rolling(window=5).std()

    # Uzun vadeli volatilite (n gün)
    df_result["long_vol"] = df_result["close"].pct_change().rolling(window=period).std()

    # Volatilite rasyosu (0'a bölünme durumlarını ele al)
    df_result["vol_ratio"] = df_result.apply(
        lambda x: x["short_vol"] / x["long_vol"] if x["long_vol"] > 0 else 1, axis=1
    )

    return df_result


def check_two_way_price_action(
    df: pd.DataFrame,
    lookback: int = 5,
    threshold_pct: float = 1.5,
    min_moves_each_direction: int = 2,
    volatility_threshold: float = 0.015,
) -> bool:
    """
    Check if there's significant two-way price action in recent bars
    This helps detect choppy/ranging markets and avoid false signals

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with 'close' column
    lookback : int
        Number of bars to look back
    threshold_pct : float
        Minimum percentage move to count as significant (1.5 = 1.5%)
    min_moves_each_direction : int
        Minimum number of moves required in each direction
    volatility_threshold : float
        Minimum volatility ratio required (stdev/mean)

    Returns
    -------
    bool
        True if two-way price action is detected, False otherwise
    """
    if len(df) < lookback + 1:
        return False

    # Get recent price data
    recent_data = df.iloc[-lookback - 1 :].copy()

    # Calculate percent changes
    recent_data["pct_change"] = recent_data["close"].pct_change() * 100

    # Count significant moves in each direction
    up_moves = sum(1 for x in recent_data["pct_change"].iloc[1:] if x > threshold_pct)
    down_moves = sum(
        1 for x in recent_data["pct_change"].iloc[1:] if x < -threshold_pct
    )

    # Calculate volatility ratio (standard deviation / mean of close prices)
    prices = recent_data["close"].values
    volatility_ratio = np.std(prices) / np.mean(prices)

    # Calculate additional metrics for advanced detection

    # 1. Calculate high-low range compared to overall move
    high = recent_data["high"].max()
    low = recent_data["low"].min()
    first_close = recent_data["close"].iloc[0]
    last_close = recent_data["close"].iloc[-1]

    total_range = high - low
    net_move = abs(last_close - first_close)

    # High range compared to net move indicates two-way action
    range_to_move_ratio = total_range / net_move if net_move > 0 else float("inf")

    # 2. Check for reversal patterns
    # Look for directional changes
    directions = np.sign(recent_data["pct_change"].iloc[1:].values)
    direction_changes = sum(
        1 for i in range(1, len(directions)) if directions[i] != directions[i - 1]
    )

    # 3. Check for wick patterns indicating rejection
    recent_data["upper_wick"] = recent_data["high"] - np.maximum(
        recent_data["open"], recent_data["close"]
    )
    recent_data["lower_wick"] = (
        np.minimum(recent_data["open"], recent_data["close"]) - recent_data["low"]
    )
    recent_data["body"] = abs(recent_data["close"] - recent_data["open"])

    # Calculate wick to body ratios
    with np.errstate(divide="ignore", invalid="ignore"):
        recent_data["upper_wick_ratio"] = (
            recent_data["upper_wick"] / recent_data["body"]
        )
        recent_data["lower_wick_ratio"] = (
            recent_data["lower_wick"] / recent_data["body"]
        )

    # Replace infinite values with large number
    recent_data.replace([np.inf, -np.inf], 10, inplace=True)
    recent_data.fillna(0, inplace=True)

    # Count significant wicks (wicks larger than body)
    significant_upper_wicks = sum(
        1 for x in recent_data["upper_wick_ratio"].iloc[1:] if x > 1
    )
    significant_lower_wicks = sum(
        1 for x in recent_data["lower_wick_ratio"].iloc[1:] if x > 1
    )

    # Combine all factors for final determination
    has_two_way_action = (
        # Must have minimum moves in each direction
        up_moves >= min_moves_each_direction
        and down_moves >= min_moves_each_direction
        and
        # Must have sufficient volatility
        volatility_ratio > volatility_threshold
        and
        # Must have significant range compared to net move
        range_to_move_ratio > 2.0
        and
        # Must have enough direction changes
        direction_changes >= min_moves_each_direction
    )

    # Log detailed analysis if enabled
    logger = logging.getLogger("turtle_trading_bot")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Two-way price action analysis:")
        logger.debug(f"  Up moves: {up_moves}, Down moves: {down_moves}")
        logger.debug(f"  Volatility ratio: {volatility_ratio:.4f}")
        logger.debug(f"  Range/Move ratio: {range_to_move_ratio:.2f}")
        logger.debug(f"  Direction changes: {direction_changes}")
        logger.debug(f"  Significant upper wicks: {significant_upper_wicks}")
        logger.debug(f"  Significant lower wicks: {significant_lower_wicks}")
        logger.debug(f"  Two-way action detected: {has_two_way_action}")

    return has_two_way_action


def calculate_market_regime(
    df: pd.DataFrame,
    atr_period: int = 14,
    ma_period: int = 50,
    adx_period: int = 14,
    adx_threshold: int = 25,
    volatility_lookback: int = 100,
    squeeze_threshold: float = 0.5,
) -> str:
    """
    Determine the current market regime: trending, ranging, or volatile

    Parameters
    ----------
    df : pd.DataFrame
        Price DataFrame with OHLC data
    atr_period : int
        ATR calculation period
    ma_period : int
        Moving average period for trend detection
    adx_period : int
        ADX calculation period
    adx_threshold : int
        ADX threshold to consider trend
    volatility_lookback : int
        Lookback period for volatility comparison
    squeeze_threshold : float
        Threshold for detecting volatility squeeze

    Returns
    -------
    str
        Market regime: 'trending', 'ranging', or 'volatile'
    """
    if len(df) < max(atr_period, ma_period, adx_period, volatility_lookback) + 10:
        return "unknown"  # Not enough data

    # Calculate needed indicators if not already present
    if "atr" not in df.columns:
        df = calculate_atr(df, atr_period)

    # Ensure we have ATR values
    if pd.isna(df["atr"].iloc[-1]):
        return "unknown"

    # 1. Trend strength - ADX
    adx_value = 0
    if "adx" not in df.columns:
        # Use our existing function or calculate ADX
        temp_df = calculate_adx(df, adx_period)
        if "adx" in temp_df.columns and not pd.isna(temp_df["adx"].iloc[-1]):
            adx_value = temp_df["adx"].iloc[-1]
    else:
        adx_value = df["adx"].iloc[-1]

    # 2. Trend direction - Moving Average
    if "ma" not in df.columns:
        df["ma"] = df["close"].rolling(window=ma_period).mean()

    # Calculate price relative to MA
    current_price = df["close"].iloc[-1]
    current_ma = df["ma"].iloc[-1]
    price_to_ma_ratio = current_price / current_ma if current_ma > 0 else 1

    # 3. Volatility analysis
    # Current vs historical volatility
    current_atr = df["atr"].iloc[-1]
    historical_atr = df["atr"].iloc[-volatility_lookback:-1].mean()
    volatility_ratio = current_atr / historical_atr if historical_atr > 0 else 1

    # 4. Detect squeeze conditions
    is_squeeze = False
    if "squeeze" in df.columns:
        # Use pre-calculated squeeze indicator
        is_squeeze = df["squeeze"].iloc[-1]
    else:
        # Check for Bollinger Bands and Keltner Channel convergence
        if "bb_upper" in df.columns and "kc_upper" in df.columns:
            bb_width = df["bb_upper"].iloc[-1] - df["bb_lower"].iloc[-1]
            kc_width = df["kc_upper"].iloc[-1] - df["kc_lower"].iloc[-1]
            if kc_width > 0:
                bb_kc_ratio = bb_width / kc_width
                is_squeeze = bb_kc_ratio < squeeze_threshold

    # 5. Check for two-way price action
    has_two_way_action = check_two_way_price_action(df)

    # Combine factors to determine market regime
    if is_squeeze:
        # Volatility squeeze - potential breakout coming
        regime = "squeeze"
    elif adx_value > adx_threshold and not has_two_way_action:
        # Strong trend
        if price_to_ma_ratio > 1.01:
            regime = "trending_up"
        elif price_to_ma_ratio < 0.99:
            regime = "trending_down"
        else:
            regime = "trending_sideways"
    elif has_two_way_action or (adx_value < 20):
        # Ranging market
        regime = "ranging"
    elif volatility_ratio > 1.5:
        # Highly volatile market
        regime = "volatile"
    else:
        # Default - low volatility ranging
        regime = "low_volatility"

    return regime


def get_optimal_parameters_for_regime(regime: str) -> dict:
    """
    Get optimal strategy parameters for the detected market regime

    Parameters
    ----------
    regime : str
        Current market regime

    Returns
    -------
    dict
        Dictionary of optimal parameters for the current regime
    """
    # Default parameters - moderate settings
    default_params = {
        "dc_length_enter": 20,
        "dc_length_exit": 10,
        "atr_multiple_entry": 0.5,
        "atr_multiple_stop": 2.0,
        "risk_per_trade": 0.01,
        "use_pyramiding": False,
        "use_trailing_stop": True,
        "use_adx_filter": True,
        "use_macd_confirmation": False,
        "use_rsi_filter": False,
        "adx_threshold": 25,
    }

    # Adjust parameters based on regime
    if regime == "trending_up" or regime == "trending_down":
        # Strong trend - longer entries, wider stops
        return {
            "dc_length_enter": 55,
            "dc_length_exit": 20,
            "atr_multiple_entry": 0.5,
            "atr_multiple_stop": 3.0,
            "risk_per_trade": 0.015,  # Slightly higher risk
            "use_pyramiding": True,  # Enable pyramiding in trends
            "use_trailing_stop": True,
            "use_adx_filter": True,
            "use_macd_confirmation": True,
            "use_rsi_filter": False,  # RSI less useful in strong trends
            "adx_threshold": 25,
        }
    elif regime == "ranging":
        # Ranging market - tighter entries and stops
        return {
            "dc_length_enter": 10,
            "dc_length_exit": 5,
            "atr_multiple_entry": 0.3,
            "atr_multiple_stop": 1.5,
            "risk_per_trade": 0.008,  # Lower risk
            "use_pyramiding": False,  # No pyramiding in ranging markets
            "use_trailing_stop": False,
            "use_adx_filter": True,  # Filter out weak trends
            "use_macd_confirmation": True,
            "use_rsi_filter": True,  # RSI good for ranges
            "adx_threshold": 20,
        }
    elif regime == "volatile":
        # Volatile market - wider stop loss, but careful entries
        return {
            "dc_length_enter": 15,
            "dc_length_exit": 7,
            "atr_multiple_entry": 0.7,
            "atr_multiple_stop": 3.5,
            "risk_per_trade": 0.007,  # Lower risk due to volatility
            "use_pyramiding": False,
            "use_trailing_stop": True,
            "use_adx_filter": False,  # ADX less reliable in volatility
            "use_macd_confirmation": True,
            "use_rsi_filter": True,
            "adx_threshold": 30,
        }
    elif regime == "squeeze":
        # Volatility squeeze - prepare for breakout
        return {
            "dc_length_enter": 10,
            "dc_length_exit": 5,
            "atr_multiple_entry": 0.5,
            "atr_multiple_stop": 2.0,
            "risk_per_trade": 0.012,  # Moderate risk for breakout potential
            "use_pyramiding": True,  # Ready to add on confirmation
            "use_trailing_stop": True,
            "use_adx_filter": False,  # Don't use ADX during squeeze
            "use_macd_confirmation": True,
            "use_rsi_filter": False,
            "adx_threshold": 20,
        }
    elif regime == "low_volatility":
        # Low volatility - tighter parameters, smaller targets
        return {
            "dc_length_enter": 10,
            "dc_length_exit": 5,
            "atr_multiple_entry": 0.3,
            "atr_multiple_stop": 1.5,
            "risk_per_trade": 0.01,
            "use_pyramiding": False,
            "use_trailing_stop": False,
            "use_adx_filter": True,
            "use_macd_confirmation": True,
            "use_rsi_filter": True,
            "adx_threshold": 20,
        }

    # Default case
    return default_params
