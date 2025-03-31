#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import math
import json
import random
import logging
import decimal
from decimal import Decimal
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import pandas_ta as ta
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

# Load environment variables from .env file
load_dotenv()

# Set up decimal precision
decimal.getcontext().prec = 8

# ==============================================
# CONFIGURATION
# ==============================================

# API Configuration
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() in ("true", "1", "t")

# Trading Parameters
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv(
    "TIMEFRAME", "1h"
)  # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

# Donchian Channel Parameters
DC_LENGTH_ENTER = int(os.getenv("DC_LENGTH_ENTER", "20"))  # Entry channel length
DC_LENGTH_EXIT = int(os.getenv("DC_LENGTH_EXIT", "10"))  # Exit channel length

# ATR Parameters
ATR_LENGTH = int(os.getenv("ATR_LENGTH", "14"))
ATR_SMOOTHING = int(os.getenv("ATR_SMOOTHING", "2"))

# Risk Management
RISK_PER_TRADE = Decimal(os.getenv("RISK_PER_TRADE", "0.02"))  # 2% of account balance
STOP_LOSS_ATR_MULTIPLE = Decimal(os.getenv("STOP_LOSS_ATR_MULTIPLE", "2"))

# Assets
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")
BASE_ASSET = os.getenv("BASE_ASSET", "BTC")

# Symbol Information (will be populated from API)
price_precision = None
quantity_precision = None
min_qty = None
step_size = None
min_notional = None

# ==============================================
# POSITION STATE TRACKING
# ==============================================

# Global position state variables
position_active = False
entry_price = Decimal("0")
position_quantity = Decimal("0")
stop_loss_price = Decimal("0")
take_profit_price = Decimal("0")
position_side = ""  # 'BUY' or 'SELL'
entry_time = 0
entry_atr = Decimal("0")

# State file path
STATE_FILE = Path("config/bot_state.json")


def save_state():
    """Save the current bot state to a JSON file"""
    global position_active, entry_price, position_quantity, stop_loss_price, take_profit_price, position_side, entry_time, entry_atr

    state = {
        "position_active": position_active,
        "entry_price": str(entry_price),
        "position_quantity": str(position_quantity),
        "stop_loss_price": str(stop_loss_price),
        "take_profit_price": str(take_profit_price),
        "position_side": position_side,
        "entry_time": entry_time,
        "entry_atr": str(entry_atr),
        "symbol": SYMBOL,
        "last_update": int(time.time() * 1000),
    }

    try:
        # Create config directory if it doesn't exist
        STATE_FILE.parent.mkdir(exist_ok=True)

        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)

        logger.info(f"Bot state saved to {STATE_FILE}")
    except Exception as e:
        logger.error(f"Error saving bot state: {e}")


def load_state():
    """Load the bot state from the state file if it exists"""
    global position_active, entry_price, position_quantity, stop_loss_price, take_profit_price, position_side, entry_time, entry_atr

    if not STATE_FILE.exists():
        logger.info(f"No state file found at {STATE_FILE}. Starting with clean state.")
        return

    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)

        # Verify the state is for the same symbol
        if state.get("symbol") != SYMBOL:
            logger.warning(
                f"State file is for a different symbol ({state.get('symbol')}, current: {SYMBOL}). Not loading state."
            )
            return

        # Load state
        position_active = state.get("position_active", False)
        entry_price = Decimal(state.get("entry_price", "0"))
        position_quantity = Decimal(state.get("position_quantity", "0"))
        stop_loss_price = Decimal(state.get("stop_loss_price", "0"))
        take_profit_price = Decimal(state.get("take_profit_price", "0"))
        position_side = state.get("position_side", "")
        entry_time = state.get("entry_time", 0)
        entry_atr = Decimal(state.get("entry_atr", "0"))

        logger.info(f"Bot state loaded from {STATE_FILE}")
        log_position_state()
    except Exception as e:
        logger.error(f"Error loading bot state: {e}")


def reset_position_state():
    """Reset the position state variables"""
    global position_active, entry_price, position_quantity, stop_loss_price, take_profit_price, position_side, entry_time, entry_atr

    position_active = False
    entry_price = Decimal("0")
    position_quantity = Decimal("0")
    stop_loss_price = Decimal("0")
    take_profit_price = Decimal("0")
    position_side = ""
    entry_time = 0
    entry_atr = Decimal("0")

    logger.info("Position state reset")
    save_state()


def log_position_state():
    """Log the current position state"""
    if position_active:
        logger.info(f"Active {position_side} position:")
        logger.info(f"  Quantity: {position_quantity} {BASE_ASSET}")
        logger.info(f"  Entry Price: {format_price(entry_price)}")
        logger.info(f"  Stop Loss: {format_price(stop_loss_price)}")
        logger.info(f"  Take Profit: {format_price(take_profit_price)}")
        logger.info(f"  Entry ATR: {entry_atr}")

        # Calculate current P&L
        try:
            ticker = Client(API_KEY, API_SECRET, testnet=USE_TESTNET).get_ticker(
                symbol=SYMBOL
            )
            current_price = Decimal(ticker["lastPrice"])

            if position_side == "BUY":
                pnl = (current_price - entry_price) * position_quantity
                pnl_percent = (current_price / entry_price - Decimal("1")) * Decimal(
                    "100"
                )
            else:
                pnl = (entry_price - current_price) * position_quantity
                pnl_percent = (Decimal("1") - current_price / entry_price) * Decimal(
                    "100"
                )

            logger.info(f"  Current Price: {format_price(current_price)}")
            logger.info(
                f"  Unrealized P&L: {format_price(pnl)} {QUOTE_ASSET} ({pnl_percent:.2f}%)"
            )
        except Exception as e:
            logger.error(f"Error calculating current P&L: {e}")
    else:
        logger.info("No active position")


def update_position_state(order_result, side, atr_value=None):
    """
    Update the position state after an order execution

    Parameters:
    -----------
    order_result : dict
        Order execution result from execute_order function
    side : str
        Order side ('BUY' or 'SELL')
    atr_value : Decimal, optional
        ATR value at entry time
    """
    global position_active, entry_price, position_quantity, stop_loss_price, take_profit_price, position_side, entry_time, entry_atr

    if side == "BUY" and not position_active:
        # Opening a long position
        position_active = True
        position_side = "BUY"
        entry_price = Decimal(order_result["avgPrice"])
        position_quantity = Decimal(order_result["executedQty"])
        entry_time = order_result["transactTime"]

        # Calculate stop loss and take profit
        if atr_value is not None:
            entry_atr = Decimal(str(atr_value))
            stop_distance = STOP_LOSS_ATR_MULTIPLE * entry_atr
            stop_loss_price = entry_price - stop_distance
            take_profit_price = entry_price + (
                stop_distance * Decimal("2")
            )  # 2:1 reward-to-risk

            logger.info(
                f"Position opened: LONG {position_quantity} {BASE_ASSET} at {format_price(entry_price)}"
            )
            logger.info(
                f"Stop loss set at {format_price(stop_loss_price)} (ATR: {entry_atr})"
            )
            logger.info(f"Take profit set at {format_price(take_profit_price)}")
        else:
            logger.warning("ATR value not provided. Stop loss and take profit not set.")

        save_state()

    elif side == "SELL" and not position_active:
        # Opening a short position
        position_active = True
        position_side = "SELL"
        entry_price = Decimal(order_result["avgPrice"])
        position_quantity = Decimal(order_result["executedQty"])
        entry_time = order_result["transactTime"]

        # Calculate stop loss and take profit
        if atr_value is not None:
            entry_atr = Decimal(str(atr_value))
            stop_distance = STOP_LOSS_ATR_MULTIPLE * entry_atr
            stop_loss_price = entry_price + stop_distance
            take_profit_price = entry_price - (
                stop_distance * Decimal("2")
            )  # 2:1 reward-to-risk

            logger.info(
                f"Position opened: SHORT {position_quantity} {BASE_ASSET} at {format_price(entry_price)}"
            )
            logger.info(
                f"Stop loss set at {format_price(stop_loss_price)} (ATR: {entry_atr})"
            )
            logger.info(f"Take profit set at {format_price(take_profit_price)}")
        else:
            logger.warning("ATR value not provided. Stop loss and take profit not set.")

        save_state()

    elif side == "SELL" and position_active and position_side == "BUY":
        # Closing a long position

        # Calculate profit/loss
        exit_price = Decimal(order_result["avgPrice"])
        pnl = (exit_price - entry_price) * position_quantity
        pnl_percent = (exit_price / entry_price - Decimal("1")) * Decimal("100")

        logger.info(f"Position closed: LONG {position_quantity} {BASE_ASSET}")
        logger.info(
            f"Entry: {format_price(entry_price)}, Exit: {format_price(exit_price)}"
        )
        logger.info(f"P&L: {format_price(pnl)} {QUOTE_ASSET} ({pnl_percent:.2f}%)")

        reset_position_state()

    elif side == "BUY" and position_active and position_side == "SELL":
        # Closing a short position

        # Calculate profit/loss
        exit_price = Decimal(order_result["avgPrice"])
        pnl = (entry_price - exit_price) * position_quantity
        pnl_percent = (Decimal("1") - exit_price / entry_price) * Decimal("100")

        logger.info(f"Position closed: SHORT {position_quantity} {BASE_ASSET}")
        logger.info(
            f"Entry: {format_price(entry_price)}, Exit: {format_price(exit_price)}"
        )
        logger.info(f"P&L: {format_price(pnl)} {QUOTE_ASSET} ({pnl_percent:.2f}%)")

        reset_position_state()

    else:
        logger.warning(
            f"Unexpected order scenario: side={side}, position_active={position_active}, position_side={position_side}"
        )


# ==============================================
# LOGGING CONFIGURATION
# ==============================================
# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)


# Configure logging
def setup_logging():
    logger = logging.getLogger("turtle_trading_bot")
    logger.setLevel(logging.INFO)

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


logger = setup_logging()

# ==============================================
# BINANCE CLIENT INITIALIZATION
# ==============================================


def initialize_binance_client():
    if not API_KEY or not API_SECRET:
        logger.error(
            "API Key or Secret Key not found. Please set them in the environment variables."
        )
        raise ValueError("API Key or Secret Key not found")

    try:
        if USE_TESTNET:
            logger.info("Initializing Binance client in TESTNET mode")
            client = Client(API_KEY, API_SECRET, testnet=True)
        else:
            logger.info("Initializing Binance client in PRODUCTION mode")
            client = Client(API_KEY, API_SECRET)

        # Test connectivity
        client.ping()
        server_time = client.get_server_time()
        logger.info(f"Connected to Binance. Server time: {server_time}")

        # Verify account access
        account_info = client.get_account()
        logger.info(
            f"Account status: {account_info['accountType']}, canTrade: {account_info['canTrade']}"
        )

        return client
    except BinanceAPIException as e:
        logger.error(f"Binance API Exception: {e}")
        raise
    except Exception as e:
        logger.error(f"Error initializing Binance client: {e}")
        raise


# ==============================================
# SYMBOL INFORMATION FUNCTIONS
# ==============================================


def get_symbol_info(client, symbol):
    """Retrieve symbol trading information from Binance API"""
    try:
        symbol_info = client.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol {symbol} not found on Binance")
            raise ValueError(f"Symbol {symbol} not found on Binance")

        logger.info(f"Retrieved symbol information for {symbol}")
        return symbol_info
    except BinanceAPIException as e:
        logger.error(f"Binance API Exception when getting symbol info: {e}")
        raise
    except Exception as e:
        logger.error(f"Error getting symbol info: {e}")
        raise


def extract_symbol_filters(symbol_info):
    """Extract and process symbol filters from symbol_info"""
    global price_precision, quantity_precision, min_qty, step_size, min_notional

    try:
        # Extract filters
        price_filter = next(
            (f for f in symbol_info["filters"] if f["filterType"] == "PRICE_FILTER"),
            None,
        )
        lot_size_filter = next(
            (f for f in symbol_info["filters"] if f["filterType"] == "LOT_SIZE"), None
        )
        min_notional_filter = next(
            (
                f
                for f in symbol_info["filters"]
                if f["filterType"] in ["MIN_NOTIONAL", "NOTIONAL"]
            ),
            None,
        )

        if not price_filter or not lot_size_filter or not min_notional_filter:
            logger.error(f"Required filters not found for {symbol_info['symbol']}")
            raise ValueError(f"Required filters not found for {symbol_info['symbol']}")

        # Calculate precision from tick size (e.g., 0.00001 -> 5 decimal places)
        def get_precision_from_step(step_str):
            decimal_str = step_str.rstrip("0")
            if "." in decimal_str:
                return len(decimal_str) - decimal_str.index(".") - 1
            return 0

        # Process price filter
        price_precision = get_precision_from_step(price_filter["tickSize"])

        # Process lot size filter
        step_size = Decimal(lot_size_filter["stepSize"])
        min_qty = Decimal(lot_size_filter["minQty"])
        quantity_precision = get_precision_from_step(lot_size_filter["stepSize"])

        # Process min notional filter
        min_notional = Decimal(min_notional_filter["minNotional"])

        # Log filter information
        logger.info(f"Symbol Filters for {symbol_info['symbol']}:")
        logger.info(f"Price Precision: {price_precision}")
        logger.info(f"Quantity Precision: {quantity_precision}")
        logger.info(f"Minimum Quantity: {min_qty}")
        logger.info(f"Step Size: {step_size}")
        logger.info(f"Minimum Notional: {min_notional}")

        return {
            "price_precision": price_precision,
            "quantity_precision": quantity_precision,
            "min_qty": min_qty,
            "step_size": step_size,
            "min_notional": min_notional,
        }
    except Exception as e:
        logger.error(f"Error processing symbol filters: {e}")
        raise


# ==============================================
# TECHNICAL INDICATORS
# ==============================================


def calculate_indicators(df, dc_enter, dc_exit, atr_len, atr_smooth):
    """
    Calculate Donchian Channels and ATR indicators on a DataFrame

    Parameters:
    -----------
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

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added indicator columns, NaN rows removed
    """
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


# ==============================================
# DATA FETCHING FUNCTIONS
# ==============================================


def fetch_data(client, symbol, interval, lookback):
    """
    Fetch historical klines (candlestick) data from Binance

    Parameters:
    -----------
    client : binance.client.Client
        Initialized Binance client
    symbol : str
        Trading pair symbol (e.g., 'BTCUSDT')
    interval : str
        Kline interval (e.g., '1h', '4h', '1d')
    lookback : int
        Number of candles to fetch

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame with columns: timestamp, open, high, low, close, volume
        None if an error occurs
    """
    try:
        # Add extra candles for calculations (e.g., for indicators that need more data)
        extra_candles = 50
        total_candles = lookback + extra_candles

        logger.info(f"Fetching {total_candles} {interval} candles for {symbol}")

        # Calculate start time based on interval and lookback
        # Convert interval to milliseconds
        interval_ms = {
            "1m": 60 * 1000,
            "3m": 3 * 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "2h": 2 * 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "6h": 6 * 60 * 60 * 1000,
            "8h": 8 * 60 * 60 * 1000,
            "12h": 12 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
            "3d": 3 * 24 * 60 * 60 * 1000,
            "1w": 7 * 24 * 60 * 60 * 1000,
            "1M": 30 * 24 * 60 * 60 * 1000,
        }

        # Fetch klines from Binance
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            limit=1000,  # Maximum allowed by Binance
            start_str=f"{total_candles} {interval} ago UTC",
        )

        if not klines:
            logger.warning(f"No data returned for {symbol} {interval}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(
            klines,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )

        # Keep only the needed columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        # Set timestamp as index
        df.set_index("timestamp", inplace=True)

        # Remove the last candle (it might be incomplete)
        df = df.iloc[:-1]

        # Make sure we have enough data
        if len(df) < lookback:
            logger.warning(
                f"Insufficient data: got {len(df)} candles, wanted {lookback}"
            )
            return df

        # Keep only the most recent data up to lookback
        df = df.iloc[-lookback:]

        logger.info(f"Successfully fetched {len(df)} {interval} candles for {symbol}")
        return df

    except BinanceAPIException as e:
        logger.error(f"Binance API error while fetching data: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return None


# ==============================================
# UTILITY FUNCTIONS
# ==============================================


# Function to round quantity according to the symbol's quantity step size
def round_step_size(quantity, step_size):
    """Round quantity to the nearest step size"""
    if step_size == 0:
        return quantity

    precision = int(round(-math.log10(float(step_size))))
    return float(
        Decimal(int(quantity * Decimal(10**precision)) / Decimal(10**precision))
    )


# Format price according to the symbol's price precision
def format_price(price, precision=None):
    """Format price with the correct number of decimal places"""
    global price_precision
    if precision is None:
        precision = price_precision

    format_str = f"{{:.{precision}f}}"
    return format_str.format(float(price))


# Format quantity according to the symbol's quantity precision
def format_quantity(quantity, precision=None):
    """Format quantity with the correct number of decimal places"""
    global quantity_precision
    if precision is None:
        precision = quantity_precision

    format_str = f"{{:.{precision}f}}"
    return format_str.format(float(quantity))


# ==============================================
# RISK MANAGEMENT AND POSITION SIZING
# ==============================================


def calculate_trade_size(
    balance,
    risk_percent,
    atr_value,
    current_price,
    min_qty,
    step_size,
    min_notional,
    price_precision,
):
    """
    Calculate position size based on risk management rules and exchange limitations

    Parameters:
    -----------
    balance : Decimal
        Available balance in quote asset
    risk_percent : Decimal
        Risk percentage per trade (e.g., 0.02 for 2%)
    atr_value : float
        Current ATR value
    current_price : Decimal
        Current market price
    min_qty : Decimal
        Minimum allowed quantity
    step_size : Decimal
        Quantity step size
    min_notional : Decimal
        Minimum notional value (quantity * price)
    price_precision : int
        Price precision for the symbol

    Returns:
    --------
    Decimal
        Position size in base asset or 0.0 if constraints can't be met
    """
    try:
        # Convert to Decimal for precision calculations
        atr_value = Decimal(str(atr_value))
        current_price = Decimal(str(current_price))

        # 1. Calculate the amount willing to risk per trade
        risk_amount = balance * risk_percent
        logger.info(
            f"Risk amount: {risk_amount} {QUOTE_ASSET} ({risk_percent*100}% of {balance})"
        )

        # 2. Calculate stop loss distance in quote asset terms
        stop_distance = STOP_LOSS_ATR_MULTIPLE * atr_value
        stop_distance_quote = stop_distance
        logger.info(
            f"Stop distance: {stop_distance} ({STOP_LOSS_ATR_MULTIPLE} * ATR {atr_value})"
        )

        # 3. Calculate position size based on risk amount and stop distance
        if stop_distance_quote == Decimal("0"):
            logger.error("Stop distance is zero. Cannot calculate position size.")
            return Decimal("0")

        position_size = risk_amount / stop_distance_quote
        logger.info(f"Initial position size calculation: {position_size} {BASE_ASSET}")

        # 4. Adjust for step size restrictions
        adjusted_position_size = round_step_size(position_size, step_size)
        logger.info(
            f"Position size adjusted for step size: {adjusted_position_size} {BASE_ASSET}"
        )

        # 5. Check against minimum quantity requirement
        if adjusted_position_size < min_qty:
            logger.warning(
                f"Calculated position size {adjusted_position_size} is below minimum quantity {min_qty}"
            )

            # Try to use minimum quantity instead
            adjusted_position_size = min_qty
            logger.info(
                f"Adjusted to minimum quantity: {adjusted_position_size} {BASE_ASSET}"
            )

        # 6. Check against minimum notional value
        notional_value = adjusted_position_size * current_price

        if notional_value < min_notional:
            logger.warning(
                f"Notional value {notional_value} is below minimum {min_notional}"
            )

            # Try to adjust position size to meet minimum notional
            required_position_size = min_notional / current_price
            # Round up to next step size
            adjusted_position_size = Decimal(
                round_step_size(required_position_size, step_size)
            )

            # Re-check step size compliance
            if adjusted_position_size % step_size != Decimal("0"):
                # Ensure it's a multiple of step_size by rounding up
                steps = (adjusted_position_size / step_size).quantize(
                    Decimal("1"), rounding="ROUND_UP"
                )
                adjusted_position_size = steps * step_size

            logger.info(
                f"Position size adjusted for min notional: {adjusted_position_size} {BASE_ASSET}"
            )
            notional_value = adjusted_position_size * current_price
            logger.info(f"New notional value: {notional_value} {QUOTE_ASSET}")

        # 7. Check if there's enough balance for the position
        required_balance = adjusted_position_size * current_price

        if required_balance > balance:
            logger.warning(
                f"Required balance {required_balance} exceeds available balance {balance}"
            )

            # Try to adjust position size to available balance
            max_affordable_size = (
                balance * Decimal("0.99")
            ) / current_price  # 99% of balance for fees
            adjusted_position_size = Decimal(
                round_step_size(max_affordable_size, step_size)
            )

            # Final check against minimum requirements
            if adjusted_position_size < min_qty:
                logger.error(
                    f"Cannot meet minimum quantity requirement with available balance"
                )
                return Decimal("0")

            notional_value = adjusted_position_size * current_price
            if notional_value < min_notional:
                logger.error(
                    f"Cannot meet minimum notional requirement with available balance"
                )
                return Decimal("0")

            logger.info(
                f"Position size adjusted for available balance: {adjusted_position_size} {BASE_ASSET}"
            )

        # 8. Final check for all requirements
        if (
            adjusted_position_size >= min_qty
            and adjusted_position_size * current_price >= min_notional
            and adjusted_position_size * current_price <= balance
        ):

            # Format to correct precision
            formatted_size = format_quantity(adjusted_position_size)
            final_size = Decimal(formatted_size)

            logger.info(f"Final position size: {final_size} {BASE_ASSET}")
            logger.info(f"Estimated cost: {final_size * current_price} {QUOTE_ASSET}")
            logger.info(
                f"Risk per trade: {risk_amount} {QUOTE_ASSET} ({risk_percent*100}%)"
            )

            return final_size
        else:
            logger.error(
                "Failed to calculate valid position size meeting all requirements"
            )
            return Decimal("0")

    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return Decimal("0")


# ==============================================
# TRADING STRATEGY CORE LOGIC
# ==============================================


def check_and_execute_trading_logic(client, df_with_indicators):
    """
    Core trading logic that checks indicators and position state to make trading decisions

    Parameters:
    -----------
    client : binance.client.Client
        Initialized Binance client
    df_with_indicators : pandas.DataFrame
        DataFrame with price data and calculated indicators
    """
    global position_active, position_side, position_quantity, stop_loss_price

    logger.info("Checking trading conditions...")

    try:
        # Get latest indicator values
        latest_row = df_with_indicators.iloc[-1]
        prev_row = df_with_indicators.iloc[-2] if len(df_with_indicators) > 1 else None

        # Get current market price
        ticker = client.get_ticker(symbol=SYMBOL)
        current_price = Decimal(ticker["lastPrice"])

        logger.info(f"Current market price: {format_price(current_price)}")
        logger.info(f"Latest indicators (timestamp {latest_row.name}):")
        logger.info(f"  DC Upper Entry: {latest_row['dc_upper_entry']}")
        logger.info(f"  DC Lower Exit: {latest_row['dc_lower_exit']}")
        logger.info(f"  ATR: {latest_row['atr']}")

        # Check current position state
        if position_active:
            logger.info("Currently in active position. Checking exit conditions...")

            if position_side == "BUY":  # We're in a LONG position
                # Check stop loss
                if current_price <= stop_loss_price:
                    logger.info(
                        f"STOP LOSS TRIGGERED: Current price {format_price(current_price)} <= Stop loss {format_price(stop_loss_price)}"
                    )

                    # Execute sell order
                    success, order_result = execute_order(
                        client=client,
                        symbol=SYMBOL,
                        side="SELL",
                        quantity=position_quantity,
                        price_precision=price_precision,
                        quantity_precision=quantity_precision,
                    )

                    if success:
                        logger.info("Successfully closed LONG position with STOP LOSS")
                    else:
                        logger.error(
                            f"Failed to execute stop loss order: {order_result}"
                        )

                # Check Donchian channel exit (price breaks below lower band)
                elif latest_row["close"] < latest_row["dc_lower_exit"]:
                    logger.info(
                        f"EXIT SIGNAL: Close price {latest_row['close']} broke below Donchian lower band {latest_row['dc_lower_exit']}"
                    )

                    # Execute sell order
                    success, order_result = execute_order(
                        client=client,
                        symbol=SYMBOL,
                        side="SELL",
                        quantity=position_quantity,
                        price_precision=price_precision,
                        quantity_precision=quantity_precision,
                    )

                    if success:
                        logger.info(
                            "Successfully closed LONG position with Donchian exit signal"
                        )
                    else:
                        logger.error(
                            f"Failed to execute Donchian exit order: {order_result}"
                        )

                else:
                    logger.info("No exit conditions met, maintaining LONG position")

            elif (
                position_side == "SELL"
            ):  # We're in a SHORT position - for future implementation
                logger.info("SHORT position handling not implemented yet")
                # Similar logic as above but for short positions (inverse conditions)
                pass

        else:
            logger.info("No active position. Checking entry conditions...")

            # Check entry signal - Breakout above Donchian upper band
            if latest_row["close"] > latest_row["dc_upper_entry"]:
                logger.info(
                    f"ENTRY SIGNAL: Close price {latest_row['close']} broke above Donchian upper band {latest_row['dc_upper_entry']}"
                )

                # Calculate position size
                account = client.get_account()
                quote_balance = Decimal("0")
                for asset in account["balances"]:
                    if asset["asset"] == QUOTE_ASSET:
                        quote_balance = Decimal(asset["free"])
                        break

                logger.info(f"Available {QUOTE_ASSET} balance: {quote_balance}")

                # Calculate quantity based on risk management
                position_size = calculate_trade_size(
                    balance=quote_balance,
                    risk_percent=RISK_PER_TRADE,
                    atr_value=latest_row["atr"],
                    current_price=current_price,
                    min_qty=min_qty,
                    step_size=step_size,
                    min_notional=min_notional,
                    price_precision=price_precision,
                )

                if position_size > Decimal("0"):
                    logger.info(
                        f"Calculated position size: {position_size} {BASE_ASSET}"
                    )

                    # Execute buy order
                    success, order_result = execute_order(
                        client=client,
                        symbol=SYMBOL,
                        side="BUY",
                        quantity=position_size,
                        price_precision=price_precision,
                        quantity_precision=quantity_precision,
                        atr_value=latest_row["atr"],
                    )

                    if success:
                        logger.info("Successfully opened LONG position")
                    else:
                        logger.error(f"Failed to execute entry order: {order_result}")
                else:
                    logger.warning(
                        "Entry signal detected but position size calculation returned zero. No order placed."
                    )
            else:
                logger.info("No entry conditions met")

    except Exception as e:
        logger.error(f"Error in trading logic: {e}")
        logger.error(f"Exception details: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


# ==============================================
# ORDER EXECUTION
# ==============================================


def execute_order(
    client,
    symbol,
    side,
    quantity,
    price_precision,
    quantity_precision,
    simulate=False,
    atr_value=None,
):
    """
    Execute a market order on Binance or simulate order execution

    Parameters:
    -----------
    client : binance.client.Client
        Initialized Binance client
    symbol : str
        Trading pair symbol (e.g., 'BTCUSDT')
    side : str
        Order side ('BUY' or 'SELL')
    quantity : Decimal
        Order quantity in base asset
    price_precision : int
        Price precision for the symbol
    quantity_precision : int
        Quantity precision for the symbol
    simulate : bool, optional
        Whether to simulate the order instead of actually placing it
    atr_value : float, optional
        Current ATR value for position state tracking

    Returns:
    --------
    tuple
        (success_bool, order_details_dict)
    """
    try:
        # Format quantity to correct precision
        formatted_quantity = format_quantity(quantity, quantity_precision)

        # Log order details
        logger.info(
            f"Preparing to execute {side} order for {formatted_quantity} {symbol}"
        )

        # Get current price for simulation or logging
        ticker = client.get_ticker(symbol=symbol)
        current_price = Decimal(ticker["lastPrice"])
        formatted_price = format_price(current_price, price_precision)

        # Simulate order or execute real order based on settings
        if simulate or USE_TESTNET:
            # Simulate order execution
            execution_type = "SIMULATED" if simulate else "TESTNET"
            logger.info(
                f"Executing {execution_type} {side} MARKET order for {formatted_quantity} {symbol} at ~{formatted_price}"
            )

            # Build a simulated order response that mimics Binance's response format
            order_time = int(time.time() * 1000)  # Current time in milliseconds
            order_id = f"simulated_{int(order_time)}_{side}_{symbol}".lower()

            # Calculate simulated execution values
            # Add small slippage for realism (0.1%)
            slippage_factor = Decimal("1.001") if side == "BUY" else Decimal("0.999")
            execution_price = current_price * slippage_factor
            formatted_execution_price = format_price(execution_price, price_precision)

            # Calculate commission (simulate 0.1% fee)
            commission_rate = Decimal("0.001")
            commission_asset = (
                symbol[-4:] if len(symbol) >= 4 else "USDT"
            )  # Use quote asset for commission
            commission_amount = (
                Decimal(formatted_quantity) * execution_price * commission_rate
            )

            # Simulate fills data
            fills = [
                {
                    "price": formatted_execution_price,
                    "qty": formatted_quantity,
                    "commission": str(commission_amount),
                    "commissionAsset": commission_asset,
                    "tradeId": int(order_time),
                }
            ]

            # Build complete simulated response
            order_result = {
                "symbol": symbol,
                "orderId": order_id,
                "clientOrderId": f"simulated_{order_time}",
                "transactTime": order_time,
                "price": "0.00000000",  # Market orders don't have a set price
                "origQty": formatted_quantity,
                "executedQty": formatted_quantity,
                "cummulativeQuoteQty": str(
                    Decimal(formatted_quantity) * execution_price
                ),
                "status": "FILLED",
                "timeInForce": "GTC",
                "type": "MARKET",
                "side": side,
                "fills": fills,
            }

            logger.info(f"{execution_type} order successfully 'executed'")

        else:
            # Execute actual order on Binance
            logger.info(
                f"Executing REAL {side} MARKET order for {formatted_quantity} {symbol}"
            )

            # Build actual order parameters
            order_params = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": formatted_quantity,
            }

            # Send order to Binance
            order_result = client.create_order(**order_params)
            logger.info(f"Real order sent to Binance, response received")

        # Process order result (same for both real and simulated)
        if order_result["status"] == "FILLED":
            # Calculate average fill price
            total_cost = Decimal("0")
            total_qty = Decimal("0")

            for fill in order_result["fills"]:
                fill_price = Decimal(fill["price"])
                fill_qty = Decimal(fill["qty"])
                fill_cost = fill_price * fill_qty

                total_cost += fill_cost
                total_qty += fill_qty

            avg_price = total_cost / total_qty if total_qty > 0 else Decimal("0")

            # Log success details
            logger.info(f"Order {order_result['orderId']} FILLED successfully:")
            logger.info(f"  Symbol: {order_result['symbol']}")
            logger.info(f"  Side: {order_result['side']}")
            logger.info(f"  Type: {order_result['type']}")
            logger.info(f"  Quantity: {order_result['executedQty']}")
            logger.info(
                f"  Average Fill Price: {format_price(avg_price, price_precision)}"
            )
            logger.info(f"  Total Cost: {format_price(total_cost, price_precision)}")

            # Add derived data to the result
            order_result["avgPrice"] = str(avg_price)
            order_result["totalCost"] = str(total_cost)

            # Update position state
            update_position_state(order_result, side, atr_value)

            return True, order_result
        else:
            # Order was created but not filled
            logger.warning(
                f"Order {order_result['orderId']} created but status is {order_result['status']}"
            )
            return False, order_result

    except BinanceAPIException as e:
        logger.error(f"Binance API Exception during order execution: {e}")
        return False, {"error": str(e), "code": getattr(e, "code", None)}

    except BinanceOrderException as e:
        logger.error(f"Binance Order Exception: {e}")
        return False, {"error": str(e), "code": getattr(e, "code", None)}

    except Exception as e:
        logger.error(f"Unexpected error executing order: {e}")
        return False, {"error": str(e)}


# ==============================================
# MAIN BOT LOOP
# ==============================================


def get_sleep_time(timeframe):
    """
    Calculate the time to sleep until the next candle close

    Parameters:
    -----------
    timeframe : str
        The timeframe of the candlesticks (e.g., '1h', '4h', '1d')

    Returns:
    --------
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


def run_bot():
    """
    Main function to run the trading bot in a continuous loop
    """
    logger.info("==============================================")
    logger.info("         TURTLE TRADING BOT STARTED          ")
    logger.info("==============================================")
    logger.info(f"Trading Configuration: Symbol={SYMBOL}, Timeframe={TIMEFRAME}")
    logger.info(
        f"Strategy Parameters: DC_Enter={DC_LENGTH_ENTER}, DC_Exit={DC_LENGTH_EXIT}, ATR_Length={ATR_LENGTH}"
    )
    logger.info(
        f"Risk Management: Risk_Per_Trade={RISK_PER_TRADE}, SL_ATR_Multiple={STOP_LOSS_ATR_MULTIPLE}"
    )

    try:
        # Initialize Binance client
        client = initialize_binance_client()

        # Get symbol information and trading rules
        symbol_info = get_symbol_info(client, SYMBOL)
        symbol_filters = extract_symbol_filters(symbol_info)

        # Load bot state from file
        load_state()

        # Check if we have an active position and log it
        if position_active:
            logger.info("Bot started with active position:")
            log_position_state()

            # Check if we need to update stop loss and take profit
            if entry_atr == Decimal("0") or stop_loss_price == Decimal("0"):
                logger.warning("Active position has no stop loss. Recalculating...")
                # Fetch latest data and calculate ATR to set stop loss
                lookback = ATR_LENGTH + 10
                df = fetch_data(client, SYMBOL, TIMEFRAME, lookback)
                if df is not None and not df.empty:
                    df_with_indicators = calculate_indicators(
                        df=df,
                        dc_enter=DC_LENGTH_ENTER,
                        dc_exit=DC_LENGTH_EXIT,
                        atr_len=ATR_LENGTH,
                        atr_smooth="RMA",
                    )
                    if df_with_indicators is not None and not df_with_indicators.empty:
                        # Update ATR and stop loss
                        latest_atr = df_with_indicators["atr"].iloc[-1]
                        entry_atr = Decimal(str(latest_atr))
                        stop_distance = STOP_LOSS_ATR_MULTIPLE * entry_atr

                        if position_side == "BUY":
                            stop_loss_price = entry_price - stop_distance
                            take_profit_price = entry_price + (
                                stop_distance * Decimal("2")
                            )
                        else:
                            stop_loss_price = entry_price + stop_distance
                            take_profit_price = entry_price - (
                                stop_distance * Decimal("2")
                            )

                        logger.info(
                            f"Updated stop loss to {format_price(stop_loss_price)}"
                        )
                        logger.info(
                            f"Updated take profit to {format_price(take_profit_price)}"
                        )
                        save_state()

        # Main bot loop
        logger.info("Starting main trading loop...")

        while True:
            try:
                loop_start_time = time.time()

                # Fetch latest data
                lookback = max(DC_LENGTH_ENTER, DC_LENGTH_EXIT, ATR_LENGTH) + 50
                df = fetch_data(client, SYMBOL, TIMEFRAME, lookback)

                if df is None or df.empty:
                    logger.error(
                        "Unable to fetch historical data. Skipping this iteration."
                    )
                    time.sleep(60)  # Wait 60 seconds before retrying
                    continue

                # Calculate indicators
                df_with_indicators = calculate_indicators(
                    df=df,
                    dc_enter=DC_LENGTH_ENTER,
                    dc_exit=DC_LENGTH_EXIT,
                    atr_len=ATR_LENGTH,
                    atr_smooth="RMA",
                )

                if df_with_indicators is None or df_with_indicators.empty:
                    logger.error(
                        "Failed to calculate indicators. Skipping this iteration."
                    )
                    time.sleep(60)  # Wait 60 seconds before retrying
                    continue

                # Get current market price
                ticker = client.get_ticker(symbol=SYMBOL)
                current_price = Decimal(ticker["lastPrice"])

                logger.info(f"Current market price: {format_price(current_price)}")

                # Check trading conditions and execute orders if needed
                check_and_execute_trading_logic(client, df_with_indicators)

                # Log position state if active
                if position_active:
                    log_position_state()

                # Calculate time to sleep until next candle close
                sleep_seconds = get_sleep_time(TIMEFRAME)
                next_check_time = time.time() + sleep_seconds
                logger.info(
                    f"Next check at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(next_check_time))}"
                )
                logger.info("Waiting for next candle close...")

                # Sleep until next candle close
                time.sleep(sleep_seconds)

            except KeyboardInterrupt:
                logger.info("Bot stopped by user (Ctrl+C)")
                break

            except BinanceAPIException as e:
                logger.error(f"Binance API Exception: {e}")

                # Handle specific API errors
                if hasattr(e, "code"):
                    if e.code == -1021:  # RECONNECT
                        logger.error(
                            "Timestamp for this request was outside the recvWindow. Server time might be different. Syncing..."
                        )
                        client.ping()  # Ping to sync time
                    elif e.code == -1003:  # TOO_MANY_REQUESTS
                        logger.error(
                            "Rate limit exceeded. Waiting longer before next request."
                        )
                        time.sleep(300)  # Wait 5 minutes
                    else:
                        logger.error(
                            f"Unhandled API error code: {e.code}. Waiting before retry."
                        )
                        time.sleep(60)  # Wait 1 minute
                else:
                    # Generic API error handling
                    logger.error("Waiting before retry...")
                    time.sleep(60)  # Wait 1 minute

            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                logger.error(f"Error details: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())

                # Wait before retrying to avoid tight error loops
                logger.info("Waiting 2 minutes before retrying...")
                time.sleep(120)

    except KeyboardInterrupt:
        logger.info("Bot initialization interrupted by user")

    except Exception as e:
        logger.error(f"Fatal error during bot initialization: {e}")
        import traceback

        logger.error(traceback.format_exc())

    finally:
        logger.info("==============================================")
        logger.info("          TURTLE TRADING BOT STOPPED         ")
        logger.info("==============================================")


if __name__ == "__main__":
    run_bot()
