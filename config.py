"""
Configuration module for the Turtle Trading Bot
This loads configuration from .env file and makes it available for import
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Binance API credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# Trading parameters
USE_TESTNET = os.getenv("USE_TESTNET", "False").lower() in ("true", "1", "t")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("TIMEFRAME", "5m")

# Multi-timeframe Analysis
USE_MULTI_TIMEFRAME = os.getenv("USE_MULTI_TIMEFRAME", "True").lower() in (
    "true",
    "1",
    "t",
)
TREND_TIMEFRAME = os.getenv("TREND_TIMEFRAME", "15m")
ENTRY_TIMEFRAME = os.getenv("ENTRY_TIMEFRAME", "5m")
TREND_ALIGNMENT_REQUIRED = os.getenv("TREND_ALIGNMENT_REQUIRED", "False").lower() in (
    "true",
    "1",
    "t",
)

# Donchian Channel Parameters
DC_LENGTH_ENTER = int(os.getenv("DC_LENGTH_ENTER", "20"))
DC_LENGTH_EXIT = int(os.getenv("DC_LENGTH_EXIT", "10"))

# ATR Parameters
ATR_LENGTH = int(os.getenv("ATR_LENGTH", "14"))
ATR_SMOOTHING = os.getenv("ATR_SMOOTHING", "RMA")

# Risk Management
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.015"))
STOP_LOSS_ATR_MULTIPLE = float(os.getenv("STOP_LOSS_ATR_MULTIPLE", "1.2"))
MAX_RISK_PERCENTAGE = float(os.getenv("MAX_RISK_PERCENTAGE", "0.08"))

# Pyramiding
USE_PYRAMIDING = os.getenv("USE_PYRAMIDING", "True").lower() in ("true", "1", "t")
PYRAMID_MAX_ENTRIES = int(os.getenv("PYRAMID_MAX_ENTRIES", "3"))
PYRAMID_SIZE_FIRST = float(os.getenv("PYRAMID_SIZE_FIRST", "0.5"))
PYRAMID_SIZE_ADDITIONAL = float(os.getenv("PYRAMID_SIZE_ADDITIONAL", "0.25"))

# Exit Strategy
USE_TRAILING_STOP = os.getenv("USE_TRAILING_STOP", "True").lower() in ("true", "1", "t")
USE_PARTIAL_EXITS = os.getenv("USE_PARTIAL_EXITS", "True").lower() in ("true", "1", "t")
FIRST_TARGET_ATR = float(os.getenv("FIRST_TARGET_ATR", "2"))
SECOND_TARGET_ATR = float(os.getenv("SECOND_TARGET_ATR", "3.5"))
PROFIT_FOR_TRAILING = float(os.getenv("PROFIT_FOR_TRAILING", "0.1"))

# Filters
USE_ADX_FILTER = os.getenv("USE_ADX_FILTER", "True").lower() in ("true", "1", "t")
ADX_PERIOD = int(os.getenv("ADX_PERIOD", "14"))
ADX_THRESHOLD = int(os.getenv("ADX_THRESHOLD", "25"))
USE_MA_FILTER = os.getenv("USE_MA_FILTER", "True").lower() in ("true", "1", "t")
MA_PERIOD = int(os.getenv("MA_PERIOD", "200"))
MA_TYPE = os.getenv("MA_TYPE", "SMA")

# Leverage
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
MAX_LEVERAGE_TREND = float(os.getenv("MAX_LEVERAGE_TREND", "3"))
MAX_LEVERAGE_COUNTER = float(os.getenv("MAX_LEVERAGE_COUNTER", "1.5"))

# Assets
QUOTE_ASSET = os.getenv("QUOTE_ASSET", "USDT")
BASE_ASSET = os.getenv("BASE_ASSET", "BTC")

# Print configuration for debugging
if __name__ == "__main__":
    print("Turtle Trading Bot Configuration:")
    print(f"API KEY: {'*' * 8}{BINANCE_API_KEY[-4:] if BINANCE_API_KEY else 'Not set'}")
    print(
        f"API SECRET: {'*' * 12}{BINANCE_API_SECRET[-4:] if BINANCE_API_SECRET else 'Not set'}"
    )
    print(f"TESTNET: {USE_TESTNET}")
    print(f"SYMBOL: {SYMBOL}")
    print(f"TIMEFRAME: {TIMEFRAME}")
    print(f"RISK PER TRADE: {RISK_PER_TRADE*100}%")
