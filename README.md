# Turtle Trading Bot

A class-based implementation of the Turtle Trading strategy for algorithmic trading on the Binance exchange.

## Overview

This project implements the Turtle Trading system, a trend-following strategy developed by Richard Dennis and William Eckhardt. The bot trades automatically based on Donchian Channel breakouts and implements proper risk management with ATR-based stop losses.

## Features

- **Class-based Architecture**: Modular, maintainable, and testable design
- **Donchian Channel Breakout**: For trend identification and trade entries
- **Risk Management**: ATR-based position sizing and stop losses
- **Dual Direction Trading**: Support for both long and short positions
- **State Persistence**: Save and load bot state between restarts
- **Comprehensive Logging**: Detailed logs for monitoring and analysis

## Project Structure

```
turtle_trading_bot/
├── turtle_trading_bot.py    # Main entry point
├── .env                     # Configuration (API keys, trading parameters)
├── .env.example             # Example configuration template
├── requirements.txt         # Dependencies
├── logs/                    # Trading logs directory
├── config/                  # Bot state and configuration files
└── bot/                     # Core modules
    ├── __init__.py          # Package initialization
    ├── core.py              # TurtleTradingBot class
    ├── exchange.py          # Binance exchange operations
    ├── indicators.py        # Technical indicators and signal detection
    ├── models.py            # Data models and type definitions
    ├── risk.py              # Risk management and position sizing
    └── utils.py             # Utility functions
```

## Setup and Configuration

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and update with your Binance API credentials:
   ```
   cp .env.example .env
   ```
4. Edit parameters in `.env` as needed (risk, timeframe, symbol, etc.)

## Usage

Run the bot:

```
python turtle_trading_bot.py
```

## Configuration Parameters

Edit these in the `.env` file:

- **API_KEY/API_SECRET**: Your Binance API credentials
- **USE_TESTNET**: Set to True for testing, False for live trading
- **SYMBOL**: Trading pair (e.g., BTCUSDT)
- **TIMEFRAME**: Candlestick interval (e.g., 1h, 4h, 1d)
- **DC_LENGTH_ENTER**: Donchian Channel period for entries
- **DC_LENGTH_EXIT**: Donchian Channel period for exits
- **ATR_LENGTH**: ATR calculation period
- **RISK_PER_TRADE**: Risk percentage per trade (0.02 = 2%)
- **STOP_LOSS_ATR_MULTIPLE**: ATR multiplier for stop loss placement

## Improvements from Original Codebase

- **Object-Oriented Design**: Proper encapsulation of state and behavior
- **Type Hints**: Enhanced code quality and IDE support
- **Modular Structure**: Separate modules for different concerns
- **Improved Error Handling**: Consistent exception handling
- **Better Documentation**: Comprehensive docstrings and code comments
- **Short Position Support**: Added ability to trade in both directions
- **Enhanced Testability**: Easier to write unit tests

## License

[MIT License](LICENSE)

## Disclaimer

Trading cryptocurrencies carries significant risk. This bot is provided for educational purposes only. Use at your own risk.
