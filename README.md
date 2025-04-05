# Advanced Turtle Trading Bot

This project is an algorithmic trading bot that employs the Turtle Trading strategy, enriched with a variety of advanced features.

## Features

- **Multi-Timeframe Analysis**: Detect trends across different timeframes
- **Pyramiding Positions**: Increase the position as the trend strengthens
- **Advanced Exit Strategy**: Partial profit-taking targets and trailing stop
- **Trend Filters**: ADX and Moving Average (MA) filtering
- **Smart Risk Management**: Limits risk per position and overall risk

## Strategy Logic

This bot implements a trading strategy based on the following key components:

1. **Trend Analysis**:
   - The main trends are determined using the 200-day moving average on a 1-day chart
   - Trend strength is measured using the ADX indicator (values above 25 indicate a strong trend)

2. **Entry Signals**:
   - Donchian Channels generate trend-following entry signals
   - Entry signals are confirmed by comparing them with the primary trend direction

3. **Pyramiding**:
   - Initial entry: 40% of the planned position size
   - Additional entries: 30% slices of the remaining size

4. **Exit Strategy**:
   - First Target: Exit 50% of the position at a distance of 3 ATR
   - Second Target: Exit 30% of the position at a distance of 5 ATR
   - For the final slice: Use a trailing stop

5. **Leverage Management**:
   - For trades in the direction of the trend: 2-3x leverage
   - For trades against the trend: Maximum 1.5x leverage

## Installation

1. Clone the repository:
   ```bash
   git clone [repo-url]
   cd TurtleTrading
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Edit the `.env` file:
   ```
   # Write your API keys here
   BINANCE_API_KEY=your_api_key_here
   BINANCE_API_SECRET=your_api_secret_here
   USE_TESTNET=True  # Run in test mode before starting live trading
   ```

4. Adjust the strategy parameters in the `.env` file as desired:
   ```
   # Adjust risk parameters according to your preference
   RISK_PER_TRADE=0.02  # 2% of your capital
   STOP_LOSS_ATR_MULTIPLE=1.5  # Stop loss distance is 1.5 times the ATR
   ```

## Running

To start the bot:

```bash
python turtle_trading_bot.py
```

## Important Considerations

- Test on the testnet before using real money
- Adjust risk management parameters according to your risk tolerance
- The bot operates with the risk of losing your entire capital; you are responsible for its use

## Customization

You can customize the behavior of the bot by modifying the strategy parameters in the `.env` file:

- `USE_MULTI_TIMEFRAME`: Enables/disables multi-timeframe analysis
- `USE_PYRAMIDING`: Enables/disables the pyramiding strategy
- `USE_TRAILING_STOP`: Enables/disables the use of trailing stop
- `USE_PARTIAL_EXITS`: Enables/disables partial profit-taking targets
- `USE_ADX_FILTER` and `USE_MA_FILTER`: Enables/disables trend filters

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
3. Copy `.env.example` to `.env` and update it with your Binance API credentials:
   ```
   cp .env.example .env
   ```
4. Edit the parameters in `.env` as needed (risk, timeframe, symbol, etc.)

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