# Turtle Trading Bot

A Python trading bot that implements the Turtle Trading System strategy on the Binance exchange.

## Features

- Connects to Binance API (supports both testnet and production)
- Implements Donchian Channel-based entry and exit signals
- Uses ATR for position sizing and stop-loss calculation
- Risk management with configurable risk per trade
- Position state tracking and persistence between sessions
- Smart candle-based timing system for efficient trading
- Robust error handling and logging

## Strategy Implementation

The bot implements the Turtle Trading System with the following rules:

1. **Entry**: Buy when price breaks above the Donchian Channel upper band (lookback period configurable)
2. **Exit**: Sell when price breaks below the Donchian Channel lower band (lookback period configurable)
3. **Stop Loss**: Set at 2× ATR below entry price (multiplier configurable)
4. **Position Sizing**: Based on account risk percentage divided by stop distance (ATR-based)

## Requirements

- Python 3.7 or higher
- Required packages:
  - python-binance
  - pandas
  - pandas-ta
  - python-dotenv

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/Akcanbasri/turtle-trading-bot.git
   cd turtle-trading-bot
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file from the example:

   ```
   cp .env.example .env
   ```

4. Edit the `.env` file with your Binance API credentials and desired parameters.

## Configuration

All configuration is done through environment variables in the `.env` file:

- `BINANCE_API_KEY`: Your Binance API key
- `BINANCE_API_SECRET`: Your Binance API secret
- `USE_TESTNET`: Set to "True" to use Binance testnet or "False" to use real trading
- `SYMBOL`: Trading pair (e.g., "BTCUSDT")
- `TIMEFRAME`: Candlestick timeframe (e.g., "1h", "4h", "1d")
- `DC_LENGTH_ENTER`: Donchian Channel period for entry signals (default: 20)
- `DC_LENGTH_EXIT`: Donchian Channel period for exit signals (default: 10)
- `ATR_LENGTH`: ATR indicator period (default: 14)
- `ATR_SMOOTHING`: ATR smoothing period (default: 2)
- `RISK_PER_TRADE`: Percentage of account balance to risk per trade (default: 0.02 = 2%)
- `STOP_LOSS_ATR_MULTIPLE`: Stop loss distance in ATR multiples (default: 2)
- `QUOTE_ASSET`: Quote asset in trading pair (e.g., "USDT")
- `BASE_ASSET`: Base asset in trading pair (e.g., "BTC")

## Usage

1. Make sure your `.env` configuration is set correctly.

2. Run the bot:

   ```
   python turtle_trading_bot.py
   ```

3. The bot will:

   - Connect to Binance API
   - Load previous position state (if exists)
   - Enter the main trading loop that runs until interrupted
   - Wait for the next candle close based on the configured timeframe
   - Make trading decisions at candle close
   - Log all operations and trading decisions

4. Press Ctrl+C to stop the bot.

## Logs and State

- Logs are stored in the `logs/` directory with daily rotating log files
- Bot state is saved in `config/bot_state.json` and restored between sessions
- Positions are tracked and persisted even if the bot is restarted

## Security

- Create a dedicated API key with trading permissions only
- Store your API credentials securely in the `.env` file (not tracked in git)
- Use testnet for testing before live trading
- Implement additional security measures if deploying in production

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk. Always test thoroughly on a testnet before using with real funds.

## License

[MIT License](LICENSE)
