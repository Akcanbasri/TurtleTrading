# TurtleTrading Bot - Usage Guide

This guide explains how to set up, configure, and effectively use the TurtleTrading bot with its advanced multi-timeframe analysis capabilities.

## Setup

### Prerequisites

- Python 3.8 or higher
- A Binance account (regular or testnet)
- Basic understanding of cryptocurrency trading concepts

### Installation Steps

1. Clone the repository and navigate to it:
   ```bash
   git clone https://github.com/yourusername/TurtleTrading.git
   cd TurtleTrading
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create your environment configuration:
   ```bash
   cp .env.example .env
   ```

4. Edit the `.env` file with your Binance API credentials and preferred settings

## Running the Bot

### Available Command-Line Options

The bot offers several command-line arguments for different modes of operation:

```bash
# Basic operation
python turtle_trading_bot.py

# Test mode - analyze without executing trades
python turtle_trading_bot.py --test

# Backtest mode
python turtle_trading_bot.py --backtest --days 30

# Demo mode with synthetic data (no API keys needed)
python turtle_trading_bot.py --demo

# Clear cache before starting
python turtle_trading_bot.py --clear-cache

# Use predefined timeframe settings
python turtle_trading_bot.py --preset crypto_standard
```

### Available Presets

The bot comes with several predefined configuration presets:

- `crypto_fast`: Optimized for short-term cryptocurrency trading
- `crypto_standard`: Balanced settings for medium-term crypto trading
- `crypto_swing`: Longer-term swing trading settings for cryptocurrencies
- `forex_standard`: Settings tuned for forex markets
- `stocks_daily`: Daily timeframe settings for stock trading 
- `original_turtle`: Classic Turtle Trading strategy parameters

## Testing Environment

### Using Binance Testnet

1. Create a Binance Futures testnet account at [testnet.binancefuture.com](https://testnet.binancefuture.com/)
2. Generate API keys from the testnet dashboard
3. Add these keys to your `.env` file
4. Ensure `USE_TESTNET=True` in your `.env` file

### Running in Test Mode

To analyze the market without executing trades:

```bash
python turtle_trading_bot.py --test
```

This will:
- Process market data and calculate indicators
- Generate entry/exit signals
- Display potential trades and position sizes
- Log analysis results
- Not place any actual orders

### Demo Mode

For testing without a Binance account:

```bash
python turtle_trading_bot.py --demo
```

This mode generates synthetic data to simulate trading, making it useful for strategy testing without any API connection.

## Understanding Multi-Timeframe Analysis

### How It Works

The bot analyzes multiple timeframes simultaneously to improve trading decisions:

1. **Trend Timeframe** (higher timeframe):
   - Used to determine the primary market trend
   - Moving average crosses identify long-term trend direction
   - ADX indicator measures trend strength

2. **Entry Timeframe** (medium timeframe):
   - Used to find optimal entry points
   - Donchian Channel breakouts generate trading signals
   - Additional filters reduce false signals

3. **Execution Timeframe** (lower timeframe):
   - Used for precise entry and exit timing
   - Helps optimize trade execution
   - Provides finer detail for stop loss placement

### Key Indicators

- **Donchian Channels**: Identify range breakouts
- **Moving Averages**: Determine trend direction
- **ATR (Average True Range)**: Measure volatility for position sizing
- **ADX (Average Directional Index)**: Evaluate trend strength
- **RSI (Relative Strength Index)**: Identify overbought/oversold conditions
- **Bollinger Bands**: Detect volatility contractions (squeezes)

## Bot Configuration

### Essential .env Variables

```
# API Configuration
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
USE_TESTNET=True

# Trading Pair and Timeframes
SYMBOL=BTCUSDT
TIMEFRAME=1h
TREND_TIMEFRAME=4h
ENTRY_TIMEFRAME=1h

# Risk Parameters
RISK_PER_TRADE=0.02
MAX_RISK_PERCENTAGE=0.05
STOP_LOSS_ATR_MULTIPLE=1.5

# Strategy Parameters
DC_LENGTH_ENTER=20
DC_LENGTH_EXIT=10
ATR_LENGTH=14
MA_PERIOD=200
ADX_THRESHOLD=25

# Feature Toggles
USE_MULTI_TIMEFRAME=True
USE_PYRAMIDING=True
USE_TRAILING_STOP=True
USE_PARTIAL_EXITS=True
USE_ADX_FILTER=True
USE_MA_FILTER=True
```

### Key Configuration Options

#### Risk Management

- `RISK_PER_TRADE`: Percentage of account balance risked per trade (e.g., 0.02 for 2%)
- `MAX_RISK_PERCENTAGE`: Maximum total risk exposure across all positions
- `STOP_LOSS_ATR_MULTIPLE`: ATR multiplier for stop loss placement

#### Strategy Parameters

- `DC_LENGTH_ENTER`: Donchian Channel period for entry signals
- `DC_LENGTH_EXIT`: Donchian Channel period for exit signals
- `ATR_LENGTH`: ATR calculation period
- `MA_PERIOD`: Moving Average period for trend determination
- `ADX_THRESHOLD`: Minimum ADX value to consider a trend strong

#### Feature Toggles

- `USE_MULTI_TIMEFRAME`: Enable multi-timeframe analysis
- `USE_PYRAMIDING`: Allow adding to existing positions in trend direction
- `USE_TRAILING_STOP`: Enable trailing stop loss
- `USE_PARTIAL_EXITS`: Take partial profits at predetermined levels

## Live Trading

### Preparation

Before running in live trading mode:

1. Test thoroughly on the Binance testnet
2. Start with small position sizes
3. Use conservative risk settings:
   - `RISK_PER_TRADE=0.01` (1% per trade)
   - `MAX_RISK_PERCENTAGE=0.03` (3% maximum total risk)
4. Monitor the bot's operation closely during the first few trades

### Configuration for Live Trading

1. Set `USE_TESTNET=False` in your `.env` file
2. Use your real Binance API credentials with appropriate permissions
3. Consider starting with simpler settings:
   - `USE_PYRAMIDING=False`
   - `USE_MULTI_TIMEFRAME=True`
   - `USE_TRAILING_STOP=True`

## Monitoring Your Bot

### Log Files

The bot generates detailed logs in the `logs` directory:

- `turtle_trading_bot.log`: Main log with all bot operations
- Error logs with specific error details
- Debug logs with detailed process information

### Performance Metrics

The bot tracks and logs these key performance metrics:

1. Win Rate: Percentage of profitable trades
2. Risk-Reward Ratio: Average profit compared to risk taken
3. Maximum Drawdown: Largest peak-to-trough decline
4. Profit Factor: Gross profit divided by gross loss

### Bot State

The bot state is saved in `config/bot_state.json` and includes:

- Current position information
- Trading history
- Performance metrics
- Last analysis results

## Troubleshooting

### Common Issues

1. **'ma' Error:**
   - Problem: Error message containing "Error during multi-timeframe analysis: 'ma'"
   - Solution: Use `--clear-cache` option to reset data cache
   
2. **Insufficient Balance Errors:**
   - Problem: Not enough funds to place orders
   - Solution: Check account balance or decrease risk percentage
   
3. **Invalid Quantity Errors:**
   - Problem: Order quantity doesn't meet exchange requirements
   - Solution: Check minimum order size for the trading pair

4. **Data Connection Issues:**
   - Problem: Errors retrieving market data
   - Solution: Check internet connection and API key permissions

### Getting Help

If you encounter problems:
1. Check detailed logs in the `logs` directory
2. Look for error messages in the terminal output
3. Open an issue on GitHub with the error details and bot configuration

## Advanced Usage

### Customizing Strategy Parameters

For experienced users, these parameters can be fine-tuned:

1. **Donchian Channel Periods:**
   - Longer periods (20-55) for fewer but more reliable signals
   - Shorter periods (10-20) for more frequent trading opportunities

2. **Moving Average Periods:**
   - Longer periods (100-200) for stable long-term trends
   - Shorter periods (50-100) for more responsive trend detection

3. **ATR Settings:**
   - Longer periods (14-21) for more stable volatility measurement
   - Shorter periods (7-14) for quicker adaptation to changing volatility

### Backtesting

Run backtests to evaluate strategy performance:

```bash
python turtle_trading_bot.py --backtest --days 30
```

Parameters:
- `--days`: Number of days to backtest (default: 30)

## Disclaimer

This bot is for educational purposes only. Trading cryptocurrencies involves significant risk of loss. Only trade with funds you can afford to lose. The developers are not responsible for any financial losses incurred from using this software. 