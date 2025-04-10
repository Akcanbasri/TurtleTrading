# Optimized Turtle Trading Bot for Cryptocurrency Markets

An enhanced Turtle Trading implementation optimized for cryptocurrency markets, with advanced signal detection, risk management, and market regime adaptability.

## Key Optimizations

### 1. Reduced Filter Requirements

- **Optional Trend Alignment**: Configure `TREND_ALIGNMENT_REQUIRED=False` to allow counter-trend trades
- **Lower ADX Threshold**: Reduced to 15 (from 25) to catch more potential moves
- **Simplified Confirmation**: Requires either RSI OR MACD confirmation, not both
- **Modified Entry Signal**: Enhanced `check_entry_signal()` to handle both long/short scenarios

### 2. Adjusted Signal Parameters

- **Shorter DC Length**: Decreased Donchian Channel entry period to 15 (from 20)
- **Wider RSI Ranges**: 
  - Long entries: 30-80 (formerly 40-70)
  - Short entries: 20-70 (formerly 30-60)
- **Simplified MACD**: Only checks MACD vs Signal line without histogram requirements

### 3. Detailed Logging

- **Condition-by-Condition Checks**: Visual ✅/❌ indicators for each filter
- **Value Displays**: Shows actual indicator values vs thresholds
- **Entry Requirement Counts**: Displays how many requirements were met (e.g., "4/6 requirements met")
- **Clear Signal Indicators**: Visual confirmation for triggered signals

### 4. Market Regime Detection

- **Regime Classification**: Automatically identifies market conditions:
  - Trending (up/down/sideways)
  - Ranging
  - Squeeze (low volatility, potential breakout)
  - Volatile
- **Dynamic Parameters**: Adjusts strategy parameters based on current regime
- **Squeeze Breakout Detection**: Optimized for breakouts during low volatility periods
- **Two-Way Price Action**: Avoids trading in choppy, ranging markets

### 5. Enhanced Risk Management

- **Dynamic Leverage**: Sets appropriate leverage (3-5x) based on:
  - Signal strength
  - Market regime
  - Weekend volatility adjustments
- **Partial Take-Profits**: Implements automatic partial exits at:
  - 2x ATR (30% of position)
  - 3x ATR (30% of position)
- **Position Sizing**: Dynamically adjusts position size based on volatility
- **Weekend Adjustments**: Reduces risk during typically volatile weekend periods

## Usage

To run the optimized bot:

```python
from bot.turtle_optimized import check_optimized_entry_signal, execute_optimized_entry, check_optimized_exit_conditions

# Check for entry signals
entry_signal, details = check_optimized_entry_signal(market_data, trend_data, "long", config)

# Execute entry if signal detected
if entry_signal:
    success, order_details = execute_optimized_entry(
        exchange=exchange,
        symbol="BTCUSDT",
        direction="long",
        current_price=details["price"],
        atr_value=details["atr"],
        signal_strength=details["signal_strength"],
        market_regime=details["market_regime"],
        position=position,
        config=config
    )
    
# Check for exit signals
exit_signal, reason = check_optimized_exit_conditions(market_data, position, current_price)
```

## Performance Dashboard

The trading bot maintains a performance dashboard that tracks:

- Win rate by market regime type
- Profit factor
- Average win/loss metrics
- Risk-adjusted returns
- Signal quality statistics

## Configuration

Modify the following environment variables to adjust the optimized bot's behavior:

```
# Signal Parameters
DC_LENGTH_ENTER=15
DC_LENGTH_EXIT=8
ADX_THRESHOLD=15
TREND_ALIGNMENT_REQUIRED=False

# Risk Management
USE_PARTIAL_EXITS=True
FIRST_TARGET_ATR=2.0
SECOND_TARGET_ATR=3.0
BASE_LEVERAGE=3
MAX_LEVERAGE=5

# Market Regime
USE_REGIME_DETECTION=True
```

## Key Features

- **Multi-Timeframe Analysis**: Identifies trends across different timeframes for better entry/exit points
- **Advanced Indicator System**: Uses Donchian Channels, ATR, Moving Averages, ADX, and more
- **Smart Pyramiding**: Strategically increases position size as the trend strengthens
- **Flexible Exit Strategies**: Implements trailing stops and partial profit taking
- **Comprehensive Risk Management**: Adaptive position sizing with maximum exposure limits
- **Trend Filters**: ADX strength indicator and Moving Average alignment filters
- **Automatic Backtesting**: Built-in capability for historical performance testing

## Strategy Logic

The TurtleTrading bot's strategy is based on these core components:

1. **Trend Analysis**:
   - Primary trend identification using Moving Averages across multiple timeframes
   - Trend strength measurement via ADX (values above 25 indicate strong trends)
   - Volatility measurement using ATR (Average True Range)

2. **Entry Signals**:
   - Donchian Channel breakouts generate trend-following entry signals
   - Signal confirmation through multi-timeframe trend alignment
   - Additional filters for reducing false signals

3. **Position Management**:
   - Dynamic position sizing based on account balance and market volatility
   - Optional pyramiding to add to positions when trend strengthens
   - Configurable risk percentage per trade

4. **Intelligent Exit Strategy**:
   - Multiple exit targets based on ATR multiples
   - Trailing stop loss activation at predefined profit levels
   - Systematic exits when trend reversal signals appear

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Akcanbasri/TurtleTrading
   cd TurtleTrading
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the `.env` file:
   ```
   # Enter your API keys
   BINANCE_API_KEY=your_api_key_here
   BINANCE_API_SECRET=your_api_secret_here
   
   # Set to True for testing, False for live trading
   USE_TESTNET=True
   
   # Trading parameters
   SYMBOL=BTCUSDT
   TIMEFRAME=1h
   
   # Risk parameters
   RISK_PER_TRADE=0.02
   STOP_LOSS_ATR_MULTIPLE=1.5
   ```

## Running the Bot

### Basic Usage

Start the bot with default settings:

```bash
python turtle_trading_bot.py
```

### Optional Arguments

```bash
# Run in test mode (no trade execution)
python turtle_trading_bot.py --test

# Run in backtest mode
python turtle_trading_bot.py --backtest --days 30

# Run in demo mode with synthetic data (no API key required)
python turtle_trading_bot.py --demo

# Clear data cache before starting
python turtle_trading_bot.py --clear-cache

# Use predefined timeframe settings
python turtle_trading_bot.py --preset crypto_standard
```

Available presets:
- `crypto_fast`: Short-term trading for cryptocurrencies
- `crypto_standard`: Medium-term strategy for crypto markets
- `crypto_swing`: Long-term swing trading for cryptocurrencies
- `forex_standard`: Optimized for forex markets
- `stocks_daily`: Daily timeframe for stock trading
- `original_turtle`: Classic Turtle Trading strategy implementation

## Configuration Options

### Core Trading Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `SYMBOL` | Trading pair | BTCUSDT |
| `TIMEFRAME` | Base timeframe | 1h |
| `RISK_PER_TRADE` | Risk percentage per trade | 0.02 (2%) |
| `STOP_LOSS_ATR_MULTIPLE` | ATR multiplier for stop loss | 1.5 |

### Advanced Features

| Feature | Parameter | Description |
|---------|-----------|-------------|
| Multi-Timeframe | `USE_MULTI_TIMEFRAME=True` | Analyze multiple timeframes |
| Pyramiding | `USE_PYRAMIDING=True` | Add to positions in trend direction |
| Trailing Stop | `USE_TRAILING_STOP=True` | Dynamic trailing stop loss |
| Partial Exits | `USE_PARTIAL_EXITS=True` | Take profits at predetermined levels |

## Project Structure

```
TurtleTrading/
├── turtle_trading_bot.py    # Main entry point
├── .env                     # Configuration file
├── requirements.txt         # Dependencies
├── README.md                # This file
├── USAGE_GUIDE.md           # Detailed usage instructions
├── logs/                    # Trading logs
├── config/                  # Bot state and configurations
└── bot/                     # Core modules
    ├── core.py              # Main bot implementation
    ├── exchange.py          # Exchange interaction
    ├── indicators.py        # Technical indicators
    ├── models.py            # Data models and state tracking
    ├── risk.py              # Risk management
    └── utils.py             # Utility functions
```

## Risk Management

The bot employs multiple layers of risk management:

1. **Per-Trade Risk Limit**: Each trade risks only a percentage of your balance
2. **Dynamic Position Sizing**: Position size calculated based on ATR volatility
3. **Maximum Exposure**: Total risk across all positions is capped
4. **Stop Loss Protection**: Automatic stop loss for every position
5. **Multi-timeframe Confirmation**: Reduces false signals and risky entries

## Troubleshooting

If you encounter problems with the bot:

1. Check the log files in the `logs` directory for detailed error messages
2. Ensure your API keys have the correct permissions
3. Verify that your account has sufficient balance
4. Make sure the market for your selected symbol is active
5. For "ma" related errors, use the `--clear-cache` option to reset cached data

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Disclaimer

Trading cryptocurrencies carries significant risk of loss. This bot is provided for educational purposes only. The developers are not responsible for any financial losses incurred from using this software. Use at your own risk.
