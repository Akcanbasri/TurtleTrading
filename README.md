# TurtleTrading - Advanced Algorithmic Trading Bot

A robust algorithmic trading bot implementing the Turtle Trading strategy with advanced features including multi-timeframe analysis, adaptive trend detection, and sophisticated risk management.

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
   git clone https://github.com/yourusername/TurtleTrading
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