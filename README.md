# TurtleTrading - Advanced Algorithmic Trading Bot

A robust algorithmic trading bot implementing the Turtle Trading strategy with advanced features including dynamic leverage, multi-timeframe analysis, and sophisticated risk management.

## Key Features

- **Dynamic Leverage Management**: Automatically adjusts leverage based on account balance and trade direction
- **Multi-Timeframe Analysis**: Identifies trends across different timeframes for better entry/exit points
- **Smart Pyramiding**: Strategically increases position size as the trend strengthens
- **Advanced Exit Strategies**: Trailing stops and partial profit taking at key targets
- **Comprehensive Risk Management**: Adaptive risk per trade with maximum exposure limits
- **Trend Filters**: ADX strength and Moving Average alignment filters

## Dynamic Leverage System

The bot implements a sophisticated leverage management system that:

1. **Adapts to Account Size**:
   - Small accounts (<20 USDT): Maximum 5x leverage
   - Medium accounts (<50 USDT): Maximum 7x leverage
   - Standard accounts (<100 USDT): Maximum 10x leverage
   - Larger accounts: Uses configured leverage

2. **Trend-Based Adjustments**:
   - Higher leverage for trend-aligned trades (up to MAX_LEVERAGE_TREND)
   - Reduced leverage for counter-trend trades (up to MAX_LEVERAGE_COUNTER)

3. **Risk Protection**:
   - Enforces minimum position sizes (5 USDT by default)
   - Maintains consistent risk exposure regardless of leverage

## Strategy Logic

The trading strategy is based on these core components:

1. **Trend Analysis**:
   - Primary trend identification using Moving Averages on higher timeframes
   - Trend strength measurement via ADX (values above 25 indicate strong trends)

2. **Entry Signals**:
   - Donchian Channel breakouts generate trend-following entry signals
   - Signal confirmation through multi-timeframe trend alignment

3. **Position Management**:
   - Initial entry: 40% of the calculated position size
   - Additional entries: 30% increments when trend strengthens
   - Dynamic stop loss placement based on ATR volatility

4. **Intelligent Exit Strategy**:
   - First Target: Exit 50% of the position at 3x ATR
   - Trailing stop: Activated at predefined profit threshold
   - Systematic exits on trend reversal signals

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

To start the bot:

```bash
python turtle_trading_bot.py
```

For analysis-only mode (no trading):

```bash
python turtle_trading_bot.py --analyze
```

## Configuration Options

### Core Trading Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `SYMBOL` | Trading pair | BTCUSDT |
| `TIMEFRAME` | Candle timeframe | 1h |
| `RISK_PER_TRADE` | Risk percentage per trade | 0.02 (2%) |
| `STOP_LOSS_ATR_MULTIPLE` | ATR multiplier for stop loss | 1.5 |

### Advanced Features

| Feature | Parameter | Description |
|---------|-----------|-------------|
| Multi-Timeframe | `USE_MULTI_TIMEFRAME=True` | Analyze multiple timeframes |
| Pyramiding | `USE_PYRAMIDING=True` | Add to positions in trend direction |
| Trailing Stop | `USE_TRAILING_STOP=True` | Dynamic trailing stop loss |
| Partial Exits | `USE_PARTIAL_EXITS=True` | Take profits at predetermined levels |

### Leverage Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `LEVERAGE` | Base leverage value | 10 |
| `MAX_LEVERAGE_TREND` | Max leverage for trend-aligned trades | 3 |
| `MAX_LEVERAGE_COUNTER` | Max leverage for counter-trend trades | 1.5 |

## Project Structure

```
TurtleTrading/
├── turtle_trading_bot.py    # Main entry point
├── .env                     # Configuration file
├── requirements.txt         # Dependencies
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
5. **Adaptive Leverage**: Lower leverage for smaller accounts and counter-trend trades

## Important Notes

- Always test on a testnet before running with real funds
- Start with small position sizes and conservative risk parameters
- The bot performs best in trending markets
- Performance may vary depending on market conditions
- Regularly check and adjust parameters based on performance

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Disclaimer

Trading cryptocurrencies carries significant risk of loss. This bot is provided for educational purposes only. The developers are not responsible for any financial losses incurred from using this software. Use at your own risk.