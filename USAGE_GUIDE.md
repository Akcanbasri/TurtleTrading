# TurtleTrading Bot - Usage Guide

This guide explains how to set up, test, and effectively use the TurtleTrading bot with its dynamic leverage features.

## Setup

### Prerequisites

- Python 3.8 or higher
- A Binance account (regular or testnet)
- Basic understanding of cryptocurrency trading concepts

### Installation Steps

1. Clone the repository and navigate to it:
   ```bash
   git clone https://github.com/Akcanbasri/TurtleTrading.git
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

## Testing

### Using Binance Testnet

1. Create a Binance Futures testnet account at [testnet.binancefuture.com](https://testnet.binancefuture.com/)
2. Generate API keys from the testnet dashboard
3. Add these keys to your `.env` file
4. Ensure `USE_TESTNET=True` in your `.env` file

### Running in Analysis Mode

To run the bot in analysis mode without executing trades:

```bash
python turtle_trading_bot.py --analyze
```

This will display market analysis, entry/exit signals, and position sizing calculations without placing actual orders.

### Monitoring Test Results

1. Check the `logs` directory for detailed operation logs
2. Monitor the terminal output for real-time information
3. Analyze backtesting results (if generated) in the `backtest_results` directory

## Live Trading

### Configuration Recommendations

When moving to live trading, consider these recommendations:

1. Start with a small account balance (20-50 USDT)
2. Use conservative risk settings: 
   - `RISK_PER_TRADE=0.01` (1% per trade)
   - `MAX_RISK_PERCENTAGE=0.05` (5% maximum risk)
3. Begin with simpler settings:
   - `USE_PYRAMIDING=False`
   - `USE_TRAILING_STOP=True`
   - `USE_PARTIAL_EXITS=True`

### Enabling Live Trading

1. Set `USE_TESTNET=False` in your `.env` file
2. Use your real Binance API credentials
3. Start the bot in regular mode:
   ```bash
   python turtle_trading_bot.py
   ```

## Understanding Dynamic Leverage

### How It Works

The bot automatically adjusts leverage based on:

1. **Account Size:**
   - <20 USDT: Maximum 5x leverage
   - <50 USDT: Maximum 7x leverage
   - <100 USDT: Maximum 10x leverage
   - >100 USDT: Uses configured leverage

2. **Trade Direction:**
   - Trades aligned with the main trend: Higher leverage (MAX_LEVERAGE_TREND)
   - Counter-trend trades: Lower leverage (MAX_LEVERAGE_COUNTER)

3. **Minimum Position Size:**
   - The bot enforces a 5 USDT minimum position value
   - For very small accounts, this may require using the maximum allowed leverage

### Example Scenarios

**Scenario 1: Small Account (20 USDT)**
- Risk per trade: 2% = 0.4 USDT
- For a BTC position with high ATR volatility, leverage might automatically adjust to 5x
- This allows meaningful position sizes while maintaining risk limits

**Scenario 2: Medium Account (60 USDT)**
- Risk per trade: 2% = 1.2 USDT
- Trend-aligned trade: May use up to 7x leverage
- Counter-trend trade: Limited to 1.5x leverage

## Monitoring Your Bot

### Real-time Monitoring

The bot outputs detailed information to:
- The console (real-time updates)
- Log files in the `logs` directory (comprehensive details)

### Important Metrics to Watch

1. **Leverage Used:** Check that the leverage is appropriate for your account size
2. **Position Size:** Verify that positions are neither too small nor too large
3. **Risk Per Trade:** Confirm that actual risk matches your configuration
4. **Win Rate & Profit:** Track performance over time

### Stopping the Bot

To stop the bot safely:
- Press `Ctrl+C` in the terminal
- The bot will complete any pending operations before shutting down

## Troubleshooting

### Common Issues

1. **Insufficient Balance Errors:**
   - Check your account balance
   - Reduce the risk percentage or leverage

2. **Invalid Quantity Errors:**
   - Some trading pairs have minimum quantity requirements
   - The bot will try to adjust automatically, but may not always succeed

3. **API Connection Issues:**
   - Verify your API keys are correct
   - Check your internet connection
   - Ensure the Binance API is operational

### Getting Help

If you encounter problems:
1. Check the detailed logs in the `logs` directory
2. Review the [GitHub Issues](https://github.com/Akcanbasri/TurtleTrading/issues) section
3. Open a new issue with detailed information about your problem

## Advanced Usage

### Customizing Strategy Parameters

For experienced users, you can fine-tune these parameters:

1. **Trend Detection:**
   - `TREND_TIMEFRAME` - Higher timeframes provide more reliable trends
   - `MA_PERIOD` - Longer periods for long-term trends, shorter for more sensitivity

2. **Entry/Exit Timing:**
   - `DC_LENGTH_ENTER` - Higher values for fewer but more reliable entries
   - `DC_LENGTH_EXIT` - Lower values for quicker exits

3. **Risk Management:**
   - `STOP_LOSS_ATR_MULTIPLE` - Higher values for wider stops (more breathing room)
   - `FIRST_TARGET_ATR` and `SECOND_TARGET_ATR` - Adjust profit targets
   
### Event-Driven Updates

1. Run with automatic updates on market events:
   ```bash
   python turtle_trading_bot.py --live
   ```

2. This mode will update positions based on real-time market events rather than candle closes

## Disclaimer

This bot is for educational purposes only. Trading cryptocurrencies involves significant risk of loss. Only trade with funds you can afford to lose. The developers are not responsible for any financial losses incurred from using this software. 