#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Turtle Trading Bot - Main entry point
"""

import os
import sys
import time
import logging
import argparse
import importlib
from dotenv import load_dotenv
import numpy as np

# Try to force-reload the bot module to ensure we get the latest changes
if "bot" in sys.modules:
    importlib.reload(sys.modules["bot"])

from bot import TurtleTradingBot, setup_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logger()
logger = logging.getLogger("turtle_trading_bot")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Turtle Trading Bot")
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Run in test mode without executing trades",
    )
    parser.add_argument(
        "-b", "--backtest", action="store_true", help="Run backtesting mode"
    )
    parser.add_argument(
        "-d",
        "--days",
        type=int,
        default=30,
        help="Number of days to backtest (default: 30)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with synthetic data (no API key needed)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=[
            "crypto_fast",
            "crypto_standard",
            "crypto_swing",
            "forex_standard",
            "stocks_daily",
            "original_turtle",
        ],
        help="Use a predefined timeframe preset for trading",
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear data cache before starting"
    )
    args = parser.parse_args()

    logger.info("Initializing Turtle Trading Bot")

    try:
        # Initialize the trading bot
        use_testnet = os.getenv("USE_TESTNET", "True").lower() in ("true", "1", "t")
        logger.info(
            f"Initializing Binance client in {'TESTNET' if use_testnet else 'PRODUCTION'} mode"
        )

        # Use demo mode if specified (no API key needed)
        if args.demo:
            logger.info("Running in DEMO mode with synthetic data - no API key needed")
            from bot.exchange import BinanceExchange

            # Create dummy API credentials for demo mode
            api_key = "demo_key"
            api_secret = "demo_secret"
            bot = TurtleTradingBot(
                use_testnet=True,
                api_key=api_key,
                api_secret=api_secret,
                demo_mode=True,
                timeframe_preset=args.preset,
            )
        else:
            # Normal mode - use API keys from environment variables
            bot = TurtleTradingBot(
                use_testnet=use_testnet, timeframe_preset=args.preset
            )

        if args.backtest:
            logger.info(f"Running backtest for the last {args.days} days")
            bot.run_backtest(days=args.days)
            return

        if args.clear_cache:
            bot.data_manager.clear_cache()
            logger.info("Data cache cleared")

        # Main bot loop
        while True:
            try:
                if args.test or args.demo:
                    mode_desc = "TEST" if args.test else "DEMO"
                    logger.info(
                        f"Running in {mode_desc} mode - analyzing market without executing trades"
                    )
                    try:
                        bot.analyze_only()
                    except Exception as e:
                        import traceback

                        logger.error(f"Error during analysis: {e}")
                        logger.error(traceback.format_exc())  # Print full traceback
                else:
                    bot.run_trading_cycle()

                # Sleep to avoid excessive API calls
                time.sleep(30)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                import traceback

                logger.error(f"Error in main loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(60)  # Wait a bit longer on error

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
