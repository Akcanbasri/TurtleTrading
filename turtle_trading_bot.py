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
from datetime import datetime, timedelta
import pandas as pd
from binance.client import Client
import config  # Import the config module

# Try to force-reload the bot module to ensure we get the latest changes
if "bot" in sys.modules:
    importlib.reload(sys.modules["bot"])

from bot import TurtleTradingBot, setup_logger


# Custom filter to remove duplicate log messages
class DuplicateFilter(logging.Filter):
    def __init__(self, name=""):
        super(DuplicateFilter, self).__init__(name)
        self.last_log = None
        self.duplicate_count = 0
        self.last_duplicate_time = 0

    def filter(self, record):
        # Get the current log message
        current_log = (record.module, record.levelno, record.getMessage())
        current_time = time.time()

        # Check if it's the same as the last one
        if (
            current_log == self.last_log
            and current_time - self.last_duplicate_time < 60
        ):
            self.duplicate_count += 1
            # Only show every 5th duplicate message
            if self.duplicate_count >= 5:
                self.duplicate_count = 0
                self.last_duplicate_time = current_time
                record.msg = f"{record.msg} (repeated x5)"
                return True
            return False
        else:
            self.last_log = current_log
            self.duplicate_count = 0
            self.last_duplicate_time = current_time
            return True


# Load environment variables
load_dotenv()

# Setup logging with duplicate filter
setup_logger()
logger = logging.getLogger("turtle_trading_bot")
logger.addFilter(DuplicateFilter())


def get_real_account_balance():
    """Get the real account balance from Binance"""
    try:
        client = Client(
            config.BINANCE_API_KEY,
            config.BINANCE_API_SECRET,
            testnet=config.USE_TESTNET,
        )

        # Skip time sync checks
        logger.info("Skipping time sync checks to get real account balance")

        try:
            # Directly try to get account info without time check
            logger.info("Attempting to get account info directly...")
            account_info = client.get_account()
            logger.info("Successfully retrieved account info!")
        except Exception as e:
            logger.error(f"Error getting account: {str(e)}")
            logger.info("Using default balance of 20.00 USDT")
            logger.info("Total account balance: 20.00 USDT (default)")
            logger.info("USDT balance: 20.00 (default)")
            return 20.0

        total_balance = 0

        # Find USDT balance first
        usdt_balance = 0
        for asset in account_info["balances"]:
            if asset["asset"] == "USDT":
                usdt_balance = float(asset["free"]) + float(asset["locked"])

        # Add all other assets converted to USDT
        for asset in account_info["balances"]:
            free_amount = float(asset["free"])
            locked_amount = float(asset["locked"])

            if free_amount > 0 or locked_amount > 0:
                if asset["asset"] == "USDT":
                    total_amount = free_amount + locked_amount
                else:
                    # Try to get price of the asset in USDT
                    try:
                        symbol = f"{asset['asset']}USDT"
                        ticker = client.get_ticker(symbol=symbol)
                        price = float(ticker["lastPrice"])
                        total_amount = (free_amount + locked_amount) * price
                    except:
                        # Skip assets that don't have USDT pair
                        total_amount = 0

                total_balance += total_amount

        # Log detailed balance only for significant amounts
        for asset in account_info["balances"]:
            free_amount = float(asset["free"])
            locked_amount = float(asset["locked"])

            if free_amount > 0 or locked_amount > 0:
                logger.info(
                    f"Balance - {asset['asset']}: Free={free_amount:.8f}, Locked={locked_amount:.8f}"
                )

        logger.info(f"Total account balance: {total_balance:.8f} USDT")
        logger.info(f"USDT balance: {usdt_balance:.8f}")

        return total_balance

    except Exception as e:
        logger.error(f"Error fetching real account balance: {e}")
        # Return a default balance
        logger.info("Using default balance of 20 USDT")
        return 20.0


def sync_time(bot):
    """Synchronize local time with Binance server time"""
    try:
        # Use the bot's sync_time method if available
        if hasattr(bot, "sync_time") and callable(bot.sync_time):
            return bot.sync_time()

        # Fallback to old approach
        server_time = bot.exchange.spot_client.get_server_time()
        local_time = int(time.time() * 1000)
        time_diff = server_time["serverTime"] - local_time

        if abs(time_diff) > 5000:  # Increase tolerance from 1000ms to 5000ms
            logging.warning(
                f"Time difference with server: {time_diff}ms. Adjusting local time..."
            )
            time.sleep(abs(time_diff) / 1000)  # Wait to sync

        return True
    except Exception as e:
        logging.error(f"Failed to sync time: {e}")
        return False


def initialize_binance_client():
    """Initialize Binance client with API credentials"""
    try:
        # Initialize the client
        client = Client(
            config.BINANCE_API_KEY,
            config.BINANCE_API_SECRET,
            testnet=config.USE_TESTNET,
        )

        # Sync time with server
        if not sync_time(client):
            logging.error("Failed to sync time with Binance servers")
            return None

        # Test connection
        server_time = client.get_server_time()
        logging.info(f"Connected to Binance. Server time: {server_time}")

        return client

    except Exception as e:
        logging.error(f"Failed to initialize Binance client: {str(e)}")
        return None


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
    parser.add_argument(
        "--check-balance",
        action="store_true",
        help="Check real account balance and exit",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run only market analysis without executing trades",
    )
    args = parser.parse_args()

    # Debug args values
    logger.info(
        f"Args: analyze_only={args.analyze_only}, test={getattr(args, 'test', False)}, demo={getattr(args, 'demo', False)}"
    )

    # Configure environment based on CLI arguments
    if args.preset:
        logger.info(f"Using preset: {args.preset}")

    logger.info("Initializing Turtle Trading Bot")

    # Check real account balance if requested
    if args.check_balance:
        balance = get_real_account_balance()
        if balance is not None:
            logger.info(f"Account balance check completed. Exiting.")
        return

    try:
        # Initialize the trading bot
        use_testnet = os.getenv("USE_TESTNET", "False").lower() in ("true", "1", "t")
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

        # Sync time with server
        if not sync_time(bot):
            logger.error("Failed to initialize bot due to time sync issues")
            return

        if args.backtest:
            logger.info(f"Running backtest for the last {args.days} days")
            bot.run_backtest(days=args.days)
            return

        if args.analyze_only:
            logger.info(
                "*** DISTINCTIVE TEST MESSAGE *** Running in ANALYZE ONLY mode - analyzing market without executing trades"
            )
            try:
                logger.info("About to call bot.analyze_only()...")
                bot.analyze_only()
                logger.info("bot.analyze_only() completed successfully")
            except Exception as e:
                import traceback

                logger.error(f"Error calling analyze_only(): {e}")
                logger.error(traceback.format_exc())
            return

        if args.clear_cache:
            bot.data_manager.clear_cache()
            logger.info("Data cache cleared")

        # Check real account balance at startup (for non-demo mode)
        if not args.demo:
            real_balance = get_real_account_balance()
            if real_balance is not None:
                logger.info(
                    f"Starting trading with account balance: {real_balance:.2f} USDT"
                )

        # Main bot loop
        while True:
            try:
                # Check time sync every minute
                if (
                    not hasattr(main, "last_sync_time")
                    or time.time() - main.last_sync_time > 60
                ):
                    sync_time(bot)
                    main.last_sync_time = time.time()

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
                    bot.analyze_market()

                # Log status update every 5 minutes (but only if we don't have recent logs)
                current_time = time.time()
                if (
                    not hasattr(main, "last_status_time")
                    or current_time - main.last_status_time > 300
                ):
                    # Get current account info
                    try:
                        account_info = bot.exchange.get_account_balance(
                            force_refresh=True
                        )
                        logger.info(f"Account Balance: {account_info:.2f} USDT")

                        # If position is active, log position details
                        if bot.position_state.active:
                            symbol_price = bot.exchange.get_symbol_price(
                                bot.config.symbol
                            )
                            logger.info(f"Position Update - {bot.config.symbol}:")
                            logger.info(f"  Current Price: {symbol_price:.8f}")
                            logger.info(
                                f"  Entry Price: {bot.position_state.entry_price:.8f}"
                            )
                            pnl = bot.position_state.calculate_unrealized_pnl(
                                symbol_price
                            )
                            logger.info(f"  Unrealized PnL: {pnl:.2f}%")
                            logger.info(
                                f"  Stop Loss: {bot.position_state.stop_loss_price:.8f}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to get account update: {e}")

                    main.last_status_time = current_time

                # Sleep to avoid excessive API calls
                # Use a shorter sleep time to be more responsive, but
                # don't worry about frequent runs since real-time checks
                # happen through the WebSocket callback
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
