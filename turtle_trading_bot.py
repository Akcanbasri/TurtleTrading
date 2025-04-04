#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Turtle Trading Bot - Main entry point
"""

import sys
from dotenv import load_dotenv

from bot import TurtleTradingBot
from bot.utils import setup_logging

# Set up logger
logger = setup_logging("turtle_trading_bot")


def main():
    """Main entry point for the Turtle Trading Bot"""
    try:
        # Load environment variables from .env file
        load_dotenv()

        # Initialize and run the Turtle Trading Bot
        logger.info("Initializing Turtle Trading Bot")
        bot = TurtleTradingBot()
        bot.run()

    except KeyboardInterrupt:
        logger.info("Bot execution interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
