"""
Core TurtleTradingBot class implementation
"""

import decimal
import time
import traceback
from decimal import Decimal
from typing import Any, Dict, Optional, List, Union

import pandas as pd
import logging
import os
import json
from pathlib import Path

from bot.exchange import BinanceExchange
from bot.indicators import (
    calculate_indicators,
    calculate_stop_loss_take_profit,
    check_exit_signal,
    check_stop_loss,
)
from bot.models import BotConfig, PositionState
from bot.risk import calculate_pnl, calculate_position_size
from bot.utils import (
    format_price,
    get_sleep_time,
    load_position_state,
    save_position_state,
    setup_logging,
)


class DataManager:
    """Manages data operations for the bot."""

    def __init__(self, exchange):
        """Initialize the data manager."""
        self.exchange = exchange
        self.data_cache = {}

    def get_historical_data(self, symbol, timeframe, limit=100):
        """Get historical candlestick data for a symbol and timeframe."""
        cache_key = f"{symbol}_{timeframe}_{limit}"

        # Check cache first
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        # Fetch from exchange
        data = self.exchange.get_historical_data(symbol, timeframe, limit)

        # Cache the result
        self.data_cache[cache_key] = data

        return data

    def clear_cache(self):
        """Clear the data cache."""
        self.data_cache = {}


class TurtleTradingBot:
    """
    Turtle Trading Bot main class.

    This bot implements the Turtle Trading strategy, a trend-following approach
    that uses Donchian Channel breakouts for entries and exits, with proper
    risk management through position sizing based on ATR.
    """

    def __init__(
        self,
        use_testnet=True,
        config_file=".env",
        api_key=None,
        api_secret=None,
        demo_mode=False,
    ):
        """
        Initialize the bot.

        Args:
            use_testnet: Whether to use the Binance testnet
            config_file: Path to the configuration file
            api_key: Optional API key (overrides config file)
            api_secret: Optional API secret (overrides config file)
            demo_mode: Whether to run in demo mode with synthetic data
        """
        # Setup logging
        self.logger = logging.getLogger("turtle_trading_bot")

        # Load configuration
        self.config_file = config_file
        self.config = BotConfig(config_file)

        # Override testnet setting if provided
        if use_testnet is not None:
            self.config.use_testnet = use_testnet

        # Store demo mode flag
        self.demo_mode = demo_mode
        if demo_mode:
            self.logger.info("Running in demo mode with synthetic data")

        # Extract common settings for easier access
        self.symbol = self.config.symbol
        self.timeframe = self.config.timeframe
        self.quote_asset = self.config.quote_asset
        self.base_asset = self.config.base_asset

        # Use provided API keys if available
        if api_key is not None and api_secret is not None:
            self.config.api_key = api_key
            self.config.api_secret = api_secret

        # Initialize exchange interface
        self.exchange = BinanceExchange(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            use_testnet=self.config.use_testnet,
        )

        # Initialize the data manager
        self.data_manager = DataManager(self.exchange)

        # Initialize position state
        self.position = self._load_position_state()

        # Initialize symbol info
        try:
            if self.demo_mode:
                # Create a default SymbolInfo for demo mode
                from bot.models import SymbolInfo

                self.symbol_info = SymbolInfo(
                    price_precision=2,
                    quantity_precision=4,
                    min_qty=Decimal("0.001"),
                    step_size=Decimal("0.001"),
                    min_notional=Decimal("10"),
                )
                self.logger.info("Using default symbol info for demo mode")
            else:
                self.symbol_info = self.exchange.get_symbol_info(self.symbol)
        except Exception as e:
            if self.demo_mode:
                # Create a default SymbolInfo for demo mode
                from bot.models import SymbolInfo

                self.symbol_info = SymbolInfo(
                    price_precision=2,
                    quantity_precision=4,
                    min_qty=Decimal("0.001"),
                    step_size=Decimal("0.001"),
                    min_notional=Decimal("10"),
                )
                self.logger.info("Using default symbol info for demo mode")
            else:
                self.logger.error(f"Error getting symbol info: {e}")
                raise

        # Log bot initialization
        self._log_initialization()

    def _load_position_state(self) -> PositionState:
        """
        Load position state from file or create new one

        Returns
        -------
        PositionState
            Current position state
        """
        loaded_position = load_position_state(self.config.symbol)
        if loaded_position:
            return loaded_position
        return PositionState()

    def _log_initialization(self) -> None:
        """Log bot initialization details"""
        self.logger.info("==============================================")
        self.logger.info("         TURTLE TRADING BOT STARTED          ")
        self.logger.info("==============================================")
        self.logger.info(
            f"Trading Configuration: Symbol={self.config.symbol}, Timeframe={self.config.timeframe}"
        )
        self.logger.info(
            f"Strategy Parameters: DC_Enter={self.config.dc_length_enter}, DC_Exit={self.config.dc_length_exit}, ATR_Length={self.config.atr_length}"
        )
        self.logger.info(
            f"Risk Management: Risk_Per_Trade={self.config.risk_per_trade}, SL_ATR_Multiple={self.config.stop_loss_atr_multiple}"
        )

        if self.position.active:
            self.logger.info("Bot started with active position:")
            self.log_position_state()

    def log_position_state(self) -> None:
        """Log the current position state"""
        if self.position.active:
            self.logger.info(f"Active {self.position.side} position:")
            self.logger.info(
                f"  Quantity: {self.position.quantity} {self.config.base_asset}"
            )
            self.logger.info(
                f"  Entry Price: {format_price(self.position.entry_price, self.symbol_info.price_precision)}"
            )
            self.logger.info(
                f"  Stop Loss: {format_price(self.position.stop_loss_price, self.symbol_info.price_precision)}"
            )
            self.logger.info(
                f"  Take Profit: {format_price(self.position.take_profit_price, self.symbol_info.price_precision)}"
            )
            self.logger.info(f"  Entry ATR: {self.position.entry_atr}")

            # Calculate current P&L
            try:
                current_price = self.exchange.get_current_price(self.config.symbol)

                if current_price:
                    if self.position.side == "BUY":
                        pnl = (
                            current_price - self.position.entry_price
                        ) * self.position.quantity
                        pnl_percent = (
                            current_price / self.position.entry_price - Decimal("1")
                        ) * Decimal("100")
                    else:
                        pnl = (
                            self.position.entry_price - current_price
                        ) * self.position.quantity
                        pnl_percent = (
                            Decimal("1") - current_price / self.position.entry_price
                        ) * Decimal("100")

                    self.logger.info(
                        f"  Current Price: {format_price(current_price, self.symbol_info.price_precision)}"
                    )
                    self.logger.info(
                        f"  Unrealized P&L: {format_price(pnl, self.symbol_info.price_precision)} {self.config.quote_asset} ({pnl_percent:.2f}%)"
                    )
            except Exception as e:
                self.logger.error(f"Error calculating current P&L: {e}")
        else:
            self.logger.info("No active position")

    def update_position_state(
        self, order_result: Dict[str, Any], side: str, atr_value: Optional[float] = None
    ) -> None:
        """
        Update the position state after an order execution

        Parameters
        ----------
        order_result : Dict[str, Any]
            Order execution result from execute_order function
        side : str
            Order side ('BUY' or 'SELL')
        atr_value : Optional[float]
            ATR value at entry time
        """
        if side == "BUY" and not self.position.active:
            # Opening a long position
            self.position.active = True
            self.position.side = "BUY"
            self.position.entry_price = Decimal(order_result["avgPrice"])
            self.position.quantity = Decimal(order_result["executedQty"])
            self.position.entry_time = order_result["transactTime"]

            # Calculate stop loss and take profit
            if atr_value is not None:
                self.position.entry_atr = Decimal(str(atr_value))
                stop_distance = (
                    self.config.stop_loss_atr_multiple * self.position.entry_atr
                )
                self.position.stop_loss_price = (
                    self.position.entry_price - stop_distance
                )
                self.position.take_profit_price = self.position.entry_price + (
                    stop_distance * Decimal("2")
                )  # 2:1 reward-to-risk

                self.logger.info(
                    f"Position opened: LONG {self.position.quantity} {self.config.base_asset} at {format_price(self.position.entry_price, self.symbol_info.price_precision)}"
                )
                self.logger.info(
                    f"Stop loss set at {format_price(self.position.stop_loss_price, self.symbol_info.price_precision)} (ATR: {self.position.entry_atr})"
                )
                self.logger.info(
                    f"Take profit set at {format_price(self.position.take_profit_price, self.symbol_info.price_precision)}"
                )
            else:
                self.logger.warning(
                    "ATR value not provided. Stop loss and take profit not set."
                )

            save_position_state(self.position, self.config.symbol)

        elif side == "SELL" and not self.position.active:
            # Opening a short position
            self.position.active = True
            self.position.side = "SELL"
            self.position.entry_price = Decimal(order_result["avgPrice"])
            self.position.quantity = Decimal(order_result["executedQty"])
            self.position.entry_time = order_result["transactTime"]

            # Calculate stop loss and take profit
            if atr_value is not None:
                self.position.entry_atr = Decimal(str(atr_value))
                stop_distance = (
                    self.config.stop_loss_atr_multiple * self.position.entry_atr
                )
                self.position.stop_loss_price = (
                    self.position.entry_price + stop_distance
                )
                self.position.take_profit_price = self.position.entry_price - (
                    stop_distance * Decimal("2")
                )  # 2:1 reward-to-risk

                self.logger.info(
                    f"Position opened: SHORT {self.position.quantity} {self.config.base_asset} at {format_price(self.position.entry_price, self.symbol_info.price_precision)}"
                )
                self.logger.info(
                    f"Stop loss set at {format_price(self.position.stop_loss_price, self.symbol_info.price_precision)} (ATR: {self.position.entry_atr})"
                )
                self.logger.info(
                    f"Take profit set at {format_price(self.position.take_profit_price, self.symbol_info.price_precision)}"
                )
            else:
                self.logger.warning(
                    "ATR value not provided. Stop loss and take profit not set."
                )

            save_position_state(self.position, self.config.symbol)

        elif side == "SELL" and self.position.active and self.position.side == "BUY":
            # Closing a long position

            # Calculate profit/loss
            exit_price = Decimal(order_result["avgPrice"])
            pnl, pnl_percent = calculate_pnl(
                entry_price=self.position.entry_price,
                exit_price=exit_price,
                position_quantity=self.position.quantity,
                position_side=self.position.side,
            )

            self.logger.info(
                f"Position closed: LONG {self.position.quantity} {self.config.base_asset}"
            )
            self.logger.info(
                f"Entry: {format_price(self.position.entry_price, self.symbol_info.price_precision)}, Exit: {format_price(exit_price, self.symbol_info.price_precision)}"
            )
            self.logger.info(
                f"P&L: {format_price(pnl, self.symbol_info.price_precision)} {self.config.quote_asset} ({pnl_percent:.2f}%)"
            )

            self.position.reset()
            save_position_state(self.position, self.config.symbol)

        elif side == "BUY" and self.position.active and self.position.side == "SELL":
            # Closing a short position

            # Calculate profit/loss
            exit_price = Decimal(order_result["avgPrice"])
            pnl, pnl_percent = calculate_pnl(
                entry_price=self.position.entry_price,
                exit_price=exit_price,
                position_quantity=self.position.quantity,
                position_side=self.position.side,
            )

            self.logger.info(
                f"Position closed: SHORT {self.position.quantity} {self.config.base_asset}"
            )
            self.logger.info(
                f"Entry: {format_price(self.position.entry_price, self.symbol_info.price_precision)}, Exit: {format_price(exit_price, self.symbol_info.price_precision)}"
            )
            self.logger.info(
                f"P&L: {format_price(pnl, self.symbol_info.price_precision)} {self.config.quote_asset} ({pnl_percent:.2f}%)"
            )

            self.position.reset()
            save_position_state(self.position, self.config.symbol)

        else:
            self.logger.warning(
                f"Unexpected order scenario: side={side}, position_active={self.position.active}, position_side={self.position.side}"
            )

    def check_and_execute_multi_timeframe_logic(self):
        """
        Implement the trading strategy logic.
        Optimized version that uses multi-timeframe analysis, pyramiding, and advanced exit strategies.
        """
        # Step 1: Check if we have an active position and manage exit if needed
        if self.position.is_active:
            self.manage_exit_strategy()
            # If position was closed during exit management, proceed to entry analysis
            if not self.position.is_active:
                self.logger.info(
                    "Position closed, now analyzing for new entry opportunities"
                )
            else:
                # Position still active, check for pyramiding opportunities
                return

        # Step 2: Analyze market for entry signals
        if self.config["use_multi_timeframe"]:
            # Use multi-timeframe analysis
            analysis = self.analyze_multi_timeframe(self.symbol)
            self.last_analysis = analysis  # Store for reference

            if not analysis["entry_signal"]:
                self.logger.info("No entry signal detected")
                return

            # Entry signal detected, apply filters
            signal_direction = analysis["signal_direction"]
            current_price = analysis["current_price"]
            atr = analysis["atr"]

            # Apply ADX filter if enabled
            if self.config["use_adx_filter"] and not analysis["adx_filter_passed"]:
                self.logger.info(
                    f"Entry signal detected but ADX filter failed. ADX: {analysis['adx_value']:.2f}"
                )
                return

            # Apply MA filter if enabled
            if self.config["use_ma_filter"] and not analysis["ma_filter_passed"]:
                self.logger.info("Entry signal detected but MA filter failed")
                return

            # All conditions met, execute entry or pyramid
            if not self.position.is_active:
                # New position
                self.logger.info(f"Entry signal confirmed for {signal_direction}")
                self._execute_entry(signal_direction, current_price, atr)
            else:
                # Check for pyramiding opportunity
                if signal_direction == self.position.side:
                    self.execute_pyramid_entry(signal_direction, current_price, atr)
        else:
            # Use simple Donchian Channel breakout for entries
            last_df = self.data_manager.get_historical_data(
                self.symbol, self.timeframe, limit=100
            )

            # Calculate indicators
            from bot.indicators import calculate_indicators, check_entry_signal

            calculate_indicators(last_df, self.config)

            # Check entry signals
            entry_long = check_entry_signal(last_df, "long")
            entry_short = check_entry_signal(last_df, "short")

            signal_direction = None
            if entry_long:
                signal_direction = "long"
            elif entry_short:
                signal_direction = "short"

            if signal_direction:
                current_price = last_df["close"].iloc[-1]
                atr = last_df["atr"].iloc[-1]

                if not self.position.is_active:
                    # New position
                    self.logger.info(f"Entry signal confirmed for {signal_direction}")
                    self._execute_entry(signal_direction, current_price, atr)
                elif (
                    signal_direction == self.position.side
                    and self.config["use_pyramiding"]
                ):
                    # Pyramiding opportunity
                    self.execute_pyramid_entry(signal_direction, current_price, atr)

    def run(self) -> None:
        """
        Main bot execution loop
        """
        try:
            # Check if we have an active position and log it
            if self.position.active:
                self.logger.info("Bot started with active position:")
                self.log_position_state()

                # Check if we need to update stop loss and take profit
                if self.position.entry_atr == Decimal(
                    "0"
                ) or self.position.stop_loss_price == Decimal("0"):
                    self.logger.warning(
                        "Active position has no stop loss. Recalculating..."
                    )
                    # Fetch latest data and calculate ATR to set stop loss
                    lookback = self.config.atr_length + 10
                    df = self.exchange.fetch_historical_data(
                        self.config.symbol, self.config.timeframe, lookback
                    )
                    if df is not None and not df.empty:
                        df_with_indicators = calculate_indicators(
                            df=df,
                            dc_enter=self.config.dc_length_enter,
                            dc_exit=self.config.dc_length_exit,
                            atr_len=self.config.atr_length,
                            atr_smooth="RMA",
                        )
                        if (
                            df_with_indicators is not None
                            and not df_with_indicators.empty
                        ):
                            # Update ATR and stop loss
                            latest_atr = df_with_indicators["atr"].iloc[-1]
                            self.position.entry_atr = Decimal(str(latest_atr))

                            # Calculate stop loss and take profit
                            stop_loss, take_profit = calculate_stop_loss_take_profit(
                                entry_price=float(self.position.entry_price),
                                atr_value=float(self.position.entry_atr),
                                atr_multiple=float(self.config.stop_loss_atr_multiple),
                                position_side=self.position.side,
                            )

                            self.position.stop_loss_price = Decimal(str(stop_loss))
                            self.position.take_profit_price = Decimal(str(take_profit))

                            self.logger.info(
                                f"Updated stop loss to {format_price(self.position.stop_loss_price, self.symbol_info.price_precision)}"
                            )
                            self.logger.info(
                                f"Updated take profit to {format_price(self.position.take_profit_price, self.symbol_info.price_precision)}"
                            )
                            save_position_state(self.position, self.config.symbol)

            # Main bot loop
            self.logger.info("Starting main trading loop...")

            while True:
                try:
                    # Fetch latest data
                    lookback = (
                        max(
                            self.config.dc_length_enter,
                            self.config.dc_length_exit,
                            self.config.atr_length,
                        )
                        + 50
                    )

                    df = self.exchange.fetch_historical_data(
                        self.config.symbol, self.config.timeframe, lookback
                    )

                    if df is None or df.empty:
                        self.logger.error(
                            "Unable to fetch historical data. Skipping this iteration."
                        )
                        time.sleep(60)  # Wait 60 seconds before retrying
                        continue

                    # Calculate indicators
                    df_with_indicators = calculate_indicators(
                        df=df,
                        dc_enter=self.config.dc_length_enter,
                        dc_exit=self.config.dc_length_exit,
                        atr_len=self.config.atr_length,
                        atr_smooth="RMA",
                    )

                    if df_with_indicators is None or df_with_indicators.empty:
                        self.logger.error(
                            "Failed to calculate indicators. Skipping this iteration."
                        )
                        time.sleep(60)  # Wait 60 seconds before retrying
                        continue

                    # Get current market price
                    current_price = self.exchange.get_current_price(self.config.symbol)
                    if current_price:
                        self.logger.info(
                            f"Current market price: {format_price(current_price, self.symbol_info.price_precision)}"
                        )

                    # Check trading conditions and execute orders if needed
                    self.check_and_execute_multi_timeframe_logic()

                    # Log position state if active
                    if self.position.active:
                        self.log_position_state()

                    # Calculate time to sleep until next candle close
                    sleep_seconds = get_sleep_time(self.config.timeframe)
                    next_check_time = time.time() + sleep_seconds
                    self.logger.info(
                        f"Next check at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(next_check_time))}"
                    )
                    self.logger.info("Waiting for next candle close...")

                    # Sleep until next candle close
                    time.sleep(sleep_seconds)

                except KeyboardInterrupt:
                    self.logger.info("Bot stopped by user (Ctrl+C)")
                    break

                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    self.logger.error(traceback.format_exc())

                    # Wait before retrying to avoid tight error loops
                    self.logger.info("Waiting 2 minutes before retrying...")
                    time.sleep(120)

        except KeyboardInterrupt:
            self.logger.info("Bot initialization interrupted by user")

        except Exception as e:
            self.logger.error(f"Fatal error during bot initialization: {e}")
            self.logger.error(traceback.format_exc())

        finally:
            self.logger.info("==============================================")
            self.logger.info("          TURTLE TRADING BOT STOPPED         ")
            self.logger.info("==============================================")

    def run_trading_cycle(self):
        """Run a complete trading cycle."""
        self.logger.info(f"Running trading cycle for {self.symbol}")
        try:
            # Get latest market data
            self.update_market_data()
            # Execute trading logic
            self.check_and_execute_multi_timeframe_logic()
            # Save the updated state
            self.save_state()
        except Exception as e:
            self.logger.error(f"Error during trading cycle: {e}")

    def analyze_only(self):
        """Run analysis without executing trades (for testing)"""
        self.logger.info(f"Analyzing market for {self.symbol} (test mode)")
        try:
            # Get latest market data
            self.update_market_data()

            # Get analysis results
            if self.config["use_multi_timeframe"]:
                analysis = self.analyze_multi_timeframe(self.symbol)
                self.logger.info("Multi-timeframe analysis results:")
                self.logger.info(f"Trend strength (ADX): {analysis['adx_value']:.2f}")
                self.logger.info(f"Trend direction: {analysis['trend_direction']}")
                self.logger.info(f"Entry signal: {analysis['entry_signal']}")
                self.logger.info(f"Signal direction: {analysis['signal_direction']}")

                # Check filters
                if self.config["use_adx_filter"]:
                    adx_passed = analysis["adx_value"] > self.config["adx_threshold"]
                    self.logger.info(f"ADX filter passed: {adx_passed}")

                if self.config["use_ma_filter"]:
                    ma_passed = analysis["ma_filter_passed"]
                    self.logger.info(f"MA filter passed: {ma_passed}")
            else:
                # Get entry signals from standard Donchian Channel breakout
                last_df = self.data_manager.get_historical_data(
                    self.symbol, self.timeframe, limit=100
                )

                # Calculate indicators
                from bot.indicators import calculate_indicators, check_entry_signal

                calculate_indicators(last_df, self.config)

                entry_long = check_entry_signal(last_df, "long")
                entry_short = check_entry_signal(last_df, "short")

                self.logger.info(f"Current price: {last_df['close'].iloc[-1]:.2f}")
                self.logger.info(
                    f"Donchian Upper Band: {last_df['dc_upper'].iloc[-1]:.2f}"
                )
                self.logger.info(
                    f"Donchian Lower Band: {last_df['dc_lower'].iloc[-1]:.2f}"
                )
                self.logger.info(f"ATR: {last_df['atr'].iloc[-1]:.2f}")
                self.logger.info(f"Entry signal long: {entry_long}")
                self.logger.info(f"Entry signal short: {entry_short}")

            # Check active position
            if self.position.is_active:
                self.logger.info(
                    f"Active position: {self.position.side} at {self.position.entry_price:.2f}"
                )
                self.logger.info(f"Current PnL: {self.calculate_unrealized_pnl():.2f}%")
                self.logger.info(f"Stop loss at: {self.position.stop_loss_price:.2f}")

                # Check exit conditions
                last_price = self.get_current_price()
                stop_triggered = self.check_stop_loss(last_price)
                exit_signal = self.check_exit_signal()

                self.logger.info(f"Stop loss triggered: {stop_triggered}")
                self.logger.info(f"Exit signal: {exit_signal}")

                if (
                    self.config["use_trailing_stop"]
                    and self.position.trailing_stop_price
                ):
                    self.logger.info(
                        f"Trailing stop at: {self.position.trailing_stop_price:.2f}"
                    )

                if self.config["use_partial_exits"]:
                    from bot.indicators import check_partial_exit

                    partial_exit = check_partial_exit(
                        last_price,
                        self.position.entry_price,
                        self.position.side,
                        self.position.atr_at_entry,
                        self.config["first_target_atr"],
                    )
                    self.logger.info(f"Partial exit signal: {partial_exit}")

        except Exception as e:
            self.logger.error(f"Error during market analysis: {e}")

    def run_backtest(self, days=30):
        """Run backtest for specified number of days"""
        self.logger.info(f"Starting backtest for {self.symbol} over {days} days")

        # Placeholder for backtest implementation
        # This would need to be implemented with historical data processing
        # and simulated trade execution
        self.logger.info("Backtesting not yet implemented")

    def analyze_multi_timeframe(self, symbol):
        """
        Perform multi-timeframe analysis to evaluate trends and entry conditions.

        Args:
            symbol: The trading symbol to analyze

        Returns:
            dict: Results of analysis including trend alignment and entry signals
        """
        self.logger.info(f"Performing multi-timeframe analysis for {symbol}")

        # Get historical data for trend timeframe (higher timeframe)
        trend_df = self.data_manager.get_historical_data(
            symbol, self.config["trend_timeframe"], limit=100
        )

        # Get historical data for entry timeframe (lower timeframe)
        entry_df = self.data_manager.get_historical_data(
            symbol, self.config["entry_timeframe"], limit=100
        )

        # Calculate indicators for both timeframes
        from bot.indicators import calculate_indicators, check_entry_signal
        from bot.indicators import check_adx_filter, check_ma_filter

        calculate_indicators(trend_df, self.config)
        calculate_indicators(entry_df, self.config)

        # Determine trend direction from higher timeframe
        if trend_df["close"].iloc[-1] > trend_df["ma"].iloc[-1]:
            trend_direction = "long"
        elif trend_df["close"].iloc[-1] < trend_df["ma"].iloc[-1]:
            trend_direction = "short"
        else:
            trend_direction = "neutral"

        # Check entry signals on entry timeframe
        entry_long = check_entry_signal(entry_df, "long")
        entry_short = check_entry_signal(entry_df, "short")

        # Check if entry aligns with trend
        entry_signal = False
        signal_direction = "none"

        if entry_long and (
            trend_direction == "long" or not self.config["trend_alignment_required"]
        ):
            entry_signal = True
            signal_direction = "long"
        elif entry_short and (
            trend_direction == "short" or not self.config["trend_alignment_required"]
        ):
            entry_signal = True
            signal_direction = "short"

        # Check ADX for trend strength
        adx_value = trend_df["adx"].iloc[-1]
        adx_passed = check_adx_filter(trend_df, self.config["adx_threshold"])

        # Check MA filter for price position relative to MA
        ma_passed = False
        if signal_direction != "none":
            ma_passed = check_ma_filter(entry_df, signal_direction)

        return {
            "trend_direction": trend_direction,
            "entry_signal": entry_signal,
            "signal_direction": signal_direction,
            "adx_value": adx_value,
            "adx_filter_passed": adx_passed,
            "ma_filter_passed": ma_passed,
            "current_price": entry_df["close"].iloc[-1],
            "atr": entry_df["atr"].iloc[-1],
        }

    def execute_pyramid_entry(self, direction, price, atr):
        """
        Implement pyramiding approach for gradual entry into positions.

        Args:
            direction: 'long' or 'short'
            price: Current market price
            atr: Current ATR value

        Returns:
            bool: Whether a pyramid entry was executed
        """
        if not self.config["use_pyramiding"]:
            return False

        # Check if we have an active position
        if not self.position.is_active:
            # No active position, can't pyramid
            return False

        # Check if direction matches current position
        if direction != self.position.side:
            return False

        # Check if we've reached maximum pyramid entries
        if self.position.entry_count >= self.config["pyramid_max_entries"]:
            self.logger.info(
                f"Maximum pyramid entries ({self.config['pyramid_max_entries']}) reached"
            )
            return False

        # Calculate time since last entry
        time_since_last_entry = time.time() - self.position.last_entry_time
        min_time_between_entries = 3600  # 1 hour minimum between entries

        if time_since_last_entry < min_time_between_entries:
            self.logger.info(
                f"Too soon for another pyramid entry (minimum time: {min_time_between_entries/3600:.1f}h)"
            )
            return False

        # Check if price has moved in our favor since last entry
        price_moved_enough = False

        if direction == "long" and price > self.position.entry_price * 1.005:
            price_moved_enough = True
        elif direction == "short" and price < self.position.entry_price * 0.995:
            price_moved_enough = True

        if not price_moved_enough:
            self.logger.info("Price hasn't moved enough for a pyramid entry")
            return False

        # All conditions met, execute pyramid entry
        self.logger.info(
            f"Executing pyramid entry #{self.position.entry_count + 1} for {direction}"
        )

        # Use smaller position size for pyramid entries
        if self.position.entry_count == 0:
            # First pyramid entry after initial entry
            size_factor = self.config["pyramid_size_first"]
        else:
            # Subsequent pyramid entries
            size_factor = self.config["pyramid_size_additional"]

        # Execute the entry
        success = self._execute_entry(direction, price, atr, size_factor=size_factor)

        if success:
            self.position.entry_count += 1
            self.position.last_entry_time = time.time()
            self.logger.info(
                f"Pyramid entry successful. Total entries: {self.position.entry_count}"
            )
            return True
        else:
            self.logger.warning("Pyramid entry failed")
            return False

    def _execute_entry(self, direction, price, atr, size_factor=1.0):
        """
        Handle the actual position opening process.

        Args:
            direction: 'long' or 'short'
            price: Entry price
            atr: Current ATR value
            size_factor: Factor to adjust position size (used for pyramiding)

        Returns:
            bool: Whether the entry was successful
        """
        try:
            # Calculate position size
            account_balance = self.exchange.get_balance(self.quote_asset)

            # Determine leverage based on trend alignment
            if self.config["use_multi_timeframe"] and hasattr(self, "last_analysis"):
                trend_direction = self.last_analysis.get("trend_direction", "neutral")

                if direction == trend_direction:
                    # Trade is in trend direction, use higher leverage
                    leverage = min(
                        self.config["max_leverage_trend"], self.config["leverage"]
                    )
                else:
                    # Counter-trend trade, use lower leverage
                    leverage = min(
                        self.config["max_leverage_counter"], self.config["leverage"]
                    )
            else:
                leverage = self.config["leverage"]

            # Calculate position size
            risk_amount = account_balance * self.config["risk_per_trade"] * size_factor
            stop_price = self.calculate_stop_loss_price(direction, price, atr)
            risk_per_unit = abs(price - stop_price)

            # Adjust for leverage
            position_size = (risk_amount / risk_per_unit) * leverage

            # Convert to asset quantity
            quantity = position_size / price

            # Check if this is the first entry or a pyramid entry
            if not self.position.is_active:
                # First entry - create new position
                self.position.side = direction
                self.position.entry_price = price
                self.position.quantity = quantity
                self.position.entry_time = time.time()
                self.position.stop_loss_price = stop_price
                self.position.atr_at_entry = atr
                self.position.is_active = True
                self.position.entry_count = 1
                self.position.last_entry_time = time.time()

                self.logger.info(
                    f"Opened new {direction} position at {price:.2f}, quantity: {quantity:.6f}"
                )

            else:
                # Pyramid entry - update existing position
                new_total_quantity = self.position.quantity + quantity
                new_avg_price = (
                    (self.position.entry_price * self.position.quantity)
                    + (price * quantity)
                ) / new_total_quantity

                self.position.entry_price = new_avg_price
                self.position.quantity = new_total_quantity

                # Keep the original stop loss for the combined position
                # We could adjust this if needed based on strategy requirements

                self.logger.info(
                    f"Added to {direction} position at {price:.2f}, "
                    f"new avg price: {new_avg_price:.2f}, "
                    f"new quantity: {new_total_quantity:.6f}"
                )

            # TODO: Set up trailing stop if enabled
            if self.config["use_trailing_stop"]:
                # Initialize trailing stop to None, will be set once in profit
                self.position.trailing_stop_price = None

            # Save position state
            self.save_state()
            return True

        except Exception as e:
            self.logger.error(f"Error executing entry: {e}")
            return False

    def manage_exit_strategy(self):
        """
        Manage position exits through various strategies.
        """
        if not self.position.is_active:
            return

        # Get current price
        current_price = self.get_current_price()

        # 1. Check stop loss
        if self.check_stop_loss(current_price):
            self._execute_full_exit(current_price, "Stop loss triggered")
            return

        # 2. Check for trailing stop if enabled
        if self.config["use_trailing_stop"]:
            # Check if we have a trailing stop price set
            if self.position.trailing_stop_price:
                if (
                    self.position.side == "long"
                    and current_price <= self.position.trailing_stop_price
                ) or (
                    self.position.side == "short"
                    and current_price >= self.position.trailing_stop_price
                ):
                    self._execute_full_exit(current_price, "Trailing stop triggered")
                    return

            # Check if we should set or update trailing stop
            self.update_trailing_stop(current_price)

        # 3. Check for partial exits if enabled
        if self.config["use_partial_exits"] and not self.position.partial_exits_done:
            from bot.indicators import check_partial_exit

            # First target (usually 50% of position)
            if not self.position.first_target_reached:
                first_target_hit = check_partial_exit(
                    current_price,
                    self.position.entry_price,
                    self.position.side,
                    self.position.atr_at_entry,
                    self.config["first_target_atr"],
                )

                if first_target_hit:
                    self._execute_partial_exit(
                        current_price, 0.5, "First target reached"
                    )
                    self.position.first_target_reached = True
                    return

            # Second target (usually 30% of original position)
            elif not self.position.second_target_reached:
                second_target_hit = check_partial_exit(
                    current_price,
                    self.position.entry_price,
                    self.position.side,
                    self.position.atr_at_entry,
                    self.config["second_target_atr"],
                )

                if second_target_hit:
                    self._execute_partial_exit(
                        current_price, 0.6, "Second target reached"
                    )  # 60% of remaining
                    self.position.second_target_reached = True
                    return

            # After second target, we leave the remaining 20% to be managed by trailing stop

        # 4. Check for exit signals from Donchian Channel
        if self.check_exit_signal():
            self._execute_full_exit(current_price, "Exit signal triggered")
            return

    def _execute_partial_exit(self, price, exit_percentage, reason=""):
        """
        Execute a partial exit from a position.

        Args:
            price: Current market price
            exit_percentage: Percentage of position to exit (0.0 to 1.0)
            reason: Reason for the exit
        """
        if not self.position.is_active:
            return False

        # Calculate exit quantity
        exit_quantity = self.position.quantity * exit_percentage

        try:
            # TODO: Execute the actual order via exchange
            # For now we'll simulate it

            # Calculate profit/loss
            if self.position.side == "long":
                pnl_percentage = (
                    (price - self.position.entry_price) / self.position.entry_price
                ) * 100
            else:  # short
                pnl_percentage = (
                    (self.position.entry_price - price) / self.position.entry_price
                ) * 100

            self.logger.info(
                f"Executed partial exit ({exit_percentage * 100:.0f}%) at {price:.2f}, "
                f"PnL: {pnl_percentage:.2f}%, Reason: {reason}"
            )

            # Update position
            remaining_quantity = self.position.quantity - exit_quantity
            self.position.quantity = remaining_quantity

            self.logger.info(
                f"Remaining position: {remaining_quantity:.6f} {self.base_asset}"
            )

            # If quantity very small, consider position closed
            if remaining_quantity * price < 5:  # Less than $5 worth
                self._execute_full_exit(price, "Remaining position too small")

            # Save updated state
            self.save_state()
            return True

        except Exception as e:
            self.logger.error(f"Error executing partial exit: {e}")
            return False

    def _execute_full_exit(self, price, reason=""):
        """
        Execute a full exit from a position.

        Args:
            price: Current market price
            reason: Reason for the exit
        """
        if not self.position.is_active:
            return False

        try:
            # TODO: Execute the actual order via exchange
            # For now we'll simulate it

            # Calculate profit/loss
            if self.position.side == "long":
                pnl_percentage = (
                    (price - self.position.entry_price) / self.position.entry_price
                ) * 100
            else:  # short
                pnl_percentage = (
                    (self.position.entry_price - price) / self.position.entry_price
                ) * 100

            # Calculate trade duration
            duration_seconds = time.time() - self.position.entry_time
            duration_hours = duration_seconds / 3600

            self.logger.info(
                f"Closed {self.position.side} position at {price:.2f}, "
                f"PnL: {pnl_percentage:.2f}%, "
                f"Duration: {duration_hours:.1f}h, "
                f"Reason: {reason}"
            )

            # Reset position state
            self.position.is_active = False
            self.position.quantity = 0
            self.position.side = None
            self.position.entry_price = 0
            self.position.stop_loss_price = 0
            self.position.trailing_stop_price = None
            self.position.atr_at_entry = 0
            self.position.entry_count = 0
            self.position.last_entry_time = 0
            self.position.first_target_reached = False
            self.position.second_target_reached = False

            # Save updated state
            self.save_state()
            return True

        except Exception as e:
            self.logger.error(f"Error executing full exit: {e}")
            return False

    def update_trailing_stop(self, current_price):
        """
        Update the trailing stop price once a position is in profit.

        Args:
            current_price: Current market price
        """
        if not self.position.is_active or not self.config["use_trailing_stop"]:
            return

        # Calculate profit threshold
        profit_threshold = self.config["profit_for_trailing"]

        # Check if position is in sufficient profit to activate trailing stop
        in_profit = False

        if self.position.side == "long":
            profit_pct = (
                current_price - self.position.entry_price
            ) / self.position.entry_price
            in_profit = profit_pct > profit_threshold
        else:  # short
            profit_pct = (
                self.position.entry_price - current_price
            ) / self.position.entry_price
            in_profit = profit_pct > profit_threshold

        if not in_profit:
            return

        # Calculate new trailing stop price
        from bot.indicators import update_trailing_stop

        new_stop = update_trailing_stop(
            current_price,
            self.position.side,
            self.position.atr_at_entry,
            self.position.trailing_stop_price,
        )

        # Update trailing stop if it's more favorable
        if self.position.trailing_stop_price is None:
            self.position.trailing_stop_price = new_stop
            self.logger.info(f"Activated trailing stop at {new_stop:.2f}")
        else:
            if (
                self.position.side == "long"
                and new_stop > self.position.trailing_stop_price
            ) or (
                self.position.side == "short"
                and new_stop < self.position.trailing_stop_price
            ):
                old_stop = self.position.trailing_stop_price
                self.position.trailing_stop_price = new_stop
                self.logger.info(
                    f"Updated trailing stop from {old_stop:.2f} to {new_stop:.2f}"
                )

    def update_market_data(self):
        """
        Update market data for current trading symbol
        """
        try:
            # Fetch latest market data
            lookback = (
                max(
                    self.config.dc_length_enter,
                    self.config.dc_length_exit,
                    self.config.atr_length,
                )
                + 50
            )

            # Fetch data for all required timeframes
            if self.config.use_multi_timeframe:
                # Get data for trend timeframe
                self.trend_data = self.exchange.fetch_historical_data(
                    self.config.symbol, self.config.trend_timeframe, lookback
                )

                # Get data for entry timeframe
                self.entry_data = self.exchange.fetch_historical_data(
                    self.config.symbol, self.config.entry_timeframe, lookback
                )

                # Get data for base timeframe
                self.market_data = self.exchange.fetch_historical_data(
                    self.config.symbol, self.config.timeframe, lookback
                )
            else:
                # Just get base timeframe data
                self.market_data = self.exchange.fetch_historical_data(
                    self.config.symbol, self.config.timeframe, lookback
                )

            # Update current price
            self.current_price = self.exchange.get_current_price(self.config.symbol)

            return True
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            return False
