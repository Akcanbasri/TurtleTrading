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
        try:
            # Use fetch_historical_data instead of get_historical_data
            data = self.exchange.fetch_historical_data(symbol, timeframe, limit)
        except Exception as e:
            # For demo/test mode, create synthetic data if exchange methods fail
            import pandas as pd
            import numpy as np

            # Log the error but continue with synthetic data
            logging.getLogger("turtle_trading_bot").warning(
                f"Error fetching data from exchange: {e}. Generating synthetic data for {symbol} - {limit} candles"
            )

            # Create timestamps
            end_date = pd.Timestamp.now()

            # Adjust timeframe to pandas frequency
            freq = "1h"  # Default
            if timeframe == "1m":
                freq = "1min"
            elif timeframe == "5m":
                freq = "5min"
            elif timeframe == "15m":
                freq = "15min"
            elif timeframe == "30m":
                freq = "30min"
            elif timeframe == "1h":
                freq = "1h"
            elif timeframe == "4h":
                freq = "4h"
            elif timeframe == "1d":
                freq = "1d"

            dates = pd.date_range(end=end_date, periods=limit, freq=freq)

            # Generate price data with random walk
            np.random.seed(42)  # For reproducibility

            # Base price for the asset (e.g., BTC ~30000)
            if "BTC" in symbol:
                base_price = 30000.0
            elif "ETH" in symbol:
                base_price = 2000.0
            else:
                base_price = 100.0

            # Generate noise and trend
            noise = np.random.normal(
                0, base_price * 0.01, limit
            ).cumsum()  # 1% daily volatility
            trend = np.linspace(0, base_price * 0.1, limit)  # 10% trend over the period
            close_prices = base_price + trend + noise

            # Create realistic OHLCV data
            open_prices = np.roll(close_prices, 1)  # Previous close is today's open
            open_prices[0] = close_prices[0] * (
                1 + np.random.normal(0, 0.01)
            )  # Random first open

            # Add some noise to create high/low
            high_prices = np.maximum(close_prices, open_prices) + np.abs(
                np.random.normal(0, base_price * 0.005, limit)
            )

            low_prices = np.minimum(close_prices, open_prices) - np.abs(
                np.random.normal(0, base_price * 0.005, limit)
            )

            # Generate volume with some relationship to price movement
            price_change = np.abs(close_prices - open_prices)
            volume = np.abs(
                price_change * 50 + np.random.normal(0, base_price * 0.1, limit)
            )

            # Create DataFrame
            data = pd.DataFrame(
                {
                    "open": open_prices,
                    "high": high_prices,
                    "low": low_prices,
                    "close": close_prices,
                    "volume": volume,
                },
                index=dates,
            )

            logging.getLogger("turtle_trading_bot").info(
                f"Successfully generated {limit} synthetic candles for {symbol}"
            )

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

    def _execute_entry(
        self,
        direction: str,
        current_price: Union[float, Decimal],
        atr_value: Union[float, Decimal],
        pyramid_level: int = 0,
    ) -> bool:
        """
        Execute a position entry with dynamic leverage based on account balance

        Parameters
        ----------
        direction : str
            Trade direction ('long' or 'short')
        current_price : Union[float, Decimal]
            Current market price
        atr_value : Union[float, Decimal]
            Current ATR value
        pyramid_level : int
            Pyramid level (0 for first entry, 1+ for additional entries)

        Returns
        -------
        bool
            Whether the entry was successful
        """
        try:
            # Convert to proper types
            current_price_dec = (
                Decimal(str(current_price))
                if not isinstance(current_price, Decimal)
                else current_price
            )

            # Get available balance
            available_balance = self.exchange.get_account_balance(self.quote_asset)
            if available_balance is None or available_balance <= Decimal("0"):
                self.logger.error(f"Invalid available balance: {available_balance}")
                return False

            self.logger.info(
                f"Available balance: {available_balance} {self.quote_asset}"
            )

            # Map direction to order side
            order_side = "BUY" if direction == "long" else "SELL"

            # Determine if trade is aligned with trend
            is_trend_aligned = True
            if hasattr(self, "last_analysis") and self.last_analysis:
                is_trend_aligned = self.last_analysis.get("trend_aligned", True)

            # Calculate minimum position size based on account balance
            min_position_value = Decimal("5")  # Minimum position size in USDT

            # Set dynamic leverage based on account balance
            # Lower leverage for smaller accounts to limit risk
            target_leverage = 1  # Default leverage

            if available_balance <= Decimal("20"):
                # For accounts with less than 20 USDT, max leverage is 5x
                target_leverage = min(5, int(self.config.leverage))
                self.logger.info(
                    f"Small account detected (<20 USDT), limiting leverage to 5x"
                )
            elif available_balance <= Decimal("50"):
                # For accounts with less than 50 USDT, max leverage is 7x
                target_leverage = min(7, int(self.config.leverage))
                self.logger.info(
                    f"Medium account detected (<50 USDT), limiting leverage to 7x"
                )
            elif available_balance <= Decimal("100"):
                # For accounts with less than 100 USDT, max leverage is 10x
                target_leverage = min(10, int(self.config.leverage))
                self.logger.info(
                    f"Standard account detected (<100 USDT), limiting leverage to 10x"
                )
            else:
                # For larger accounts, use the configured leverage
                target_leverage = int(self.config.leverage)
                self.logger.info(
                    f"Large account detected (>100 USDT), using configured leverage {target_leverage}x"
                )

            # For trend trades, we can use higher leverage
            if is_trend_aligned:
                actual_leverage = min(
                    target_leverage, int(self.config.max_leverage_trend)
                )
            else:
                # For counter-trend trades, we use lower leverage
                actual_leverage = min(
                    target_leverage, int(self.config.max_leverage_counter)
                )

            # Set leverage on exchange
            self.logger.info(
                f"Setting leverage to {actual_leverage}x for {self.symbol}"
            )
            leverage_set = self.exchange.set_leverage(self.symbol, actual_leverage)

            if not leverage_set:
                self.logger.warning(
                    f"Failed to set leverage. Using default leverage from exchange."
                )

            # Calculate position size based on risk management
            position_size, status = calculate_position_size(
                available_balance=available_balance,
                risk_percent=Decimal(str(self.config.risk_per_trade)),
                atr_value=atr_value,
                current_price=current_price_dec,
                symbol_info=self.symbol_info.__dict__,
                max_risk_percentage=Decimal(str(self.config.max_risk_percentage)),
                leverage=actual_leverage,
                position_side=order_side,
                pyramid_level=pyramid_level,
                pyramid_size_first=Decimal(str(self.config.pyramid_size_first)),
                pyramid_size_additional=Decimal(
                    str(self.config.pyramid_size_additional)
                ),
                is_trend_aligned=is_trend_aligned,
            )

            if status != "success" or position_size <= Decimal("0"):
                self.logger.error(f"Position sizing failed: {status}")
                return False

            # Check if position value meets minimum requirements
            position_value = position_size * current_price_dec
            if position_value < min_position_value:
                self.logger.warning(
                    f"Position value {position_value} is below minimum {min_position_value}. "
                    f"Consider increasing risk percentage or account size."
                )
                # Try to adjust position size to meet minimum value
                adjusted_size = min_position_value / current_price_dec
                if (
                    adjusted_size > Decimal("0")
                    and adjusted_size
                    * current_price_dec
                    / Decimal(str(actual_leverage))
                    <= available_balance
                ):
                    position_size = adjusted_size
                    self.logger.info(
                        f"Adjusted position size to meet minimum value: {position_size}"
                    )
                else:
                    self.logger.error(
                        f"Cannot meet minimum position size with current balance"
                    )
                    return False

            # Execute the order
            self.logger.info(
                f"Executing {order_side} order for {position_size} {self.base_asset} at ~{format_price(current_price_dec, self.symbol_info.price_precision)}"
            )

            success, order_result = self.exchange.execute_order(
                symbol=self.symbol,
                side=order_side,
                quantity=position_size,
                order_type="MARKET",
            )

            if not success:
                self.logger.error(f"Order execution failed: {order_result}")
                return False

            # Update position state
            self.update_position_state(order_result, order_side, float(atr_value))

            # Update position entry count and time for pyramiding
            self.position.entry_count += 1
            self.position.last_entry_time = time.time()

            return True

        except Exception as e:
            self.logger.error(f"Error executing entry: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def manage_exit_strategy(self) -> bool:
        """
        Manage position exit strategies including stop loss, trailing stop, and take profits

        Returns
        -------
        bool
            Whether an exit action was taken
        """
        if not self.position.active:
            return False

        try:
            # Get current price
            current_price = self.get_current_price()
            if current_price is None:
                self.logger.error("Failed to get current price for exit strategy")
                return False

            # 1. Check stop loss
            stop_triggered = self.check_stop_loss(current_price)
            if stop_triggered:
                self.logger.info(
                    f"Stop loss triggered at {current_price} (SL: {self.position.stop_loss_price})"
                )
                return self._execute_exit("Stop Loss")

            # 2. Check for trailing stop
            if (
                self.config.use_trailing_stop
                and self.position.trailing_stop_price is not None
            ):
                trailing_stop_triggered = False

                if (
                    self.position.side == "BUY"
                    and current_price <= self.position.trailing_stop_price
                ):
                    trailing_stop_triggered = True
                elif (
                    self.position.side == "SELL"
                    and current_price >= self.position.trailing_stop_price
                ):
                    trailing_stop_triggered = True

                if trailing_stop_triggered:
                    self.logger.info(
                        f"Trailing stop triggered at {current_price} (TS: {self.position.trailing_stop_price})"
                    )
                    return self._execute_exit("Trailing Stop")

            # 3. Check if we need to activate or update trailing stop
            if self.config.use_trailing_stop:
                # Calculate price movement since entry
                profit_percent = self.calculate_unrealized_pnl()

                # If profit exceeds threshold, activate or update trailing stop
                if profit_percent >= self.config.profit_for_trailing:
                    # Calculate trailing stop price
                    atr_value = self.position.entry_atr
                    trailing_distance = (
                        atr_value / 2
                    )  # Use half ATR for trailing stop distance

                    if self.position.side == "BUY":
                        new_trailing_stop = current_price - trailing_distance
                        # Only update if new trailing stop is higher
                        if (
                            self.position.trailing_stop_price is None
                            or new_trailing_stop > self.position.trailing_stop_price
                        ):
                            self.position.trailing_stop_price = new_trailing_stop
                            self.logger.info(
                                f"Updated trailing stop to {self.position.trailing_stop_price}"
                            )
                            save_position_state(self.position, self.symbol)
                    else:  # SELL position
                        new_trailing_stop = current_price + trailing_distance
                        # Only update if new trailing stop is lower
                        if (
                            self.position.trailing_stop_price is None
                            or new_trailing_stop < self.position.trailing_stop_price
                        ):
                            self.position.trailing_stop_price = new_trailing_stop
                            self.logger.info(
                                f"Updated trailing stop to {self.position.trailing_stop_price}"
                            )
                            save_position_state(self.position, self.symbol)

            # 4. Check for take profit targets (partial exits)
            if self.config.use_partial_exits and not self.position.partial_exit_taken:
                # Calculate profit in ATR multiples
                price_movement = abs(current_price - self.position.entry_price)
                atr_movement = price_movement / self.position.entry_atr

                # Check first take profit target
                if atr_movement >= self.config.first_target_atr:
                    self.logger.info(
                        f"First take profit target reached at {current_price} ({atr_movement:.2f} ATR)"
                    )
                    # Take 50% off the position
                    self.position.partial_exit_taken = True
                    save_position_state(self.position, self.symbol)
                    return self._execute_partial_exit(0.5, "First Target")

            # 5. Check for exit signal based on system rules
            exit_signal = self.check_exit_signal()
            if exit_signal:
                self.logger.info(f"Exit signal triggered at {current_price}")
                return self._execute_exit("Exit Signal")

            return False

        except Exception as e:
            self.logger.error(f"Error in exit strategy management: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def _execute_exit(self, reason: str) -> bool:
        """
        Execute a full position exit

        Parameters
        ----------
        reason : str
            Reason for the exit

        Returns
        -------
        bool
            Whether the exit was successful
        """
        if not self.position.active:
            return False

        try:
            # Determine the exit side opposite to position side
            exit_side = "SELL" if self.position.side == "BUY" else "BUY"

            self.logger.info(
                f"Executing {exit_side} order to close position ({reason})"
            )

            # Execute the order
            success, order_result = self.exchange.execute_order(
                symbol=self.symbol,
                side=exit_side,
                quantity=self.position.quantity,
                order_type="MARKET",
            )

            if not success:
                self.logger.error(f"Exit order execution failed: {order_result}")
                return False

            # Calculate PnL
            exit_price = Decimal(order_result["avgPrice"])
            pnl, pnl_percent = calculate_pnl(
                entry_price=self.position.entry_price,
                exit_price=exit_price,
                position_quantity=self.position.quantity,
                position_side=self.position.side,
            )

            self.logger.info(
                f"Position closed: {self.position.side} {self.position.quantity} {self.base_asset}"
            )
            self.logger.info(
                f"Entry: {format_price(self.position.entry_price, self.symbol_info.price_precision)}, "
                f"Exit: {format_price(exit_price, self.symbol_info.price_precision)}"
            )
            self.logger.info(
                f"P&L: {format_price(pnl, self.symbol_info.price_precision)} {self.quote_asset} ({pnl_percent:.2f}%)"
            )
            self.logger.info(f"Exit reason: {reason}")

            # Reset position
            self.position.reset()
            save_position_state(self.position, self.symbol)

            return True

        except Exception as e:
            self.logger.error(f"Error executing exit: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def _execute_partial_exit(self, exit_portion: float, reason: str) -> bool:
        """
        Execute a partial position exit

        Parameters
        ----------
        exit_portion : float
            Portion of the position to exit (0.0 to 1.0)
        reason : str
            Reason for the partial exit

        Returns
        -------
        bool
            Whether the partial exit was successful
        """
        if not self.position.active or exit_portion <= 0 or exit_portion >= 1:
            return False

        try:
            # Calculate the quantity to exit
            exit_quantity = self.position.quantity * Decimal(str(exit_portion))

            # Ensure it meets minimum requirements
            if exit_quantity < self.symbol_info.min_qty:
                self.logger.warning(
                    f"Partial exit quantity {exit_quantity} is below minimum {self.symbol_info.min_qty}. "
                    f"Skipping partial exit."
                )
                return False

            # Round to the correct precision
            from bot.utils import format_quantity

            exit_quantity = Decimal(
                format_quantity(exit_quantity, self.symbol_info.quantity_precision)
            )

            # Determine the exit side opposite to position side
            exit_side = "SELL" if self.position.side == "BUY" else "BUY"

            self.logger.info(
                f"Executing partial {exit_side} for {exit_quantity} {self.base_asset} ({exit_portion*100:.1f}% of position)"
            )

            # Execute the order
            success, order_result = self.exchange.execute_order(
                symbol=self.symbol,
                side=exit_side,
                quantity=exit_quantity,
                order_type="MARKET",
            )

            if not success:
                self.logger.error(
                    f"Partial exit order execution failed: {order_result}"
                )
                return False

            # Calculate PnL for the exited portion
            exit_price = Decimal(order_result["avgPrice"])
            pnl, pnl_percent = calculate_pnl(
                entry_price=self.position.entry_price,
                exit_price=exit_price,
                position_quantity=exit_quantity,
                position_side=self.position.side,
            )

            self.logger.info(
                f"Partial position closed: {exit_quantity} {self.base_asset} of {self.position.quantity} total"
            )
            self.logger.info(
                f"Entry: {format_price(self.position.entry_price, self.symbol_info.price_precision)}, "
                f"Exit: {format_price(exit_price, self.symbol_info.price_precision)}"
            )
            self.logger.info(
                f"P&L: {format_price(pnl, self.symbol_info.price_precision)} {self.quote_asset} ({pnl_percent:.2f}%)"
            )
            self.logger.info(f"Partial exit reason: {reason}")

            # Update position quantity
            self.position.quantity -= exit_quantity
            save_position_state(self.position, self.symbol)

            return True

        except Exception as e:
            self.logger.error(f"Error executing partial exit: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def check_and_execute_multi_timeframe_logic(self):
        """
        Implement the trading strategy logic.
        Optimized version that uses multi-timeframe analysis, pyramiding, and advanced exit strategies.
        """
        # Step 1: Check if we have an active position and manage exit if needed
        if self.position.active:
            self.manage_exit_strategy()
            # If position was closed during exit management, proceed to entry analysis
            if not self.position.active:
                self.logger.info(
                    "Position closed, now analyzing for new entry opportunities"
                )
            else:
                # Position still active, check for pyramiding opportunities
                return

        # Step 2: Analyze market for entry signals
        if self.config.use_multi_timeframe:
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
            if self.config.use_adx_filter and not analysis["adx_filter_passed"]:
                self.logger.info(
                    f"Entry signal detected but ADX filter failed. ADX: {analysis['adx_value']:.2f}"
                )
                return

            # Apply MA filter if enabled
            if self.config.use_ma_filter and not analysis["ma_filter_passed"]:
                self.logger.info("Entry signal detected but MA filter failed")
                return

            # All conditions met, execute entry or pyramid
            if not self.position.active:
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

                if not self.position.active:
                    # New position
                    self.logger.info(f"Entry signal confirmed for {signal_direction}")
                    self._execute_entry(signal_direction, current_price, atr)
                elif (
                    signal_direction == self.position.side
                    and self.config.use_pyramiding
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
            if self.config.use_multi_timeframe:
                analysis = self.analyze_multi_timeframe(self.symbol)
                self.logger.info("Multi-timeframe analysis results:")
                self.logger.info(f"Trend strength (ADX): {analysis['adx_value']:.2f}")
                self.logger.info(f"Trend direction: {analysis['trend_direction']}")
                self.logger.info(f"Entry signal: {analysis['entry_signal']}")
                self.logger.info(f"Signal direction: {analysis['signal_direction']}")

                # Check filters
                if self.config.use_adx_filter:
                    adx_passed = analysis["adx_value"] > self.config.adx_threshold
                    self.logger.info(f"ADX filter passed: {adx_passed}")

                if self.config.use_ma_filter:
                    ma_passed = analysis["ma_filter_passed"]
                    self.logger.info(f"MA filter passed: {ma_passed}")
            else:
                # Get entry signals from standard Donchian Channel breakout
                last_df = self.data_manager.get_historical_data(
                    self.symbol, self.timeframe, limit=100
                )

                # Calculate indicators
                from bot.indicators import calculate_indicators, check_entry_signal

                # Calculate indicators with proper parameters
                last_df = calculate_indicators(
                    df=last_df,
                    dc_enter=self.config.dc_length_enter,
                    dc_exit=self.config.dc_length_exit,
                    atr_len=self.config.atr_length,
                    atr_smooth=self.config.atr_smoothing,
                    ma_period=self.config.ma_period,
                    adx_period=self.config.adx_period,
                )

                entry_long = check_entry_signal(last_df.iloc[-1], "long")
                entry_short = check_entry_signal(last_df.iloc[-1], "short")

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
            if self.position.active:
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

                if self.config.use_trailing_stop and self.position.trailing_stop_price:
                    self.logger.info(
                        f"Trailing stop at: {self.position.trailing_stop_price:.2f}"
                    )

                if self.config.use_partial_exits:
                    from bot.indicators import check_partial_exit

                    partial_exit = check_partial_exit(
                        last_price,
                        self.position.entry_price,
                        self.position.entry_atr,
                        self.position.side,
                        self.config.first_target_atr,
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

        try:
            # Get historical data for trend timeframe (higher timeframe)
            trend_df = self.data_manager.get_historical_data(
                symbol, self.config.trend_timeframe, limit=100
            )

            # Get historical data for entry timeframe (lower timeframe)
            entry_df = self.data_manager.get_historical_data(
                symbol, self.config.entry_timeframe, limit=100
            )

            # Calculate indicators for both timeframes
            from bot.indicators import calculate_indicators, check_entry_signal
            from bot.indicators import check_adx_filter, check_ma_filter

            # Make sure the DataFrames have the required columns
            if trend_df is None or trend_df.empty:
                self.logger.warning(f"No trend data available for {symbol}")
                trend_df = self._create_default_dataframe()

            if entry_df is None or entry_df.empty:
                self.logger.warning(f"No entry data available for {symbol}")
                entry_df = self._create_default_dataframe()

            # Calculate indicators with proper parameters
            trend_df = calculate_indicators(
                df=trend_df,
                dc_enter=self.config.dc_length_enter,
                dc_exit=self.config.dc_length_exit,
                atr_len=self.config.atr_length,
                atr_smooth=self.config.atr_smoothing,
                ma_period=self.config.ma_period,
                adx_period=self.config.adx_period,
            )

            entry_df = calculate_indicators(
                df=entry_df,
                dc_enter=self.config.dc_length_enter,
                dc_exit=self.config.dc_length_exit,
                atr_len=self.config.atr_length,
                atr_smooth=self.config.atr_smoothing,
                ma_period=self.config.ma_period,
                adx_period=self.config.adx_period,
            )

            # Ensure 'ma' column exists
            if "ma" not in trend_df.columns:
                trend_df["ma"] = trend_df["close"].rolling(self.config.ma_period).mean()

            # Ensure 'adx' column exists
            if "adx" not in trend_df.columns:
                trend_df["adx"] = 25.0  # Default ADX value

            # Determine trend direction from higher timeframe
            if trend_df["close"].iloc[-1] > trend_df["ma"].iloc[-1]:
                trend_direction = "long"
            elif trend_df["close"].iloc[-1] < trend_df["ma"].iloc[-1]:
                trend_direction = "short"
            else:
                trend_direction = "neutral"

            # Check entry signals on entry timeframe
            try:
                entry_long = check_entry_signal(entry_df.iloc[-1], "long")
                entry_short = check_entry_signal(entry_df.iloc[-1], "short")
            except Exception as e:
                self.logger.error(f"Error checking entry signals: {e}")
                entry_long = False
                entry_short = False

            # Check if entry aligns with trend
            entry_signal = False
            signal_direction = "none"

            if entry_long and (
                trend_direction == "long" or not self.config.trend_alignment_required
            ):
                entry_signal = True
                signal_direction = "long"
            elif entry_short and (
                trend_direction == "short" or not self.config.trend_alignment_required
            ):
                entry_signal = True
                signal_direction = "short"

            # Check ADX for trend strength
            try:
                adx_value = trend_df["adx"].iloc[-1]
                adx_passed = check_adx_filter(
                    trend_df.iloc[-1], self.config.adx_threshold
                )
            except Exception as e:
                self.logger.error(f"Error checking ADX: {e}")
                adx_value = 0.0
                adx_passed = False

            # Check MA filter for price position relative to MA
            ma_passed = False
            if signal_direction != "none":
                try:
                    ma_passed = check_ma_filter(entry_df.iloc[-1], signal_direction)
                except Exception as e:
                    self.logger.error(f"Error checking MA filter: {e}")

            return {
                "trend_direction": trend_direction,
                "entry_signal": entry_signal,
                "signal_direction": signal_direction,
                "adx_value": adx_value,
                "adx_filter_passed": adx_passed,
                "ma_filter_passed": ma_passed,
                "current_price": entry_df["close"].iloc[-1],
                "atr": entry_df["atr"].iloc[-1] if "atr" in entry_df.columns else 0.0,
            }
        except Exception as e:
            self.logger.error(f"Error during multi-timeframe analysis: {e}")
            # Return default values in case of error
            return {
                "trend_direction": "neutral",
                "entry_signal": False,
                "signal_direction": "none",
                "adx_value": 0.0,
                "adx_filter_passed": False,
                "ma_filter_passed": False,
                "current_price": 30000.0,  # Default price for BTC
                "atr": 1000.0,  # Default ATR for BTC
            }

    def _create_default_dataframe(self):
        """Create a default DataFrame with basic OHLCV data for testing"""
        import numpy as np
        import pandas as pd
        from bot.indicators import calculate_indicators

        # Create timestamps
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="1h")

        # Create synthetic data
        base_price = 30000.0  # Default for BTC

        # Generate price data with some random walk
        np.random.seed(42)
        noise = np.random.normal(0, 500, 100).cumsum()
        trend = np.linspace(base_price * 0.8, base_price, 100)
        close_prices = trend + noise

        # Generate OHLCV data
        data = {
            "open": close_prices - np.random.randn(100) * 100,
            "high": close_prices + np.abs(np.random.randn(100) * 200),
            "low": close_prices - np.abs(np.random.randn(100) * 200),
            "close": close_prices,
            "volume": np.abs(np.random.randn(100) * 10 + 50),
        }

        df = pd.DataFrame(data, index=dates)

        # Calculate all indicators with proper parameters
        df = calculate_indicators(
            df=df,
            dc_enter=self.config.dc_length_enter,
            dc_exit=self.config.dc_length_exit,
            atr_len=self.config.atr_length,
            atr_smooth=self.config.atr_smoothing,
            ma_period=self.config.ma_period,
            adx_period=self.config.adx_period,
        )

        return df

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
        if not self.config.use_pyramiding:
            return False

        # Check if we have an active position
        if not self.position.active:
            # No active position, can't pyramid
            return False

        # Check if direction matches current position
        if direction != self.position.side:
            return False

        # Check if we've reached maximum pyramid entries
        if self.position.entry_count >= self.config.pyramid_max_entries:
            self.logger.info(
                f"Maximum pyramid entries ({self.config.pyramid_max_entries}) reached"
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

        # Execute the pyramid entry
        return self._execute_entry(
            direction=direction,
            current_price=price,
            atr_value=atr,
            pyramid_level=self.position.entry_count,
        )

    def check_exit_signal(self):
        """Check for exit signals based on Donchian Channel breakout"""
        try:
            # Get latest data
            df = self.data_manager.get_historical_data(
                self.symbol, self.timeframe, limit=100
            )

            # Calculate indicators if not already present
            if "dc_upper" not in df.columns or "dc_lower" not in df.columns:
                from bot.indicators import calculate_indicators

                df = calculate_indicators(
                    df=df,
                    dc_enter=self.config.dc_length_enter,
                    dc_exit=self.config.dc_length_exit,
                    atr_len=self.config.atr_length,
                    atr_smooth=self.config.atr_smoothing,
                    ma_period=self.config.ma_period,
                    adx_period=self.config.adx_period,
                )

            from bot.indicators import check_exit_signal

            # Check the last row for exit signal
            return check_exit_signal(df.iloc[-1], self.position.side)
        except Exception as e:
            self.logger.error(f"Error checking exit signal: {e}")
            return False

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
                self.trend_data = self.data_manager.get_historical_data(
                    self.symbol, self.config.trend_timeframe, lookback
                )

                # Get data for entry timeframe
                self.entry_data = self.data_manager.get_historical_data(
                    self.symbol, self.config.entry_timeframe, lookback
                )

                # Get data for base timeframe
                self.market_data = self.data_manager.get_historical_data(
                    self.symbol, self.config.timeframe, lookback
                )
            else:
                # Just get base timeframe data
                self.market_data = self.data_manager.get_historical_data(
                    self.symbol, self.config.timeframe, lookback
                )

            # Update current price
            self.current_price = self.get_current_price()

            return True
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            return False

    def get_current_price(self):
        """Get current price of the trading symbol"""
        try:
            return self.exchange.get_current_price(self.symbol)
        except Exception as e:
            self.logger.warning(f"Error getting current price: {e}")
            # Return last price from market data if available
            if (
                hasattr(self, "market_data")
                and self.market_data is not None
                and not self.market_data.empty
            ):
                return self.market_data["close"].iloc[-1]
            # Return default price for demo mode
            return 30000.0  # Default BTC price

    def calculate_unrealized_pnl(self):
        """Calculate unrealized profit/loss percentage for current position"""
        if not self.position.active:
            return 0.0

        try:
            current_price = self.get_current_price()

            if self.position.side == "BUY" or self.position.side == "long":
                pnl_percent = ((current_price / self.position.entry_price) - 1.0) * 100
            else:  # SELL or short
                pnl_percent = (1.0 - (current_price / self.position.entry_price)) * 100

            return float(pnl_percent)
        except Exception as e:
            self.logger.error(f"Error calculating unrealized PnL: {e}")
            return 0.0

    def check_stop_loss(self, current_price):
        """Check if stop loss is triggered"""
        if not self.position.active or not self.position.stop_loss_price:
            return False

        if self.position.side == "BUY" or self.position.side == "long":
            return current_price <= self.position.stop_loss_price
        else:  # SELL or short
            return current_price >= self.position.stop_loss_price

    def save_state(self):
        """Save the current position state to file"""
        from bot.models import save_position_state

        save_position_state(self.position, self.symbol)
        self.logger.info(f"Bot state saved for {self.symbol}")
