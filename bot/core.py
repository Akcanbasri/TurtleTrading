"""
Core TurtleTradingBot class implementation
"""

import decimal
import time
import traceback
from decimal import Decimal
from typing import Any, Dict, Optional

import pandas as pd

from bot.exchange import BinanceExchange
from bot.indicators import (
    calculate_indicators,
    calculate_stop_loss_take_profit,
    check_entry_signal,
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


class TurtleTradingBot:
    """
    Turtle Trading Bot implementation

    This class implements the Turtle Trading strategy using the Binance exchange.
    It provides methods for initializing the bot, loading state, fetching data,
    calculating indicators, executing trades, and managing positions.
    """

    def __init__(self, config: Optional[BotConfig] = None):
        """
        Initialize the Turtle Trading Bot

        Parameters
        ----------
        config : Optional[BotConfig]
            Bot configuration, if None load from environment variables
        """
        # Set up decimal precision
        decimal.getcontext().prec = 8

        # Initialize logger
        self.logger = setup_logging("turtle_trading_bot")

        # Load configuration
        self.config = config if config else BotConfig.from_env()

        # Initialize exchange client
        self.exchange = BinanceExchange(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            use_testnet=self.config.use_testnet,
        )

        # Get symbol information
        self.symbol_info = self.exchange.get_symbol_info(self.config.symbol)

        # Initialize position state
        self.position = self._load_position_state()

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

    def check_and_execute_trading_logic(self, df_with_indicators: pd.DataFrame) -> None:
        """
        Core trading logic that checks indicators and position state to make trading decisions

        Parameters
        ----------
        df_with_indicators : pd.DataFrame
            DataFrame with price data and calculated indicators
        """
        self.logger.info("Checking trading conditions...")

        try:
            # Get latest indicator values
            latest_row = df_with_indicators.iloc[-1]

            # Get current market price
            current_price = self.exchange.get_current_price(self.config.symbol)
            if not current_price:
                self.logger.error("Failed to get current market price")
                return

            self.logger.info(
                f"Current market price: {format_price(current_price, self.symbol_info.price_precision)}"
            )
            self.logger.info(f"Latest indicators (timestamp {latest_row.name}):")
            self.logger.info(f"  DC Upper Entry: {latest_row['dc_upper_entry']}")
            self.logger.info(f"  DC Lower Entry: {latest_row['dc_lower_entry']}")
            self.logger.info(f"  DC Upper Exit: {latest_row['dc_upper_exit']}")
            self.logger.info(f"  DC Lower Exit: {latest_row['dc_lower_exit']}")
            self.logger.info(f"  ATR: {latest_row['atr']}")

            # Check current position state
            if self.position.active:
                self.logger.info(
                    "Currently in active position. Checking exit conditions..."
                )

                if self.position.side == "BUY":  # We're in a LONG position
                    # Check stop loss
                    if check_stop_loss(
                        float(current_price),
                        float(self.position.stop_loss_price),
                        self.position.side,
                    ):
                        self.logger.info(
                            f"STOP LOSS TRIGGERED: Current price {format_price(current_price, self.symbol_info.price_precision)} <= Stop loss {format_price(self.position.stop_loss_price, self.symbol_info.price_precision)}"
                        )

                        # Execute sell order
                        success, order_result = self.exchange.execute_order(
                            symbol=self.config.symbol,
                            side="SELL",
                            quantity=self.position.quantity,
                            symbol_info=self.symbol_info,
                        )

                        if success:
                            self.logger.info(
                                "Successfully closed LONG position with STOP LOSS"
                            )
                            self.update_position_state(order_result, "SELL")
                        else:
                            self.logger.error(
                                f"Failed to execute stop loss order: {order_result}"
                            )

                    # Check Donchian channel exit (price breaks below lower band)
                    elif check_exit_signal(latest_row, self.position.side):
                        self.logger.info(
                            f"EXIT SIGNAL: Close price {latest_row['close']} broke below Donchian lower band {latest_row['dc_lower_exit']}"
                        )

                        # Execute sell order
                        success, order_result = self.exchange.execute_order(
                            symbol=self.config.symbol,
                            side="SELL",
                            quantity=self.position.quantity,
                            symbol_info=self.symbol_info,
                        )

                        if success:
                            self.logger.info(
                                "Successfully closed LONG position with Donchian exit signal"
                            )
                            self.update_position_state(order_result, "SELL")
                        else:
                            self.logger.error(
                                f"Failed to execute Donchian exit order: {order_result}"
                            )

                    else:
                        self.logger.info(
                            "No exit conditions met, maintaining LONG position"
                        )

                elif self.position.side == "SELL":  # We're in a SHORT position
                    # Check stop loss
                    if check_stop_loss(
                        float(current_price),
                        float(self.position.stop_loss_price),
                        self.position.side,
                    ):
                        self.logger.info(
                            f"STOP LOSS TRIGGERED: Current price {format_price(current_price, self.symbol_info.price_precision)} >= Stop loss {format_price(self.position.stop_loss_price, self.symbol_info.price_precision)}"
                        )

                        # Execute buy order to close position
                        success, order_result = self.exchange.execute_order(
                            symbol=self.config.symbol,
                            side="BUY",
                            quantity=self.position.quantity,
                            symbol_info=self.symbol_info,
                        )

                        if success:
                            self.logger.info(
                                "Successfully closed SHORT position with STOP LOSS"
                            )
                            self.update_position_state(order_result, "BUY")
                        else:
                            self.logger.error(
                                f"Failed to execute stop loss order: {order_result}"
                            )

                    # Check Donchian channel exit (price breaks above upper band)
                    elif check_exit_signal(latest_row, self.position.side):
                        self.logger.info(
                            f"EXIT SIGNAL: Close price {latest_row['close']} broke above Donchian upper band {latest_row['dc_upper_exit']}"
                        )

                        # Execute buy order to close position
                        success, order_result = self.exchange.execute_order(
                            symbol=self.config.symbol,
                            side="BUY",
                            quantity=self.position.quantity,
                            symbol_info=self.symbol_info,
                        )

                        if success:
                            self.logger.info(
                                "Successfully closed SHORT position with Donchian exit signal"
                            )
                            self.update_position_state(order_result, "BUY")
                        else:
                            self.logger.error(
                                f"Failed to execute Donchian exit order: {order_result}"
                            )

                    else:
                        self.logger.info(
                            "No exit conditions met, maintaining SHORT position"
                        )

            else:
                self.logger.info("No active position. Checking entry conditions...")

                # Check long entry signal - Breakout above Donchian upper band
                if check_entry_signal(latest_row, "BUY"):
                    self.logger.info(
                        f"LONG ENTRY SIGNAL: Close price {latest_row['close']} broke above Donchian upper band {latest_row['dc_upper_entry']}"
                    )

                    # Calculate position size
                    quote_balance = self.exchange.get_account_balance(
                        self.config.quote_asset
                    )
                    if not quote_balance:
                        self.logger.error(
                            f"Failed to get {self.config.quote_asset} balance"
                        )
                        return

                    self.logger.info(
                        f"Available {self.config.quote_asset} balance: {quote_balance}"
                    )

                    # Calculate quantity based on risk management
                    position_size, message = calculate_position_size(
                        available_balance=quote_balance,
                        risk_percent=self.config.risk_per_trade,
                        atr_value=latest_row["atr"],
                        current_price=current_price,
                        symbol_info={
                            "min_qty": self.symbol_info.min_qty,
                            "step_size": self.symbol_info.step_size,
                            "min_notional": self.symbol_info.min_notional,
                            "price_precision": self.symbol_info.price_precision,
                            "quantity_precision": self.symbol_info.quantity_precision,
                        },
                    )

                    if position_size > Decimal("0"):
                        self.logger.info(
                            f"Calculated position size: {position_size} {self.config.base_asset}"
                        )

                        # Execute buy order
                        success, order_result = self.exchange.execute_order(
                            symbol=self.config.symbol,
                            side="BUY",
                            quantity=position_size,
                            symbol_info=self.symbol_info,
                        )

                        if success:
                            self.logger.info("Successfully opened LONG position")
                            self.update_position_state(
                                order_result, "BUY", latest_row["atr"]
                            )
                        else:
                            self.logger.error(
                                f"Failed to execute entry order: {order_result}"
                            )
                    else:
                        self.logger.warning(
                            f"Entry signal detected but position size calculation failed: {message}"
                        )

                # Check short entry signal - Breakout below Donchian lower band
                elif check_entry_signal(latest_row, "SELL"):
                    self.logger.info(
                        f"SHORT ENTRY SIGNAL: Close price {latest_row['close']} broke below Donchian lower band {latest_row['dc_lower_entry']}"
                    )

                    # Calculate position size
                    quote_balance = self.exchange.get_account_balance(
                        self.config.quote_asset
                    )
                    if not quote_balance:
                        self.logger.error(
                            f"Failed to get {self.config.quote_asset} balance"
                        )
                        return

                    self.logger.info(
                        f"Available {self.config.quote_asset} balance: {quote_balance}"
                    )

                    # Calculate quantity based on risk management
                    position_size, message = calculate_position_size(
                        available_balance=quote_balance,
                        risk_percent=self.config.risk_per_trade,
                        atr_value=latest_row["atr"],
                        current_price=current_price,
                        symbol_info={
                            "min_qty": self.symbol_info.min_qty,
                            "step_size": self.symbol_info.step_size,
                            "min_notional": self.symbol_info.min_notional,
                            "price_precision": self.symbol_info.price_precision,
                            "quantity_precision": self.symbol_info.quantity_precision,
                        },
                    )

                    if position_size > Decimal("0"):
                        self.logger.info(
                            f"Calculated position size: {position_size} {self.config.base_asset}"
                        )

                        # Execute sell order to open short position
                        success, order_result = self.exchange.execute_order(
                            symbol=self.config.symbol,
                            side="SELL",
                            quantity=position_size,
                            symbol_info=self.symbol_info,
                        )

                        if success:
                            self.logger.info("Successfully opened SHORT position")
                            self.update_position_state(
                                order_result, "SELL", latest_row["atr"]
                            )
                        else:
                            self.logger.error(
                                f"Failed to execute short entry order: {order_result}"
                            )
                    else:
                        self.logger.warning(
                            f"Short entry signal detected but position size calculation failed: {message}"
                        )
                else:
                    self.logger.info("No entry conditions met")

        except Exception as e:
            self.logger.error(f"Error in trading logic: {e}")
            self.logger.error(f"Exception details: {str(e)}")
            self.logger.error(traceback.format_exc())

    def run(self) -> None:
        """
        Main function to run the trading bot in a continuous loop
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
                    self.check_and_execute_trading_logic(df_with_indicators)

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
