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
import math

from bot.exchange import BinanceExchange
from bot.indicators import (
    calculate_indicators,
    calculate_stop_loss_take_profit,
    check_exit_signal,
    check_stop_loss,
    calculate_indicators_incremental,
    check_two_way_price_action,
    calculate_ma,
)
from bot.models import BotConfig, PositionState
from bot.risk import calculate_pnl, calculate_position_size
from bot.utils import (
    format_price,
    get_sleep_time,
    load_position_state,
    save_position_state,
    setup_logging,
    format_quantity,
)


class DataManager:
    """Manages data operations for the bot."""

    def __init__(self, exchange):
        """Initialize the data manager."""
        self.exchange = exchange
        self.data_cache = {}
        self.cache_access_times = {}  # Track when each cache entry was last accessed

    def get_historical_data(self, symbol, timeframe, limit=100):
        """Get historical candlestick data for a symbol and timeframe."""
        cache_key = f"{symbol}_{timeframe}_{limit}"

        # Check cache first
        if cache_key in self.data_cache:
            # Update access time
            self.cache_access_times[cache_key] = time.time()
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

        # Cache the result and update access time
        self.data_cache[cache_key] = data
        self.cache_access_times[cache_key] = time.time()

        # Clean up old cache entries if cache is getting too large
        self.clear_old_cache_entries()

        # Veriyi döndürmeden önce gerekli sütunların varlığını kontrol edin
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in data.columns:
                logger = logging.getLogger("turtle_trading_bot")
                logger.error(f"Eksik veri sütunu: {col}")
                # Eksik sütunu uygun değerlerle doldurun veya hata fırlatın

        return data

    def clear_cache(self):
        """Clear the data cache."""
        self.data_cache = {}
        self.cache_access_times = {}

    def clear_old_cache_entries(self, max_cache_size=100, max_age_hours=24):
        """
        Clear old cache entries to prevent memory issues on long runs

        Parameters
        ----------
        max_cache_size : int
            Maximum number of entries to keep in cache
        max_age_hours : int
            Maximum age of cache entries in hours
        """
        # If cache is smaller than max size, only remove by age
        if len(self.data_cache) <= max_cache_size:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            # Remove entries older than max age
            expired_keys = [
                key
                for key, access_time in self.cache_access_times.items()
                if current_time - access_time > max_age_seconds
            ]

            for key in expired_keys:
                if key in self.data_cache:
                    del self.data_cache[key]
                if key in self.cache_access_times:
                    del self.cache_access_times[key]
        else:
            # If cache is larger than max size, keep most recently accessed entries
            sorted_keys = sorted(
                self.cache_access_times.keys(),
                key=lambda k: self.cache_access_times[k],
                reverse=True,
            )

            # Keep only max_cache_size most recent entries
            keys_to_remove = sorted_keys[max_cache_size:]

            for key in keys_to_remove:
                if key in self.data_cache:
                    del self.data_cache[key]
                if key in self.cache_access_times:
                    del self.cache_access_times[key]

    def initialize_real_time_data(self, symbol, timeframe):
        """
        Gerçek zamanlı veri akışı için WebSocket başlatır

        Parameters
        ----------
        symbol : str
            İzlenecek sembol
        timeframe : str
            Zaman dilimi
        """
        self.real_time_data = {}
        self.current_candle = None

        def websocket_callback(data):
            # Kline verisi geldiğinde
            if "k" in data:
                k = data["k"]
                if k["x"]:  # Mum tamamlandıysa
                    # Yeni mumu dataframe'e ekle
                    new_candle = {
                        "open_time": k["t"] // 1000,  # milisaniyeden saniyeye çevir
                        "open": float(k["o"]),
                        "high": float(k["h"]),
                        "low": float(k["l"]),
                        "close": float(k["c"]),
                        "volume": float(k["v"]),
                        "close_time": k["T"] // 1000,
                    }
                    # Mum verisini sakla
                    self.real_time_data[timeframe] = new_candle

                    # Geçerli DataFrame'e yeni veriyi ekle ve önbelleği güncelle
                    self._update_dataframe_with_new_candle(
                        symbol, timeframe, new_candle
                    )
                else:
                    # Devam eden mumu sakla
                    self.current_candle = {
                        "open_time": k["t"] // 1000,
                        "open": float(k["o"]),
                        "high": float(k["h"]),
                        "low": float(k["l"]),
                        "close": float(k["c"]),
                        "volume": float(k["v"]),
                        "close_time": k["T"] // 1000,
                    }

            # Ticker verisi geldiğinde
            elif "lastPrice" in data:
                self.last_price = float(data["lastPrice"])

        # WebSocket başlat
        self.ws_manager = self.exchange.initialize_websocket(
            symbol.lower(), websocket_callback
        )

    def _update_dataframe_with_new_candle(self, symbol, timeframe, candle_data):
        """Yeni bir mum verisini DataFrame'e ekler ve göstergeleri inkremental olarak günceller"""
        cache_key = f"{symbol}_{timeframe}"

        # Yeni satır oluştur
        new_row = {
            "timestamp": candle_data["open_time"],
            "open": candle_data["open"],
            "high": candle_data["high"],
            "low": candle_data["low"],
            "close": candle_data["close"],
            "volume": candle_data["volume"],
        }

        if cache_key in self.data_cache:
            df = self.data_cache[cache_key].copy()

            # Göstergeleri inkremental olarak güncelle
            # Bot yapılandırmasını al (bu örnek için sabit değerler)
            dc_enter = 20
            dc_exit = 10
            atr_len = 14

            # Göstergeleri inkremental olarak hesapla
            updated_df = calculate_indicators_incremental(
                df, new_row, dc_enter, dc_exit, atr_len
            )

            # Önbelleği güncelle
            self.data_cache[cache_key] = updated_df
            self.cache_timestamps[cache_key] = time.time()
        else:
            # Önbellekte yoksa, normal hesaplama yap
            new_df = pd.DataFrame([new_row])
            self.data_cache[cache_key] = new_df
            self.cache_timestamps[cache_key] = time.time()


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
        timeframe_preset=None,
    ):
        """
        Initialize the bot.

        Args:
            use_testnet: Whether to use the Binance testnet
            config_file: Path to the configuration file
            api_key: Optional API key (overrides config file)
            api_secret: Optional API secret (overrides config file)
            demo_mode: Whether to run in demo mode with synthetic data
            timeframe_preset: Optional timeframe preset to load (e.g., 'crypto_standard')
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

        # Load timeframe preset if specified
        if timeframe_preset:
            if self.config.load_timeframe_preset(timeframe_preset):
                # Update local variables with new values from preset
                self.timeframe = self.config.timeframe
                self.logger.info(f"Using timeframe preset: {timeframe_preset}")
            else:
                self.logger.warning(
                    f"Failed to load timeframe preset: {timeframe_preset}"
                )

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

        # __init__ metoduna eklenecek
        self.use_websocket = True  # WebSocket kullanımını etkinleştir

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
                f"P&L: {format_price(pnl, self.symbol_info.price_precision)} {self.quote_asset} ({pnl_percent:.2f}%)"
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
        atr_average: Union[float, Decimal] = None,
    ) -> bool:
        """
        Execute trade entry

        Args:
            direction: Trade direction ('long' or 'short')
            current_price: Current market price
            atr_value: Current ATR value
            pyramid_level: Pyramid level (0 = first entry)
            atr_average: Average ATR over longer period for volatility-based risk adjustment

        Returns:
            bool: Whether the entry was executed successfully
        """
        # Normalize direction
        direction = (
            direction.upper() if direction.lower() in ["buy", "long"] else "SELL"
        )

        # Standardize direction to match Binance API
        side = "BUY" if direction.upper() in ["BUY", "LONG"] else "SELL"

        # Log entry plan
        self.logger.info(f"Planning {side} entry for {self.symbol} at {current_price}")

        # Don't enter if the price is NaN or zero
        if current_price == 0 or (
            isinstance(current_price, float) and math.isnan(current_price)
        ):
            self.logger.error("Invalid price (0 or NaN). Cannot execute entry.")
            return False

        try:
            # Get account balance for the quote asset
            balance = self.exchange.get_account_balance(self.config.quote_asset)
            if balance is None or balance == 0:
                self.logger.error(
                    f"No {self.config.quote_asset} balance available. Cannot execute entry."
                )
                return False

            self.logger.info(f"Available balance: {balance} {self.config.quote_asset}")

            # Build symbol info dictionary for position sizing
            symbol_info = {
                "min_qty": self.symbol_info.min_qty,
                "step_size": self.symbol_info.step_size,
                "min_notional": self.symbol_info.min_notional,
                "price_precision": self.symbol_info.price_precision,
                "quantity_precision": self.symbol_info.quantity_precision,
            }

            # Check if entry is aligned with higher timeframe trend
            is_trend_aligned = (
                direction == "BUY"
                and self.last_analysis.get("trend_direction") == "long"
            ) or (
                direction == "SELL"
                and self.last_analysis.get("trend_direction") == "short"
            )

            # Determine leverage based on trend alignment
            if is_trend_aligned:
                leverage = min(
                    int(self.config.leverage), int(self.config.max_leverage_trend)
                )
                self.logger.info(
                    f"Trend-aligned trade, using up to {leverage}x leverage"
                )
            else:
                leverage = min(
                    int(self.config.leverage), int(self.config.max_leverage_counter)
                )
                self.logger.info(
                    f"Counter-trend trade, limiting to {leverage}x leverage"
                )

            # Set leverage on exchange
            if leverage > 1:
                try:
                    self.logger.info(
                        f"Setting leverage to {leverage}x for {self.symbol}"
                    )
                    self.exchange.set_leverage(self.symbol, leverage)
                except Exception as e:
                    self.logger.warning(f"Failed to set leverage: {e}. Using 1x.")
                    leverage = 1

            # Calculate position size using risk management rules
            from bot.risk import calculate_position_size

            risk_percent = Decimal(str(self.config.risk_per_trade))
            max_risk_percentage = Decimal(str(self.config.max_risk_percentage))

            # Pass average ATR for volatility-based risk adjustment
            position_size, status = calculate_position_size(
                available_balance=balance,
                risk_percent=risk_percent,
                atr_value=atr_value,
                current_price=Decimal(str(current_price)),
                symbol_info=symbol_info,
                max_risk_percentage=max_risk_percentage,
                leverage=leverage,
                position_side=side,
                pyramid_level=pyramid_level,
                pyramid_size_first=Decimal(str(self.config.pyramid_size_first)),
                pyramid_size_additional=Decimal(
                    str(self.config.pyramid_size_additional)
                ),
                is_trend_aligned=is_trend_aligned,
                atr_average=atr_average,  # Pass for volatility adjustment
            )

            if status != "success" or position_size <= 0:
                self.logger.error(f"Position sizing failed: {status}")
                return False

            # Format quantity for the order
            quantity_str = format_quantity(
                position_size, self.symbol_info.quantity_precision
            )
            self.logger.info(f"Calculated position size: {quantity_str}")

            # Execute the order
            self.logger.info(
                f"Executing {side} order for {quantity_str} {self.symbol} at market price"
            )

            try:
                # Will be implemented in exchange.py
                order_result = self.exchange.execute_market_order(
                    symbol=self.symbol,
                    side=side,
                    quantity=position_size,
                )

                if not order_result:
                    self.logger.error("Order execution failed")
                    return False

                self.logger.info(f"Order executed successfully: {order_result}")

                # Calculate stop loss and take profit levels
                from bot.indicators import calculate_stop_loss_take_profit

                stop_loss, take_profit = calculate_stop_loss_take_profit(
                    entry_price=float(current_price),
                    atr_value=float(atr_value),
                    atr_multiple=float(self.config.stop_loss_atr_multiple),
                    position_side=side,
                )

                # Update position state
                self.update_position_state(order_result, side, float(atr_value))
                self.position.stop_loss_price = Decimal(str(stop_loss))
                self.position.take_profit_price = Decimal(str(take_profit))
                self.position.entry_count = pyramid_level + 1
                self.position.current_entry_level = pyramid_level

                # Save state
                self.save_state()

                self.logger.info(
                    f"Entry successful: {side} {quantity_str} {self.symbol} at {current_price}"
                )
                self.logger.info(
                    f"Stop loss set at {stop_loss} | Take profit at {take_profit}"
                )
                return True

            except Exception as e:
                self.logger.error(f"Error executing order: {e}")
                return False

        except Exception as e:
            self.logger.error(f"Error in _execute_entry: {e}")
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

        # run metoduna eklenecek (self.run() fonksiyonu içinde)
        if self.use_websocket:
            self.data_manager.initialize_real_time_data(
                self.config.symbol, self.config.timeframe
            )
            self.logger.info(f"WebSocket initialized for {self.config.symbol}")

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

            # Get historical data for execution timeframe (lowest timeframe)
            execution_df = self.data_manager.get_historical_data(
                symbol, self.config.timeframe, limit=100
            )

            # Calculate indicators for all timeframes
            from bot.indicators import (
                calculate_indicators,
                check_entry_signal,
                check_adx_filter,
                check_ma_filter,
                check_bb_squeeze,
                check_rsi_conditions,
                check_macd_confirmation,
            )

            # Make sure the DataFrames have the required columns
            if trend_df is None or trend_df.empty:
                self.logger.warning(f"No trend data available for {symbol}")
                trend_df = self._create_default_dataframe()

            if entry_df is None or entry_df.empty:
                self.logger.warning(f"No entry data available for {symbol}")
                entry_df = self._create_default_dataframe()

            if execution_df is None or execution_df.empty:
                self.logger.warning(f"No execution data available for {symbol}")
                execution_df = self._create_default_dataframe()

            # Ensure 'ma' column exists in trend_df before using it
            if "ma" not in trend_df.columns:
                trend_df["ma"] = trend_df["close"].rolling(self.config.ma_period).mean()

            # Determine trend direction from higher timeframe
            if trend_df["close"].iloc[-1] > trend_df["ma"].iloc[-1]:
                trend_direction = "long"
            elif trend_df["close"].iloc[-1] < trend_df["ma"].iloc[-1]:
                trend_direction = "short"
            else:
                trend_direction = "neutral"

            # Calculate indicators with proper parameters for both long and short
            # For short positions, we want to use a slightly modified set of parameters
            trend_df = calculate_indicators(
                df=trend_df,
                dc_enter=self.config.dc_length_enter,
                dc_exit=self.config.dc_length_exit,
                atr_len=self.config.atr_length,
                atr_smooth=self.config.atr_smoothing,
                ma_period=self.config.ma_period,
                adx_period=self.config.adx_period,
                include_additional=True,  # Include BB and RSI
                position_side=(
                    "SELL" if trend_direction == "short" else "BUY"
                ),  # Optimize for trend direction
            )

            entry_df = calculate_indicators(
                df=entry_df,
                dc_enter=self.config.dc_length_enter,
                dc_exit=self.config.dc_length_exit,
                atr_len=self.config.atr_length,
                atr_smooth=self.config.atr_smoothing,
                ma_period=self.config.ma_period,
                adx_period=self.config.adx_period,
                include_additional=True,  # Include BB and RSI
                position_side=(
                    "SELL" if trend_direction == "short" else "BUY"
                ),  # Optimize for trend direction
            )

            execution_df = calculate_indicators(
                df=execution_df,
                dc_enter=self.config.dc_length_enter,
                dc_exit=self.config.dc_length_exit,
                atr_len=self.config.atr_length,
                atr_smooth=self.config.atr_smoothing,
                ma_period=self.config.ma_period,
                adx_period=self.config.adx_period,
                include_additional=True,  # Include BB and RSI
                position_side=(
                    "SELL" if trend_direction == "short" else "BUY"
                ),  # Optimize for trend direction
            )

            # Ensure 'adx' column exists
            if "adx" not in trend_df.columns:
                trend_df["adx"] = 25.0  # Default ADX value

            # Calculate average ATR for volatility analysis (over 30 periods)
            atr_avg_period = 30
            if len(entry_df) >= atr_avg_period and "atr" in entry_df.columns:
                atr_average = (
                    entry_df["atr"].rolling(window=atr_avg_period).mean().iloc[-1]
                )
            else:
                atr_average = (
                    entry_df["atr"].mean() if "atr" in entry_df.columns else 0.0
                )

            # Check entry signals on entry timeframe
            try:
                entry_long = check_entry_signal(entry_df.iloc[-1], "BUY")
                entry_short = check_entry_signal(entry_df.iloc[-1], "SELL")
            except Exception as e:
                self.logger.error(f"Error checking entry signals: {e}")
                entry_long = False
                entry_short = False

            # Check for Bollinger Band squeeze (low volatility, potential breakout)
            try:
                bb_squeeze_detected = check_bb_squeeze(entry_df.iloc[-1])
                if bb_squeeze_detected:
                    self.logger.info(
                        "Bollinger Band squeeze detected - potential breakout setup"
                    )
            except Exception as e:
                self.logger.error(f"Error checking BB squeeze: {e}")
                bb_squeeze_detected = False

            # Check if entry aligns with trend
            entry_signal = False
            signal_direction = "none"

            if entry_long and (
                trend_direction == "long" or not self.config.trend_alignment_required
            ):
                # Confirm with RSI for long entries
                rsi_confirms = check_rsi_conditions(entry_df.iloc[-1], "BUY")
                # Also confirm with MACD for long entries
                macd_confirms = check_macd_confirmation(entry_df.iloc[-1], "BUY")

                if rsi_confirms and macd_confirms:
                    entry_signal = True
                    signal_direction = "long"
                else:
                    self.logger.info(
                        "Long entry signal rejected by filter: "
                        + ("RSI failed" if not rsi_confirms else "")
                        + ("MACD failed" if not macd_confirms else "")
                    )
            elif entry_short and (
                trend_direction == "short" or not self.config.trend_alignment_required
            ):
                # For short entries, we need stronger confirmation

                # 1. First, check if ADX is high enough for a strong trend
                adx_value = trend_df["adx"].iloc[-1]
                adx_strong = adx_value >= 30  # Require stronger trend for shorts

                # 2. Check if price is below MA by a significant margin (1.5% lower)
                price_below_ma = (
                    entry_df["close"].iloc[-1] < entry_df["ma"].iloc[-1] * 0.985
                )

                # 3. Check RSI conditions for short entries
                rsi_confirms = check_rsi_conditions(entry_df.iloc[-1], "SELL")

                # 4. Check MACD confirmation for short entries
                macd_confirms = check_macd_confirmation(entry_df.iloc[-1], "SELL")

                # 5. For shorts, verify the downtrend on multiple timeframes
                downtrend_confirmed = (
                    trend_df["close"].iloc[-1] < trend_df["ma"].iloc[-1]
                    and entry_df["close"].iloc[-1] < entry_df["ma"].iloc[-1]
                    and trend_df["macd"].iloc[-1] < trend_df["macd_signal"].iloc[-1]
                )

                # Combine all confirmations for short entry
                if (
                    rsi_confirms
                    and macd_confirms
                    and adx_strong
                    and price_below_ma
                    and downtrend_confirmed
                ):
                    entry_signal = True
                    signal_direction = "short"
                    self.logger.info(
                        "Short entry confirmed with multiple confirmations"
                    )
                else:
                    self.logger.info(
                        f"Short entry rejected. ADX: {adx_value:.1f} (strong: {adx_strong}), "
                        f"Price/MA: {price_below_ma}, RSI: {rsi_confirms}, "
                        f"MACD: {macd_confirms}, Downtrend: {downtrend_confirmed}"
                    )

            # Check ADX for trend strength
            try:
                adx_value = trend_df["adx"].iloc[-1]
                # Different ADX thresholds for long and short
                adx_threshold = self.config.adx_threshold
                if signal_direction == "short":
                    adx_threshold = max(
                        30, self.config.adx_threshold
                    )  # Higher threshold for shorts

                adx_passed = check_adx_filter(trend_df.iloc[-1], adx_threshold)
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

            # Get entry price and ATR from execution timeframe
            entry_price = execution_df["close"].iloc[-1]
            entry_atr = (
                execution_df["atr"].iloc[-1] if "atr" in execution_df.columns else 0.0
            )

            # İki yönlü fiyat hareketi kontrolü
            two_way_price_action = check_two_way_price_action(entry_df, lookback=5)
            if two_way_price_action:
                self.logger.info(
                    "İki yönlü fiyat hareketi tespit edildi - sinyal filtreleniyor"
                )
                entry_signal = False
                signal_direction = "none"

            return {
                "trend_direction": trend_direction,
                "entry_signal": entry_signal,
                "signal_direction": signal_direction,
                "adx_value": adx_value,
                "adx_filter_passed": adx_passed,
                "ma_filter_passed": ma_passed,
                "current_price": entry_price,
                "atr": entry_atr,
                "atr_average": atr_average,
                "bb_squeeze": bb_squeeze_detected,
                "rsi_value": (
                    entry_df["rsi"].iloc[-1] if "rsi" in entry_df.columns else 50.0
                ),
                "macd_confirms": (
                    True
                    if signal_direction == "none"
                    else check_macd_confirmation(
                        entry_df.iloc[-1],
                        "BUY" if signal_direction == "long" else "SELL",
                    )
                ),
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
                "atr_average": 1000.0,
                "bb_squeeze": False,
                "rsi_value": 50.0,
                "macd_confirms": False,
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
            atr_average=atr_average,
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

    def execute_trade(self, symbol, side, analysis_results):
        """
        Execute a trade based on the signal and analysis results.

        Args:
            symbol: The trading symbol
            side: Position side ("BUY" or "SELL")
            analysis_results: Results from multi-timeframe analysis
        """
        try:
            current_price = analysis_results.get("current_price", 0.0)
            atr = analysis_results.get("atr", 0.0)

            # Skip if price or ATR is zero (error case)
            if current_price <= 0 or atr <= 0:
                self.logger.warning(
                    f"Invalid price ({current_price}) or ATR ({atr}). Skipping trade."
                )
                return

            # Get balance and calculate position size with risk management
            account_balance = self.exchange.get_balance()

            # Base risk percentage on volatility (use higher risk for lower volatility)
            # For short positions, reduce risk further due to potentially unlimited loss
            base_risk_percentage = self.config.risk_per_trade

            # Adjust risk based on volatility
            volatility_adjusted_risk = self.adjust_risk_based_on_volatility(
                symbol, analysis_results.get("atr_average", atr), base_risk_percentage
            )

            # Further reduce risk for short positions
            if side == "SELL":
                # Reduce risk by 20% for short positions
                volatility_adjusted_risk *= 0.8
                self.logger.info(
                    f"Reduced risk for short position: {volatility_adjusted_risk:.2f}%"
                )

            # Calculate position size based on risk and ATR
            risk_amount = account_balance * (volatility_adjusted_risk / 100)

            # Calculate stop loss distance - use ATR multiplier
            # For shorts, use a slightly wider stop loss
            sl_multiplier = self.config.sl_atr_multiplier
            if side == "SELL":
                sl_multiplier *= 1.2  # 20% wider stop loss for shorts

            stop_loss_distance = atr * sl_multiplier

            # Calculate position size based on risk and stop loss
            position_size = (
                risk_amount / stop_loss_distance if stop_loss_distance > 0 else 0
            )

            # Calculate take profit based on reward/risk ratio
            # For shorts, use a more conservative target
            tp_multiplier = self.config.tp_atr_multiplier
            if side == "SELL":
                tp_multiplier *= (
                    0.8  # 20% closer target for shorts to account for faster rebounds
                )

            take_profit_distance = atr * tp_multiplier

            # Calculate stop loss and take profit levels
            if side == "BUY":
                stop_loss = current_price - stop_loss_distance
                take_profit = current_price + take_profit_distance
            else:  # SELL
                stop_loss = current_price + stop_loss_distance
                take_profit = current_price - take_profit_distance

            # Execute the trade through the exchange
            trade_result = self.exchange.execute_order(
                symbol=symbol,
                side=side,
                quantity=position_size,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

            # Log detailed trade information
            self.logger.info(
                f"Trade executed: {side} {symbol} at {current_price}. "
                f"Pos Size: {position_size:.8f}, "
                f"Stop Loss: {stop_loss:.2f}, "
                f"Take Profit: {take_profit:.2f}, "
                f"Risk: {volatility_adjusted_risk:.2f}%, "
                f"ATR: {atr:.2f}"
            )

            # Save the trade to database
            self.save_trade(
                symbol=symbol,
                side=side,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=position_size,
                risk_percentage=volatility_adjusted_risk,
            )

            return trade_result

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None

    def adjust_risk_based_on_volatility(self, symbol, atr_average, base_risk_percent):
        """
        Adjust risk percentage based on current market volatility and market conditions.
        Reduces risk in highly volatile markets and for short positions.

        Args:
            symbol: Trading symbol
            atr_average: Average ATR value
            base_risk_percent: Base risk percentage from configuration

        Returns:
            float: Adjusted risk percentage
        """
        try:
            # Get current ATR
            df = self.data_manager.get_historical_data(
                symbol, self.config.timeframe, limit=30
            )
            if df is None or df.empty or "atr" not in df.columns:
                self.logger.warning(f"Could not get ATR for {symbol}. Using base risk.")
                return base_risk_percent

            current_atr = df["atr"].iloc[-1]

            # Prevent division by zero
            if atr_average <= 0:
                return base_risk_percent

            # Calculate volatility ratio
            volatility_ratio = current_atr / atr_average

            # Adjust risk based on volatility ratio
            if volatility_ratio > 1.5:
                # High volatility - reduce risk
                adjusted_risk = base_risk_percent * 0.7
                self.logger.info(
                    f"High volatility detected (ratio: {volatility_ratio:.2f}). Reducing risk to {adjusted_risk:.2f}%"
                )
            elif volatility_ratio < 0.7:
                # Low volatility - can increase risk slightly
                adjusted_risk = base_risk_percent * 1.2
                self.logger.info(
                    f"Low volatility detected (ratio: {volatility_ratio:.2f}). Increasing risk to {adjusted_risk:.2f}%"
                )
            else:
                # Normal volatility - use base risk
                adjusted_risk = base_risk_percent
                self.logger.info(
                    f"Normal volatility detected (ratio: {volatility_ratio:.2f}). Using base risk of {adjusted_risk:.2f}%"
                )

            # Ensure risk doesn't exceed the maximum allowed
            max_risk = self.config.max_risk_percentage
            if adjusted_risk > max_risk:
                adjusted_risk = max_risk
                self.logger.info(f"Capping risk at maximum allowed: {max_risk:.2f}%")

            return adjusted_risk

        except Exception as e:
            self.logger.error(f"Error in risk adjustment: {e}. Using base risk.")
            return base_risk_percent

    def detect_volatility_squeeze(self, symbol=None, timeframe=None):
        """
        Detect volatility squeeze conditions (narrowing Bollinger Bands and Keltner Channels)

        Parameters
        ----------
        symbol : str, optional
            Trading symbol, defaults to self.symbol
        timeframe : str, optional
            Timeframe to analyze, defaults to self.timeframe

        Returns
        -------
        bool
            True if a volatility squeeze is detected
        """
        if symbol is None:
            symbol = self.symbol

        if timeframe is None:
            timeframe = self.timeframe

        # Get market data
        data = self.data_manager.get_historical_data(symbol, timeframe, limit=100)

        if data is None or len(data) < 20:
            return False

        # Calculate Bollinger Bands
        bb_period = 20
        bb_stdev = 2.0

        sma = data["close"].rolling(window=bb_period).mean()
        stdev = data["close"].rolling(window=bb_period).std()

        bb_upper = sma + (stdev * bb_stdev)
        bb_lower = sma - (stdev * bb_stdev)
        bb_width = bb_upper - bb_lower

        # Calculate Keltner Channels
        kc_period = 20
        kc_multiplier = 1.5

        ema = data["close"].ewm(span=kc_period, adjust=False).mean()

        # Calculate ATR if not already in data
        if "atr" not in data.columns:
            atr_period = 14
            data = calculate_indicators(data, 20, 10, atr_period)

        if "atr" not in data.columns:
            return False

        atr = data["atr"].iloc[-1]

        kc_upper = ema + (atr * kc_multiplier)
        kc_lower = ema - (atr * kc_multiplier)
        kc_width = kc_upper - kc_lower

        # Calculate BB width to KC width ratio for last candle
        try:
            bb_kc_ratio = (
                bb_width.iloc[-1] / kc_width.iloc[-1] if kc_width.iloc[-1] > 0 else 1
            )
        except:
            return False

        # Check for squeeze conditions
        is_squeeze = bb_kc_ratio < 0.8

        # Log squeeze information
        logger = logging.getLogger("turtle_trading_bot")
        if is_squeeze:
            logger.info(
                f"Volatility squeeze detected for {symbol} on {timeframe} timeframe"
            )
            logger.info(f"  BB/KC ratio: {bb_kc_ratio:.2f}")

        return is_squeeze

    def run_monte_carlo_simulation(self, backtest_results, num_simulations=1000):
        """
        Run Monte Carlo simulation on backtest results to analyze risk and robustness

        Parameters
        ----------
        backtest_results : pd.DataFrame
            DataFrame with backtest trade results
        num_simulations : int
            Number of Monte Carlo simulations to run

        Returns
        -------
        dict
            Dictionary with simulation results and statistics
        """
        logger = logging.getLogger("turtle_trading_bot")
        logger.info(f"Running Monte Carlo simulation with {num_simulations} iterations")

        if isinstance(backtest_results, list):
            # Convert to DataFrame if we received a list of trades
            import pandas as pd

            backtest_results = pd.DataFrame(backtest_results)

        if backtest_results.empty or "profit_pct" not in backtest_results.columns:
            logger.error("Invalid backtest results for Monte Carlo simulation")
            return {"error": "Invalid backtest results"}

        import numpy as np

        # Extract profit percentages from backtest results
        returns = backtest_results["profit_pct"].values

        # Create array to store simulation results
        simulation_results = np.zeros((num_simulations, len(returns)))

        # Run simulations with random resampling (bootstrap method)
        for i in range(num_simulations):
            # Resample returns with replacement
            sampled_returns = np.random.choice(returns, size=len(returns), replace=True)

            # Calculate cumulative returns for this simulation
            simulation_results[i, :] = (1 + sampled_returns / 100).cumprod()

        # Calculate statistics from simulations
        final_values = simulation_results[:, -1]

        # Calculate key statistics
        mean_final = np.mean(final_values)
        median_final = np.median(final_values)
        min_final = np.min(final_values)
        max_final = np.max(final_values)

        # Calculate percentiles
        percentiles = {
            "5%": np.percentile(final_values, 5),
            "25%": np.percentile(final_values, 25),
            "50%": np.percentile(final_values, 50),
            "75%": np.percentile(final_values, 75),
            "95%": np.percentile(final_values, 95),
        }

        # Calculate worst drawdown for each simulation
        max_drawdowns = []
        for sim in simulation_results:
            peaks = np.maximum.accumulate(sim)
            drawdowns = (sim - peaks) / peaks
            max_drawdowns.append(np.min(drawdowns))

        avg_max_drawdown = np.mean(max_drawdowns)
        worst_max_drawdown = np.min(max_drawdowns)

        # Log results
        logger.info("Monte Carlo Simulation Results:")
        logger.info(f"  Mean final equity: {mean_final:.2f}x initial capital")
        logger.info(f"  Median final equity: {median_final:.2f}x initial capital")
        logger.info(f"  Range: {min_final:.2f}x to {max_final:.2f}x")
        logger.info(f"  Average maximum drawdown: {avg_max_drawdown*100:.2f}%")
        logger.info(f"  Worst maximum drawdown: {worst_max_drawdown*100:.2f}%")

        # Assemble results dictionary
        results = {
            "mean_final": mean_final,
            "median_final": median_final,
            "min_final": min_final,
            "max_final": max_final,
            "percentiles": percentiles,
            "avg_max_drawdown": avg_max_drawdown,
            "worst_max_drawdown": worst_max_drawdown,
            "simulations": simulation_results.tolist(),
        }

        return results

    def evaluate_signal_quality_with_ml(
        self, market_data, signal_type="entry", position_side="BUY"
    ):
        """
        Use a simple machine learning model to evaluate the quality of a trading signal

        Parameters
        ----------
        market_data : pd.DataFrame
            Market data with indicators
        signal_type : str
            Type of signal to evaluate ("entry" or "exit")
        position_side : str
            Position side ("BUY" or "SELL")

        Returns
        -------
        dict
            Signal quality assessment with probability of success
        """
        logger = logging.getLogger("turtle_trading_bot")

        # Check if we have enough data
        if market_data is None or len(market_data) < 30:
            return {"score": 0.5, "confidence": "low", "reason": "Insufficient data"}

        try:
            # Import ML libraries if available
            try:
                import numpy as np
                import pandas as pd
                from sklearn.ensemble import RandomForestClassifier
            except ImportError:
                logger.warning(
                    "Machine learning libraries not available, using heuristic evaluation"
                )
                return self._evaluate_signal_heuristically(
                    market_data, signal_type, position_side
                )

            # Extract features for ML model
            features = self._extract_signal_features(market_data)

            if not features:
                return {
                    "score": 0.5,
                    "confidence": "low",
                    "reason": "Could not extract features",
                }

            # Train or load model
            # In a real implementation, we would load a pre-trained model from disk
            # For this example, we'll create a simple model based on historical patterns

            # Extract recent market conditions
            recent_data = market_data.iloc[-5:]

            if signal_type == "entry":
                if position_side == "BUY":
                    # Features important for long entries
                    feature_importance = {
                        "trend_alignment": 0.3,  # Price above MA
                        "volatility": 0.2,  # Lower volatility is better for entries
                        "rsi": 0.15,  # RSI > 50 for uptrend confirmation
                        "macd": 0.2,  # MACD histogram positive
                        "volume": 0.15,  # Higher volume on breakout
                    }
                else:  # SELL
                    # Features important for short entries
                    feature_importance = {
                        "trend_alignment": 0.3,  # Price below MA
                        "volatility": 0.2,  # Lower volatility is better for entries
                        "rsi": 0.15,  # RSI < 50 for downtrend confirmation
                        "macd": 0.2,  # MACD histogram negative
                        "volume": 0.15,  # Higher volume on breakdown
                    }
            else:  # exit signals
                # Features important for exits
                feature_importance = {
                    "trend_reversal": 0.35,  # Signs of trend reversal
                    "volatility": 0.25,  # Increase in volatility
                    "take_profit": 0.25,  # Profit target reached
                    "momentum": 0.15,  # Loss of momentum
                }

            # Calculate score for each feature
            scores = {}

            # 1. Trend alignment
            if "trend_alignment" in feature_importance:
                ma_col = "ma" if "ma" in market_data.columns else None
                if ma_col:
                    price = recent_data["close"].iloc[-1]
                    ma = recent_data[ma_col].iloc[-1]
                    if position_side == "BUY":
                        scores["trend_alignment"] = 1.0 if price > ma else 0.0
                    else:
                        scores["trend_alignment"] = 1.0 if price < ma else 0.0
                else:
                    scores["trend_alignment"] = 0.5  # Neutral if no MA

            # 2. Volatility
            if "volatility" in feature_importance:
                atr_col = "atr" if "atr" in market_data.columns else None
                if atr_col:
                    current_atr = recent_data[atr_col].iloc[-1]
                    avg_atr = market_data[atr_col].mean()

                    if signal_type == "entry":
                        # Lower volatility is better for entries (less slippage)
                        scores["volatility"] = 1.0 if current_atr < avg_atr else 0.5
                    else:
                        # Higher volatility can be good for exits (faster moves)
                        scores["volatility"] = 1.0 if current_atr > avg_atr else 0.5
                else:
                    scores["volatility"] = 0.5  # Neutral if no ATR

            # 3. RSI
            if "rsi" in feature_importance:
                rsi_col = "rsi" if "rsi" in market_data.columns else None
                if rsi_col:
                    rsi = recent_data[rsi_col].iloc[-1]
                    if position_side == "BUY":
                        # Higher RSI is better for longs (uptrend momentum)
                        if rsi > 60:
                            scores["rsi"] = 1.0
                        elif rsi > 50:
                            scores["rsi"] = 0.8
                        elif rsi > 40:
                            scores["rsi"] = 0.5
                        else:
                            scores["rsi"] = 0.2
                    else:
                        # Lower RSI is better for shorts (downtrend momentum)
                        if rsi < 40:
                            scores["rsi"] = 1.0
                        elif rsi < 50:
                            scores["rsi"] = 0.8
                        elif rsi < 60:
                            scores["rsi"] = 0.5
                        else:
                            scores["rsi"] = 0.2
                else:
                    scores["rsi"] = 0.5  # Neutral if no RSI

            # 4. MACD
            if "macd" in feature_importance:
                macd_col = "macd_hist" if "macd_hist" in market_data.columns else None
                if macd_col:
                    macd_hist = recent_data[macd_col].iloc[-1]
                    macd_hist_prev = (
                        recent_data[macd_col].iloc[-2] if len(recent_data) > 1 else 0
                    )

                    if position_side == "BUY":
                        # Positive and increasing MACD histogram is good for longs
                        if macd_hist > 0 and macd_hist > macd_hist_prev:
                            scores["macd"] = 1.0
                        elif macd_hist > 0:
                            scores["macd"] = 0.8
                        elif macd_hist > macd_hist_prev:
                            scores["macd"] = 0.6
                        else:
                            scores["macd"] = 0.3
                    else:
                        # Negative and decreasing MACD histogram is good for shorts
                        if macd_hist < 0 and macd_hist < macd_hist_prev:
                            scores["macd"] = 1.0
                        elif macd_hist < 0:
                            scores["macd"] = 0.8
                        elif macd_hist < macd_hist_prev:
                            scores["macd"] = 0.6
                        else:
                            scores["macd"] = 0.3
                else:
                    scores["macd"] = 0.5  # Neutral if no MACD

            # 5. Volume
            if "volume" in feature_importance:
                if "volume" in market_data.columns:
                    current_volume = recent_data["volume"].iloc[-1]
                    avg_volume = market_data["volume"].iloc[-10:].mean()

                    if current_volume > avg_volume * 1.5:
                        scores["volume"] = 1.0  # Strong volume
                    elif current_volume > avg_volume:
                        scores["volume"] = 0.8  # Above average volume
                    else:
                        scores["volume"] = 0.5  # Average or below volume
                else:
                    scores["volume"] = 0.5  # Neutral if no volume data

            # 6. Trend reversal for exits
            if "trend_reversal" in feature_importance:
                if all(col in market_data.columns for col in ["dc_upper", "dc_lower"]):
                    dc_upper = recent_data["dc_upper"].iloc[-1]
                    dc_lower = recent_data["dc_lower"].iloc[-1]
                    price = recent_data["close"].iloc[-1]

                    if position_side == "BUY":
                        # For longs, price approaching lower DC is a warning
                        distance_to_lower = (price - dc_lower) / price
                        if distance_to_lower < 0.01:
                            scores["trend_reversal"] = 0.9  # Very close to lower DC
                        elif distance_to_lower < 0.02:
                            scores["trend_reversal"] = 0.7  # Approaching lower DC
                        else:
                            scores["trend_reversal"] = 0.3  # Not near lower DC
                    else:
                        # For shorts, price approaching upper DC is a warning
                        distance_to_upper = (dc_upper - price) / price
                        if distance_to_upper < 0.01:
                            scores["trend_reversal"] = 0.9  # Very close to upper DC
                        elif distance_to_upper < 0.02:
                            scores["trend_reversal"] = 0.7  # Approaching upper DC
                        else:
                            scores["trend_reversal"] = 0.3  # Not near upper DC
                else:
                    scores["trend_reversal"] = 0.5  # Neutral if no DC channels

            # 7. Take profit for exits
            if "take_profit" in feature_importance:
                if (
                    "entry_price" in self.position_state.__dict__
                    and self.position_state.active
                ):
                    entry_price = self.position_state.entry_price
                    current_price = recent_data["close"].iloc[-1]
                    atr = (
                        recent_data["atr"].iloc[-1]
                        if "atr" in recent_data.columns
                        else 0
                    )

                    if position_side == "BUY":
                        profit_atr = (
                            (current_price - entry_price) / atr if atr > 0 else 0
                        )
                        if profit_atr > 3:
                            scores["take_profit"] = (
                                1.0  # 3+ ATR profit, good exit point
                            )
                        elif profit_atr > 2:
                            scores["take_profit"] = 0.8  # 2+ ATR profit, consider exit
                        elif profit_atr > 1:
                            scores["take_profit"] = 0.6  # 1+ ATR profit, watch closely
                        else:
                            scores["take_profit"] = 0.3  # Less than 1 ATR, hold
                    else:
                        profit_atr = (
                            (entry_price - current_price) / atr if atr > 0 else 0
                        )
                        if profit_atr > 3:
                            scores["take_profit"] = (
                                1.0  # 3+ ATR profit, good exit point
                            )
                        elif profit_atr > 2:
                            scores["take_profit"] = 0.8  # 2+ ATR profit, consider exit
                        elif profit_atr > 1:
                            scores["take_profit"] = 0.6  # 1+ ATR profit, watch closely
                        else:
                            scores["take_profit"] = 0.3  # Less than 1 ATR, hold
                else:
                    scores["take_profit"] = 0.5  # Neutral if no position or ATR

            # 8. Momentum for exits
            if "momentum" in feature_importance:
                if "rsi" in market_data.columns:
                    rsi = recent_data["rsi"].iloc[-1]
                    rsi_prev = recent_data["rsi"].iloc[-3:].mean()  # Average of last 3

                    if position_side == "BUY":
                        # For longs, decreasing RSI may signal loss of momentum
                        if rsi < rsi_prev * 0.9:
                            scores["momentum"] = 0.9  # Significant RSI decrease
                        elif rsi < rsi_prev:
                            scores["momentum"] = 0.7  # Mild RSI decrease
                        else:
                            scores["momentum"] = 0.3  # RSI steady or increasing
                    else:
                        # For shorts, increasing RSI may signal loss of downward momentum
                        if rsi > rsi_prev * 1.1:
                            scores["momentum"] = 0.9  # Significant RSI increase
                        elif rsi > rsi_prev:
                            scores["momentum"] = 0.7  # Mild RSI increase
                        else:
                            scores["momentum"] = 0.3  # RSI steady or decreasing
                else:
                    scores["momentum"] = 0.5  # Neutral if no RSI

            # Calculate weighted score
            weighted_score = 0
            for feature, weight in feature_importance.items():
                if feature in scores:
                    weighted_score += scores[feature] * weight

            # Classify the signal
            if weighted_score > 0.7:
                confidence = "high"
                classification = "strong_signal"
            elif weighted_score > 0.5:
                confidence = "medium"
                classification = "moderate_signal"
            else:
                confidence = "low"
                classification = "weak_signal"

            # Format return value
            result = {
                "score": weighted_score,
                "confidence": confidence,
                "classification": classification,
                "feature_scores": scores,
            }

            logger.info(
                f"Signal quality assessment ({signal_type}/{position_side}): {weighted_score:.2f} ({confidence})"
            )

            return result

        except Exception as e:
            logger.error(f"Error in ML signal evaluation: {e}")
            return {"score": 0.5, "confidence": "low", "reason": f"Error: {str(e)}"}

    def _extract_signal_features(self, market_data):
        """
        Extract features for machine learning model from market data

        Parameters
        ----------
        market_data : pd.DataFrame
            Market data with indicators

        Returns
        -------
        dict
            Dictionary of extracted features
        """
        # Ensure we have enough data
        if market_data is None or len(market_data) < 5:
            return {}

        # Get the most recent data points
        recent = market_data.iloc[-5:]

        features = {}

        # Price action features
        features["close"] = recent["close"].iloc[-1]
        features["open"] = recent["open"].iloc[-1]

        # Trend features
        if "ma" in recent.columns:
            features["price_to_ma"] = recent["close"].iloc[-1] / recent["ma"].iloc[-1]

        # Volatility features
        if "atr" in recent.columns:
            features["atr"] = recent["atr"].iloc[-1]
            if len(market_data) > 20:
                features["atr_percentile"] = (
                    sum(1 for x in market_data["atr"].iloc[-20:] if x < features["atr"])
                    / 20
                )

        # Momentum features
        if "rsi" in recent.columns:
            features["rsi"] = recent["rsi"].iloc[-1]
            features["rsi_slope"] = recent["rsi"].iloc[-1] - recent["rsi"].iloc[-3]

        # Trend features
        if all(col in recent.columns for col in ["dc_upper", "dc_lower"]):
            dc_middle = (recent["dc_upper"] + recent["dc_lower"]) / 2
            features["price_to_dc_middle"] = (
                recent["close"].iloc[-1] / dc_middle.iloc[-1]
            )

        # MACD features
        if all(col in recent.columns for col in ["macd", "macd_signal"]):
            features["macd_hist"] = (
                recent["macd"].iloc[-1] - recent["macd_signal"].iloc[-1]
            )
            if len(recent) > 1:
                prev_hist = recent["macd"].iloc[-2] - recent["macd_signal"].iloc[-2]
                features["macd_hist_change"] = features["macd_hist"] - prev_hist

        # Volume features
        if "volume" in recent.columns:
            features["volume"] = recent["volume"].iloc[-1]
            if len(market_data) > 20:
                avg_volume = market_data["volume"].iloc[-20:].mean()
                features["volume_ratio"] = (
                    features["volume"] / avg_volume if avg_volume > 0 else 1
                )

        # Check for two-way price action
        features["has_two_way_action"] = check_two_way_price_action(market_data)

        # Check for squeeze
        if "squeeze" in recent.columns:
            features["is_squeeze"] = recent["squeeze"].iloc[-1]

        return features

    def _evaluate_signal_heuristically(self, market_data, signal_type, position_side):
        """
        Evaluate signal quality using simple heuristics when ML is not available

        Parameters
        ----------
        market_data : pd.DataFrame
            Market data with indicators
        signal_type : str
            Type of signal to evaluate ("entry" or "exit")
        position_side : str
            Position side ("BUY" or "SELL")

        Returns
        -------
        dict
            Signal quality assessment
        """
        # Extract features
        features = self._extract_signal_features(market_data)

        score = 0.5  # Neutral starting point
        reasons = []

        # Entry signal evaluation
        if signal_type == "entry":
            # Check trend alignment
            if "price_to_ma" in features:
                if position_side == "BUY" and features["price_to_ma"] > 1.02:
                    score += 0.1
                    reasons.append("Price well above MA")
                elif position_side == "BUY" and features["price_to_ma"] > 1.0:
                    score += 0.05
                    reasons.append("Price above MA")
                elif position_side == "SELL" and features["price_to_ma"] < 0.98:
                    score += 0.1
                    reasons.append("Price well below MA")
                elif position_side == "SELL" and features["price_to_ma"] < 1.0:
                    score += 0.05
                    reasons.append("Price below MA")

            # Check RSI
            if "rsi" in features:
                if position_side == "BUY" and features["rsi"] > 60:
                    score += 0.1
                    reasons.append("Strong RSI")
                elif position_side == "SELL" and features["rsi"] < 40:
                    score += 0.1
                    reasons.append("Weak RSI")

            # Check MACD
            if "macd_hist" in features and "macd_hist_change" in features:
                if (
                    position_side == "BUY"
                    and features["macd_hist"] > 0
                    and features["macd_hist_change"] > 0
                ):
                    score += 0.1
                    reasons.append("Positive MACD histogram")
                elif (
                    position_side == "SELL"
                    and features["macd_hist"] < 0
                    and features["macd_hist_change"] < 0
                ):
                    score += 0.1
                    reasons.append("Negative MACD histogram")

            # Penalty for two-way price action
            if features.get("has_two_way_action", False):
                score -= 0.15
                reasons.append("Two-way price action detected")

        # Exit signal evaluation
        else:
            # Check trend reversal
            if "price_to_ma" in features:
                if position_side == "BUY" and features["price_to_ma"] < 0.98:
                    score += 0.15
                    reasons.append("Price below MA")
                elif position_side == "SELL" and features["price_to_ma"] > 1.02:
                    score += 0.15
                    reasons.append("Price above MA")

            # Check RSI reversal
            if "rsi" in features and "rsi_slope" in features:
                if position_side == "BUY" and features["rsi"] < 40:
                    score += 0.1
                    reasons.append("RSI below 40")
                elif position_side == "BUY" and features["rsi_slope"] < -10:
                    score += 0.1
                    reasons.append("RSI dropping")
                elif position_side == "SELL" and features["rsi"] > 60:
                    score += 0.1
                    reasons.append("RSI above 60")
                elif position_side == "SELL" and features["rsi_slope"] > 10:
                    score += 0.1
                    reasons.append("RSI rising")

            # Check volume
            if "volume_ratio" in features and features["volume_ratio"] > 1.5:
                score += 0.1
                reasons.append("High volume")

        # Cap score between 0 and 1
        score = max(0, min(1, score))

        # Classify confidence
        if score > 0.7:
            confidence = "high"
        elif score > 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        return {"score": score, "confidence": confidence, "reasons": reasons}

    def multi_timeframe_analysis(self, symbol):
        logger = logging.getLogger("turtle_trading_bot")

        try:
            # Emin olmak için veriyi güncelle
            self.update_market_data()

            # Veri yoksa işlemi durdur
            trend_data = self.data_manager.get_historical_data(
                symbol, self.trend_timeframe, limit=100
            )
            if trend_data is None or len(trend_data) < 20:
                logger.warning(f"Insufficient trend data for {symbol}")
                return False, None, None

            # MA sütununu kontrol et, yoksa hesapla
            if "ma" not in trend_data.columns:
                logger.info(f"MA column not found in trend data, calculating...")
                trend_data = calculate_ma(trend_data, 200)

            entry_data = self.data_manager.get_historical_data(
                symbol, self.entry_timeframe, limit=100
            )
            if entry_data is None or len(entry_data) < 20:
                logger.warning(f"Insufficient entry data for {symbol}")
                return False, None, None

            # Entry verisi için MA hesapla
            if "ma" not in entry_data.columns:
                logger.info(f"MA column not found in entry data, calculating...")
                entry_data = calculate_ma(entry_data, 200)

            base_data = self.data_manager.get_historical_data(
                symbol, self.timeframe, limit=100
            )
            if base_data is None or len(base_data) < 20:
                logger.warning(f"Insufficient base data for {symbol}")
                return False, None, None

            # Base verisi için MA hesapla
            if "ma" not in base_data.columns:
                logger.info(f"MA column not found in base data, calculating...")
                base_data = calculate_ma(base_data, 200)

            # Göstergeleri hesapla - MA'dan önce
            config = self.config

            trend_data = calculate_indicators(
                trend_data,
                config.dc_length_enter,
                config.dc_length_exit,
                config.atr_length,
                atr_smooth=config.atr_smoothing,
                ma_period=config.ma_period,
                include_additional=True,
            )

            entry_data = calculate_indicators(
                entry_data,
                config.dc_length_enter,
                config.dc_length_exit,
                config.atr_length,
                atr_smooth=config.atr_smoothing,
                ma_period=config.ma_period,
                include_additional=True,
            )

            base_data = calculate_indicators(
                base_data,
                config.dc_length_enter,
                config.dc_length_exit,
                config.atr_length,
                atr_smooth=config.atr_smoothing,
                ma_period=config.ma_period,
                include_additional=True,
            )

            # Her bir veri kümesinde MA olduğunu kontrol et
            for df_name, df in [
                ("trend_data", trend_data),
                ("entry_data", entry_data),
                ("base_data", base_data),
            ]:
                if "ma" not in df.columns or df["ma"].isna().all():
                    logger.warning(
                        f"MA column missing or all NaN in {df_name}, recalculating..."
                    )
                    # MA sütununu tekrar hesapla
                    df["ma"] = df["close"].rolling(window=200).mean()

            # Analiz koduna devam et...
            # ... (Mevcut analiz kodu) ...

            return True, None, None  # Başarı durumunu ve sonuçları döndür

        except KeyError as e:
            if str(e) == "'ma'":
                logger.warning("MA indicator not found, calculating...")

                # Verileri çektikten sonra doğrudan MA hesapla
                trend_data = self.data_manager.get_historical_data(
                    symbol, self.trend_timeframe, limit=100
                )
                entry_data = self.data_manager.get_historical_data(
                    symbol, self.entry_timeframe, limit=100
                )
                base_data = self.data_manager.get_historical_data(
                    symbol, self.timeframe, limit=100
                )

                # Her veri kümesi için MA hesapla
                trend_data["ma"] = trend_data["close"].rolling(window=200).mean()
                entry_data["ma"] = entry_data["close"].rolling(window=200).mean()
                base_data["ma"] = base_data["close"].rolling(window=200).mean()

                # Diğer göstergeleri hesapla
                config = self.config
                trend_data = calculate_indicators(
                    trend_data,
                    config.dc_length_enter,
                    config.dc_length_exit,
                    config.atr_length,
                )
                entry_data = calculate_indicators(
                    entry_data,
                    config.dc_length_enter,
                    config.dc_length_exit,
                    config.atr_length,
                )
                base_data = calculate_indicators(
                    base_data,
                    config.dc_length_enter,
                    config.dc_length_exit,
                    config.atr_length,
                )

                # Analizi tekrar çalıştır
                logger.info("Retrying analysis with recalculated MA...")
                return self.multi_timeframe_analysis(symbol)
            else:
                raise
        except Exception as e:
            logger.error(f"Error during multi-timeframe analysis: {e}")
            import traceback

            logger.error(traceback.format_exc())  # Tüm hata izini logla
            return False, None, None
