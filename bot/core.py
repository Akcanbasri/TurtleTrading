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
from datetime import datetime
import numpy as np

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
            symbol=self.config.symbol.lower(), callback=websocket_callback
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
            self.cache_access_times[cache_key] = time.time()
        else:
            # Önbellekte yoksa, normal hesaplama yap
            new_df = pd.DataFrame([new_row])
            self.data_cache[cache_key] = new_df
            self.cache_access_times[cache_key] = time.time()


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
        """Initialize the bot with configuration and connect to Binance"""
        self.logger = logging.getLogger("turtle_trading_bot")

        # Initialize configuration
        self.config = BotConfig(env_file=config_file)

        # Override config values if provided
        if api_key and api_secret:
            self.config.api_key = api_key
            self.config.api_secret = api_secret
        if use_testnet is not None:
            self.config.use_testnet = use_testnet

        # Initialize exchange
        self.exchange = BinanceExchange(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            testnet=self.config.use_testnet,
            demo_mode=demo_mode,
        )

        # Initialize data manager
        self.data_manager = DataManager(self.exchange)

        # Load initial state
        self.position_state = self._load_position_state()

        # Initialize WebSocket connection
        self._initialize_websocket()

        # Log initialization
        self._log_initialization()

    def _initialize_websocket(self):
        """Initialize WebSocket connection for real-time data"""
        try:

            def websocket_callback(data):
                """Handle WebSocket messages"""
                try:
                    # Process kline/candlestick data
                    if "k" in data:  # Kline/Candlestick data
                        kline = data["k"]
                        if kline["x"]:  # Candle is closed
                            self.logger.info(
                                f"New candle closed for {self.config.symbol}"
                            )
                            self.logger.debug(f"Candle data: {kline}")

                            # Update cache with new candle data
                            self._update_cache_with_new_candle(kline)

                            # Trigger analysis on new candle
                            self.analyze_market(force_update=True)
                        else:
                            # Update current price from open candle
                            self._update_real_time_price(
                                float(kline["c"]), candle_data=kline
                            )

                    # Process ticker data (more frequent price updates)
                    elif "data" in data and "a" in data["data"]:  # BookTicker data
                        ticker = data["data"]
                        if "a" in ticker:  # Ask price
                            self._update_real_time_price(float(ticker["a"]))

                    elif "e" in data and data["e"] == "error":
                        self.logger.error(f"WebSocket error: {data}")
                except Exception as e:
                    self.logger.error(f"Error in websocket callback: {e}")
                    import traceback

                    self.logger.error(
                        f"WebSocket callback error details: {traceback.format_exc()}"
                    )

            # Initialize WebSocket with callbacks
            self.exchange.initialize_websocket(
                symbol=self.config.symbol.lower(),
                callback=websocket_callback,
                on_ticker_callback=self.real_time_price_check,  # Add real-time price check
            )
            self.logger.info("WebSocket connection initialized successfully")

            # Initialize real-time price tracking
            self.current_price = None
            self.last_price_log_time = 0
            self.last_full_analysis_time = 0
            self.last_detailed_log_time = 0
            self.last_price_update_time = 0
            self.price_history = []  # Keep track of recent price movements

            # Initialize last candle data for tracking open positions
            self.last_candle = None

        except Exception as e:
            self.logger.error(f"Failed to initialize WebSocket: {e}")

    def _update_cache_with_new_candle(self, kline):
        """Update the data cache with new candle data from WebSocket"""
        try:
            # Create a new candle record
            symbol = self.config.symbol
            timeframe = self.config.timeframe

            # Format the candle data for the DataFrame
            new_candle = {
                "timestamp": pd.to_datetime(kline["t"], unit="ms"),
                "open": float(kline["o"]),
                "high": float(kline["h"]),
                "low": float(kline["l"]),
                "close": float(kline["c"]),
                "volume": float(kline["v"]),
                "close_time": pd.to_datetime(kline["T"], unit="ms"),
            }

            # Save the current candle data for later use
            self.last_candle = new_candle

            # Get the current data from cache
            cache_key = f"{symbol}_{timeframe}_100"

            if cache_key in self.data_manager.data_cache:
                # Get existing data
                df = self.data_manager.data_cache[cache_key].copy()

                # Create a new row with the timestamp as index
                new_row = pd.DataFrame([new_candle]).set_index("timestamp")

                # Check if this timestamp already exists in the dataframe
                if new_row.index[0] in df.index:
                    # Replace the existing row
                    df.loc[new_row.index[0]] = new_row.iloc[0]
                else:
                    # Append the new row
                    df = pd.concat([df, new_row])

                # Sort by timestamp and keep only the last 100 rows
                df = df.sort_index().iloc[-100:]

                # Update cache
                self.data_manager.data_cache[cache_key] = df
                self.data_manager.cache_access_times[cache_key] = time.time()

                self.logger.debug(
                    f"Updated data cache with new candle: Close={new_candle['close']}"
                )
            else:
                # No existing data, request a fresh set
                self.logger.debug(
                    "No existing data in cache, triggering a fresh data fetch"
                )
                self.data_manager.get_historical_data(symbol, timeframe, limit=100)

        except Exception as e:
            self.logger.error(f"Error updating cache with new candle: {e}")
            import traceback

            self.logger.error(f"Cache update error details: {traceback.format_exc()}")

    def _update_real_time_price(self, price, candle_data=None):
        """Update the current price in real-time with more detailed tracking"""
        if price is None or price <= 0:
            return

        # Record the previous price before updating
        prev_price = getattr(self, "current_price", None)
        self.current_price = price

        # Update price history for tracking short-term trends (keep last 100 prices)
        current_time = time.time()
        if not hasattr(self, "price_history"):
            self.price_history = []

        # Add only if sufficient time has passed (avoid too frequent updates)
        if (
            not hasattr(self, "last_price_update_time")
            or current_time - self.last_price_update_time > 5
        ):
            self.price_history.append((current_time, price))
            if len(self.price_history) > 100:  # Keep last 100 price points
                self.price_history.pop(0)
            self.last_price_update_time = current_time

        # Log price updates periodically (not for every tick)
        if (
            current_time - getattr(self, "last_price_log_time", 0) > 60
        ):  # Log price every minute
            # Calculate price change since last log
            price_change = 0
            if prev_price:
                price_change = (price - prev_price) / prev_price * 100  # Percentage

            self.logger.info(
                f"Current {self.config.symbol} price: {price:.8f} ({'+' if price_change >= 0 else ''}{price_change:.2f}%)"
            )
            self.last_price_log_time = current_time

            # If we have an active position, calculate and log current P&L
            if hasattr(self, "position_state") and self.position_state.active:
                # Calculate PnL
                entry_price = self.position_state.entry_price
                side = self.position_state.side
                stop_loss = self.position_state.stop_loss_price

                if side == "BUY":
                    pnl_pct = (price - entry_price) / entry_price * 100
                    risk = (
                        (entry_price - stop_loss) / entry_price * 100
                        if stop_loss
                        else 0
                    )
                else:
                    pnl_pct = (entry_price - price) / entry_price * 100
                    risk = (
                        (stop_loss - entry_price) / entry_price * 100
                        if stop_loss
                        else 0
                    )

                # Calculate reward-to-risk ratio
                reward_to_risk = abs(pnl_pct / risk) if risk != 0 else 0

                self.logger.info(
                    f"Position: {side} Entry: {entry_price:.8f} Current P&L: {pnl_pct:.2f}% R:R: {reward_to_risk:.2f}"
                )

                # Check for stop loss in real-time
                if self.check_stop_loss(price):
                    self.logger.warning(
                        f"Stop loss would be triggered at current price {price:.8f}"
                    )

                # Calculate and log days/hours the position has been open
                if (
                    hasattr(self.position_state, "entry_time")
                    and self.position_state.entry_time
                ):
                    entry_time = self.position_state.entry_time
                    if isinstance(entry_time, int):  # milliseconds timestamp
                        elapsed_hours = (current_time - entry_time / 1000) / 3600
                    else:  # datetime object
                        elapsed_hours = (current_time - entry_time.timestamp()) / 3600

                    if elapsed_hours > 24:
                        self.logger.info(f"Position age: {elapsed_hours/24:.1f} days")
                    else:
                        self.logger.info(f"Position age: {elapsed_hours:.1f} hours")

        # Trigger additional checks or analysis periodically
        # Use shorter interval if we have an active position
        analysis_interval = (
            150
            if (hasattr(self, "position_state") and self.position_state.active)
            else 300
        )

        if (
            current_time - getattr(self, "last_full_analysis_time", 0)
            > analysis_interval
        ):
            self.logger.debug("Triggering market analysis based on price update...")
            self.analyze_market(
                force_update=False
            )  # Don't force a data update if not needed
            self.last_full_analysis_time = current_time

    def analyze_market(self, force_update=False):
        """Analyze market conditions and execute trading logic"""
        try:
            # Get latest market data
            df = self.data_manager.get_historical_data(
                self.config.symbol, self.config.timeframe, limit=100
            )

            if df is None or len(df) < 20:
                self.logger.warning("Insufficient market data for analysis")
                return

            # Check if df needs updating with current price (for real-time sensitivity)
            if force_update or (
                hasattr(self, "current_price") and self.current_price is not None
            ):
                # Update last row with current price if it's different
                last_row_price = df["close"].iloc[-1]
                current_price = self.current_price

                # If the last candle price and current price are significantly different
                if (
                    abs((current_price - last_row_price) / last_row_price) > 0.0001
                ):  # 0.01% change
                    self.logger.debug(
                        f"Updating analysis with current price: {current_price} (was {last_row_price})"
                    )
                    # Update last row in dataframe
                    df.loc[df.index[-1], "close"] = current_price

                    # If current price is higher than previous high, update the high
                    if current_price > df["high"].iloc[-1]:
                        df.loc[df.index[-1], "high"] = current_price

                    # If current price is lower than previous low, update the low
                    if current_price < df["low"].iloc[-1]:
                        df.loc[df.index[-1], "low"] = current_price

            # Calculate indicators
            df = calculate_indicators(
                df=df,
                dc_enter=self.config.dc_length_enter,
                dc_exit=self.config.dc_length_exit,
                atr_len=self.config.atr_length,
                atr_smooth=self.config.atr_smoothing,
                ma_period=self.config.ma_period,
                adx_period=self.config.adx_period,
            )

            # Update the current price from the latest candle or WebSocket
            if hasattr(self, "current_price") and self.current_price is not None:
                current_price = self.current_price
            else:
                current_price = df["close"].iloc[-1]
                self.current_price = current_price

            # Log detailed market condition (with reduced frequency)
            current_time = time.time()
            if (
                not hasattr(self, "last_detailed_log_time")
                or current_time - self.last_detailed_log_time > 300
                or force_update  # Log details on force update
            ):
                self._log_market_condition(df)
                self.last_detailed_log_time = current_time
            else:
                # Log only essential info if we logged details recently
                self.logger.info(
                    f"Market update - Price: {current_price:.8f}, DC Upper: {df['dc_upper'].iloc[-1]:.8f}, DC Lower: {df['dc_lower'].iloc[-1]:.8f}"
                )

            # Check for entry/exit signals
            current_atr = df["atr"].iloc[-1]
            if not self.position_state.active:
                self._check_entry_signals(df, current_price, current_atr)
            else:
                self._check_exit_signals(df, current_price, current_atr)

        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

    def _log_initialization(self):
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

        if self.position_state.active:
            self.logger.info("Bot started with active position:")
            self.log_position_state()

    def _log_market_condition(self, df):
        """Log detailed market condition including trend direction, key indicators and position status"""
        try:
            # Get latest candle data
            current_price = df["close"].iloc[-1]
            current_atr = df["atr"].iloc[-1]

            # Trend analysis
            ma_period = (
                self.config.ma_period if hasattr(self.config, "ma_period") else 50
            )
            if "ma" not in df.columns and len(df) > ma_period:
                df["ma"] = df["close"].rolling(window=ma_period).mean()

            # Calculate trend direction
            trend_direction = "NEUTRAL"
            if "ma" in df.columns:
                ma_value = df["ma"].iloc[-1]
                price_vs_ma = current_price / ma_value - 1

                if price_vs_ma > 0.02:  # 2% above MA
                    trend_direction = "STRONG_BULLISH"
                elif price_vs_ma > 0.005:  # 0.5% above MA
                    trend_direction = "BULLISH"
                elif price_vs_ma < -0.02:  # 2% below MA
                    trend_direction = "STRONG_BEARISH"
                elif price_vs_ma < -0.005:  # 0.5% below MA
                    trend_direction = "BEARISH"

            # Calculate ADX for trend strength if available
            trend_strength = "UNKNOWN"
            if "adx" in df.columns:
                adx_value = df["adx"].iloc[-1]
                if adx_value > 30:
                    trend_strength = "STRONG"
                elif adx_value > 20:
                    trend_strength = "MODERATE"
                else:
                    trend_strength = "WEAK"

            # Get Donchian Channel values
            dc_upper = df["dc_upper"].iloc[-1]
            dc_lower = df["dc_lower"].iloc[-1]
            dc_width = (
                (dc_upper - dc_lower) / current_price * 100
            )  # As percentage of price

            # Calculate distance from channel boundaries
            upper_distance = (dc_upper - current_price) / current_atr
            lower_distance = (current_price - dc_lower) / current_atr

            # Log market status header
            self.logger.info("=" * 50)
            self.logger.info(
                f"MARKET ANALYSIS - {self.config.symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            self.logger.info("=" * 50)

            # Price information
            self.logger.info(f"PRICE DATA:")
            self.logger.info(f"  Current Price: {current_price:.8f}")
            self.logger.info(
                f"  ATR (Volatility): {current_atr:.8f} ({current_atr/current_price*100:.2f}%)"
            )
            self.logger.info(f"  Donchian Upper: {dc_upper:.8f}")
            self.logger.info(f"  Donchian Lower: {dc_lower:.8f}")
            self.logger.info(f"  Channel Width: {dc_width:.2f}% of price")

            # Trend information
            self.logger.info(f"TREND ANALYSIS:")
            self.logger.info(f"  Direction: {trend_direction}")
            self.logger.info(f"  Strength: {trend_strength}")
            self.logger.info(f"  Price vs Upper Channel: {upper_distance:.2f} ATR")
            self.logger.info(f"  Price vs Lower Channel: {lower_distance:.2f} ATR")

            # Position information if active
            if self.position_state.active:
                position_age = (
                    datetime.now() - self.position_state.entry_time
                ).total_seconds() / 3600  # in hours
                entry_price = self.position_state.entry_price
                side = self.position_state.side
                stop_loss = self.position_state.stop_loss_price
                quantity = self.position_state.quantity

                # Calculate profit/loss
                if side == "BUY":
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    risk_reward = (
                        pnl_pct / (abs(entry_price - stop_loss) / entry_price * 100)
                        if stop_loss
                        else 0
                    )
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                    risk_reward = (
                        pnl_pct / (abs(entry_price - stop_loss) / entry_price * 100)
                        if stop_loss
                        else 0
                    )

                self.logger.info(f"POSITION STATUS:")
                self.logger.info(f"  Side: {side}")
                self.logger.info(f"  Entry Price: {entry_price:.8f}")
                self.logger.info(f"  Current P&L: {pnl_pct:.2f}%")
                self.logger.info(
                    f"  Stop Loss: {stop_loss:.8f} ({abs(stop_loss-entry_price)/entry_price*100:.2f}%)"
                )
                self.logger.info(f"  Risk/Reward: {risk_reward:.2f}")
                self.logger.info(f"  Quantity: {quantity}")
                self.logger.info(f"  Position Age: {position_age:.1f} hours")
            else:
                self.logger.info("POSITION STATUS: No active position")

            # Account information
            try:
                account_balance = float(self.exchange.get_account_balance())
                self.logger.info(f"ACCOUNT STATUS:")

                # Determine if we're in demo mode
                mode_label = (
                    "DEMO" if getattr(self.exchange, "demo_mode", False) else "REAL"
                )
                self.logger.info(f"  Trading Mode: {mode_label}")
                self.logger.info(
                    f"  Balance: {account_balance:.8f} USDT ({mode_label} ACCOUNT)"
                )

                # Calculate max position size based on available balance
                max_pos_notional = (
                    account_balance * 0.95
                )  # Use up to 95% of available balance
                max_pos_size = max_pos_notional / current_price

                # Calculate potential position size based on risk parameters
                risk_per_trade = float(self.config.risk_per_trade)
                risk_amount = account_balance * risk_per_trade
                stop_loss_pct = (
                    float(self.config.stop_loss_atr_multiple)
                    * float(current_atr)
                    / current_price
                )

                self.logger.info(
                    f"  Risk per trade: ${risk_amount:.8f} ({risk_per_trade*100:.1f}% of balance)"
                )
                self.logger.info(
                    f"  Stop loss distance: {current_atr * self.config.stop_loss_atr_multiple:.8f} ({self.config.stop_loss_atr_multiple} ATR)"
                )

                if stop_loss_pct > 0:
                    potential_position_size = risk_amount / (
                        current_price * stop_loss_pct
                    )
                else:
                    potential_position_size = 0

                # Adjust if potential position size is too large
                if potential_position_size * current_price > max_pos_notional:
                    potential_position_size = max_pos_size

                self.logger.info(
                    f"  Potential Position Size: {potential_position_size:.8f} {self.config.symbol.split('USDT')[0]}"
                )
            except Exception as e:
                self.logger.warning(f"Failed to retrieve account information: {e}")
                import traceback

                self.logger.debug(
                    f"Account info error details: {traceback.format_exc()}"
                )

            # Signal quality log if calculated
            try:
                signal_quality = self.evaluate_signal_quality_with_ml(df, "analysis")
                if signal_quality and isinstance(signal_quality, dict):
                    self.logger.info(f"SIGNAL ANALYSIS:")
                    self.logger.info(
                        f"  Overall Quality: {signal_quality.get('overall_score', 0):.2f}/10"
                    )
                    self.logger.info(
                        f"  Classification: {signal_quality.get('classification', 'Unknown')}"
                    )

                    # Log component scores if available
                    components = signal_quality.get("component_scores", {})
                    if components:
                        self.logger.info(
                            f"  Trend Score: {components.get('trend', 0):.2f}/10"
                        )
                        self.logger.info(
                            f"  Momentum Score: {components.get('momentum', 0):.2f}/10"
                        )
                        self.logger.info(
                            f"  Volatility Score: {components.get('volatility', 0):.2f}/10"
                        )
                        self.logger.info(
                            f"  Volume Score: {components.get('volume', 0):.2f}/10"
                        )
            except Exception as e:
                self.logger.debug(f"Error calculating signal quality: {e}")

            self.logger.info("=" * 50)

        except Exception as e:
            self.logger.error(f"Error in market condition logging: {e}")
            import traceback

            self.logger.error(
                f"Market condition logging traceback: {traceback.format_exc()}"
            )

    def _check_entry_signals(self, df, current_price, current_atr):
        """Check for entry signals"""
        try:
            # Check for long entry
            if current_price > df["dc_upper"].iloc[-1]:
                self.logger.info("Long entry signal detected")
                signal_quality = self.evaluate_signal_quality_with_ml(
                    df, "entry", "BUY"
                )
                self.logger.info(f"Signal quality: {signal_quality}")

                if signal_quality["score"] > 0.7:  # High quality signal
                    self.logger.info("Executing long entry")
                    self._execute_entry(
                        direction="long",
                        current_price=current_price,
                        atr_value=current_atr,
                        signal_strength=signal_quality["score"],
                    )

            # Check for short entry
            elif current_price < df["dc_lower"].iloc[-1]:
                self.logger.info("Short entry signal detected")
                signal_quality = self.evaluate_signal_quality_with_ml(
                    df, "entry", "SELL"
                )
                self.logger.info(f"Signal quality: {signal_quality}")

                if signal_quality["score"] > 0.7:  # High quality signal
                    self.logger.info("Executing short entry")
                    self._execute_entry(
                        direction="short",
                        current_price=current_price,
                        atr_value=current_atr,
                        signal_strength=signal_quality["score"],
                    )

        except Exception as e:
            self.logger.error(f"Error checking entry signals: {e}")

    def _check_exit_signals(self, df, current_price, current_atr):
        """Check for exit signals"""
        try:
            if self.position_state.active:
                # Check stop loss
                if self.check_stop_loss(current_price):
                    self.logger.info("Stop loss triggered")
                    self._execute_exit(current_price, "stop_loss")
                    return

                # Check take profit
                if self.position_state.side == "BUY":
                    if current_price >= self.position_state.take_profit_price:
                        self.logger.info("Take profit triggered for long position")
                        self._execute_exit(
                            self.position_state.take_profit_price, "take_profit"
                        )
                        return
                else:  # SELL
                    if current_price <= self.position_state.take_profit_price:
                        self.logger.info("Take profit triggered for short position")
                        self._execute_exit(
                            self.position_state.take_profit_price, "take_profit"
                        )
                        return

                # Check Donchian Channel exit
                if self.check_exit_signal():
                    self.logger.info("Donchian Channel exit signal detected")
                    signal_quality = self.evaluate_signal_quality_with_ml(
                        df, "exit", self.position_state.side
                    )
                    self.logger.info(f"Exit signal quality: {signal_quality}")

                    if signal_quality["score"] > 0.6:  # Medium to high quality signal
                        self.logger.info("Executing exit")
                        self._execute_exit(current_price, "dc_exit")

        except Exception as e:
            self.logger.error(f"Error checking exit signals: {e}")

    def _execute_exit(self, exit_price, exit_reason):
        """
        Execute exit order for current position

        Parameters
        ----------
        exit_price : float
            Current market price for exit
        exit_reason : str
            Reason for exiting the position (stop_loss, take_profit, exit_signal)

        Returns
        -------
        bool
            True if exit was successful, False otherwise
        """
        try:
            if not self.position_state.active:
                self.logger.warning("Attempted to exit non-existent position")
                return False

            self.logger.info(
                f"Executing position exit at {exit_price}, reason: {exit_reason}"
            )

            # Calculate profit/loss
            entry_price = self.position_state.entry_price
            position_size = self.position_state.position_size
            side = self.position_state.side

            # Determine exit side (opposite of entry)
            exit_side = "SELL" if side == "BUY" else "BUY"

            # Calculate P&L in percentage and absolute terms
            if side == "BUY":
                pnl_percent = (exit_price - entry_price) / entry_price * 100
                pnl_absolute = (exit_price - entry_price) * position_size
            else:
                pnl_percent = (entry_price - exit_price) / entry_price * 100
                pnl_absolute = (entry_price - exit_price) * position_size

            self.logger.info(
                f"P&L for this trade: {pnl_percent:.2f}% ({pnl_absolute:.2f} units)"
            )

            # Record trade metrics
            trade_duration_ms = int(time.time() * 1000) - self.position_state.entry_time
            trade_duration_hours = trade_duration_ms / (1000 * 60 * 60)
            self.logger.info(f"Trade duration: {trade_duration_hours:.2f} hours")

            # Execute exit order
            order_result = self.exchange.create_market_order(
                symbol=self.config.symbol, side=exit_side, quantity=position_size
            )

            if order_result and "orderId" in order_result:
                # Update trade history
                trade_record = {
                    "entry_time": self.position_state.entry_time,
                    "exit_time": int(time.time() * 1000),
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "position_size": position_size,
                    "pnl_percent": pnl_percent,
                    "pnl_absolute": pnl_absolute,
                    "duration_ms": trade_duration_ms,
                    "exit_reason": exit_reason,
                    "atr_at_entry": self.position_state.atr_at_entry,
                }

                # Append trade record to history
                self._update_trade_history(trade_record)

                # Reset position state
                self.position_state.active = False
                self.position_state.side = None
                self.position_state.entry_price = 0
                self.position_state.stop_loss_price = 0
                self.position_state.position_size = 0
                self.position_state.entry_time = 0
                self.position_state.atr_at_entry = 0
                self.position_state.order_id = None

                # Save updated position state
                self.save_state()

                self.logger.info(
                    f"Exit successful: Closed {side} position of {position_size} units at {exit_price}"
                )
                return True
            else:
                self.logger.error(f"Exit order failed: {order_result}")
                return False

        except Exception as e:
            self.logger.error(f"Error executing exit: {e}")
            return False

    def _update_trade_history(self, trade_record):
        """
        Update the trade history with a new trade record

        Parameters
        ----------
        trade_record : dict
            Record of the completed trade
        """
        try:
            # Load existing trade history or create empty list
            trade_history_file = os.path.join(
                self.config.data_dir, f"{self.config.symbol}_trade_history.json"
            )

            if os.path.exists(trade_history_file):
                with open(trade_history_file, "r") as f:
                    trade_history = json.load(f)
            else:
                trade_history = []

            # Add new trade to history
            trade_history.append(trade_record)

            # Save updated history
            with open(trade_history_file, "w") as f:
                json.dump(trade_history, f, indent=4)

            self.logger.info(
                f"Updated trade history, now contains {len(trade_history)} trades"
            )

            # Log recent performance
            if len(trade_history) >= 5:
                recent_trades = trade_history[-5:]
                win_count = sum(1 for t in recent_trades if t["pnl_percent"] > 0)
                total_pnl = sum(t["pnl_percent"] for t in recent_trades)
                avg_pnl = total_pnl / len(recent_trades)

                self.logger.info(
                    f"Recent performance (last 5 trades): Win rate: {win_count}/5, Avg PnL: {avg_pnl:.2f}%"
                )

        except Exception as e:
            self.logger.error(f"Error updating trade history: {e}")

    def _load_position_state(self):
        """Load position state from file or create new one"""
        try:
            # Try to load the state from file
            position_state = load_position_state(self.config.symbol)

            # If no state file or error loading, create a new one
            if position_state is None:
                self.logger.info("No saved position state found, creating new state")
                position_state = PositionState()

            return position_state
        except Exception as e:
            self.logger.error(f"Error loading bot state: {e}")
            # Return a new position state if there's an error
            return PositionState()

    def save_state(self):
        """Save the current position state to file"""
        try:
            save_position_state(self.position_state, self.config.symbol)
        except Exception as e:
            self.logger.error(f"Error saving bot state: {e}")

    def check_stop_loss(self, current_price):
        """Check if stop loss has been triggered for the current position"""
        if not self.position_state.active:
            return False

        if self.position_state.side == "BUY":
            # For long positions, stop loss is below entry
            return current_price <= self.position_state.stop_loss_price
        else:
            # For short positions, stop loss is above entry
            return current_price >= self.position_state.stop_loss_price

    def check_exit_signal(self):
        """Check for Donchian Channel exit signals for the current position"""
        try:
            if not self.position_state.active:
                return False

            # Get latest market data
            df = self.data_manager.get_historical_data(
                self.config.symbol, self.config.timeframe, limit=50
            )

            if df is None or len(df) < 20:
                self.logger.warning("Insufficient data for exit signal check")
                return False

            # Calculate indicators if not already present
            if "dc_upper" not in df.columns or "dc_lower" not in df.columns:
                df = calculate_indicators(
                    df=df,
                    dc_enter=self.config.dc_length_enter,
                    dc_exit=self.config.dc_length_exit,
                    atr_len=self.config.atr_length,
                )

            # Get current price and exit channel levels
            current_price = df["close"].iloc[-1]
            dc_upper = df["dc_upper"].iloc[-1]
            dc_lower = df["dc_lower"].iloc[-1]

            # Check exit conditions based on position side
            if self.position_state.side == "BUY":
                # Exit long position when price breaks below the lower exit channel
                exit_signal = current_price < dc_lower
                if exit_signal:
                    self.logger.info(
                        f"Long exit signal: price {current_price} broke below DC lower {dc_lower}"
                    )
            else:  # SELL
                # Exit short position when price breaks above the upper exit channel
                exit_signal = current_price > dc_upper
                if exit_signal:
                    self.logger.info(
                        f"Short exit signal: price {current_price} broke above DC upper {dc_upper}"
                    )

            return exit_signal

        except Exception as e:
            self.logger.error(f"Error checking exit signal: {e}")
            return False

    def evaluate_signal_quality_with_ml(self, market_data, signal_type, side=None):
        """
        Evaluate the quality of a trading signal using ML features

        Parameters
        ----------
        market_data : pd.DataFrame
            Market data with indicators
        signal_type : str
            Type of signal to evaluate ("entry" or "exit")
        side : str, optional
            Position side for entry signals ("BUY" or "SELL")

        Returns
        -------
        dict
            Signal quality evaluation result
        """
        try:
            self.logger.info(
                f"Evaluating {signal_type} signal quality{f' for {side}' if side else ''}"
            )

            # Extract features for signal quality evaluation
            features = self._extract_signal_features(market_data)

            # Log extracted features for debugging
            self.logger.debug(f"Signal features: {features}")

            # Calculate trend strength
            trend_score = self._calculate_trend_score(market_data, side)
            self.logger.info(f"Trend score: {trend_score:.2f} (Range: 0-10)")

            # Calculate momentum indicators
            momentum_score = self._calculate_momentum_score(market_data, side)
            self.logger.info(f"Momentum score: {momentum_score:.2f} (Range: 0-10)")

            # Calculate volatility assessment
            volatility_score = self._calculate_volatility_score(market_data)
            self.logger.info(f"Volatility score: {volatility_score:.2f} (Range: 0-10)")

            # Calculate volume confirmation
            volume_score = self._calculate_volume_score(market_data, side)
            self.logger.info(f"Volume score: {volume_score:.2f} (Range: 0-10)")

            # Calculate overall signal score
            overall_score = (
                trend_score * 0.4
                + momentum_score * 0.3
                + volatility_score * 0.15
                + volume_score * 0.15
            )

            self.logger.info(
                f"Overall signal quality score: {overall_score:.2f} (Range: 0-10)"
            )

            # Classify signal based on overall score
            if overall_score >= 7.5:
                classification = "strong_signal"
                confidence = 0.9
            elif overall_score >= 5.0:
                classification = "moderate_signal"
                confidence = 0.7
            else:
                classification = "weak_signal"
                confidence = 0.5

            # Log market condition context
            self._log_market_condition(market_data)

            return {
                "classification": classification,
                "confidence": confidence,
                "overall_score": overall_score,
                "components": {
                    "trend": trend_score,
                    "momentum": momentum_score,
                    "volatility": volatility_score,
                    "volume": volume_score,
                },
            }

        except Exception as e:
            self.logger.error(f"Error evaluating signal quality: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

            # Return default conservative evaluation on error
            return {
                "classification": "weak_signal",
                "confidence": 0.3,
                "overall_score": 3.0,
                "components": {
                    "trend": 3.0,
                    "momentum": 3.0,
                    "volatility": 3.0,
                    "volume": 3.0,
                },
            }

    def _extract_signal_features(self, market_data):
        """Extract ML features from market data for signal evaluation"""
        # Last row contains most recent data
        latest = market_data.iloc[-1]

        # Calculate basic features
        features = {
            # Price action features
            "close": latest["close"],
            "high_low_ratio": (
                latest["high"] / latest["low"] if latest["low"] > 0 else 1.0
            ),
            # Indicator features
            "atr": latest["atr"] if "atr" in latest else None,
            "rsi": latest["rsi"] if "rsi" in latest else None,
            "macd": latest["macd"] if "macd" in latest else None,
            "macd_signal": latest["macd_signal"] if "macd_signal" in latest else None,
            "bb_width": latest["bb_width"] if "bb_width" in latest else None,
            # Volume features
            "volume": latest["volume"] if "volume" in latest else None,
            "volume_ma": latest["volume_ma"] if "volume_ma" in latest else None,
        }

        # Include Donchian Channel features if available
        if "dc_upper" in latest and "dc_lower" in latest:
            dc_width = (
                (latest["dc_upper"] - latest["dc_lower"]) / latest["close"]
                if latest["close"] > 0
                else 0
            )
            features["dc_width"] = dc_width
            features["price_dc_upper_ratio"] = (
                latest["close"] / latest["dc_upper"] if latest["dc_upper"] > 0 else 1.0
            )
            features["price_dc_lower_ratio"] = (
                latest["close"] / latest["dc_lower"] if latest["dc_lower"] > 0 else 1.0
            )

        # Calculate trend direction based on moving averages if available
        if "sma_20" in latest and "sma_50" in latest and "sma_100" in latest:
            features["ma_trend"] = (
                (1 if latest["sma_20"] > latest["sma_50"] else -1)
                + (1 if latest["sma_50"] > latest["sma_100"] else -1)
            ) / 2  # Range: -1 to 1

        return features

    def _calculate_trend_score(self, market_data, side):
        """Calculate trend strength score (0-10)"""
        latest = market_data.iloc[-1]
        score = 5.0  # Neutral starting point

        # Check for moving average alignment
        if "sma_20" in latest and "sma_50" in latest and "sma_100" in latest:
            # For uptrend (BUY signals)
            if side == "BUY":
                if latest["sma_20"] > latest["sma_50"] > latest["sma_100"]:
                    score += 2.5  # Strong uptrend
                elif latest["sma_20"] > latest["sma_50"]:
                    score += 1.5  # Moderate uptrend
                elif latest["sma_20"] < latest["sma_50"]:
                    score -= 1.5  # Potential downtrend
            # For downtrend (SELL signals)
            elif side == "SELL":
                if latest["sma_20"] < latest["sma_50"] < latest["sma_100"]:
                    score += 2.5  # Strong downtrend
                elif latest["sma_20"] < latest["sma_50"]:
                    score += 1.5  # Moderate downtrend
                elif latest["sma_20"] > latest["sma_50"]:
                    score -= 1.5  # Potential uptrend

        # Check for trend continuation with Donchian Channel breakout
        if "dc_upper" in latest and "dc_lower" in latest:
            if side == "BUY" and latest["close"] >= latest["dc_upper"] * 0.995:
                score += 1.5  # Price near upper Donchian Channel
            elif side == "SELL" and latest["close"] <= latest["dc_lower"] * 1.005:
                score += 1.5  # Price near lower Donchian Channel

        # Adjust based on ADX if available
        if "adx" in latest:
            adx_value = latest["adx"]
            if adx_value > 30:
                score += 1.0  # Strong trend
            elif adx_value > 20:
                score += 0.5  # Moderate trend
            else:
                score -= 0.5  # Weak trend

        # Ensure score stays within 0-10 range
        return max(0, min(10, score))

    def _calculate_momentum_score(self, market_data, side):
        """Calculate momentum indicators score (0-10)"""
        latest = market_data.iloc[-1]
        score = 5.0  # Neutral starting point

        # Check RSI
        if "rsi" in latest:
            rsi = latest["rsi"]
            # For BUY signals
            if side == "BUY":
                if 40 <= rsi <= 60:
                    score += 1.0  # Neutral zone
                elif 60 < rsi <= 70:
                    score += 1.5  # Bullish momentum without being overbought
                elif rsi > 70:
                    score -= 1.0  # Overbought
                elif rsi < 30:
                    score -= 0.5  # Oversold but against trend
            # For SELL signals
            elif side == "SELL":
                if 40 <= rsi <= 60:
                    score += 1.0  # Neutral zone
                elif 30 <= rsi < 40:
                    score += 1.5  # Bearish momentum without being oversold
                elif rsi < 30:
                    score -= 1.0  # Oversold
                elif rsi > 70:
                    score -= 0.5  # Overbought but against trend

        # Check MACD
        if "macd" in latest and "macd_signal" in latest:
            macd = latest["macd"]
            macd_signal = latest["macd_signal"]
            # For BUY signals
            if side == "BUY":
                if macd > macd_signal and macd > 0:
                    score += 1.5  # Bullish MACD crossover above zero
                elif macd > macd_signal:
                    score += 0.75  # Bullish MACD crossover below zero
                elif macd < macd_signal:
                    score -= 1.0  # Bearish MACD
            # For SELL signals
            elif side == "SELL":
                if macd < macd_signal and macd < 0:
                    score += 1.5  # Bearish MACD crossover below zero
                elif macd < macd_signal:
                    score += 0.75  # Bearish MACD crossover above zero
                elif macd > macd_signal:
                    score -= 1.0  # Bullish MACD

        # Check recent price action momentum
        if len(market_data) >= 5:
            price_5_periods_ago = market_data["close"].iloc[-5]
            current_price = latest["close"]
            price_change_pct = (
                (current_price - price_5_periods_ago) / price_5_periods_ago
            ) * 100

            # For BUY signals
            if side == "BUY":
                if price_change_pct > 3:
                    score += 1.0  # Strong recent upward momentum
                elif price_change_pct > 1:
                    score += 0.5  # Moderate recent upward momentum
                elif price_change_pct < -2:
                    score -= 1.0  # Recent downward momentum
            # For SELL signals
            elif side == "SELL":
                if price_change_pct < -3:
                    score += 1.0  # Strong recent downward momentum
                elif price_change_pct < -1:
                    score += 0.5  # Moderate recent downward momentum
                elif price_change_pct > 2:
                    score -= 1.0  # Recent upward momentum

        # Ensure score stays within 0-10 range
        return max(0, min(10, score))

    def _calculate_volatility_score(self, market_data):
        """Calculate volatility assessment score (0-10)"""
        latest = market_data.iloc[-1]
        score = 5.0  # Neutral starting point

        # Check ATR relative to price
        if "atr" in latest:
            atr_pct = (latest["atr"] / latest["close"]) * 100

            # Prefer moderate volatility, penalize extreme values
            if 0.5 <= atr_pct <= 2.0:
                score += 2.0  # Ideal volatility
            elif 2.0 < atr_pct <= 3.0:
                score += 1.0  # Higher but still acceptable
            elif atr_pct > 3.0:
                score -= 1.0  # Too volatile
            elif atr_pct < 0.3:
                score -= 0.5  # Too low volatility

        # Check Bollinger Band width if available
        if "bb_width" in latest:
            bb_width = latest["bb_width"]

            # Score based on Bollinger Band width (normalized values assumed)
            if 0.5 <= bb_width <= 1.5:
                score += 1.5  # Normal volatility
            elif 1.5 < bb_width <= 2.5:
                score += 0.75  # Higher volatility
            elif bb_width > 2.5:
                score -= 1.0  # Extremely high volatility
            elif bb_width < 0.3:
                score -= 1.0  # Very low volatility

        # Check recent historical volatility
        if len(market_data) >= 10:
            recent_close_values = market_data["close"].iloc[-10:].values
            historical_volatility = (
                np.std(np.diff(recent_close_values) / recent_close_values[:-1]) * 100
            )

            if 0.5 <= historical_volatility <= 2.0:
                score += 1.5  # Ideal historical volatility
            elif 2.0 < historical_volatility <= 3.0:
                score += 0.5  # Higher but tradeable
            elif historical_volatility > 3.0:
                score -= 1.0  # Too volatile historically
            elif historical_volatility < 0.3:
                score -= 0.5  # Too stable historically

        # Ensure score stays within 0-10 range
        return max(0, min(10, score))

    def _calculate_volume_score(self, market_data, side):
        """Calculate volume confirmation score (0-10)"""
        if "volume" not in market_data.columns:
            return 5.0  # Neutral score if volume data not available

        latest = market_data.iloc[-1]
        previous = market_data.iloc[-2] if len(market_data) > 1 else latest
        score = 5.0  # Neutral starting point

        # Check volume trend
        if "volume_ma" in latest:
            volume = latest["volume"]
            volume_ma = latest["volume_ma"]

            # High volume is good for confirming moves
            if volume > volume_ma * 1.5:
                score += 2.0  # Very high volume
            elif volume > volume_ma * 1.2:
                score += 1.0  # Above average volume
            elif volume < volume_ma * 0.8:
                score -= 0.5  # Below average volume

        # Check volume with price direction
        if len(market_data) >= 2:
            price_change = latest["close"] - previous["close"]

            # For BUY signals
            if (
                side == "BUY"
                and price_change > 0
                and latest["volume"] > previous["volume"]
            ):
                score += 1.5  # Increasing price with increasing volume (confirmation)
            # For SELL signals
            elif (
                side == "SELL"
                and price_change < 0
                and latest["volume"] > previous["volume"]
            ):
                score += 1.5  # Decreasing price with increasing volume (confirmation)
            # Volume not confirming price movement
            elif (
                side == "BUY"
                and price_change > 0
                and latest["volume"] < previous["volume"]
            ) or (
                side == "SELL"
                and price_change < 0
                and latest["volume"] < previous["volume"]
            ):
                score -= (
                    0.5  # Price moving in expected direction but with decreasing volume
                )

        # Check for volume spikes (can indicate exhaustion or strong momentum)
        if len(market_data) >= 5:
            avg_volume_5_periods = market_data["volume"].iloc[-5:].mean()
            if latest["volume"] > avg_volume_5_periods * 2:
                # Volume spike - could be exhaustion or strong confirmation
                # For BUY signals at resistance or SELL signals at support
                if (
                    side == "BUY"
                    and "dc_upper" in latest
                    and latest["close"] > latest["dc_upper"] * 0.95
                ) or (
                    side == "SELL"
                    and "dc_lower" in latest
                    and latest["close"] < latest["dc_lower"] * 1.05
                ):
                    score += 1.0  # Volume spike confirming breakout

        # Ensure score stays within 0-10 range
        return max(0, min(10, score))

    def _calculate_position_size(self, current_price, stop_loss_amount):
        """
        Calculate position size based on risk parameters

        Parameters
        ----------
        current_price : float
            Current market price
        stop_loss_amount : float
            Stop loss amount in price units

        Returns
        -------
        float
            Position size in base currency
        """
        try:
            # Get account balance
            account_balance = float(self.exchange.get_account_balance())

            # Calculate risk amount in USDT
            risk_amount = account_balance * float(
                self.config.risk_per_trade
            )  # Convert to float if needed

            # Calculate risk per unit
            risk_per_unit = float(abs(stop_loss_amount))  # Convert to float for safety

            # Calculate position size in base currency
            if risk_per_unit > 0:
                position_size = risk_amount / risk_per_unit

                # Get symbol information for rounding
                symbol_info = self.exchange.get_symbol_info(self.config.symbol)
                if symbol_info:
                    # Get quantity precision from symbol info
                    qty_precision = getattr(symbol_info, "quantity_precision", 0)
                    min_qty = float(
                        getattr(symbol_info, "min_qty", 0.001)
                    )  # Convert to float

                    # Round to appropriate precision
                    position_size = max(min_qty, position_size)
                    position_size = round(position_size, qty_precision)

                return position_size
            else:
                self.logger.warning("Invalid risk_per_unit (zero or negative)")
                return 0

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            import traceback

            self.logger.error(
                f"Position size calculation traceback: {traceback.format_exc()}"
            )
            return 0

    def real_time_price_check(self, ticker_data):
        """
        Real-time price check based on WebSocket ticker data
        This method checks for potential stop loss triggers and
        other real-time conditions without waiting for candle close.

        Parameters
        ----------
        ticker_data : dict
            Real-time ticker data from WebSocket
        """
        try:
            # Extract price from ticker data
            if "ask_price" in ticker_data:
                current_price = float(ticker_data["ask_price"])
            elif "close" in ticker_data:
                current_price = float(ticker_data["close"])
            else:
                return  # No price data available

            # Only check if position is active
            if not self.position_state.active:
                return

            # Check for stop loss trigger
            if self.check_stop_loss(current_price):
                self.logger.info(
                    f"Stop loss triggered in real-time at price {current_price}"
                )
                self._execute_exit(current_price, "stop_loss")

            # Log periodic price updates (every 5 minutes)
            current_time = time.time()
            if (
                not hasattr(self, "last_price_log_time")
                or current_time - self.last_price_log_time > 300
            ):
                # Calculate unrealized PnL
                entry_price = self.position_state.entry_price
                side = self.position_state.side

                if side == "BUY":
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100

                self.logger.info(
                    f"Current Price: {current_price:.8f}, Unrealized PnL: {pnl_pct:.2f}%"
                )
                self.last_price_log_time = current_time

        except Exception as e:
            self.logger.error(f"Error in real-time price check: {e}")

    def analyze_only(self):
        """Analyze market conditions without executing trades (for testing/demo)"""
        try:
            # For analyze-only mode, we'll run analyses in a loop with pauses
            self.logger.info("Starting ANALYZE ONLY mode with real-time price tracking")

            # Get latest market data
            df = self.data_manager.get_historical_data(
                self.config.symbol, self.config.timeframe, limit=100
            )

            if df is None or len(df) < 20:
                self.logger.warning("Insufficient market data for analysis")
                return

            # Update with current price if available
            current_price = self.current_price or df["close"].iloc[-1]

            # Update last row with current price
            df.loc[df.index[-1], "close"] = current_price

            # Calculate indicators
            df = calculate_indicators(
                df=df,
                dc_enter=self.config.dc_length_enter,
                dc_exit=self.config.dc_length_exit,
                atr_len=self.config.atr_length,
                atr_smooth=self.config.atr_smoothing,
                ma_period=self.config.ma_period,
                adx_period=self.config.adx_period,
            )

            # Log real-time price
            self.logger.info(f"=== PRICE UPDATE ({time.strftime('%H:%M:%S')}) ===")
            self.logger.info(f"Current Price: {current_price:.8f}")

            # Log detailed market condition
            self._log_market_condition(df)

            # Check for signals
            current_atr = df["atr"].iloc[-1]

            # Check for entry signals
            if current_price > df["dc_upper"].iloc[-1]:
                self.logger.info("Long entry signal detected (Analyze-only mode)")
                signal_quality = self.evaluate_signal_quality_with_ml(
                    df, "entry", "BUY"
                )
                self.logger.info(
                    f"Signal quality: {signal_quality.get('overall_score', 0):.2f}/10"
                )
            elif current_price < df["dc_lower"].iloc[-1]:
                self.logger.info("Short entry signal detected (Analyze-only mode)")
                signal_quality = self.evaluate_signal_quality_with_ml(
                    df, "entry", "SELL"
                )
                self.logger.info(
                    f"Signal quality: {signal_quality.get('overall_score', 0):.2f}/10"
                )

            # Check for exit signals if we have an active position
            if self.position_state.active:
                if self.check_exit_signal():
                    self.logger.info("Exit signal detected (Analyze-only mode)")
                elif self.check_stop_loss(current_price):
                    self.logger.info("Stop loss would be triggered (Analyze-only mode)")

        except Exception as e:
            self.logger.error(f"Error in analysis: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
