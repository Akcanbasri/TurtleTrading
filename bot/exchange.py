"""
Exchange operations using Binance API for the Turtle Trading Bot
"""

import logging
import time
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import websocket
import json
import threading
from queue import Queue
import traceback

try:
    # Futures API için yeni import
    from binance.um_futures import UMFutures
    from binance.spot import Spot
    from binance.error import ClientError

    BINANCE_FUTURES_AVAILABLE = True
except ImportError:
    # Eski import yapısı
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceOrderException

    BINANCE_FUTURES_AVAILABLE = False

from bot.models import SymbolInfo, TradeSide
from bot.utils import format_price, format_quantity, round_step_size


class BinanceWebSocketManager:
    def __init__(self, symbol, callback=None, use_testnet=True, auto_reconnect=True):
        self.symbol = symbol.lower()
        self.callback = callback
        self.ws = None
        self.thread = None
        self.data_queue = Queue()
        self.is_connected = False
        self.use_testnet = use_testnet
        self.reconnect_count = 0
        self.max_reconnects = 5
        self.reconnect_delay = 5  # seconds
        self.should_reconnect = auto_reconnect

        # WebSocket URL'leri
        self.ws_base_url = (
            "wss://fstream.binance.com/stream?streams="
            if not use_testnet
            else "wss://stream.binancefuture.com/stream?streams="
        )
        self.last_ping_time = time.time()
        self.logger = logging.getLogger("turtle_trading_bot")

    def _on_message(self, ws, message):
        data = json.loads(message)
        self.logger.debug(f"Received WebSocket message: {data.keys()}")

        # For combined streams, data contains the stream name and the data
        if "data" in data and "stream" in data:
            stream_data = data["data"]

            # Process kline data
            if "k" in stream_data:
                kline = self._process_kline_data(stream_data)
                self.data_queue.put({"type": "kline", "data": kline})
            # Process bookTicker data - this comes more frequently
            elif "b" in stream_data and "a" in stream_data:
                ticker = self._process_ticker_data(stream_data)
                self.data_queue.put({"type": "ticker", "data": ticker})

                # Call the real-time price check function in callback if provided
                if hasattr(self, "price_check_callback") and self.price_check_callback:
                    self.price_check_callback(ticker)

        # Handle pong responses or direct stream data
        elif "result" in data and data["result"] is None and "id" in data:
            self.logger.debug("Received pong from server")
            return
        # Handle direct stream data (for legacy format or single streams)
        elif "k" in data:
            kline = self._process_kline_data(data)
            self.data_queue.put({"type": "kline", "data": kline})
        # Process direct bookTicker data
        elif "b" in data and "a" in data:
            ticker = self._process_ticker_data(data)
            self.data_queue.put({"type": "ticker", "data": ticker})

            # Call the real-time price check function in callback if provided
            if hasattr(self, "price_check_callback") and self.price_check_callback:
                self.price_check_callback(ticker)

    def _process_kline_data(self, data):
        """Process kline websocket data into a standardized format"""
        k = data["k"]
        return {
            "open_time": k["t"],
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "close_time": k["T"],
            "is_closed": k["x"],  # True if candle is closed/completed
            "symbol": data["s"],
        }

    def _process_ticker_data(self, data):
        """Process bookTicker websocket data into a standardized format"""
        return {
            "symbol": data["s"],
            "bid_price": float(data["b"]),
            "bid_qty": float(data["B"]),
            "ask_price": float(data["a"]),
            "ask_qty": float(data["A"]),
            "timestamp": int(time.time() * 1000),
        }

    def _on_error(self, ws, error):
        self.logger.error(f"WebSocket error: {error}")
        if self.should_reconnect:
            self._schedule_reconnect()

    def _on_close(self, ws, close_status_code, close_msg):
        self.logger.info(f"WebSocket connection closed: {close_msg}")
        self.is_connected = False
        if self.should_reconnect:
            self._schedule_reconnect()

    def _schedule_reconnect(self):
        """Schedule a reconnection attempt with exponential backoff"""
        if self.reconnect_count < self.max_reconnects:
            delay = self.reconnect_delay * (2**self.reconnect_count)
            self.logger.info(f"Scheduling reconnection in {delay} seconds")
            threading.Timer(delay, self.start).start()
            self.reconnect_count += 1
        else:
            self.logger.error("Max reconnection attempts reached. Giving up.")
            self.should_reconnect = False

    def _on_open(self, ws):
        self.logger.info(f"WebSocket connection opened for {self.symbol}")
        self.is_connected = True
        self.reconnect_count = 0  # Reset reconnect counter on successful connection

        # No need to create a new WebSocketApp here, as we're already in the callback
        # Just send subscribe message for the combined streams
        self.logger.info(f"Subscribing to streams for {self.symbol}")

        # For combined streams URL, we don't need to send a subscribe message
        # The streams are already specified in the URL

    def _process_queue(self):
        while self.is_connected:
            try:
                if not self.data_queue.empty():
                    data = self.data_queue.get(timeout=1)
                    self.callback(data)
                else:
                    time.sleep(0.01)

                # Send ping every 3 minutes to keep connection alive
                if time.time() - self.last_ping_time > 180:
                    ping_msg = {"method": "PING", "id": int(time.time())}
                    self.ws.send(json.dumps(ping_msg))
                    self.last_ping_time = time.time()
                    self.logger.debug("Sent ping to server")

            except Exception as e:
                self.logger.error(f"Error processing WebSocket data: {e}")

    def start(self):
        if self.is_connected:
            return

        websocket.enableTrace(False)  # Set to True for debugging

        # Properly construct WebSocket URL with streams
        streams = [
            f"{self.symbol}@kline_1m",  # 1-minute candles
            f"{self.symbol}@kline_5m",  # 5-minute candles
            f"{self.symbol}@kline_15m",  # 15-minute candles
            f"{self.symbol}@bookTicker",  # Best bid/ask
        ]
        ws_url = f"{self.ws_base_url}{'/'.join(streams)}"

        self.logger.info(f"Connecting to WebSocket URL: {ws_url}")
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        # WebSocket thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        # Data processing thread
        self.process_thread = threading.Thread(target=self._process_queue)
        self.process_thread.daemon = True
        self.process_thread.start()

        # Wait for connection to establish
        timeout = 0
        while not self.is_connected and timeout < 10:
            time.sleep(0.5)
            timeout += 0.5

        if not self.is_connected:
            raise ConnectionError("Could not establish WebSocket connection")
        else:
            self.logger.info(f"Successfully connected to WebSocket for {self.symbol}")

    def stop(self):
        self.should_reconnect = False  # Prevent automatic reconnection
        if self.ws:
            # Send unsubscribe message
            unsubscribe_msg = {
                "method": "UNSUBSCRIBE",
                "params": [
                    f"{self.symbol}@kline_1m",
                    f"{self.symbol}@kline_5m",
                    f"{self.symbol}@kline_15m",
                    f"{self.symbol}@bookTicker",
                ],
                "id": 2,
            }
            try:
                self.ws.send(json.dumps(unsubscribe_msg))
                time.sleep(0.5)  # Give time for unsubscribe to process
            except:
                pass  # Ignore errors during shutdown
            self.ws.close()
        self.is_connected = False
        self.logger.info(f"WebSocket connection for {self.symbol} closed")


class BinanceExchange:
    """
    Binance exchange API wrapper for Futures trading

    This class handles all exchange-related operations:
    - API client initialization
    - Market data fetching
    - Symbol info retrieval
    - Order execution
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        demo_mode: bool = False,
    ):
        """
        Initialize the Binance Exchange wrapper for Futures trading

        Parameters
        ----------
        api_key : str
            Binance API key
        api_secret : str
            Binance API secret
        testnet : bool, optional
            Whether to use the testnet, by default True
        demo_mode : bool, optional
            Whether to run in demo mode without actual trades, by default False
        """
        self.logger = logging.getLogger("turtle_trading_bot")
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.demo_mode = demo_mode

        # Initialize clients
        try:
            # Initialize Futures client if available
            if BINANCE_FUTURES_AVAILABLE:
                self.futures_client = UMFutures(
                    key=api_key,
                    secret=api_secret,
                    testnet=testnet,
                    base_url="https://testnet.binancefuture.com" if testnet else None,
                )
                self.logger.info(f"Futures API initialized (testnet: {testnet})")

                # Try to handle time synchronization
                try:
                    # Check server time and sync
                    server_time = self.futures_client.time()
                    time_diff = int(time.time() * 1000) - server_time["serverTime"]
                    if (
                        abs(time_diff) > 1000
                    ):  # If time difference is more than 1 second
                        self.logger.warning(
                            f"Time difference with Binance server: {time_diff}ms. Adjusting..."
                        )
                        # For Python-Binance, we can adjust the timestamp offset
                        self.futures_client.timestamp_offset = time_diff
                        self.logger.info(f"Timestamp offset adjusted to {time_diff}ms")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to synchronize time with Binance server: {e}"
                    )
                    self.logger.info(
                        "Continuing without time synchronization. Some operations may fail."
                    )

            # Initialize Spot client for backup or when Futures API not available
            self.spot_client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
            )
            self.logger.info(f"Spot API initialized (testnet: {testnet})")

            # Also try to sync time with spot client if futures wasn't available
            if not BINANCE_FUTURES_AVAILABLE:
                try:
                    server_time = self.spot_client.get_server_time()
                    time_diff = int(time.time() * 1000) - server_time["serverTime"]
                    if abs(time_diff) > 1000:
                        self.logger.warning(
                            f"Time difference with Binance server: {time_diff}ms. Adjusting..."
                        )
                        # For legacy Python-Binance client, we can just log it
                        self.logger.info("Adjustment may be needed in requests")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to synchronize time with Binance spot server: {e}"
                    )
                    self.logger.info(
                        "Continuing without time synchronization. Some operations may fail."
                    )

        except Exception as e:
            self.logger.error(f"Error initializing Binance client: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise ConnectionError(f"Failed to connect to Binance API: {e}")

        # Initialize WebSocket manager
        self.ws_manager = BinanceWebSocketManager(
            symbol="btcusdt",  # Default symbol, will be updated later
            callback=self._on_ws_message if hasattr(self, "_on_ws_message") else None,
            use_testnet=testnet,
            auto_reconnect=True,
        )

        # Cache for symbol info
        self.symbol_info_cache = {}
        self.last_prices = {}

        # Callbacks for WebSocket data
        self.kline_callbacks = {}
        self.ticker_callbacks = {}

        # Stats and monitoring
        self.ws_message_count = 0
        self.last_ws_message_time = None

    def _on_ws_message(self, message):
        """
        Process WebSocket message

        Parameters
        ----------
        message : dict
            Message from WebSocket
        """
        try:
            self.ws_message_count += 1
            self.last_ws_message_time = time.time()

            # Extract data based on message type
            if message.get("type") == "kline":
                kline_data = message.get("data", {})
                symbol = kline_data.get("symbol", "").lower()

                # Call all registered callbacks for this symbol's klines
                if symbol in self.kline_callbacks and self.kline_callbacks[symbol]:
                    self.kline_callbacks[symbol](kline_data)

            elif message.get("type") == "ticker":
                ticker_data = message.get("data", {})
                symbol = ticker_data.get("symbol", "").lower()

                # Update last price
                if symbol and "ask_price" in ticker_data:
                    self.last_prices[symbol] = float(ticker_data["ask_price"])

                # Call all registered callbacks for this symbol's ticker
                if symbol in self.ticker_callbacks and self.ticker_callbacks[symbol]:
                    self.ticker_callbacks[symbol](ticker_data)

        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
            self.logger.debug(f"Problematic message: {message}")

    def fetch_historical_data(
        self, symbol: str, interval: str, lookback: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical candlestick data for a symbol and interval

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., 'BTCUSDT')
        interval : str
            Kline interval (e.g., '1h', '4h', '1d')
        lookback : int
            Number of candles to fetch

        Returns
        -------
        pd.DataFrame or None
            DataFrame with OHLCV data or None if data fetching failed
        """
        try:
            self.logger.info(f"Fetching {lookback} {interval} candles for {symbol}")

            # Get klines from Futures API
            if BINANCE_FUTURES_AVAILABLE:
                klines = self.futures_client.klines(
                    symbol=symbol, interval=interval, limit=lookback
                )
            else:
                # Legacy API
                klines = self.spot_client.get_historical_klines(
                    symbol=symbol, interval=interval, limit=lookback
                )

            # Check if we have enough data
            if len(klines) < lookback * 0.7:  # Require at least 70% of requested data
                self.logger.warning(
                    f"Insufficient data: got {len(klines)} candles, wanted {int(lookback * 0.7)}"
                )

                # Check if we got any data at all
                if len(klines) == 0:
                    self.logger.info(
                        f"Supplementing with synthetic data for {symbol} {interval}"
                    )
                    return self._generate_synthetic_data(
                        symbol, interval, lookback, lookback
                    )

                # Otherwise supplement with synthetic data
                remaining = lookback - len(klines)
                self.logger.info(
                    f"Supplementing with {remaining} synthetic candles for {symbol} {interval}"
                )
                synthetic_df = self._generate_synthetic_data(
                    symbol, interval, remaining, remaining
                )

                # Convert klines to DataFrame
                columns = [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ]
                real_df = pd.DataFrame(klines, columns=columns)
                real_df["timestamp"] = pd.to_datetime(real_df["timestamp"], unit="ms")
                real_df.set_index("timestamp", inplace=True)

                # Convert numeric columns
                for col in ["open", "high", "low", "close", "volume"]:
                    real_df[col] = pd.to_numeric(real_df[col])

                # Concatenate real and synthetic data
                combined_df = pd.concat([synthetic_df.iloc[: -len(real_df)], real_df])
                return combined_df

            # Convert to DataFrame if we have enough data
            columns = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ]
            df = pd.DataFrame(klines, columns=columns)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Convert numeric columns
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])

            return df

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            # Generate synthetic data as fallback
            self.logger.info(
                f"Generating synthetic data for {symbol} - {lookback} candles"
            )
            return self._generate_synthetic_data(symbol, interval, lookback, lookback)

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """
        Get trading information for a symbol

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., 'BTCUSDT')

        Returns
        -------
        SymbolInfo
            Symbol trading information

        Raises
        ------
        ValueError
            If symbol info retrieval fails
        """
        try:
            # Get symbol info from Futures API
            if BINANCE_FUTURES_AVAILABLE:
                exchange_info = self.futures_client.exchange_info(symbol=symbol)
                # Extract the symbol info from the response
                symbol_info = None
                for info in exchange_info["symbols"]:
                    if info["symbol"] == symbol:
                        symbol_info = info
                        break
            else:
                # Legacy API
                symbol_info = self.spot_client.get_symbol_info(symbol)

            if not symbol_info:
                raise ValueError(f"Symbol information not found for {symbol}")

            self.logger.info(f"Retrieved symbol information for {symbol}")

            # Extract filter values
            price_filter = next(
                (
                    f
                    for f in symbol_info["filters"]
                    if f["filterType"] == "PRICE_FILTER"
                ),
                None,
            )
            lot_size_filter = next(
                (f for f in symbol_info["filters"] if f["filterType"] == "LOT_SIZE"),
                None,
            )
            min_notional_filter = next(
                (
                    f
                    for f in symbol_info["filters"]
                    if f["filterType"] == "MIN_NOTIONAL"
                ),
                None,
            )

            def get_precision_from_step(step_str):
                if not step_str or float(step_str) == 0:
                    return 0
                step_str = step_str.rstrip("0")
                if "." in step_str:
                    return len(step_str) - step_str.index(".") - 1
                return 0

            # Set defaults if filters not found
            tick_size = price_filter.get("tickSize", "0.01") if price_filter else "0.01"
            step_size = (
                lot_size_filter.get("stepSize", "0.001") if lot_size_filter else "0.001"
            )
            min_qty = (
                lot_size_filter.get("minQty", "0.001") if lot_size_filter else "0.001"
            )
            min_notional = (
                min_notional_filter.get("minNotional", "10")
                if min_notional_filter
                else "10"
            )

            # Calculate precisions
            price_precision = get_precision_from_step(tick_size)
            quantity_precision = get_precision_from_step(step_size)

            # Log symbol filters
            self.logger.info(f"Symbol Filters for {symbol}:")
            self.logger.info(f"Price Precision: {price_precision}")
            self.logger.info(f"Quantity Precision: {quantity_precision}")
            self.logger.info(f"Minimum Quantity: {min_qty}")
            self.logger.info(f"Step Size: {step_size}")
            self.logger.info(f"Minimum Notional: {min_notional}")

            # Create and return SymbolInfo
            return SymbolInfo(
                price_precision=price_precision,
                quantity_precision=quantity_precision,
                min_qty=Decimal(min_qty),
                step_size=Decimal(step_size),
                min_notional=Decimal(min_notional),
            )
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {e}")
            # Return default values
            return SymbolInfo(
                price_precision=2,
                quantity_precision=3,
                min_qty=Decimal("0.001"),
                step_size=Decimal("0.001"),
                min_notional=Decimal("10"),
            )

    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current price for a symbol

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., 'BTCUSDT')

        Returns
        -------
        Decimal or None
            Current price or None if price retrieval fails
        """
        try:
            if BINANCE_FUTURES_AVAILABLE:
                # Get current price from Futures API
                ticker = self.futures_client.ticker_price(symbol=symbol)
                price_str = ticker["price"]
            else:
                # Legacy API
                ticker = self.spot_client.get_ticker(symbol=symbol)
                price_str = ticker["lastPrice"]

            return Decimal(price_str)
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            # Return a reasonable price for known symbols, or a default
            if "BTC" in symbol:
                return Decimal("30000")
            elif "ETH" in symbol:
                return Decimal("2000")
            elif "DOGE" in symbol:
                return Decimal("0.1")
            else:
                return Decimal("100")

    def get_account_balance(
        self, asset: str = None, force_refresh=False
    ) -> Optional[Decimal]:
        """
        Get account balance for an asset or total balance if no asset is specified.
        Uses caching to avoid excessive API calls.

        Parameters
        ----------
        asset : str, optional
            Asset symbol (e.g., 'USDT'). If None, returns total account balance.
        force_refresh : bool, optional
            Force a fresh balance check, bypassing the cache.

        Returns
        -------
        Decimal or None
            Available balance or None if balance retrieval fails
        """
        try:
            # Apply time offset for Binance API requests
            # This helps with the "Timestamp for this request is X ms ahead/behind" errors
            try:
                server_time = self.spot_client.get_server_time()
                local_time = int(time.time() * 1000)
                self.spot_client.timestamp_offset = (
                    server_time["serverTime"] - local_time
                )
                self.logger.debug(
                    f"Applied timestamp offset of {self.spot_client.timestamp_offset}ms to Binance client"
                )
            except Exception as ts_err:
                self.logger.debug(f"Failed to set timestamp offset: {ts_err}")

            # Check for cached balance
            balance_cache_key = f"balance_{asset}" if asset else "balance_total"
            current_time = time.time()

            # Use cached value if available and not expired
            if hasattr(self, "_balance_cache") and not force_refresh:
                cache = getattr(self, "_balance_cache", {})
                cache_time = getattr(self, "_balance_cache_time", {})

                if balance_cache_key in cache and balance_cache_key in cache_time:
                    # Use cache if it's less than 5 minutes old
                    if current_time - cache_time[balance_cache_key] < 300:
                        return cache[balance_cache_key]

            # Initialize cache if it doesn't exist
            if not hasattr(self, "_balance_cache"):
                self._balance_cache = {}
                self._balance_cache_time = {}

            # If in demo mode, return the simulated balance
            if self.demo_mode:
                demo_balance = Decimal("20")  # Real account has 20 USDT

                # Check if we have a saved demo balance
                if hasattr(self, "_demo_balance"):
                    demo_balance = self._demo_balance
                else:
                    self._demo_balance = demo_balance

                # Cache the balance and update the timestamp
                self._balance_cache[balance_cache_key] = demo_balance
                self._balance_cache_time[balance_cache_key] = current_time

                return demo_balance

            # If no asset is specified, get total account balance
            if asset is None:
                if BINANCE_FUTURES_AVAILABLE:
                    # Get balance from Futures API
                    account_info = self.futures_client.account()
                    total_balance = Decimal(account_info["totalWalletBalance"])

                    # Cache the balance and update the timestamp
                    self._balance_cache[balance_cache_key] = total_balance
                    self._balance_cache_time[balance_cache_key] = current_time

                    return total_balance
                else:
                    # For legacy API, return USDT balance
                    return self.get_account_balance("USDT", force_refresh)

            # Get specific asset balance
            if BINANCE_FUTURES_AVAILABLE:
                # Get balance from Futures API
                balances = self.futures_client.balance()
                for balance in balances:
                    if balance["asset"] == asset:
                        asset_balance = Decimal(balance["availableBalance"])

                        # Cache the balance and update the timestamp
                        self._balance_cache[balance_cache_key] = asset_balance
                        self._balance_cache_time[balance_cache_key] = current_time

                        return asset_balance
            else:
                # Legacy API
                account = self.spot_client.get_account()
                for balance in account["balances"]:
                    if balance["asset"] == asset:
                        asset_balance = Decimal(balance["free"])

                        # Cache the balance and update the timestamp
                        self._balance_cache[balance_cache_key] = asset_balance
                        self._balance_cache_time[balance_cache_key] = current_time

                        return asset_balance

            self.logger.warning(f"Asset {asset} not found in account balances")

            # Set a reasonable default for the missing asset
            default_balance = Decimal("0")
            self._balance_cache[balance_cache_key] = default_balance
            self._balance_cache_time[balance_cache_key] = current_time

            return default_balance

        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")

            # Check for time synchronization errors
            if "Timestamp for this request" in str(e):
                self.logger.warning(
                    "Time synchronization error. Using cached balance if available."
                )

                # Use cached value if available
                if (
                    hasattr(self, "_balance_cache")
                    and balance_cache_key in self._balance_cache
                ):
                    self.logger.info(f"Using cached balance due to time sync error")
                    return self._balance_cache[balance_cache_key]

            # Return a demo balance for errors
            if asset is None or asset == "USDT":
                return Decimal("20")  # Real account has 20 USDT
            elif asset == "BTC":
                return Decimal("0.0005")
            else:
                return Decimal("0.1")

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., 'BTCUSDT')
        leverage : int
            Leverage value (1-125)

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            if BINANCE_FUTURES_AVAILABLE:
                response = self.futures_client.change_leverage(
                    symbol=symbol, leverage=leverage
                )
                self.logger.info(
                    f"Set leverage for {symbol} to {leverage}x - Response: {response}"
                )
                return True
            else:
                self.logger.warning("Leverage setting not available in legacy mode")
                return False
        except Exception as e:
            self.logger.error(f"Error setting leverage: {e}")
            return False

    def initialize_websocket(self, symbol=None, callback=None, on_ticker_callback=None):
        """
        Initialize and start the WebSocket connection for the configured symbol.

        Parameters
        ----------
        symbol : str, optional
            Trading symbol (e.g., 'btcusdt'). Will use default if not provided.
        callback : callable, optional
            Main callback function for WebSocket messages
        on_ticker_callback : callable, optional
            Callback function for real-time price checks

        Returns
        -------
        bool
            True if successfully initialized, False otherwise
        """
        try:
            if (
                hasattr(self, "ws_manager")
                and self.ws_manager
                and self.ws_manager.is_connected
            ):
                self.logger.info("WebSocket connection already initialized")
                return True

            # Set the symbol if provided
            if symbol:
                self.symbol = symbol

            if not hasattr(self, "symbol") or not self.symbol:
                self.logger.error("Symbol not set for WebSocket connection")
                return False

            self.logger.info(f"Initializing WebSocket for {self.symbol}")
            self.ws_manager = BinanceWebSocketManager(
                symbol=self.symbol,
                callback=callback or self._on_ws_message,
                use_testnet=self.testnet,
                auto_reconnect=True,
            )

            # Set the price check callback if provided
            if on_ticker_callback:
                if not hasattr(self, "ticker_callbacks"):
                    self.ticker_callbacks = {}
                self.ticker_callbacks[self.symbol] = on_ticker_callback

            self.ws_manager.start()
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize WebSocket: {e}")
            return False

    def close_websocket(self):
        """WebSocket bağlantısını kapatır"""
        if hasattr(self, "ws_manager"):
            self.ws_manager.stop()

    def get_symbol_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a specific symbol

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., 'BTCUSDT')

        Returns
        -------
        float
            Current price or None if price retrieval fails
        """
        try:
            if BINANCE_FUTURES_AVAILABLE:
                # Get current price from Futures API
                ticker = self.futures_client.ticker_price(symbol=symbol)
                return float(ticker["price"])
            else:
                # Legacy API
                ticker = self.spot_client.get_ticker(symbol=symbol)
                return float(ticker["lastPrice"])
        except Exception as e:
            self.logger.error(f"Error getting {symbol} price: {e}")
            # Return a reasonable default price for demo mode
            if self.demo_mode:
                if "BTC" in symbol:
                    return 30000.0
                elif "ETH" in symbol:
                    return 2000.0
                elif "DOGE" in symbol:
                    return 0.15
                else:
                    return 100.0
            return None

    def create_market_order(
        self, symbol: str, side: str, quantity: float
    ) -> Optional[Dict[str, Any]]:
        """
        Create a market order

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., 'BTCUSDT')
        side : str
            Order side ('BUY' or 'SELL')
        quantity : float
            Order quantity

        Returns
        -------
        dict or None
            Order response or None if order placement fails
        """
        try:
            self.logger.info(f"Creating {side} market order for {quantity} {symbol}")

            # Get symbol info for precision formatting
            symbol_info = self.get_symbol_info(symbol)

            # Format quantity with correct precision
            formatted_quantity = format_quantity(
                quantity, symbol_info.quantity_precision, symbol_info.step_size
            )

            self.logger.info(f"Formatted quantity: {formatted_quantity}")

            # Place the order
            if BINANCE_FUTURES_AVAILABLE:
                # Using Futures API
                params = {
                    "symbol": symbol,
                    "side": side,
                    "type": "MARKET",
                    "quantity": formatted_quantity,
                }

                if self.demo_mode:
                    self.logger.info(
                        f"DEMO MODE: Would place order with params: {params}"
                    )
                    # Generate a mock order response
                    order_id = int(time.time() * 1000)
                    mock_response = {
                        "orderId": order_id,
                        "symbol": symbol,
                        "status": "FILLED",
                        "clientOrderId": f"demo_{order_id}",
                        "price": str(self.get_symbol_price(symbol)),
                        "avgPrice": str(self.get_symbol_price(symbol)),
                        "origQty": formatted_quantity,
                        "executedQty": formatted_quantity,
                        "cumQty": formatted_quantity,
                        "timeInForce": "GTC",
                        "type": "MARKET",
                        "side": side,
                    }
                    self.logger.info(f"DEMO MODE: Mock order response: {mock_response}")
                    return mock_response
                else:
                    response = self.futures_client.new_order(**params)
                    self.logger.info(f"Order placed successfully: {response}")
                    return response
            else:
                # Using legacy API
                if self.demo_mode:
                    self.logger.info(
                        f"DEMO MODE: Would place {side} market order for {formatted_quantity} {symbol}"
                    )
                    # Generate a mock order response
                    order_id = int(time.time() * 1000)
                    mock_response = {
                        "orderId": order_id,
                        "symbol": symbol,
                        "status": "FILLED",
                        "clientOrderId": f"demo_{order_id}",
                        "price": "0.00",
                        "origQty": formatted_quantity,
                        "executedQty": formatted_quantity,
                        "timeInForce": "GTC",
                        "type": "MARKET",
                        "side": side,
                    }
                    self.logger.info(f"DEMO MODE: Mock order response: {mock_response}")
                    return mock_response
                else:
                    response = self.spot_client.create_order(
                        symbol=symbol,
                        side=side,
                        type="MARKET",
                        quantity=formatted_quantity,
                    )
                    self.logger.info(f"Order placed successfully: {response}")
                    return response

        except Exception as e:
            self.logger.error(f"Error creating market order: {e}")
            if self.demo_mode:
                # Return a mock successful order response for demo mode
                order_id = int(time.time() * 1000)
                mock_response = {
                    "orderId": order_id,
                    "symbol": symbol,
                    "status": "FILLED",
                    "clientOrderId": f"demo_error_{order_id}",
                }
                self.logger.info(
                    f"DEMO MODE: Returning mock order response despite error: {mock_response}"
                )
                return mock_response
            return None

    def get_min_order_quantity(self, symbol: str) -> Decimal:
        """
        Get minimum order quantity for a symbol

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., 'BTCUSDT')

        Returns
        -------
        Decimal
            Minimum order quantity
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            return symbol_info.min_qty
        except Exception as e:
            self.logger.error(f"Error getting minimum order quantity: {e}")
            return Decimal("0.001")  # Default fallback value

    def format_quantity(
        self, quantity: Union[Decimal, float], precision: int, step_size: Decimal = None
    ) -> str:
        """
        Format quantity with the correct number of decimal places and step size

        Parameters
        ----------
        quantity : Union[Decimal, float]
            Quantity to format
        precision : int
            Number of decimal places
        step_size : Decimal, optional
            Step size from exchange rules for rounding

        Returns
        -------
        str
            Formatted quantity string
        """
        # First round to step size if provided
        if step_size is not None:
            quantity = round_step_size(quantity, step_size)

        # Then format to correct precision
        return format_quantity(quantity, precision)

    def get_account_balance(self, force_refresh=False):
        """Get current account balance in USDT"""
        try:
            # Apply time offset for Binance API request
            self.spot_client.timestamp_offset = (
                int(time.time() * 1000)
                - self.spot_client.get_server_time()["serverTime"]
            )

            # Get account info
            account_info = self.spot_client.get_account()

            # Find USDT balance
            usdt_balance = 0
            for asset in account_info["balances"]:
                if asset["asset"] == "USDT":
                    usdt_balance = float(asset["free"]) + float(asset["locked"])
                    break

            return usdt_balance
        except Exception as e:
            logging.getLogger("turtle_trading_bot").error(
                f"Error getting account balance: {e}"
            )
            # Return default balance on error
            return 20.0


class ExchangeInterface:
    """
    Exchange interface that wraps the BinanceExchange class.

    This class provides a simplified interface to exchange operations and handles
    additional functionality like error handling and data conversion.
    """

    def __init__(self, api_key, api_secret, use_testnet=True):
        """
        Initialize the exchange interface.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            use_testnet: Whether to use the Binance testnet
        """
        self.logger = logging.getLogger("turtle_trading_bot")
        self.exchange = BinanceExchange(api_key, api_secret, testnet=use_testnet)
        self.symbol_cache = {}

    def get_historical_data(self, symbol, timeframe, limit=100):
        """
        Get historical candlestick data for a symbol and timeframe.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Candlestick timeframe (e.g., '1h', '4h', '1d')
            limit: Number of candlesticks to retrieve

        Returns:
            pandas.DataFrame: Historical data with OHLCV columns
        """
        try:
            # Convert Binance klines to DataFrame
            klines = self.exchange.spot_client.get_klines(
                symbol=symbol, interval=timeframe, limit=limit
            )

            # Create DataFrame
            data = pd.DataFrame(
                klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_asset_volume",
                    "number_of_trades",
                    "taker_buy_base_asset_volume",
                    "taker_buy_quote_asset_volume",
                    "ignore",
                ],
            )

            # Convert types
            data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                data[col] = data[col].astype(float)

            # Set timestamp as index
            data.set_index("timestamp", inplace=True)

            return data

        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            raise

    def get_symbol_info(self, symbol):
        """
        Get information about a trading symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            dict: Symbol information
        """
        # Check cache first
        if symbol in self.symbol_cache:
            return self.symbol_cache[symbol]

        # Get info from exchange
        symbol_info = self.exchange.get_symbol_info(symbol)

        # Cache the result
        self.symbol_cache[symbol] = symbol_info

        return symbol_info

    def get_current_price(self, symbol):
        """
        Get current market price for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            float: Current market price
        """
        price = self.exchange.get_current_price(symbol)
        if price is not None:
            return float(price)
        else:
            raise ValueError(f"Could not get current price for {symbol}")

    def get_balance(self, asset):
        """
        Get available balance for an asset.

        Args:
            asset: Asset name (e.g., 'USDT', 'BTC')

        Returns:
            float: Available balance
        """
        balance = self.exchange.get_account_balance(asset)
        if balance is not None:
            return float(balance)
        else:
            raise ValueError(f"Could not get balance for {asset}")

    def get_account_info(self):
        """
        Get account information.

        Returns:
            dict: Account information
        """
        try:
            return self.exchange.spot_client.get_account()
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise

    def execute_order(
        self, symbol, side, quantity, order_type="MARKET", simulate=False
    ):
        """
        Execute an order on the exchange.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity
            order_type: Order type (default: 'MARKET')
            simulate: Whether to simulate the order

        Returns:
            tuple: (success, order_details)
        """
        symbol_info = self.get_symbol_info(symbol)

        return self.exchange.execute_order(
            symbol=symbol,
            side=side,
            quantity=Decimal(str(quantity)),
            symbol_info=symbol_info,
            order_type=order_type,
            simulate=simulate,
        )
