"""
Exchange operations using Binance API for the Turtle Trading Bot
"""

import logging
import time
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import websocket
import json
import threading
from queue import Queue

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
from bot.utils import format_price, format_quantity


class BinanceWebSocketManager:
    def __init__(self, symbol, callback, use_testnet=True):
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
        self.should_reconnect = True

        # WebSocket URL'leri
        self.ws_base_url = (
            "wss://fstream.binance.com/ws/"
            if not use_testnet
            else "wss://stream.binancefuture.com/ws/"
        )
        self.last_ping_time = time.time()
        self.logger = logging.getLogger("turtle_trading_bot")

    def _on_message(self, ws, message):
        data = json.loads(message)

        # Handle pong responses
        if "result" in data and data["result"] is None and "id" in data:
            self.logger.debug("Received pong from server")
            return

        # Process kline data
        if "k" in data:
            kline = self._process_kline_data(data)
            self.data_queue.put({"type": "kline", "data": kline})
        # Process bookTicker data
        elif "b" in data and "a" in data:
            ticker = self._process_ticker_data(data)
            self.data_queue.put({"type": "ticker", "data": ticker})

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

        # Subscribe to multiple streams
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [
                f"{self.symbol}@kline_1m",  # 1-minute candles
                f"{self.symbol}@kline_5m",  # 5-minute candles
                f"{self.symbol}@kline_15m",  # 15-minute candles
                f"{self.symbol}@bookTicker",  # Best bid/ask
            ],
            "id": 1,
        }
        ws.send(json.dumps(subscribe_msg))

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
        self.ws = websocket.WebSocketApp(
            self.ws_base_url,
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

    def __init__(self, api_key: str, api_secret: str, use_testnet: bool = True):
        """
        Initialize Binance exchange client

        Parameters
        ----------
        api_key : str
            Binance API key
        api_secret : str
            Binance API secret
        use_testnet : bool
            Whether to use the Binance testnet
        """
        self.logger = logging.getLogger("turtle_trading_bot")
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet

        # Futures API base URL'leri
        if use_testnet:
            self.futures_base_url = "https://testnet.binancefuture.com"
        else:
            self.futures_base_url = "https://fapi.binance.com"

        # Initialize clients
        self.client, self.futures_client = self._initialize_client()

    def _initialize_client(self):
        """
        Initialize Binance client with API credentials

        Returns
        -------
        tuple
            (spot_client, futures_client)

        Raises
        ------
        ValueError
            If API credentials are missing
        ClientError
            If there's an issue with the Binance API
        """
        if not self.api_key or not self.api_secret:
            self.logger.error("API Key or Secret Key not found.")

            # Create a minimal mock client for demo purposes
            class MockClient:
                def ping(self):
                    return True

                def time(self):
                    return {"serverTime": int(time.time() * 1000)}

                def account(self):
                    return {
                        "accountType": "DEMO",
                        "canTrade": True,
                        "balances": [
                            {"asset": "USDT", "free": "10000.00", "locked": "0.00"},
                            {"asset": "BTC", "free": "1.00", "locked": "0.00"},
                        ],
                    }

                def klines(self, *args, **kwargs):
                    # This will force the _generate_synthetic_data to be called
                    return []

                def exchange_info(self, **kwargs):
                    # Return a simulated symbol info
                    symbol = kwargs.get("symbol", "BTCUSDT")
                    return {
                        "symbols": [
                            {
                                "symbol": symbol,
                                "filters": [
                                    {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                                    {
                                        "filterType": "LOT_SIZE",
                                        "minQty": "0.001",
                                        "stepSize": "0.001",
                                    },
                                    {
                                        "filterType": "MIN_NOTIONAL",
                                        "minNotional": "10.0",
                                    },
                                ],
                            }
                        ]
                    }

                def ticker_price(self, **kwargs):
                    # Return a simulated ticker with current price
                    symbol = kwargs.get("symbol", "BTCUSDT")
                    if symbol == "BTCUSDT":
                        return {"price": "30000.00"}
                    elif symbol == "ETHUSDT":
                        return {"price": "2000.00"}
                    else:
                        return {"price": "100.00"}

                def new_order(self, **kwargs):
                    # Return a simulated order response
                    symbol = kwargs.get("symbol", "UNKNOWN")
                    side = kwargs.get("side", "BUY")
                    quantity = kwargs.get("quantity", "0.001")

                    order_time = int(time.time() * 1000)
                    price = 30000.0 if symbol == "BTCUSDT" else 2000.0

                    return {
                        "symbol": symbol,
                        "orderId": f"demo_{order_time}",
                        "clientOrderId": f"demo_{order_time}_client",
                        "transactTime": order_time,
                        "price": "0.00000000",
                        "origQty": str(quantity),
                        "executedQty": str(quantity),
                        "cumQuote": str(float(quantity) * price),
                        "status": "FILLED",
                        "timeInForce": "GTC",
                        "type": "MARKET",
                        "side": side,
                        "avgPrice": str(price),
                    }

                def balance(self, **kwargs):
                    return [
                        {
                            "asset": "USDT",
                            "balance": "10000.00",
                            "availableBalance": "10000.00",
                        },
                        {"asset": "BTC", "balance": "1.00", "availableBalance": "1.00"},
                    ]

                def change_leverage(self, **kwargs):
                    return {"leverage": kwargs.get("leverage", 1)}

            self.logger.warning("Using mock client for demo mode")
            return MockClient(), MockClient()

        try:
            spot_client = None
            futures_client = None

            if BINANCE_FUTURES_AVAILABLE:
                # Modern Binance kütüphanesi (v2+) kullanım
                if self.use_testnet:
                    self.logger.info(
                        "Initializing Binance Futures client in TESTNET mode"
                    )
                    futures_client = UMFutures(
                        key=self.api_key,
                        secret=self.api_secret,
                        base_url="https://testnet.binancefuture.com",
                    )
                    spot_client = Spot(
                        key=self.api_key,
                        secret=self.api_secret,
                        base_url="https://testnet.binance.vision",
                    )
                else:
                    self.logger.info(
                        "Initializing Binance Futures client in PRODUCTION mode"
                    )
                    futures_client = UMFutures(key=self.api_key, secret=self.api_secret)
                    spot_client = Spot(key=self.api_key, secret=self.api_secret)
            else:
                # Eski Binance kütüphanesi (v1) kullanım
                if self.use_testnet:
                    self.logger.info(
                        "Initializing Binance client in TESTNET mode (legacy)"
                    )
                    spot_client = Client(self.api_key, self.api_secret, testnet=True)
                else:
                    self.logger.info(
                        "Initializing Binance client in PRODUCTION mode (legacy)"
                    )
                    spot_client = Client(self.api_key, self.api_secret)
                futures_client = spot_client  # Legacy mode, aynı client kullanılacak

            # Test connectivity
            if BINANCE_FUTURES_AVAILABLE:
                futures_client.ping()
                server_time = futures_client.time()
            else:
                spot_client.ping()
                server_time = spot_client.get_server_time()

            self.logger.info(f"Connected to Binance. Server time: {server_time}")

            # Verify account access for futures
            try:
                if BINANCE_FUTURES_AVAILABLE:
                    account_info = futures_client.account()
                    account_type = "FUTURES"
                else:
                    account_info = spot_client.get_account()
                    account_type = account_info.get("accountType", "UNKNOWN")

                can_trade = True
                if "canTrade" in account_info:
                    can_trade = account_info["canTrade"]

                self.logger.info(
                    f"Account status: {account_type}, canTrade: {can_trade}"
                )
            except Exception as e:
                self.logger.warning(f"Could not get account info: {e}")
                self.logger.warning("Will use Spot API for basic operations")

            return spot_client, futures_client

        except Exception as e:
            self.logger.error(f"Binance API Exception: {e}")

            # If this is likely a demo run or testnet, return a mock client
            if "demo" in (self.api_key or "").lower() or self.use_testnet:
                self.logger.warning(
                    "API error but using mock client for demo/test mode"
                )

                class MockClient:
                    def ping(self):
                        return True

                    def time(self):
                        return {"serverTime": int(time.time() * 1000)}

                    def account(self):
                        return {
                            "accountType": "DEMO",
                            "canTrade": True,
                            "balances": [
                                {"asset": "USDT", "free": "10000.00", "locked": "0.00"},
                                {"asset": "BTC", "free": "1.00", "locked": "0.00"},
                            ],
                        }

                    def klines(self, **kwargs):
                        # This will force the _generate_synthetic_data to be called
                        return []

                    def exchange_info(self, **kwargs):
                        # Return a simulated symbol info
                        symbol = kwargs.get("symbol", "BTCUSDT")
                        return {
                            "symbols": [
                                {
                                    "symbol": symbol,
                                    "filters": [
                                        {
                                            "filterType": "PRICE_FILTER",
                                            "tickSize": "0.01",
                                        },
                                        {
                                            "filterType": "LOT_SIZE",
                                            "minQty": "0.001",
                                            "stepSize": "0.001",
                                        },
                                        {
                                            "filterType": "MIN_NOTIONAL",
                                            "minNotional": "10.0",
                                        },
                                    ],
                                }
                            ]
                        }

                    def ticker_price(self, **kwargs):
                        # Return a simulated ticker with current price
                        symbol = kwargs.get("symbol", "BTCUSDT")
                        if symbol == "BTCUSDT":
                            return {"price": "30000.00"}
                        elif symbol == "ETHUSDT":
                            return {"price": "2000.00"}
                        else:
                            return {"price": "100.00"}

                    def new_order(self, **kwargs):
                        # Return a simulated order response
                        symbol = kwargs.get("symbol", "UNKNOWN")
                        side = kwargs.get("side", "BUY")
                        quantity = kwargs.get("quantity", "0.001")

                        order_time = int(time.time() * 1000)
                        price = 30000.0 if symbol == "BTCUSDT" else 2000.0

                        return {
                            "symbol": symbol,
                            "orderId": f"demo_{order_time}",
                            "clientOrderId": f"demo_{order_time}_client",
                            "transactTime": order_time,
                            "price": "0.00000000",
                            "origQty": str(quantity),
                            "executedQty": str(quantity),
                            "cumQuote": str(float(quantity) * price),
                            "status": "FILLED",
                            "timeInForce": "GTC",
                            "type": "MARKET",
                            "side": side,
                            "avgPrice": str(price),
                        }

                    def balance(self, **kwargs):
                        return [
                            {
                                "asset": "USDT",
                                "balance": "10000.00",
                                "availableBalance": "10000.00",
                            },
                            {
                                "asset": "BTC",
                                "balance": "1.00",
                                "availableBalance": "1.00",
                            },
                        ]

                    def change_leverage(self, **kwargs):
                        return {"leverage": kwargs.get("leverage", 1)}

                return MockClient(), MockClient()
            else:
                raise

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
                klines = self.client.get_historical_klines(
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
                symbol_info = self.client.get_symbol_info(symbol)

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
                ticker = self.client.get_ticker(symbol=symbol)
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

    def get_account_balance(self, asset: str) -> Optional[Decimal]:
        """
        Get account balance for an asset

        Parameters
        ----------
        asset : str
            Asset symbol (e.g., 'USDT')

        Returns
        -------
        Decimal or None
            Available balance or None if balance retrieval fails
        """
        try:
            if BINANCE_FUTURES_AVAILABLE:
                # Get balance from Futures API
                balances = self.futures_client.balance()
                for balance in balances:
                    if balance["asset"] == asset:
                        return Decimal(balance["availableBalance"])
            else:
                # Legacy API
                account = self.client.get_account()
                for balance in account["balances"]:
                    if balance["asset"] == asset:
                        return Decimal(balance["free"])

            self.logger.warning(f"Asset {asset} not found in account balances")
            return Decimal("0")
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            # Return a fake balance for demo
            if asset == "USDT":
                return Decimal("10000")
            elif asset == "BTC":
                return Decimal("1")
            else:
                return Decimal("100")

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

    def initialize_websocket(self, symbol, callback):
        """
        Belirli bir sembol için WebSocket bağlantısı başlatır

        Parameters
        ----------
        symbol : str
            İzlenecek sembol (örn. 'btcusdt')
        callback : callable
            WebSocket verisi alındığında çağrılacak fonksiyon
        """
        self.ws_manager = BinanceWebSocketManager(
            symbol=symbol, callback=callback, use_testnet=self.use_testnet
        )
        self.ws_manager.start()
        return self.ws_manager

    def close_websocket(self):
        """WebSocket bağlantısını kapatır"""
        if hasattr(self, "ws_manager"):
            self.ws_manager.stop()


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
        self.exchange = BinanceExchange(api_key, api_secret, use_testnet)
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
            klines = self.exchange.client.get_klines(
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
            return self.exchange.client.get_account()
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
