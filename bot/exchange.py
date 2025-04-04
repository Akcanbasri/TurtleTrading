"""
Exchange operations using Binance API for the Turtle Trading Bot
"""

import logging
import time
from decimal import Decimal
from typing import Dict, Any, Optional, Tuple
import pandas as pd

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

from bot.models import SymbolInfo, TradeSide
from bot.utils import format_price, format_quantity


class BinanceExchange:
    """
    Binance exchange API wrapper

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
        self.client = self._initialize_client()

    def _initialize_client(self) -> Client:
        """
        Initialize Binance client with API credentials

        Returns
        -------
        Client
            Initialized Binance client

        Raises
        ------
        ValueError
            If API credentials are missing
        BinanceAPIException
            If there's an issue with the Binance API
        """
        if not self.api_key or not self.api_secret:
            self.logger.error("API Key or Secret Key not found.")
            raise ValueError("API Key or Secret Key not found")

        try:
            if self.use_testnet:
                self.logger.info("Initializing Binance client in TESTNET mode")
                client = Client(self.api_key, self.api_secret, testnet=True)
            else:
                self.logger.info("Initializing Binance client in PRODUCTION mode")
                client = Client(self.api_key, self.api_secret)

            # Test connectivity
            client.ping()
            server_time = client.get_server_time()
            self.logger.info(f"Connected to Binance. Server time: {server_time}")

            # Verify account access
            account_info = client.get_account()
            self.logger.info(
                f"Account status: {account_info['accountType']}, canTrade: {account_info['canTrade']}"
            )

            return client
        except BinanceAPIException as e:
            self.logger.error(f"Binance API Exception: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing Binance client: {e}")
            raise

    def get_symbol_info(self, symbol: str) -> SymbolInfo:
        """
        Retrieve and process symbol trading information

        Parameters
        ----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')

        Returns
        -------
        SymbolInfo
            Processed symbol trading rules

        Raises
        ------
        ValueError
            If symbol not found or required filters missing
        """
        try:
            symbol_info = self.client.get_symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Symbol {symbol} not found on Binance")
                raise ValueError(f"Symbol {symbol} not found on Binance")

            self.logger.info(f"Retrieved symbol information for {symbol}")

            # Extract filters
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
                    if f["filterType"] in ["MIN_NOTIONAL", "NOTIONAL"]
                ),
                None,
            )

            if not price_filter or not lot_size_filter or not min_notional_filter:
                self.logger.error(
                    f"Required filters not found for {symbol_info['symbol']}"
                )
                raise ValueError(
                    f"Required filters not found for {symbol_info['symbol']}"
                )

            # Calculate precision from tick size (e.g., 0.00001 -> 5 decimal places)
            def get_precision_from_step(step_str):
                decimal_str = step_str.rstrip("0")
                if "." in decimal_str:
                    return len(decimal_str) - decimal_str.index(".") - 1
                return 0

            # Process price filter
            price_precision = get_precision_from_step(price_filter["tickSize"])

            # Process lot size filter
            step_size = Decimal(lot_size_filter["stepSize"])
            min_qty = Decimal(lot_size_filter["minQty"])
            quantity_precision = get_precision_from_step(lot_size_filter["stepSize"])

            # Process min notional filter
            min_notional = Decimal(min_notional_filter["minNotional"])

            # Log filter information
            self.logger.info(f"Symbol Filters for {symbol_info['symbol']}:")
            self.logger.info(f"Price Precision: {price_precision}")
            self.logger.info(f"Quantity Precision: {quantity_precision}")
            self.logger.info(f"Minimum Quantity: {min_qty}")
            self.logger.info(f"Step Size: {step_size}")
            self.logger.info(f"Minimum Notional: {min_notional}")

            return SymbolInfo(
                price_precision=price_precision,
                quantity_precision=quantity_precision,
                min_qty=min_qty,
                step_size=step_size,
                min_notional=min_notional,
            )
        except Exception as e:
            self.logger.error(f"Error processing symbol filters: {e}")
            raise

    def fetch_historical_data(
        self, symbol: str, interval: str, lookback: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical klines (candlestick) data from Binance

        Parameters
        ----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        interval : str
            Kline interval (e.g., '1h', '4h', '1d')
        lookback : int
            Number of candles to fetch

        Returns
        -------
        Optional[pd.DataFrame]
            DataFrame with columns: timestamp, open, high, low, close, volume
            None if an error occurs
        """
        try:
            # Add extra candles for calculations (e.g., for indicators that need more data)
            extra_candles = 50
            total_candles = lookback + extra_candles

            self.logger.info(
                f"Fetching {total_candles} {interval} candles for {symbol}"
            )

            # Fetch klines from Binance
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,  # Maximum allowed by Binance
                start_str=f"{total_candles} {interval} ago UTC",
            )

            if not klines:
                self.logger.warning(f"No data returned for {symbol} {interval}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(
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

            # Keep only the needed columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)

            # Set timestamp as index
            df.set_index("timestamp", inplace=True)

            # Remove the last candle (it might be incomplete)
            df = df.iloc[:-1]

            # Make sure we have enough data
            if len(df) < lookback:
                self.logger.warning(
                    f"Insufficient data: got {len(df)} candles, wanted {lookback}"
                )
                return df

            # Keep only the most recent data up to lookback
            df = df.iloc[-lookback:]

            self.logger.info(
                f"Successfully fetched {len(df)} {interval} candles for {symbol}"
            )
            return df

        except BinanceAPIException as e:
            self.logger.error(f"Binance API error while fetching data: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get current market price for a symbol

        Parameters
        ----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')

        Returns
        -------
        Optional[Decimal]
            Current market price or None if error
        """
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            return Decimal(ticker["lastPrice"])
        except BinanceAPIException as e:
            self.logger.error(f"Binance API Exception while getting price: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None

    def get_account_balance(self, asset: str) -> Optional[Decimal]:
        """
        Get available balance for an asset

        Parameters
        ----------
        asset : str
            Asset name (e.g., 'USDT', 'BTC')

        Returns
        -------
        Optional[Decimal]
            Available balance or None if error
        """
        try:
            account = self.client.get_account()
            for balance in account["balances"]:
                if balance["asset"] == asset:
                    return Decimal(balance["free"])
            return Decimal("0")
        except BinanceAPIException as e:
            self.logger.error(f"Binance API Exception while getting balance: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return None

    def execute_order(
        self,
        symbol: str,
        side: TradeSide,
        quantity: Decimal,
        symbol_info: SymbolInfo,
        order_type: str = "MARKET",
        simulate: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a market order on Binance or simulate order execution

        Parameters
        ----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        side : TradeSide
            Order side ('BUY' or 'SELL')
        quantity : Decimal
            Order quantity in base asset
        symbol_info : SymbolInfo
            Symbol trading rules
        order_type : str, optional
            Order type, default is 'MARKET'
        simulate : bool, optional
            Whether to simulate the order instead of actually placing it

        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            (success_bool, order_details_dict)
        """
        try:
            # Format quantity to correct precision
            formatted_quantity = format_quantity(
                quantity, symbol_info.quantity_precision
            )

            # Log order details
            self.logger.info(
                f"Preparing to execute {side} order for {formatted_quantity} {symbol}"
            )

            # Get current price for simulation or logging
            ticker = self.client.get_ticker(symbol=symbol)
            current_price = Decimal(ticker["lastPrice"])
            formatted_price = format_price(current_price, symbol_info.price_precision)

            # Simulate order or execute real order based on settings
            if simulate or self.use_testnet:
                # Simulate order execution
                execution_type = "SIMULATED" if simulate else "TESTNET"
                self.logger.info(
                    f"Executing {execution_type} {side} MARKET order for {formatted_quantity} {symbol} at ~{formatted_price}"
                )

                # Build a simulated order response that mimics Binance's response format
                order_time = int(time.time() * 1000)  # Current time in milliseconds
                order_id = f"simulated_{int(order_time)}_{side}_{symbol}".lower()

                # Calculate simulated execution values
                # Add small slippage for realism (0.1%)
                slippage_factor = (
                    Decimal("1.001") if side == "BUY" else Decimal("0.999")
                )
                execution_price = current_price * slippage_factor
                formatted_execution_price = format_price(
                    execution_price, symbol_info.price_precision
                )

                # Calculate commission (simulate 0.1% fee)
                commission_rate = Decimal("0.001")
                commission_asset = (
                    symbol[-4:] if len(symbol) >= 4 else "USDT"
                )  # Use quote asset for commission
                commission_amount = (
                    Decimal(formatted_quantity) * execution_price * commission_rate
                )

                # Simulate fills data
                fills = [
                    {
                        "price": formatted_execution_price,
                        "qty": formatted_quantity,
                        "commission": str(commission_amount),
                        "commissionAsset": commission_asset,
                        "tradeId": int(order_time),
                    }
                ]

                # Build complete simulated response
                order_result = {
                    "symbol": symbol,
                    "orderId": order_id,
                    "clientOrderId": f"simulated_{order_time}",
                    "transactTime": order_time,
                    "price": "0.00000000",  # Market orders don't have a set price
                    "origQty": formatted_quantity,
                    "executedQty": formatted_quantity,
                    "cummulativeQuoteQty": str(
                        Decimal(formatted_quantity) * execution_price
                    ),
                    "status": "FILLED",
                    "timeInForce": "GTC",
                    "type": "MARKET",
                    "side": side,
                    "fills": fills,
                }

                self.logger.info(f"{execution_type} order successfully 'executed'")

            else:
                # Execute actual order on Binance
                self.logger.info(
                    f"Executing REAL {side} MARKET order for {formatted_quantity} {symbol}"
                )

                # Build actual order parameters
                order_params = {
                    "symbol": symbol,
                    "side": side,
                    "type": order_type,
                    "quantity": formatted_quantity,
                }

                # Send order to Binance
                order_result = self.client.create_order(**order_params)
                self.logger.info("Real order sent to Binance, response received")

            # Process order result (same for both real and simulated)
            if order_result["status"] == "FILLED":
                # Calculate average fill price
                total_cost = Decimal("0")
                total_qty = Decimal("0")

                for fill in order_result["fills"]:
                    fill_price = Decimal(fill["price"])
                    fill_qty = Decimal(fill["qty"])
                    fill_cost = fill_price * fill_qty

                    total_cost += fill_cost
                    total_qty += fill_qty

                avg_price = total_cost / total_qty if total_qty > 0 else Decimal("0")

                # Log success details
                self.logger.info(
                    f"Order {order_result['orderId']} FILLED successfully:"
                )
                self.logger.info(f"  Symbol: {order_result['symbol']}")
                self.logger.info(f"  Side: {order_result['side']}")
                self.logger.info(f"  Type: {order_result['type']}")
                self.logger.info(f"  Quantity: {order_result['executedQty']}")
                self.logger.info(
                    f"  Average Fill Price: {format_price(avg_price, symbol_info.price_precision)}"
                )
                self.logger.info(
                    f"  Total Cost: {format_price(total_cost, symbol_info.price_precision)}"
                )

                # Add derived data to the result
                order_result["avgPrice"] = str(avg_price)
                order_result["totalCost"] = str(total_cost)

                return True, order_result
            else:
                # Order was created but not filled
                self.logger.warning(
                    f"Order {order_result['orderId']} created but status is {order_result['status']}"
                )
                return False, order_result

        except BinanceAPIException as e:
            self.logger.error(f"Binance API Exception during order execution: {e}")
            return False, {"error": str(e), "code": getattr(e, "code", None)}

        except BinanceOrderException as e:
            self.logger.error(f"Binance Order Exception: {e}")
            return False, {"error": str(e), "code": getattr(e, "code", None)}

        except Exception as e:
            self.logger.error(f"Unexpected error executing order: {e}")
            return False, {"error": str(e)}
