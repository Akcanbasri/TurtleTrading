"""
Data models and type definitions for the Turtle Trading Bot
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Union, Any, Literal


@dataclass
class SymbolInfo:
    """Symbol information and trading rules"""

    price_precision: int
    quantity_precision: int
    min_qty: Decimal
    step_size: Decimal
    min_notional: Decimal


@dataclass
class PositionState:
    """Position state tracking"""

    active: bool = False
    entry_price: Decimal = Decimal("0")
    quantity: Decimal = Decimal("0")
    stop_loss_price: Decimal = Decimal("0")
    take_profit_price: Decimal = Decimal("0")
    side: str = ""  # 'BUY' or 'SELL'
    entry_time: int = 0
    entry_atr: Decimal = Decimal("0")

    def reset(self) -> None:
        """Reset position state to default values"""
        self.active = False
        self.entry_price = Decimal("0")
        self.quantity = Decimal("0")
        self.stop_loss_price = Decimal("0")
        self.take_profit_price = Decimal("0")
        self.side = ""
        self.entry_time = 0
        self.entry_atr = Decimal("0")


@dataclass
class BotConfig:
    """Configuration for the Turtle Trading Bot"""

    # API Configuration
    api_key: str
    api_secret: str
    use_testnet: bool

    # Trading Parameters
    symbol: str
    timeframe: str

    # Donchian Channel Parameters
    dc_length_enter: int
    dc_length_exit: int

    # ATR Parameters
    atr_length: int
    atr_smoothing: Union[int, str]

    # Risk Management
    risk_per_trade: Decimal
    stop_loss_atr_multiple: Decimal

    # Assets
    quote_asset: str
    base_asset: str

    @classmethod
    def from_env(cls) -> "BotConfig":
        """Create configuration from environment variables"""
        import os
        from decimal import Decimal

        # Helper to safely get and clean env values
        def get_clean_env(key, default):
            value = os.getenv(key, default)
            # Remove any trailing comments if present
            if isinstance(value, str) and "#" in value:
                value = value.split("#")[0].strip()
            return value

        return cls(
            api_key=get_clean_env("BINANCE_API_KEY", ""),
            api_secret=get_clean_env("BINANCE_API_SECRET", ""),
            use_testnet=get_clean_env("USE_TESTNET", "True").lower()
            in ("true", "1", "t"),
            symbol=get_clean_env("SYMBOL", "BTCUSDT"),
            timeframe=get_clean_env("TIMEFRAME", "1h"),
            dc_length_enter=int(get_clean_env("DC_LENGTH_ENTER", "20")),
            dc_length_exit=int(get_clean_env("DC_LENGTH_EXIT", "10")),
            atr_length=int(get_clean_env("ATR_LENGTH", "14")),
            atr_smoothing=int(get_clean_env("ATR_SMOOTHING", "2")),
            risk_per_trade=Decimal(get_clean_env("RISK_PER_TRADE", "0.02")),
            stop_loss_atr_multiple=Decimal(
                get_clean_env("STOP_LOSS_ATR_MULTIPLE", "2")
            ),
            quote_asset=get_clean_env("QUOTE_ASSET", "USDT"),
            base_asset=get_clean_env("BASE_ASSET", "BTC"),
        )


TradeSide = Literal["BUY", "SELL"]
OrderResult = Dict[str, Any]
