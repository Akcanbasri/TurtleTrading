"""
Data models and type definitions for the Turtle Trading Bot
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Union, Any, Literal, List
import os
import json
from pathlib import Path
from dotenv import load_dotenv


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
    """Current position state and information."""

    def __init__(self):
        """Initialize an empty position state."""
        self.active = False
        self.side = None  # "BUY" for long, "SELL" for short
        self.entry_price = 0.0
        self.quantity = 0.0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.entry_time = None
        self.exit_time = None
        self.exit_price = 0.0
        self.pnl_amount = 0.0
        self.pnl_percentage = 0.0
        self.exit_reason = None
        self.signal_strength = 0.0
        self.max_running_profit = 0.0
        self.max_running_loss = 0.0
        self.trades_count = 0
        self.entry_count = 0  # For pyramiding
        self.profit_trades = 0
        self.loss_trades = 0

        # Added fields for enhanced tracking
        self.trailing_stop_price = None
        self.trailing_stop_activated = False
        self.partial_exit_executed = False
        self.partial_exit_price = 0.0
        self.partial_exit_time = None
        self.first_target_price = 0.0
        self.second_target_price = 0.0
        self.max_price = 0.0  # Highest price reached during the trade (for longs)
        self.min_price = 0.0  # Lowest price reached during the trade (for shorts)
        self.entry_atr = 0.0  # ATR at time of entry
        self.entry_signal_type = None  # e.g., "dc_breakout", "pullback"
        self.entry_signal_timeframe = None  # e.g., "5m", "15m"

    def update(self, **kwargs):
        """Update position state attributes."""
        # Only update attributes that are passed in kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def reset(self):
        """Reset position state."""
        self.active = False
        self.side = None
        self.entry_price = 0.0
        self.quantity = 0.0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.entry_time = None
        self.exit_time = None
        self.exit_price = 0.0
        self.pnl_amount = 0.0
        self.pnl_percentage = 0.0
        self.exit_reason = None
        self.signal_strength = 0.0
        self.max_running_profit = 0.0
        self.max_running_loss = 0.0
        self.entry_count = 0

        # Reset enhanced tracking fields
        self.trailing_stop_price = None
        self.trailing_stop_activated = False
        self.partial_exit_executed = False
        self.partial_exit_price = 0.0
        self.partial_exit_time = None
        self.first_target_price = 0.0
        self.second_target_price = 0.0
        self.max_price = 0.0
        self.min_price = 0.0
        self.entry_atr = 0.0
        self.entry_signal_type = None
        self.entry_signal_timeframe = None

    def calculate_unrealized_pnl(self, current_price):
        """
        Calculate unrealized profit/loss percentage based on current price

        Parameters
        ----------
        current_price : float
            Current market price

        Returns
        -------
        float
            Unrealized PnL percentage
        """
        if not self.active or self.entry_price == 0:
            return 0.0

        if self.side == "BUY":
            pnl_percentage = (
                (current_price - self.entry_price) / self.entry_price
            ) * 100
        else:  # SELL
            pnl_percentage = (
                (self.entry_price - current_price) / self.entry_price
            ) * 100

        # Update max profit/drawdown tracking
        if pnl_percentage > 0 and pnl_percentage > self.max_running_profit:
            self.max_running_profit = pnl_percentage
        elif pnl_percentage < 0 and abs(pnl_percentage) > abs(self.max_running_loss):
            self.max_running_loss = pnl_percentage

        # Update max/min price tracking
        if self.side == "BUY" and current_price > self.max_price:
            self.max_price = current_price
        elif self.side == "SELL" and current_price < self.min_price:
            self.min_price = current_price

        return pnl_percentage


def save_position_state(position: PositionState, symbol: str) -> None:
    """
    Save position state to file

    Parameters
    ----------
    position : PositionState
        Position state to save
    symbol : str
        Trading symbol
    """
    # Create config directory if it doesn't exist
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    # Path to state file
    state_file = config_dir / "bot_state.json"

    # Convert position to dictionary
    position_dict = {
        "active": position.active,
        "entry_price": str(position.entry_price),
        "quantity": str(position.quantity),
        "stop_loss_price": str(position.stop_loss_price),
        "take_profit_price": str(position.take_profit_price),
        "side": position.side,
        "entry_time": position.entry_time,
        "entry_atr": str(position.entry_atr),
        "entry_count": position.entry_count,
        "exit_time": position.exit_time,
        "exit_price": str(position.exit_price),
        "pnl_amount": str(position.pnl_amount),
        "pnl_percentage": str(position.pnl_percentage),
        "exit_reason": position.exit_reason,
        "signal_strength": position.signal_strength,
        "max_running_profit": str(position.max_running_profit),
        "max_running_loss": str(position.max_running_loss),
        "trades_count": position.trades_count,
        "profit_trades": position.profit_trades,
        "loss_trades": position.loss_trades,
        "trailing_stop_price": (
            str(position.trailing_stop_price) if position.trailing_stop_price else None
        ),
        "trailing_stop_activated": position.trailing_stop_activated,
        "partial_exit_executed": position.partial_exit_executed,
        "partial_exit_price": str(position.partial_exit_price),
        "partial_exit_time": position.partial_exit_time,
        "first_target_price": str(position.first_target_price),
        "second_target_price": str(position.second_target_price),
        "max_price": str(position.max_price),
        "min_price": str(position.min_price),
        "entry_signal_type": position.entry_signal_type,
        "entry_signal_timeframe": position.entry_signal_timeframe,
        "symbol": symbol,
    }

    # Save to file
    with open(state_file, "w") as f:
        json.dump(position_dict, f, indent=4)


class BotConfig:
    """Bot configuration parameters."""

    def __init__(self, env_file=".env"):
        """Initialize configuration from environment variables or .env file."""
        # Load environment variables
        load_dotenv(env_file)

        # API settings
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.use_testnet = os.getenv("USE_TESTNET", "True").lower() in (
            "true",
            "1",
            "t",
        )

        # Trading settings
        self.symbol = os.getenv("SYMBOL", "BTCUSDT")
        self.timeframe = os.getenv("TIMEFRAME", "1h")

        # Multi-timeframe settings
        self.use_multi_timeframe = os.getenv(
            "USE_MULTI_TIMEFRAME", "False"
        ).lower() in ("true", "1", "t")
        self.trend_timeframe = os.getenv("TREND_TIMEFRAME", "1d")
        self.entry_timeframe = os.getenv("ENTRY_TIMEFRAME", "4h")
        self.trend_alignment_required = os.getenv(
            "TREND_ALIGNMENT_REQUIRED", "True"
        ).lower() in ("true", "1", "t")

        # Parse the quote and base assets from the symbol
        self.symbol_info = None  # Will be set later by exchange
        self.quote_asset = os.getenv("QUOTE_ASSET", "USDT")
        self.base_asset = os.getenv("BASE_ASSET", "BTC")

        # Helper function to clean values and handle comments
        def clean_value(value):
            if not value:
                return value
            # Remove any trailing comments if present
            if "#" in value:
                value = value.split("#")[0].strip()
            return value

        # Donchian Channel settings
        try:
            dc_length_enter = clean_value(os.getenv("DC_LENGTH_ENTER", "20"))
            dc_length_exit = clean_value(os.getenv("DC_LENGTH_EXIT", "10"))

            self.dc_length_enter = int(dc_length_enter)
            self.dc_length_exit = int(dc_length_exit)
        except ValueError as e:
            raise ValueError(f"Error parsing Donchian Channel parameters: {e}")

        # ATR settings
        try:
            atr_length = clean_value(os.getenv("ATR_LENGTH", "14"))
            self.atr_length = int(atr_length)
            self.atr_smoothing = os.getenv("ATR_SMOOTHING", "RMA")
        except ValueError as e:
            raise ValueError(f"Error parsing ATR parameters: {e}")

        # Risk management settings
        try:
            risk_per_trade = clean_value(os.getenv("RISK_PER_TRADE", "0.01"))
            stop_loss_atr = clean_value(os.getenv("STOP_LOSS_ATR_MULTIPLE", "2.0"))
            max_risk = clean_value(os.getenv("MAX_RISK_PERCENTAGE", "0.15"))

            self.risk_per_trade = float(risk_per_trade)
            self.stop_loss_atr_multiple = float(stop_loss_atr)
            self.max_risk_percentage = float(max_risk)
        except ValueError as e:
            raise ValueError(f"Error parsing risk management parameters: {e}")

        # Pyramiding settings
        self.use_pyramiding = os.getenv("USE_PYRAMIDING", "False").lower() in (
            "true",
            "1",
            "t",
        )
        try:
            pyramid_max = clean_value(os.getenv("PYRAMID_MAX_ENTRIES", "3"))
            pyramid_first = clean_value(os.getenv("PYRAMID_SIZE_FIRST", "0.4"))
            pyramid_add = clean_value(os.getenv("PYRAMID_SIZE_ADDITIONAL", "0.3"))

            self.pyramid_max_entries = int(pyramid_max)
            self.pyramid_size_first = float(pyramid_first)
            self.pyramid_size_additional = float(pyramid_add)
        except ValueError as e:
            raise ValueError(f"Error parsing pyramiding parameters: {e}")

        # Exit strategy settings
        self.use_trailing_stop = os.getenv("USE_TRAILING_STOP", "False").lower() in (
            "true",
            "1",
            "t",
        )
        self.use_partial_exits = os.getenv("USE_PARTIAL_EXITS", "False").lower() in (
            "true",
            "1",
            "t",
        )
        try:
            first_target = clean_value(os.getenv("FIRST_TARGET_ATR", "3.0"))
            second_target = clean_value(os.getenv("SECOND_TARGET_ATR", "5.0"))
            profit_trailing = clean_value(os.getenv("PROFIT_FOR_TRAILING", "0.02"))

            self.first_target_atr = float(first_target)
            self.second_target_atr = float(second_target)
            self.profit_for_trailing = float(profit_trailing)
        except ValueError as e:
            raise ValueError(f"Error parsing exit strategy parameters: {e}")

        # Filter settings
        self.use_adx_filter = os.getenv("USE_ADX_FILTER", "False").lower() in (
            "true",
            "1",
            "t",
        )
        try:
            adx_period = clean_value(os.getenv("ADX_PERIOD", "14"))
            adx_thresh = clean_value(os.getenv("ADX_THRESHOLD", "25.0"))

            self.adx_period = int(adx_period)
            self.adx_threshold = float(adx_thresh)
        except ValueError as e:
            raise ValueError(f"Error parsing ADX filter parameters: {e}")

        self.use_ma_filter = os.getenv("USE_MA_FILTER", "False").lower() in (
            "true",
            "1",
            "t",
        )
        try:
            ma_period = clean_value(os.getenv("MA_PERIOD", "200"))

            self.ma_period = int(ma_period)
            self.ma_type = os.getenv("MA_TYPE", "SMA")
        except ValueError as e:
            raise ValueError(f"Error parsing MA filter parameters: {e}")

        # Leverage settings
        try:
            leverage = clean_value(os.getenv("LEVERAGE", "1.0"))
            max_lev_trend = clean_value(os.getenv("MAX_LEVERAGE_TREND", "3.0"))
            max_lev_counter = clean_value(os.getenv("MAX_LEVERAGE_COUNTER", "1.5"))

            self.leverage = float(leverage)
            self.max_leverage_trend = float(max_lev_trend)
            self.max_leverage_counter = float(max_lev_counter)
        except ValueError as e:
            raise ValueError(f"Error parsing leverage parameters: {e}")

    def __getitem__(self, key):
        """Allow dictionary-like access to configuration attributes."""
        return getattr(self, key)

    def set_symbol_info(self, symbol_info):
        """Set the symbol information."""
        self.symbol_info = symbol_info

    def get_all_settings(self):
        """Get all settings as a dictionary."""
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith("__") and not callable(getattr(self, attr))
        }

    def load_timeframe_preset(self, preset_name):
        """
        Load timeframe preset settings from config file

        Args:
            preset_name: Name of the preset to load

        Returns:
            bool: Whether the preset was loaded successfully
        """
        import json
        import logging
        from pathlib import Path

        logger = logging.getLogger("turtle_trading_bot")

        preset_file = Path("bot/config/timeframe_presets.json")

        if not preset_file.exists():
            logger.error(f"Preset file {preset_file} does not exist")
            return False

        try:
            with open(preset_file, "r") as f:
                presets = json.load(f)

            if preset_name not in presets:
                logger.error(f"Preset '{preset_name}' not found in presets file")
                logger.info(f"Available presets: {list(presets.keys())}")
                return False

            preset = presets[preset_name]
            logger.info(
                f"Loading timeframe preset: {preset_name} - {preset['description']}"
            )

            # Set timeframes
            self.timeframe = preset.get("timeframe", self.timeframe)
            self.trend_timeframe = preset.get("trend_timeframe", self.trend_timeframe)
            self.entry_timeframe = preset.get("entry_timeframe", self.entry_timeframe)

            # Set indicator parameters
            self.dc_length_enter = preset.get("dc_length_enter", self.dc_length_enter)
            self.dc_length_exit = preset.get("dc_length_exit", self.dc_length_exit)
            self.atr_length = preset.get("atr_length", self.atr_length)

            # Ensure multi-timeframe is enabled if we're using different timeframes
            if (
                self.timeframe != self.trend_timeframe
                or self.timeframe != self.entry_timeframe
            ):
                self.use_multi_timeframe = True

            logger.info(f"Timeframe preset '{preset_name}' loaded successfully")
            logger.info(
                f"New timeframes: Base={self.timeframe}, Trend={self.trend_timeframe}, Entry={self.entry_timeframe}"
            )
            return True

        except Exception as e:
            logger.error(f"Error loading timeframe preset: {e}")
            return False


TradeSide = Literal["BUY", "SELL"]
OrderResult = Dict[str, Any]
