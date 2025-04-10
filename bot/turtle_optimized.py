"""
Optimized Turtle Trading Bot implementation with enhanced signal detection,
risk management, and market regime adaptability.
"""

import logging
from decimal import Decimal
from typing import Dict, Any, Union, List, Tuple, Optional
import numpy as np
import pandas as pd
import time
import datetime

from bot.models import PositionState, BotConfig
from bot.indicators import (
    calculate_indicators,
    check_entry_signal,
    check_rsi_conditions,
    check_macd_confirmation,
    check_ma_filter,
    calculate_market_regime,
    check_two_way_price_action,
    check_adx_filter,
    log_signal_conditions,
)
from bot.risk import (
    calculate_position_size,
    adjust_leverage_by_signal_strength,
    calculate_partial_take_profit_levels,
    is_weekend,
)

logger = logging.getLogger("turtle_trading_bot")


def check_optimized_entry_signal(
    market_data: pd.DataFrame,
    trend_data: pd.DataFrame,
    side: str,
    config: BotConfig,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check for entry signals with optimized parameters and detailed logging

    Parameters
    ----------
    market_data : pd.DataFrame
        Market data with indicators calculated
    trend_data : pd.DataFrame
        Higher timeframe data for trend detection
    side : str
        Trading side ("BUY"/"long" or "SELL"/"short")
    config : BotConfig
        Bot configuration

    Returns
    -------
    tuple
        (entry_signal, details_dict)
    """
    if market_data.empty or trend_data.empty:
        logger.warning("Empty data provided to check_optimized_entry_signal")
        return False, {"error": "Empty data"}

    # Get the most recent candle
    current_candle = market_data.iloc[-1]
    trend_candle = trend_data.iloc[-1]

    # Convert side to standard format
    position_side = "BUY" if side.upper() in ["LONG", "BUY"] else "SELL"

    # 1. Check base entry signal (Donchian channel breakout)
    base_signal = check_entry_signal(current_candle, position_side)

    # 2. Determine trend direction
    if "ma" in trend_candle:
        trend_direction = (
            "long" if trend_candle["close"] > trend_candle["ma"] else "short"
        )
    else:
        trend_direction = "unknown"

    # 3. Check if trend is aligned with entry direction
    trend_aligned = (
        trend_direction == side.lower() or not config.trend_alignment_required
    )

    # 4. Check technical indicators
    rsi_confirms = check_rsi_conditions(current_candle, position_side)
    macd_confirms = check_macd_confirmation(current_candle, position_side)

    # Lower ADX threshold to 15 as specified
    adx_threshold = 15
    adx_confirms = check_adx_filter(trend_candle, adx_threshold)

    # 5. Check MA filter
    ma_confirms = check_ma_filter(current_candle, side.lower())

    # 6. Check for two-way price action (avoid ranging markets)
    two_way_action = check_two_way_price_action(market_data)

    # 7. Detect market regime
    market_regime = calculate_market_regime(market_data)

    # Create conditions dictionary for detailed logging
    conditions = {
        "DC Breakout": base_signal,
        "Trend Alignment": trend_aligned,
        "RSI Filter": rsi_confirms,
        "MACD Filter": macd_confirms,
        "ADX Filter": adx_confirms,
        "MA Filter": ma_confirms,
        "Clean Price Action": not two_way_action,
    }

    # Log conditions with detailed values
    conditions_met = log_signal_conditions(
        logger, current_candle, "ENTRY", conditions, side.lower()
    )

    # Modified confirmation logic as requested - require EITHER RSI OR MACD
    indicator_confirms = rsi_confirms or macd_confirms

    # Final entry signal decision
    entry_signal = (
        base_signal
        and trend_aligned
        and indicator_confirms
        and adx_confirms
        and not two_way_action
    )

    if entry_signal:
        logger.info(f"🎯 {side.upper()} ENTRY SIGNAL CONFIRMED!")
        triggers = []
        if rsi_confirms:
            triggers.append("RSI")
        if macd_confirms:
            triggers.append("MACD")
        logger.info(f"Entry confirmed by: {', '.join(triggers)}")

    # Calculate signal strength based on number of conditions met
    signal_strength = conditions_met / len(conditions)

    # Return detailed results
    return entry_signal, {
        "signal": entry_signal,
        "direction": side.lower(),
        "trend_direction": trend_direction,
        "trend_aligned": trend_aligned,
        "rsi_confirms": rsi_confirms,
        "macd_confirms": macd_confirms,
        "adx_confirms": adx_confirms,
        "adx_value": trend_candle.get("adx", 0),
        "ma_confirms": ma_confirms,
        "two_way_action": two_way_action,
        "market_regime": market_regime,
        "signal_strength": signal_strength,
        "price": current_candle["close"],
        "atr": current_candle.get("atr", 0),
    }


def check_optimized_exit_conditions(
    market_data: pd.DataFrame,
    position: PositionState,
    current_price: float,
) -> Tuple[bool, str]:
    """
    Check for exit conditions with enhanced exit strategies

    Parameters
    ----------
    market_data : pd.DataFrame
        Market data with indicators calculated
    position : PositionState
        Current position state
    current_price : float
        Current market price

    Returns
    -------
    tuple
        (exit_signal, reason)
    """
    if not position.active:
        return False, "No active position"

    exit_reason = None

    # 1. Check standard Donchian Channel exit
    if "dc_lower" in market_data.columns and "dc_upper" in market_data.columns:
        last_candle = market_data.iloc[-1]
        if position.side == "BUY" and current_price < last_candle["dc_lower"]:
            exit_reason = "Donchian Channel exit signal"
        elif position.side == "SELL" and current_price > last_candle["dc_upper"]:
            exit_reason = "Donchian Channel exit signal"

    # 2. Check for reversal patterns
    if (
        "two_way_action" in market_data.columns
        and market_data["two_way_action"].iloc[-1]
    ):
        # Strong reversal pattern detected
        if (
            position.side == "BUY"
            and market_data["close"].iloc[-1] < market_data["close"].iloc[-2]
            and market_data["close"].iloc[-2] < market_data["close"].iloc[-3]
        ):
            exit_reason = "Reversal pattern (consecutive lower closes)"
        elif (
            position.side == "SELL"
            and market_data["close"].iloc[-1] > market_data["close"].iloc[-2]
            and market_data["close"].iloc[-2] > market_data["close"].iloc[-3]
        ):
            exit_reason = "Reversal pattern (consecutive higher closes)"

    # 3. Check for stop loss hit
    if position.stop_loss_price > 0:
        if (
            position.side == "BUY" and current_price <= float(position.stop_loss_price)
        ) or (
            position.side == "SELL" and current_price >= float(position.stop_loss_price)
        ):
            exit_reason = "Stop loss triggered"

    # 4. Check for partial take profits
    if position.partial_tp_levels and len(position.partial_tp_levels) > 0:
        for i, (tp_level, tp_taken) in enumerate(
            zip(position.partial_tp_levels, position.partial_exits_taken)
        ):
            if not tp_taken:  # Only check levels that haven't been taken yet
                if (position.side == "BUY" and current_price >= tp_level) or (
                    position.side == "SELL" and current_price <= tp_level
                ):
                    # Partial exit should be taken
                    return True, f"Partial take profit {i+1} reached"

    # 5. Check for trend change on higher timeframe
    if "ma" in market_data.columns:
        if (
            position.side == "BUY"
            and market_data["close"].iloc[-1] < market_data["ma"].iloc[-1]
            and market_data["close"].iloc[-2] > market_data["ma"].iloc[-2]
        ):
            exit_reason = "Trend change (price crossed below MA)"
        elif (
            position.side == "SELL"
            and market_data["close"].iloc[-1] > market_data["ma"].iloc[-1]
            and market_data["close"].iloc[-2] < market_data["ma"].iloc[-2]
        ):
            exit_reason = "Trend change (price crossed above MA)"

    return exit_reason is not None, exit_reason or "No exit signal"


def execute_optimized_entry(
    exchange: Any,
    symbol: str,
    direction: str,
    current_price: float,
    atr_value: float,
    signal_strength: float,
    market_regime: str,
    position: PositionState,
    config: BotConfig,
    pyramid_level: int = 0,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Execute an optimized entry with enhanced risk management

    Parameters
    ----------
    exchange : Any
        Exchange interface for executing orders
    symbol : str
        Trading symbol
    direction : str
        Position direction ("long"/"BUY" or "short"/"SELL")
    current_price : float
        Current market price
    atr_value : float
        ATR value for position sizing
    signal_strength : float
        Signal strength (0.0-1.0) for leverage adjustment
    market_regime : str
        Current market regime
    position : PositionState
        Current position state
    config : BotConfig
        Bot configuration
    pyramid_level : int
        Current pyramid level (0 = first entry)

    Returns
    -------
    tuple
        (success, details)
    """
    try:
        # Convert direction to standard format
        side = "BUY" if direction.upper() in ["LONG", "BUY"] else "SELL"

        logger.info(f"Executing optimized {side} entry at price {current_price}")
        logger.info(
            f"Market regime: {market_regime}, Signal strength: {signal_strength:.2f}"
        )

        # Check if weekend for crypto volatility adjustment
        weekend_trading = is_weekend()
        if weekend_trading:
            logger.info("Weekend trading detected - adjusting risk parameters")

        # Get account balance
        balance = exchange.get_balance()
        logger.info(f"Available balance: {balance}")

        # Optimize leverage based on signal strength and market regime
        leverage = adjust_leverage_by_signal_strength(
            signal_strength=signal_strength,
            market_regime=market_regime,
            is_weekend=weekend_trading,
            base_leverage=3,
            max_leverage=5,
            position_side=side,
        )

        # Set leverage on exchange
        try:
            logger.info(f"Setting leverage to {leverage}x")
            exchange.set_leverage(symbol, leverage)
        except Exception as e:
            logger.warning(f"Failed to set leverage: {e}. Using 1x.")
            leverage = 1

        # Calculate position size with risk management
        risk_percent = Decimal(str(config.risk_per_trade))
        position_size, status = calculate_position_size(
            available_balance=balance,
            risk_percent=risk_percent,
            atr_value=atr_value,
            current_price=Decimal(str(current_price)),
            symbol_info=exchange.get_symbol_info(symbol),
            max_risk_percentage=Decimal(str(config.max_risk_percentage)),
            leverage=leverage,
            position_side=side,
            pyramid_level=pyramid_level,
        )

        if status != "success" or position_size <= 0:
            logger.error(f"Position sizing failed: {status}")
            return False, {"error": f"Position sizing failed: {status}"}

        # Calculate stop loss level (using ATR)
        stop_loss_distance = atr_value * config.stop_loss_atr_multiple
        if side == "BUY":
            stop_loss = current_price - stop_loss_distance
        else:
            stop_loss = current_price + stop_loss_distance

        # Calculate partial take profit levels (at 2x and 3x ATR)
        tp_levels = calculate_partial_take_profit_levels(
            entry_price=current_price,
            atr_value=atr_value,
            position_side=side,
            levels=[2.0, 3.0],  # 2x and 3x ATR
        )

        # Log take profit levels
        for tp_price, tp_multiple in tp_levels:
            logger.info(f"Partial take profit ({tp_multiple}x ATR): {tp_price:.4f}")

        # Format order quantities and prices
        symbol_info = exchange.get_symbol_info(symbol)
        qty_precision = symbol_info.quantity_precision
        price_precision = symbol_info.price_precision

        # Format values with correct precision
        qty_str = (
            format(float(position_size), f".{qty_precision}f").rstrip("0").rstrip(".")
        )
        stop_loss_str = format(stop_loss, f".{price_precision}f")

        # Execute the order
        order_result = exchange.create_order(
            symbol=symbol, side=side, quantity=qty_str, stop_loss=stop_loss_str
        )

        if order_result:
            # Update position state with enhanced information
            position.active = True
            position.side = side
            position.entry_price = Decimal(str(current_price))
            position.quantity = Decimal(str(position_size))
            position.stop_loss_price = Decimal(str(stop_loss))
            position.entry_time = int(time.time())
            position.entry_atr = Decimal(str(atr_value))
            position.market_regime = market_regime
            position.signal_strength = signal_strength
            position.leverage_used = leverage

            # Set partial take profit levels
            position.partial_tp_levels = [float(level[0]) for level in tp_levels]
            position.partial_tp_percentages = [0.3, 0.3]  # Take 30% at each level
            position.partial_exits_taken = [False, False]

            # Update pyramid tracking if applicable
            if pyramid_level > 0:
                position.entry_count += 1
                position.last_entry_time = int(time.time())

            logger.info(f"Entry order executed successfully: {order_result}")
            return True, {
                "order": order_result,
                "position": position,
                "leverage": leverage,
                "stop_loss": stop_loss,
                "tp_levels": tp_levels,
            }
        else:
            logger.error("Failed to execute entry order")
            return False, {"error": "Order execution failed"}

    except Exception as e:
        logger.error(f"Error executing optimized entry: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False, {"error": str(e)}


def execute_partial_exit(
    exchange: Any,
    symbol: str,
    position: PositionState,
    tp_index: int,
    current_price: float,
) -> bool:
    """
    Execute a partial exit when a take profit level is reached

    Parameters
    ----------
    exchange : Any
        Exchange interface for executing orders
    symbol : str
        Trading symbol
    position : PositionState
        Current position state
    tp_index : int
        Index of the take profit level that was reached
    current_price : float
        Current market price

    Returns
    -------
    bool
        True if exit executed successfully
    """
    try:
        if not position.active:
            logger.warning("No active position to exit")
            return False

        if tp_index >= len(position.partial_tp_levels) or tp_index >= len(
            position.partial_tp_percentages
        ):
            logger.error(f"Invalid take profit index: {tp_index}")
            return False

        # Check if this level has already been taken
        if position.partial_exits_taken[tp_index]:
            logger.info(f"Take profit level {tp_index+1} already taken")
            return True

        # Calculate exit quantity (percentage of current position)
        exit_percentage = position.partial_tp_percentages[tp_index]
        exit_quantity = float(position.quantity) * exit_percentage

        # Format with correct precision
        symbol_info = exchange.get_symbol_info(symbol)
        qty_precision = symbol_info.quantity_precision
        exit_qty_str = (
            format(exit_quantity, f".{qty_precision}f").rstrip("0").rstrip(".")
        )

        # Execute the order (opposite side from position)
        exit_side = "SELL" if position.side == "BUY" else "BUY"
        logger.info(
            f"Executing partial exit {tp_index+1} ({exit_percentage*100}% of position)"
        )

        order_result = exchange.create_order(
            symbol=symbol,
            side=exit_side,
            quantity=exit_qty_str,
            price=None,  # Market order
        )

        if order_result:
            # Update position state
            position.quantity = position.quantity * Decimal(str(1 - exit_percentage))
            position.partial_exits_taken[tp_index] = True

            # Set first or second target reached flags
            if tp_index == 0:
                position.first_target_reached = True
            elif tp_index == 1:
                position.second_target_reached = True

            # Mark partial exit taken
            position.partial_exit_taken = True

            logger.info(
                f"Partial exit executed successfully. Remaining position: {position.quantity}"
            )
            return True
        else:
            logger.error("Failed to execute partial exit order")
            return False

    except Exception as e:
        logger.error(f"Error executing partial exit: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def get_performance_metrics(trading_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate performance metrics for the trading strategy

    Parameters
    ----------
    trading_history : list
        List of trade dictionaries with trade details

    Returns
    -------
    dict
        Dictionary with performance metrics
    """
    if not trading_history:
        return {
            "win_rate": 0,
            "profit_factor": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "total_trades": 0,
            "regime_performance": {},
        }

    # Extract trade results
    wins = [t for t in trading_history if t.get("pnl", 0) > 0]
    losses = [t for t in trading_history if t.get("pnl", 0) <= 0]

    # Basic metrics
    total_trades = len(trading_history)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_trades if total_trades > 0 else 0

    # Profit metrics
    total_profit = sum(t.get("pnl", 0) for t in wins)
    total_loss = abs(sum(t.get("pnl", 0) for t in losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    # Average metrics
    avg_win = total_profit / win_count if win_count > 0 else 0
    avg_loss = total_loss / loss_count if loss_count > 0 else 0

    # Extreme values
    largest_win = max([t.get("pnl", 0) for t in wins]) if wins else 0
    largest_loss = min([t.get("pnl", 0) for t in losses]) if losses else 0

    # Performance by market regime
    regime_trades = {}
    for trade in trading_history:
        regime = trade.get("market_regime", "unknown")
        if regime not in regime_trades:
            regime_trades[regime] = []
        regime_trades[regime].append(trade)

    regime_performance = {}
    for regime, trades in regime_trades.items():
        regime_wins = [t for t in trades if t.get("pnl", 0) > 0]
        regime_win_rate = len(regime_wins) / len(trades) if trades else 0
        regime_performance[regime] = {
            "win_rate": regime_win_rate,
            "trade_count": len(trades),
            "avg_pnl": (
                sum(t.get("pnl", 0) for t in trades) / len(trades) if trades else 0
            ),
        }

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "total_trades": total_trades,
        "regime_performance": regime_performance,
    }
