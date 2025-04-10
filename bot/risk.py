"""
Risk management and position sizing for the Turtle Trading Bot
"""

import logging
from decimal import Decimal
from typing import Tuple, Union, Dict, Any
import pandas as pd

from bot.utils import round_step_size, format_quantity
from bot.models import SymbolInfo


def adjust_risk_based_on_volatility(
    base_risk_percent: Decimal,
    atr_current: Decimal,
    atr_average: Decimal,
    max_risk_percent: Decimal = Decimal("0.05"),
) -> Decimal:
    """
    Adjust risk percentage based on current market volatility

    Parameters
    ----------
    base_risk_percent : Decimal
        Base risk percentage from configuration
    atr_current : Decimal
        Current ATR value
    atr_average : Decimal
        Average ATR value over a longer period
    max_risk_percent : Decimal
        Maximum allowable risk percentage

    Returns
    -------
    Decimal
        Adjusted risk percentage
    """
    logger = logging.getLogger("turtle_trading_bot")

    # Prevent division by zero
    if atr_average == Decimal("0"):
        return base_risk_percent

    # Calculate volatility ratio
    volatility_ratio = atr_current / atr_average

    # Adjust risk based on volatility
    if volatility_ratio > Decimal("1.5"):
        # High volatility - reduce risk
        adjusted_risk = base_risk_percent * Decimal("0.7")
        logger.info(
            f"High volatility detected (ratio: {volatility_ratio}). Reducing risk to {adjusted_risk * 100}%"
        )
    elif volatility_ratio < Decimal("0.7"):
        # Low volatility - can increase risk slightly
        adjusted_risk = base_risk_percent * Decimal("1.2")
        logger.info(
            f"Low volatility detected (ratio: {volatility_ratio}). Increasing risk to {adjusted_risk * 100}%"
        )
    else:
        # Normal volatility - use base risk
        adjusted_risk = base_risk_percent
        logger.info(
            f"Normal volatility detected (ratio: {volatility_ratio}). Using base risk of {adjusted_risk * 100}%"
        )

    # Ensure risk doesn't exceed maximum
    if adjusted_risk > max_risk_percent:
        adjusted_risk = max_risk_percent
        logger.info(f"Capping risk at maximum allowed: {max_risk_percent * 100}%")

    return adjusted_risk


def calculate_position_size(
    balance: float,
    risk_pct: float,
    atr_value: float,
    current_price: float,
    symbol_info: SymbolInfo,
    leverage: int = 1,
    existing_position_count: int = 0,
    price_precision: int = 2,
    max_positions: int = 4,
) -> Tuple[float, float, float]:
    """
    Calculate position size based on risk percentage and ATR value

    Parameters
    ----------
    balance : float
        Available balance
    risk_pct : float
        Risk percentage (0-100)
    atr_value : float
        ATR value for volatility calculation
    current_price : float
        Current price of the asset
    symbol_info : SymbolInfo
        Symbol information including min notional and quantity
    leverage : int, optional
        Leverage multiplier, by default 1
    existing_position_count : int, optional
        Number of existing positions, by default 0
    price_precision : int, optional
        Price precision, by default 2
    max_positions : int, optional
        Maximum number of positions to open, by default 4

    Returns
    -------
    Tuple[float, float, float]
        (position_size, risk_per_unit, stop_loss_distance)
    """
    logger = logging.getLogger("turtle_trading_bot")

    # Convert percentage to decimal
    risk_pct_decimal = risk_pct / 100.0

    # Get risk amount based on balance
    risk_amount = balance * risk_pct_decimal

    # Calculate max risk per position based on pyramid strategy
    if max_positions > 0:
        # Adjust risk based on existing positions
        position_risk_factor = 1.0 - (existing_position_count / max_positions)
        position_risk_amount = risk_amount * position_risk_factor
    else:
        position_risk_amount = risk_amount

    logger.info(
        f"Risk calculation: Balance=${balance:.2f}, Risk %={risk_pct}%, "
        f"Risk Amount=${risk_amount:.2f}, Position Risk=${position_risk_amount:.2f}, "
        f"Existing Positions={existing_position_count}/{max_positions}, Leverage={leverage}x"
    )

    # Calculate stop loss distance in price units
    stop_loss_distance = atr_value * 2

    # Calculate max position size based on risk
    if stop_loss_distance > 0:
        # Calculate risk per unit of price movement
        risk_per_unit = position_risk_amount / stop_loss_distance

        # Calculate position size from risk per unit
        position_size = risk_per_unit

        # Adjust for leverage if applicable
        if leverage > 1:
            position_size *= leverage
            logger.debug(
                f"Position size adjusted for {leverage}x leverage: {position_size:.8f}"
            )
    else:
        logger.warning(
            "Stop loss distance is zero or negative, using minimum position size"
        )
        position_size = float(symbol_info.min_qty)
        risk_per_unit = 0.0

    # Convert to quote currency value
    position_value = position_size * current_price

    # Check if position size meets minimum notional value
    min_notional = float(symbol_info.min_notional)
    if position_value < min_notional:
        logger.warning(
            f"Position value (${position_value:.2f}) below min notional (${min_notional}), "
            f"adjusting position size"
        )
        # Adjust position size to meet minimum notional
        position_size = min_notional / current_price

    # Check if position size meets minimum quantity
    min_qty = float(symbol_info.min_qty)
    if position_size < min_qty:
        logger.warning(
            f"Position size ({position_size:.8f}) below min qty ({min_qty}), "
            f"adjusting to minimum quantity"
        )
        position_size = min_qty

    # Ensure position size respects the step size
    step_size = symbol_info.step_size
    if step_size:
        original_position_size = position_size
        position_size = round_step_size(position_size, step_size)

        if position_size != original_position_size:
            logger.debug(
                f"Position size adjusted from {original_position_size:.8f} to {position_size:.8f} "
                f"to respect step size {step_size}"
            )

    # Final position size logging
    logger.info(
        f"Position size: {position_size:.8f} ({format_quantity(position_size, symbol_info.quantity_precision, symbol_info.step_size)}), "
        f"Value: ${position_size * current_price:.2f}, "
        f"Stop loss distance: {stop_loss_distance:.{price_precision}f}"
    )

    return position_size, risk_per_unit, stop_loss_distance


def calculate_pnl(
    entry_price: Decimal,
    exit_price: Decimal,
    position_quantity: Decimal,
    position_side: str,
) -> Tuple[Decimal, Decimal]:
    """
    Calculate profit and loss for a closed position

    Parameters
    ----------
    entry_price : Decimal
        Position entry price
    exit_price : Decimal
        Position exit price
    position_quantity : Decimal
        Position size
    position_side : str
        Position side ('BUY' or 'SELL')

    Returns
    -------
    Tuple[Decimal, Decimal]
        (profit_loss_amount, profit_loss_percent)
    """
    if position_side == "BUY":
        # Long position
        pnl_amount = (exit_price - entry_price) * position_quantity
        pnl_percent = (exit_price / entry_price - Decimal("1")) * Decimal("100")
    else:
        # Short position
        pnl_amount = (entry_price - exit_price) * position_quantity
        pnl_percent = (Decimal("1") - exit_price / entry_price) * Decimal("100")

    return pnl_amount, pnl_percent


def calculate_partial_exit_quantity(
    position_quantity: Decimal,
    exit_level: int = 1,
    first_exit_percent: Decimal = Decimal("0.5"),
    second_exit_percent: Decimal = Decimal("0.3"),
) -> Decimal:
    """
    Calculate quantity to exit for partial profit taking

    Parameters
    ----------
    position_quantity : Decimal
        Total position quantity
    exit_level : int
        Exit level (1 = first exit, 2 = second exit)
    first_exit_percent : Decimal
        Percentage to exit at first target (e.g., 0.5 = 50%)
    second_exit_percent : Decimal
        Percentage to exit at second target

    Returns
    -------
    Decimal
        Quantity to exit
    """
    if exit_level == 1:
        return position_quantity * first_exit_percent
    elif exit_level == 2:
        return position_quantity * second_exit_percent
    else:
        # Final exit - all remaining
        return position_quantity


def calculate_kelly_criterion(
    win_rate: Decimal,
    avg_win_pct: Decimal,
    avg_loss_pct: Decimal,
    max_leverage: int = 1,
) -> Decimal:
    """
    Calculate optimal position size using Kelly Criterion

    Parameters
    ----------
    win_rate : Decimal
        Win rate (probability of winning) between 0 and 1
    avg_win_pct : Decimal
        Average win percentage (e.g., 0.05 for 5%)
    avg_loss_pct : Decimal
        Average loss percentage (positive value, e.g., 0.02 for 2%)
    max_leverage : int
        Maximum leverage to use

    Returns
    -------
    Decimal
        Kelly fraction (optimal position size as a fraction of capital)
    """
    # Ensure inputs are valid
    if win_rate <= Decimal("0") or win_rate > Decimal("1"):
        raise ValueError("Win rate must be between 0 and 1")

    if avg_win_pct <= Decimal("0"):
        raise ValueError("Average win percentage must be positive")

    if avg_loss_pct <= Decimal("0"):
        raise ValueError("Average loss percentage must be positive")

    # Calculate Kelly fraction: K = W/R - (1-W)/L
    # Where W is win rate, R is win/loss ratio, L is loss rate
    # This simplifies to: K = (W*R - (1-W))/R
    win_loss_ratio = avg_win_pct / avg_loss_pct

    kelly_fraction = (
        win_rate * win_loss_ratio - (Decimal("1") - win_rate)
    ) / win_loss_ratio

    # Apply half-Kelly for more conservative sizing (common practice)
    half_kelly = kelly_fraction / Decimal("2")

    # Ensure result is not negative
    if half_kelly < Decimal("0"):
        half_kelly = Decimal("0")

    # Apply leverage if specified
    kelly_with_leverage = half_kelly * Decimal(str(max_leverage))

    # Cap at 100% of capital (or leverage-adjusted maximum)
    max_position = Decimal(str(max_leverage))
    if kelly_with_leverage > max_position:
        kelly_with_leverage = max_position

    return kelly_with_leverage


def calculate_optimal_position_size(
    available_balance: Decimal,
    trade_history: list,
    atr_value: Decimal,
    current_price: Decimal,
    symbol_info: dict,
    max_risk_percentage: Decimal = Decimal("0.1"),
    leverage: int = 1,
    position_side: str = "BUY",
    is_trend_aligned: bool = True,
    min_trades_required: int = 10,
    default_risk_percent: Decimal = Decimal("0.01"),
) -> Tuple[Decimal, str]:
    """
    Calculate optimal position size using Kelly Criterion if enough trade history is available,
    otherwise fallback to standard risk-based position sizing

    Parameters
    ----------
    available_balance : Decimal
        Available balance in quote asset
    trade_history : list
        List of historical trade results (list of dicts with 'profit_pct' key)
    atr_value : Decimal
        Current ATR value
    current_price : Decimal
        Current market price
    symbol_info : dict
        Trading rules for the symbol
    max_risk_percentage : Decimal
        Maximum risk percentage of all open positions
    leverage : int
        Leverage to use (1 = no leverage)
    position_side : str
        Position side ('BUY' or 'SELL')
    is_trend_aligned : bool
        Whether the trade is aligned with the main trend
    min_trades_required : int
        Minimum number of trades required to use Kelly Criterion
    default_risk_percent : Decimal
        Default risk percentage to use if not enough trade history

    Returns
    -------
    Tuple[Decimal, str]
        Calculated position size and status message
    """
    logger = logging.getLogger("turtle_trading_bot")

    # Extract symbol info parameters
    min_qty = symbol_info["min_qty"]
    step_size = symbol_info["step_size"]
    min_notional = symbol_info["min_notional"]

    # If we have enough trade history, use Kelly Criterion
    if len(trade_history) >= min_trades_required:
        # Calculate win rate and average profit/loss
        wins = [t for t in trade_history if t["profit_pct"] > 0]
        losses = [t for t in trade_history if t["profit_pct"] <= 0]

        if len(wins) > 0 and len(losses) > 0:
            win_rate = Decimal(str(len(wins) / len(trade_history)))
            avg_win_pct = Decimal(str(sum(t["profit_pct"] for t in wins) / len(wins)))
            avg_loss_pct = Decimal(
                str(abs(sum(t["profit_pct"] for t in losses) / len(losses)))
            )

            logger.info(
                f"Trade statistics - Win rate: {win_rate*100:.1f}%, Avg win: {avg_win_pct*100:.2f}%, Avg loss: {avg_loss_pct*100:.2f}%"
            )

            # Calculate Kelly fraction
            try:
                kelly_pct = calculate_kelly_criterion(
                    win_rate, avg_win_pct, avg_loss_pct, max_leverage=leverage
                )

                # Apply risk limits
                risk_pct = min(kelly_pct, max_risk_percentage)

                # Calculate position size
                position_value = available_balance * risk_pct
                position_size = position_value / current_price

                logger.info(
                    f"Using Kelly Criterion - Optimal position size: {kelly_pct*100:.2f}% of capital"
                )
                logger.info(
                    f"Applied risk limits - Using {risk_pct*100:.2f}% of capital"
                )

                # Apply exchange constraints
                position_size = round_step_size(position_size, step_size)
                if position_size < min_qty:
                    return min_qty, "Rounded up to minimum quantity."

                # Check if position value meets minimum notional
                position_value = position_size * current_price
                if position_value < min_notional:
                    min_position_size = min_notional / current_price
                    position_size = round_step_size(min_position_size, step_size)
                    return position_size, "Increased to meet minimum notional value."

                return position_size, "Position size calculated using Kelly Criterion."
            except Exception as e:
                logger.error(
                    f"Error calculating Kelly Criterion: {e}, falling back to standard risk model."
                )

    # Fallback to standard risk-based position sizing
    return calculate_position_size(
        available_balance,
        default_risk_percent,
        atr_value,
        current_price,
        symbol_info,
        max_risk_percentage,
        leverage,
        position_side,
        pyramid_level=0,
        is_trend_aligned=is_trend_aligned,
    )


def calculate_portfolio_correlation(
    symbols: list, price_data: dict, lookback_period: int = 30
) -> pd.DataFrame:
    """
    Calculate correlation matrix between multiple trading symbols

    Parameters
    ----------
    symbols : list
        List of trading symbols
    price_data : dict
        Dictionary with symbol as key and DataFrame with 'close' prices as value
    lookback_period : int
        Number of periods to use for correlation calculation

    Returns
    -------
    pd.DataFrame
        Correlation matrix between symbols
    """
    # Create a DataFrame to hold returns for each symbol
    returns_data = {}

    for symbol in symbols:
        if symbol in price_data and len(price_data[symbol]) > lookback_period:
            # Calculate percentage returns
            returns = price_data[symbol]["close"].pct_change().dropna()
            if len(returns) > lookback_period:
                returns_data[symbol] = returns.iloc[-lookback_period:]

    # If we have data for at least 2 symbols, calculate correlation
    if len(returns_data) > 1:
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()
        return correlation_matrix

    # Return empty DataFrame if not enough data
    return pd.DataFrame()


def optimize_position_sizes_for_portfolio(
    symbols: list,
    correlations: pd.DataFrame,
    individual_positions: dict,
    max_portfolio_risk: Decimal = Decimal("0.2"),
    max_correlated_exposure: Decimal = Decimal("0.15"),
) -> dict:
    """
    Optimize position sizes for a portfolio of symbols based on correlations

    Parameters
    ----------
    symbols : list
        List of trading symbols
    correlations : pd.DataFrame
        Correlation matrix between symbols
    individual_positions : dict
        Dictionary with symbol as key and calculated position size as value
    max_portfolio_risk : Decimal
        Maximum portfolio-wide risk
    max_correlated_exposure : Decimal
        Maximum exposure to highly correlated assets

    Returns
    -------
    dict
        Dictionary with symbol as key and adjusted position size as value
    """
    logger = logging.getLogger("turtle_trading_bot")

    # If correlation matrix is empty or we have only one symbol, return original positions
    if correlations.empty or len(symbols) < 2:
        return individual_positions

    # Copy original positions
    adjusted_positions = individual_positions.copy()

    # Group highly correlated symbols (correlation > 0.7)
    correlated_groups = []
    processed_symbols = set()

    for symbol in symbols:
        if symbol in processed_symbols:
            continue

        if symbol not in correlations.columns:
            processed_symbols.add(symbol)
            continue

        # Find symbols correlated with this one
        correlated = [
            s
            for s in symbols
            if s in correlations.columns
            and symbol != s
            and abs(correlations.loc[symbol, s]) > 0.7
        ]

        if correlated:
            group = [symbol] + correlated
            correlated_groups.append(group)
            processed_symbols.update(group)
        else:
            processed_symbols.add(symbol)

    # Adjust position sizes for correlated groups
    for group in correlated_groups:
        group_symbols = [s for s in group if s in adjusted_positions]

        if not group_symbols:
            continue

        # Calculate total risk in this correlated group
        total_group_risk = sum(
            Decimal(str(adjusted_positions[s])) for s in group_symbols
        )

        # If total risk exceeds maximum for correlated assets, scale down
        if total_group_risk > max_correlated_exposure:
            scale_factor = max_correlated_exposure / total_group_risk

            for symbol in group_symbols:
                adjusted_positions[symbol] = adjusted_positions[symbol] * scale_factor

            logger.info(
                f"Scaling down correlated group {group} by factor {scale_factor} to limit risk"
            )

    # Calculate total portfolio risk after group adjustments
    total_portfolio_risk = sum(
        Decimal(str(adjusted_positions[s])) for s in adjusted_positions
    )

    # If total portfolio risk exceeds maximum, scale down all positions
    if total_portfolio_risk > max_portfolio_risk:
        scale_factor = max_portfolio_risk / total_portfolio_risk

        for symbol in adjusted_positions:
            adjusted_positions[symbol] = adjusted_positions[symbol] * scale_factor

        logger.info(
            f"Scaling down entire portfolio by factor {scale_factor} to limit total risk to {max_portfolio_risk*100}%"
        )

    return adjusted_positions


def adjust_leverage_by_signal_strength(
    signal_strength: float,
    market_regime: str,
    is_weekend: bool = False,
    base_leverage: int = 3,
    max_leverage: int = 5,
    position_side: str = "BUY",
) -> int:
    """
    Adjust leverage based on signal strength, market regime and time of week

    Parameters
    ----------
    signal_strength : float
        Strength of the signal from 0.0-1.0
    market_regime : str
        Current market regime (trending, ranging, squeeze, etc.)
    is_weekend : bool
        Whether it's weekend trading (typically higher volatility in crypto)
    base_leverage : int
        Base leverage level to start with
    max_leverage : int
        Maximum allowed leverage
    position_side : str
        Position side ('BUY' or 'SELL')

    Returns
    -------
    int
        Recommended leverage level (1-5)
    """
    logger = logging.getLogger("turtle_trading_bot")

    # Adjust base leverage based on signal strength
    if signal_strength >= 0.8:
        adjusted_leverage = max(base_leverage + 2, max_leverage)
    elif signal_strength >= 0.6:
        adjusted_leverage = base_leverage + 1
    elif signal_strength >= 0.4:
        adjusted_leverage = base_leverage
    elif signal_strength >= 0.2:
        adjusted_leverage = base_leverage - 1
    else:
        adjusted_leverage = 1  # Minimum leverage for weak signals

    # Market regime adjustments
    if market_regime == "trending_up" and position_side == "BUY":
        adjusted_leverage += 1  # Increase leverage for aligned trend
    elif market_regime == "trending_down" and position_side == "SELL":
        adjusted_leverage += 1  # Increase leverage for aligned trend
    elif market_regime == "ranging":
        adjusted_leverage -= 1  # Reduce leverage in ranging markets
    elif market_regime == "squeeze":
        # Keep normal leverage for squeeze breakouts
        pass
    elif market_regime == "volatile":
        adjusted_leverage -= 1  # Reduce leverage in volatile markets

    # Weekend adjustment for crypto markets
    if is_weekend:
        weekend_reduction = 2  # More reduction for weekends
        adjusted_leverage = max(1, adjusted_leverage - weekend_reduction)
        logger.info(
            f"Weekend trading detected - reducing leverage by {weekend_reduction}"
        )

    # Ensure leverage is in valid range
    adjusted_leverage = max(1, min(adjusted_leverage, max_leverage))

    # Reduce leverage for short positions (more risky)
    if position_side == "SELL":
        adjusted_leverage = max(1, adjusted_leverage - 1)

    logger.info(
        f"Adjusted leverage: {adjusted_leverage}x (Signal strength: {signal_strength:.2f}, Regime: {market_regime})"
    )

    return adjusted_leverage


def calculate_partial_take_profit_levels(
    entry_price: float,
    atr_value: float,
    position_side: str,
    levels: list = [2.0, 3.0],  # Default to 2x and 3x ATR
) -> list:
    """
    Calculate partial take profit levels based on ATR

    Parameters
    ----------
    entry_price : float
        Position entry price
    atr_value : float
        ATR value at entry
    position_side : str
        Position side ('BUY' or 'SELL')
    levels : list
        List of ATR multiples for take profit levels

    Returns
    -------
    list
        List of take profit price levels
    """
    tp_levels = []

    for level_multiple in levels:
        target_distance = atr_value * level_multiple

        if position_side == "BUY":
            # Long position - take profit is above entry
            tp_price = entry_price + target_distance
        else:
            # Short position - take profit is below entry
            tp_price = entry_price - target_distance

        tp_levels.append((tp_price, level_multiple))

    return tp_levels


def is_weekend() -> bool:
    """
    Check if current time is weekend (Saturday or Sunday)
    Returns true if it's Saturday or Sunday

    Returns
    -------
    bool
        True if current day is weekend, False otherwise
    """
    import datetime

    # Get current day of week (0 = Monday, 6 = Sunday)
    current_day = datetime.datetime.now().weekday()

    # 5 = Saturday, 6 = Sunday
    return current_day >= 5
