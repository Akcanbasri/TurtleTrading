"""
Risk management and position sizing for the Turtle Trading Bot
"""

import logging
from decimal import Decimal
from typing import Tuple, Union, Dict, Any
import pandas as pd

from bot.utils import round_step_size


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
    available_balance: Decimal,
    risk_percent: Decimal,
    atr_value: Union[float, Decimal],
    current_price: Decimal,
    symbol_info: dict,
    max_risk_percentage: Decimal = Decimal("0.1"),
    leverage: int = 1,
    position_side: str = "BUY",
    pyramid_level: int = 0,
    pyramid_size_first: Decimal = Decimal("0.4"),
    pyramid_size_additional: Decimal = Decimal("0.3"),
    is_trend_aligned: bool = True,
    atr_average: Union[float, Decimal] = None,
) -> Tuple[Decimal, str]:
    """
    Calculate position size based on risk management rules and exchange limitations

    Parameters
    ----------
    available_balance : Decimal
        Available balance in quote asset
    risk_percent : Decimal
        Risk percentage per trade (e.g., 0.02 for 2%)
    atr_value : Union[float, Decimal]
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
    pyramid_level : int
        Current pyramid level (0 = first entry)
    pyramid_size_first : Decimal
        Portion of planned size for first entry (e.g., 0.4 = 40%)
    pyramid_size_additional : Decimal
        Portion of planned size for additional entries
    is_trend_aligned : bool
        Whether the trade is aligned with the main trend
    atr_average : Union[float, Decimal], optional
        Average ATR value over a longer period for volatility-based risk adjustment

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
    quantity_precision = symbol_info["quantity_precision"]

    try:
        # Convert inputs to Decimal for precision calculations
        atr_value_dec = (
            Decimal(str(atr_value)) if not isinstance(atr_value, Decimal) else atr_value
        )

        # Apply volatility-based risk adjustment if average ATR is provided
        adjusted_risk_percent = risk_percent
        if atr_average is not None:
            atr_average_dec = (
                Decimal(str(atr_average))
                if not isinstance(atr_average, Decimal)
                else atr_average
            )
            adjusted_risk_percent = adjust_risk_based_on_volatility(
                risk_percent,
                atr_value_dec,
                atr_average_dec,
                Decimal(str(max_risk_percentage)),
            )
        else:
            adjusted_risk_percent = risk_percent

        # 1. Calculate the amount willing to risk per trade
        risk_amount = available_balance * adjusted_risk_percent
        logger.info(
            f"Risk amount: {risk_amount} ({adjusted_risk_percent*100}% of {available_balance})"
        )

        # 2. Adjust leverage based on trend alignment
        effective_leverage = Decimal(str(leverage))
        if not is_trend_aligned:
            effective_leverage = Decimal("1.0")  # No leverage for counter-trend trades
            logger.info("Counter-trend trade detected, reducing leverage to 1x")

        logger.info(f"Using effective leverage: {effective_leverage}x")

        # 3. Calculate stop loss distance in quote asset terms
        stop_distance = atr_value_dec
        stop_distance_quote = stop_distance

        # 4. Adjust for leverage - risk stays the same but position size increases
        risk_with_leverage = risk_amount * effective_leverage

        logger.info(f"Stop distance: {stop_distance} (ATR {atr_value_dec})")
        logger.info(f"Leveraged risk amount: {risk_with_leverage}")

        # 5. Calculate base position size based on risk amount and stop distance
        if stop_distance_quote == Decimal("0"):
            return (
                Decimal("0"),
                "Stop distance is zero. Cannot calculate position size.",
            )

        base_position_size = risk_with_leverage / stop_distance_quote
        logger.info(f"Base position size calculation: {base_position_size}")

        # 6. Apply position sizing based on pyramid level
        if pyramid_level == 0:
            # First entry - use the specified percentage
            position_size = base_position_size * pyramid_size_first
            logger.info(f"First pyramid entry: using {pyramid_size_first*100}% of size")
        else:
            # Subsequent entries - use the additional percentage
            position_size = base_position_size * pyramid_size_additional
            logger.info(
                f"Pyramid level {pyramid_level+1}: using {pyramid_size_additional*100}% of size"
            )

        logger.info(f"Position size after pyramid adjustment: {position_size}")

        # 7. Adjust for step size restrictions
        adjusted_position_size = Decimal(str(round_step_size(position_size, step_size)))
        logger.info(f"Position size adjusted for step size: {adjusted_position_size}")

        # 8. Check against minimum quantity requirement
        if adjusted_position_size < min_qty:
            logger.warning(
                f"Calculated position size {adjusted_position_size} is below minimum quantity {min_qty}"
            )

            # Try to use minimum quantity instead
            adjusted_position_size = min_qty
            logger.info(f"Adjusted to minimum quantity: {adjusted_position_size}")

        # 9. Check against minimum notional value
        notional_value = adjusted_position_size * current_price

        if notional_value < min_notional:
            logger.warning(
                f"Notional value {notional_value} is below minimum {min_notional}"
            )

            # Try to adjust position size to meet minimum notional
            required_position_size = min_notional / current_price
            # Round up to next step size
            adjusted_position_size = Decimal(
                round_step_size(required_position_size, step_size)
            )

            # Re-check step size compliance
            if adjusted_position_size % step_size != Decimal("0"):
                # Ensure it's a multiple of step_size by rounding up
                steps = (adjusted_position_size / step_size).quantize(
                    Decimal("1"), rounding="ROUND_UP"
                )
                adjusted_position_size = steps * step_size

            logger.info(
                f"Position size adjusted for min notional: {adjusted_position_size}"
            )
            notional_value = adjusted_position_size * current_price
            logger.info(f"New notional value: {notional_value}")

        # 10. Check if there's enough balance for the position
        margin_required = (adjusted_position_size * current_price) / effective_leverage

        if margin_required > available_balance:
            logger.warning(
                f"Required margin {margin_required} exceeds available balance {available_balance}"
            )

            # Try to adjust position size to available balance
            max_affordable_size = (
                available_balance * Decimal("0.99") * effective_leverage
            ) / current_price  # 99% of balance for fees
            adjusted_position_size = Decimal(
                round_step_size(max_affordable_size, step_size)
            )

            # Final check against minimum requirements
            if adjusted_position_size < min_qty:
                return (
                    Decimal("0"),
                    f"Cannot meet minimum quantity requirement ({min_qty}) with available balance",
                )

            notional_value = adjusted_position_size * current_price
            if notional_value < min_notional:
                return (
                    Decimal("0"),
                    f"Cannot meet minimum notional requirement ({min_notional}) with available balance",
                )

            logger.info(
                f"Position size adjusted for available balance: {adjusted_position_size}"
            )

        # 11. Final check for all requirements
        if (
            adjusted_position_size >= min_qty
            and adjusted_position_size * current_price >= min_notional
            and (adjusted_position_size * current_price) / effective_leverage
            <= available_balance
        ):
            # Format to correct precision
            from bot.utils import format_quantity

            formatted_size = format_quantity(adjusted_position_size, quantity_precision)
            final_size = Decimal(formatted_size)

            logger.info(f"Final position size: {final_size}")
            logger.info(f"Estimated cost: {final_size * current_price}")
            logger.info(
                f"Margin required: {(final_size * current_price) / effective_leverage}"
            )
            logger.info(f"Risk per trade: {risk_amount} ({adjusted_risk_percent*100}%)")
            logger.info(f"Leverage used: {effective_leverage}x")

            return final_size, "success"
        else:
            return (
                Decimal("0"),
                "Failed to calculate valid position size meeting all requirements",
            )

    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return Decimal("0"), f"Error calculating position size: {str(e)}"


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
