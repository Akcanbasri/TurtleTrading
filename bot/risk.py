"""
Risk management and position sizing for the Turtle Trading Bot
"""

import logging
from decimal import Decimal
from typing import Tuple, Union

from bot.utils import round_step_size


def calculate_position_size(
    available_balance: Decimal,
    risk_percent: Decimal,
    atr_value: Union[float, Decimal],
    current_price: Decimal,
    symbol_info: dict,
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
        Trading rules for the symbol with keys:
        - min_qty: minimum allowed quantity
        - step_size: quantity step size
        - min_notional: minimum notional value
        - price_precision: price precision
        - quantity_precision: quantity precision

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

        # 1. Calculate the amount willing to risk per trade
        risk_amount = available_balance * risk_percent
        logger.info(
            f"Risk amount: {risk_amount} ({risk_percent*100}% of {available_balance})"
        )

        # 2. Calculate stop loss distance in quote asset terms
        stop_distance = atr_value_dec
        stop_distance_quote = stop_distance
        logger.info(f"Stop distance: {stop_distance} (ATR {atr_value_dec})")

        # 3. Calculate position size based on risk amount and stop distance
        if stop_distance_quote == Decimal("0"):
            return (
                Decimal("0"),
                "Stop distance is zero. Cannot calculate position size.",
            )

        position_size = risk_amount / stop_distance_quote
        logger.info(f"Initial position size calculation: {position_size}")

        # 4. Adjust for step size restrictions
        adjusted_position_size = Decimal(str(round_step_size(position_size, step_size)))
        logger.info(f"Position size adjusted for step size: {adjusted_position_size}")

        # 5. Check against minimum quantity requirement
        if adjusted_position_size < min_qty:
            logger.warning(
                f"Calculated position size {adjusted_position_size} is below minimum quantity {min_qty}"
            )

            # Try to use minimum quantity instead
            adjusted_position_size = min_qty
            logger.info(f"Adjusted to minimum quantity: {adjusted_position_size}")

        # 6. Check against minimum notional value
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

        # 7. Check if there's enough balance for the position
        required_balance = adjusted_position_size * current_price

        if required_balance > available_balance:
            logger.warning(
                f"Required balance {required_balance} exceeds available balance {available_balance}"
            )

            # Try to adjust position size to available balance
            max_affordable_size = (
                available_balance * Decimal("0.99")
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

        # 8. Final check for all requirements
        if (
            adjusted_position_size >= min_qty
            and adjusted_position_size * current_price >= min_notional
            and adjusted_position_size * current_price <= available_balance
        ):

            # Format to correct precision
            from bot.utils import format_quantity

            formatted_size = format_quantity(adjusted_position_size, quantity_precision)
            final_size = Decimal(formatted_size)

            logger.info(f"Final position size: {final_size}")
            logger.info(f"Estimated cost: {final_size * current_price}")
            logger.info(f"Risk per trade: {risk_amount} ({risk_percent*100}%)")

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
