"""Offline simulated orders exercise Kraken funding through native bot and wallets."""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import select

from freqtrade.enums import CandleType, ExitCheckTuple, ExitType, RPCMessageType
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.leverage.liquidation_price import update_liquidation_prices
from freqtrade.persistence import Order, Trade
from freqtrade.util import dt_utc
from tests.conftest import EXMS, patch_exchange, patch_freqtradebot


START = dt_utc(2024, 1, 1, 12)
PAIR = "ETH/USDT"
FUNDING = (PAIR, "1h", CandleType.FUNDING_RATE)


def history(hours: int = 4) -> pd.DataFrame:
    """Public funding history: the first `hours` hourly rates from START."""
    return pd.DataFrame(
        {
            "date": pd.date_range(START, periods=hours, freq="h"),
            "funding_rate": [0.0001] * hours,
            "funding_rate_absolute": [0.01] * hours,
        }
    )


def make_bot(mocker, conf):
    conf.update(
        trading_mode="futures",
        margin_mode="isolated",
        dry_run=True,
        dry_run_wallet=10000.0,
        stake_amount=1000.0,
        position_adjustment_enable=True,
    )
    conf["exchange"]["name"] = "krakenfutures"
    patch_freqtradebot(mocker, conf)
    patch_exchange(mocker, exchange="krakenfutures")
    mocker.patch.multiple(
        EXMS,
        get_rate=MagicMock(return_value=100.0),
        get_min_pair_stake_amount=MagicMock(return_value=1),
        get_max_pair_stake_amount=MagicMock(return_value=1e6),
        get_fee=MagicMock(return_value=0.0),
        get_max_leverage=MagicMock(return_value=1),
        get_maintenance_ratio_and_amt=MagicMock(return_value=(0.01, 0.01)),
        get_dry_market_fill_price=MagicMock(return_value=100.0),
    )
    mocker.patch("freqtrade.freqtradebot.update_liquidation_prices")
    bot = FreqtradeBot(conf)
    mocker.patch.object(
        bot.exchange,
        "fetch_l2_order_book",
        return_value={
            "bids": [[100.0, 1000]],
            "asks": [[100.0, 1000]],
        },
    )
    mocker.patch.object(bot.exchange, "refresh_latest_ohlcv", return_value={FUNDING: history()})
    return bot


@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("refresh", [None, "changed", "unchanged"])
def test_dry_entry_add_reduce_close_and_repeated_poll(
    mocker, default_conf_usdt, time_machine, short, refresh
):
    time_machine.move_to(START, tick=False)
    bot = make_bot(mocker, default_conf_usdt)
    scale = 200 if refresh else 1
    if refresh:
        bot.exchange.refresh_latest_ohlcv.return_value = {
            FUNDING: history().assign(funding_rate_absolute=2.0)
        }
    assert bot.execute_entry(PAIR, 1000.0, ordertype="market", is_short=short)
    trade = Trade.session.scalars(select(Trade)).one()
    assert trade.amount == 10
    assert trade.funding_fees == 0
    sign = 1 if short else -1

    time_machine.move_to(START + timedelta(minutes=30), tick=False)
    bot.update_funding_fees()
    bot.wallets.update()
    assert trade.funding_fees == pytest.approx(sign * 0.05 * scale)
    assert bot.wallets.get_total("USDT") == pytest.approx(10000 + sign * 0.05 * scale)
    assert bot.execute_entry(PAIR, 1000.0, trade=trade, ordertype="market", is_short=short)
    assert trade.amount == 20
    assert trade.funding_fees == pytest.approx(sign * 0.05 * scale)
    addition = trade.orders[-1]
    assert addition.funding_fee == pytest.approx(sign * 0.05 * scale)
    bot.update_trade_state(trade, addition.order_id)
    assert addition.funding_fee == pytest.approx(sign * 0.05 * scale)

    time_machine.move_to(START + timedelta(minutes=45), tick=False)
    assert bot.execute_trade_exit(
        trade, 100, ExitCheckTuple(ExitType.EXIT_SIGNAL), ordertype="market", sub_trade_amt=10
    )
    assert trade.amount == 10
    assert trade.realized_profit == pytest.approx(sign * 0.1 * scale)
    assert trade.funding_fees == pytest.approx(sign * 0.1 * scale)
    partial_fill = [
        call.args[0]
        for call in bot.rpc.send_msg.call_args_list
        if call.args[0]["type"] == RPCMessageType.EXIT_FILL and call.args[0]["sub_trade"]
    ][-1]
    assert partial_fill["profit_amount"] == pytest.approx(sign * 0.1 * scale)
    assert partial_fill["profit_ratio"] == pytest.approx(sign * 0.0001 * scale)
    if refresh:
        # Closed orders already carry signed funding of 20; the new segment is also 20.
        end = START + timedelta(minutes=105)
        time_machine.move_to(end, tick=False)
        if refresh == "unchanged":
            bot.update_funding_fees()
            assert trade.funding_fees == sign * 40
        trade.set_liquidation_price(130 if short else 70)
        position_price = 120 if short else 80
        expected = 119 if short else 81  # Existing 5% liquidation buffer around entry 100.
        with (
            patch.dict(bot.config, {"dry_run": False}),
            patch.object(bot.exchange, "exchange_has", return_value=True),
            patch.object(
                bot.exchange, "fetch_positions", return_value=[{"liquidationPrice": position_price}]
            ),
            patch("freqtrade.freqtradebot.update_liquidation_prices", update_liquidation_prices),
        ):
            assert bot._update_continuous_funding(trade, end, update_wallet=False)
        assert trade.funding_fees == sign * 40
        assert trade.liquidation_price == (
            expected if refresh == "changed" else (130 if short else 70)
        )
        return
    time_machine.move_to(START + timedelta(minutes=60), tick=False)
    bot.update_funding_fees()
    bot.wallets.update()
    assert trade.funding_fees == pytest.approx(sign * 0.125)
    assert bot.wallets.get_total("USDT") == pytest.approx(10000 + sign * 0.125)
    remaining = trade.calculate_profit(100)
    assert remaining.profit_abs == pytest.approx(sign * 0.025)
    assert remaining.total_profit == pytest.approx(sign * 0.125)
    assert trade.calc_profit_ratio(100) == pytest.approx(sign * 0.000025)
    assert trade.calc_profit_ratio(trade.calc_close_rate_for_roi(0)) == 0
    assert bot.execute_trade_exit(
        trade, 100, ExitCheckTuple(ExitType.EXIT_SIGNAL), ordertype="market"
    )
    assert not trade.is_open
    assert trade.realized_profit == pytest.approx(sign * 0.125)
    exit_fills = [
        call.args[0]
        for call in bot.rpc.send_msg.call_args_list
        if call.args[0]["type"] == RPCMessageType.EXIT_FILL
    ]
    assert exit_fills[-1]["profit_amount"] == pytest.approx(sign * 0.025)
    assert exit_fills[-1]["cumulative_profit"] == pytest.approx(sign * 0.125)
    assert sum(message["profit_amount"] for message in exit_fills) == pytest.approx(sign * 0.125)
    closing = trade.orders[-1]
    bot.update_trade_state(trade, closing.order_id)
    assert trade.realized_profit == pytest.approx(sign * 0.125)
    assert bot.wallets.get_total("USDT") == pytest.approx(10000 + sign * 0.125)


@pytest.mark.parametrize("short", [False, True])
def test_dry_partial_exits_clear_profit_when_funding_offsets(
    mocker, default_conf_usdt, time_machine, short
):
    time_machine.move_to(START, tick=False)
    bot = make_bot(mocker, default_conf_usdt)
    frame = pd.DataFrame(
        {
            "date": pd.date_range(START, periods=4, freq="h"),
            "funding_rate": [0.0001, -0.00015, 0, 0],
            "funding_rate_absolute": [0.01, -0.015, 0, 0],
        }
    )
    bot.exchange.refresh_latest_ohlcv.return_value = {(PAIR, "1h", CandleType.FUNDING_RATE): frame}
    assert bot.execute_entry(PAIR, 300, ordertype="market", is_short=short)
    trade = Trade.session.scalars(select(Trade)).one()
    sign = 1 if short else -1
    time_machine.move_to(START + timedelta(hours=1), tick=False)
    assert bot.execute_trade_exit(
        trade, 100, ExitCheckTuple(ExitType.EXIT_SIGNAL), ordertype="market", sub_trade_amt=1
    )
    assert trade.realized_profit == pytest.approx(sign * 0.03)
    time_machine.move_to(START + timedelta(hours=2), tick=False)
    assert bot.execute_trade_exit(
        trade, 100, ExitCheckTuple(ExitType.EXIT_SIGNAL), ordertype="market", sub_trade_amt=1
    )
    assert trade.amount == 1
    assert trade.funding_fees == pytest.approx(0)
    assert trade.realized_profit == pytest.approx(0)
    assert trade.calculate_profit(100).total_profit == pytest.approx(0)
    assert bot.wallets.get_total("USDT") == pytest.approx(10000, abs=1e-9, rel=0)
    partial_fill = [
        call.args[0]
        for call in bot.rpc.send_msg.call_args_list
        if call.args[0]["type"] == RPCMessageType.EXIT_FILL and call.args[0]["sub_trade"]
    ][-1]
    assert partial_fill["profit_amount"] == pytest.approx(-sign * 0.03)
    assert partial_fill["cumulative_profit"] == pytest.approx(0)


def test_dry_delayed_fill_accrues_only_after_fill(mocker, default_conf_usdt, time_machine):
    time_machine.move_to(START, tick=False)
    bot = make_bot(mocker, default_conf_usdt)
    crossed = mocker.patch.object(bot.exchange, "_dry_is_price_crossed", return_value=False)
    assert bot.execute_entry(PAIR, 1000.0, ordertype="limit")
    trade = Trade.session.scalars(select(Trade)).one()
    entry = trade.orders[-1]
    time_machine.move_to(START + timedelta(minutes=30), tick=False)
    bot.update_funding_fees()
    assert trade.funding_fees == 0
    crossed.return_value = True
    bot.update_trade_state(trade, entry.order_id)
    assert entry.order_filled_utc == START + timedelta(minutes=30)
    assert trade.funding_fees == 0
    time_machine.move_to(START + timedelta(minutes=45), tick=False)
    assert bot.execute_trade_exit(
        trade, 100, ExitCheckTuple(ExitType.EXIT_SIGNAL), ordertype="market"
    )
    assert trade.funding_fees == pytest.approx(-0.025)
    assert trade.realized_profit == pytest.approx(-0.025)


def test_dry_delayed_exit_pays_until_actual_fill(mocker, default_conf_usdt, time_machine):
    time_machine.move_to(START, tick=False)
    bot = make_bot(mocker, default_conf_usdt)
    assert bot.execute_entry(PAIR, 1000, ordertype="market")
    trade = Trade.session.scalars(select(Trade)).one()
    time_machine.move_to(START + timedelta(minutes=15), tick=False)
    crossed = mocker.patch.object(bot.exchange, "_dry_is_price_crossed", return_value=False)
    assert bot.execute_trade_exit(
        trade, 100, ExitCheckTuple(ExitType.EXIT_SIGNAL), ordertype="limit"
    )
    closing = trade.orders[-1]
    assert closing.ft_is_open
    time_machine.move_to(START + timedelta(minutes=45), tick=False)
    crossed.return_value = True
    bot.update_trade_state(trade, closing.order_id)
    assert closing.order_filled_utc == START + timedelta(minutes=45)
    assert trade.funding_fees == pytest.approx(-0.075)
    assert trade.realized_profit == pytest.approx(-0.075)


@pytest.mark.parametrize("short", [False, True])
def test_dry_funding_plus_both_order_fees(mocker, default_conf_usdt, time_machine, short):
    time_machine.move_to(START, tick=False)
    bot = make_bot(mocker, default_conf_usdt)
    mocker.patch.object(bot.exchange, "get_fee", return_value=0.001)
    assert bot.execute_entry(PAIR, 1000, ordertype="market", is_short=short)
    trade = Trade.session.scalars(select(Trade)).one()
    time_machine.move_to(START + timedelta(minutes=30), tick=False)
    assert bot.execute_trade_exit(
        trade, 100, ExitCheckTuple(ExitType.EXIT_SIGNAL), ordertype="market"
    )
    expected = -2 + (0.05 if short else -0.05)
    assert trade.realized_profit == pytest.approx(expected)
    assert bot.wallets.get_total("USDT") == pytest.approx(10000 + expected)


def test_dry_stoploss_books_funding_once(mocker, default_conf_usdt, time_machine):
    time_machine.move_to(START, tick=False)
    bot = make_bot(mocker, default_conf_usdt)
    assert bot.execute_entry(PAIR, 1000, ordertype="market")
    trade = Trade.session.scalars(select(Trade)).one()
    crossed = mocker.patch.object(bot.exchange, "_dry_is_price_crossed", return_value=False)
    raw = bot.exchange.create_dry_run_order(
        PAIR, "market", trade.exit_side, trade.amount, 100, 1, stop_loss=True, stop_price=90
    )
    order = Order.parse_from_ccxt_object(raw, PAIR, "stoploss", trade.amount, 90)
    trade.orders.append(order)
    Trade.commit()
    time_machine.move_to(START + timedelta(minutes=45), tick=False)
    crossed.return_value = True
    mocker.patch.object(bot.exchange, "get_dry_market_fill_price", return_value=90)
    bot.update_trade_state(trade, order.order_id, stoploss_order=True)
    assert not trade.is_open
    assert order.funding_fee == pytest.approx(-0.075)
    assert trade.funding_fees == pytest.approx(-0.075)
    bot.update_trade_state(trade, order.order_id, stoploss_order=True)
    assert order.funding_fee == pytest.approx(-0.075)


def test_dry_global_callback_sees_current_funding(mocker, default_conf_usdt, time_machine):
    time_machine.move_to(START, tick=False)
    bot = make_bot(mocker, default_conf_usdt)
    assert bot.execute_entry(PAIR, 1000, ordertype="market")
    trade = Trade.session.scalars(select(Trade)).one()
    time_machine.move_to(START + timedelta(minutes=30), tick=False)
    mocker.patch.object(bot.dataprovider, "refresh")
    mocker.patch.object(bot.strategy, "analyze")
    mocker.patch.object(bot, "exit_positions")
    mocker.patch.object(bot, "process_open_trade_positions")
    mocker.patch.object(bot, "enter_positions")
    observed = []

    def observe(**kwargs):
        observed.append((trade.funding_fees, bot.wallets.get_total("USDT")))

    bot.strategy.bot_loop_start = observe
    bot.process()
    assert observed == [pytest.approx((-0.05, 9999.95))]


def stub_strategy_steps(mocker, bot):
    """Run the bot loop without market data or strategy; return the exit step to count calls."""
    mocker.patch.object(bot.dataprovider, "refresh")
    mocker.patch.object(bot.strategy, "analyze")
    mocker.patch.object(bot, "process_open_trade_positions")
    mocker.patch.object(bot, "enter_positions")
    return mocker.patch.object(bot, "exit_positions")


@pytest.mark.parametrize("problem", ["not_yet_published", "unreachable"])
def test_dry_funding_not_yet_available_is_retried_without_stopping(
    mocker, default_conf_usdt, time_machine, problem
):
    """Kraken publishes each hour's rate a moment after the hour starts, and can be unreachable.

    The bot keeps trading meanwhile and catches up with the funding on a later loop.
    """
    time_machine.move_to(START, tick=False)
    bot = make_bot(mocker, default_conf_usdt)
    exit_positions = stub_strategy_steps(mocker, bot)
    assert bot.execute_entry(PAIR, 1000.0, ordertype="market")
    trade = Trade.session.scalars(select(Trade)).one()
    bot.exchange.refresh_latest_ohlcv.return_value = {FUNDING: history(2)}
    time_machine.move_to(START + timedelta(minutes=110), tick=False)
    bot.process()
    assert trade.funding_fees == pytest.approx(-0.1 * 110 / 60)

    if problem == "unreachable":
        bot.exchange.refresh_latest_ohlcv.return_value = {}
    time_machine.move_to(START + timedelta(hours=2, milliseconds=800), tick=False)
    bot.process()
    assert exit_positions.call_count == 2
    assert trade.funding_fees == pytest.approx(-0.1 * 110 / 60)

    bot.exchange.refresh_latest_ohlcv.return_value = {FUNDING: history(3)}
    time_machine.move_to(START + timedelta(hours=2, seconds=5), tick=False)
    bot.process()
    assert trade.funding_fees == pytest.approx(-0.1 * (2 + 5 / 3600))


def test_dry_exit_just_after_the_hour_is_booked_once_its_funding_is_published(
    mocker, default_conf_usdt, time_machine
):
    """The exit fills at once and keeps its fill time; booking it waits for the hour's rate."""
    time_machine.move_to(START, tick=False)
    bot = make_bot(mocker, default_conf_usdt)
    assert bot.execute_entry(PAIR, 1000.0, ordertype="market")
    trade = Trade.session.scalars(select(Trade)).one()
    bot.exchange.refresh_latest_ohlcv.return_value = {FUNDING: history(2)}
    filled = START + timedelta(hours=2, milliseconds=500)
    time_machine.move_to(filled, tick=False)
    assert bot.execute_trade_exit(
        trade, 100, ExitCheckTuple(ExitType.EXIT_SIGNAL), ordertype="market"
    )
    closing = trade.orders[-1]
    assert trade.is_open
    assert closing.ft_is_open

    time_machine.move_to(filled + timedelta(seconds=5), tick=False)
    bot.manage_open_orders()
    assert trade.is_open
    assert not bot.execute_trade_exit(
        trade, 99, ExitCheckTuple(ExitType.STOP_LOSS), ordertype="market"
    )
    bot.exchange.refresh_latest_ohlcv.return_value = {FUNDING: history(3)}
    bot.update_funding_fees()
    bot.manage_open_orders()
    assert not trade.is_open
    assert trade.orders == [trade.orders[0], closing]
    assert trade.exit_reason == ExitType.EXIT_SIGNAL.value
    assert trade.close_date_utc == filled
    expected = -0.1 * (2 + 0.5 / 3600)
    assert closing.funding_fee == pytest.approx(expected)
    assert trade.realized_profit == pytest.approx(expected)
    bot.wallets.update()
    assert bot.wallets.get_total("USDT") == pytest.approx(10000 + expected)


def test_dry_added_entry_just_after_the_hour_waits_for_its_funding(
    mocker, default_conf_usdt, time_machine
):
    time_machine.move_to(START, tick=False)
    bot = make_bot(mocker, default_conf_usdt)
    assert bot.execute_entry(PAIR, 1000.0, ordertype="market")
    trade = Trade.session.scalars(select(Trade)).one()
    bot.exchange.refresh_latest_ohlcv.return_value = {FUNDING: history(2)}
    added = START + timedelta(hours=2, milliseconds=500)
    time_machine.move_to(added, tick=False)
    assert bot.execute_entry(PAIR, 1000.0, trade=trade, ordertype="market")
    addition = trade.orders[-1]
    assert addition.ft_is_open

    bot.exchange.refresh_latest_ohlcv.return_value = {FUNDING: history(3)}
    time_machine.move_to(added + timedelta(seconds=5), tick=False)
    bot.manage_open_orders()
    assert not addition.ft_is_open
    assert trade.amount == 20
    assert addition.funding_fee == pytest.approx(-0.1 * (2 + 0.5 / 3600))
    assert trade.funding_fees == pytest.approx(-0.1 * (2 + 0.5 / 3600) - 0.2 * 5 / 3600)


@pytest.mark.parametrize("ordertype", ["market", "limit"])
def test_dry_fill_waiting_for_funding_survives_a_restart(
    mocker, default_conf_usdt, time_machine, ordertype
):
    """A restart forgets dry-run fills, so a waiting fill is saved with its time, price and fee."""
    time_machine.move_to(START, tick=False)
    bot = make_bot(mocker, default_conf_usdt)
    mocker.patch.object(
        bot.exchange,
        "get_fee",
        side_effect=lambda *args, taker_or_maker="maker", **kwargs: (
            0.0002 if taker_or_maker == "maker" else 0.0005
        ),
    )
    bot.exchange.refresh_latest_ohlcv.return_value = {FUNDING: history(2)}
    crossed = mocker.patch.object(bot.exchange, "_dry_is_price_crossed", return_value=False)
    assert bot.execute_entry(PAIR, 1000.0, ordertype="limit")
    trade = Trade.session.scalars(select(Trade)).one()
    time_machine.move_to(START + timedelta(minutes=1), tick=False)
    crossed.return_value = True
    bot.manage_open_orders()  # the entry fills at the maker fee
    filled = START + timedelta(hours=2, milliseconds=500)
    if ordertype == "limit":
        crossed.return_value = False
        time_machine.move_to(START + timedelta(minutes=110), tick=False)
        assert bot.execute_trade_exit(
            trade, 101, ExitCheckTuple(ExitType.EXIT_SIGNAL), ordertype="limit"
        )
        time_machine.move_to(filled, tick=False)
        crossed.return_value = True
        bot.manage_open_orders()
    else:
        time_machine.move_to(filled, tick=False)
        assert bot.execute_trade_exit(
            trade, 100, ExitCheckTuple(ExitType.EXIT_SIGNAL), ordertype="market"
        )
    closing = trade.orders[-1]
    assert closing.ft_is_open
    assert closing.status == "closed"
    assert not bot.execute_trade_exit(
        trade, 99, ExitCheckTuple(ExitType.STOP_LOSS), ordertype="market"
    )

    bot.exchange._dry_run_open_orders.clear()  # restart, after the price moved back
    Trade.session.expire_all()
    crossed.return_value = False
    bot.exchange.refresh_latest_ohlcv.return_value = {FUNDING: history(3)}
    time_machine.move_to(filled + timedelta(seconds=30), tick=False)
    bot.update_funding_fees()
    bot.manage_open_orders()
    assert not trade.is_open
    assert len(trade.orders) == 2
    assert trade.exit_reason == ExitType.EXIT_SIGNAL.value
    assert trade.close_date_utc == filled
    funding = -0.1 * (1 + 59 / 60 + 0.5 / 3600)
    assert trade.funding_fees == pytest.approx(funding)
    # Entry fee 0.2 at maker; exit fee 0.202 at maker (limit, at 101) or 0.5 at taker (market).
    expected = 10 - 0.2 - 0.202 + funding if ordertype == "limit" else -0.2 - 0.5 + funding
    assert trade.realized_profit == pytest.approx(expected)


def test_dry_stop_on_exchange_waiting_for_funding_sends_one_fill_message(
    mocker, default_conf_usdt, time_machine
):
    time_machine.move_to(START, tick=False)
    bot = make_bot(mocker, default_conf_usdt)
    assert bot.execute_entry(PAIR, 1000, ordertype="market")
    trade = Trade.session.scalars(select(Trade)).one()
    crossed = mocker.patch.object(bot.exchange, "_dry_is_price_crossed", return_value=False)
    raw = bot.exchange.create_dry_run_order(
        PAIR, "market", trade.exit_side, trade.amount, 100, 1, stop_loss=True, stop_price=90
    )
    stop = Order.parse_from_ccxt_object(raw, PAIR, "stoploss", trade.amount, 90)
    trade.orders.append(stop)
    Trade.commit()
    bot.exchange.refresh_latest_ohlcv.return_value = {FUNDING: history(2)}
    time_machine.move_to(START + timedelta(hours=2, milliseconds=500), tick=False)
    crossed.return_value = True
    mocker.patch.object(bot.exchange, "get_dry_market_fill_price", return_value=90)

    def fill_messages():
        return [
            call.args[0]
            for call in bot.rpc.send_msg.call_args_list
            if call.args[0]["type"] == RPCMessageType.EXIT_FILL
        ]

    assert bot.handle_stoploss_on_exchange(trade)
    assert trade.is_open
    assert fill_messages() == []
    bot.exchange.refresh_latest_ohlcv.return_value = {FUNDING: history(3)}
    time_machine.move_to(START + timedelta(hours=2, seconds=5), tick=False)
    assert bot.handle_stoploss_on_exchange(trade)
    assert not trade.is_open
    assert stop.funding_fee == pytest.approx(-0.1 * (2 + 0.5 / 3600))
    assert len(fill_messages()) == 1
