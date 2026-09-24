"""Offline simulated orders exercise Kraken funding through native bot and wallets."""

from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import select

from freqtrade.enums import CandleType, ExitCheckTuple, ExitType, RPCMessageType
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import Order, Trade
from freqtrade.util import dt_utc
from tests.conftest import EXMS, patch_exchange, patch_freqtradebot


START = dt_utc(2024, 1, 1, 12)
PAIR = "ETH/USDT"


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
    df = pd.DataFrame(
        {
            "date": pd.date_range(START, periods=4, freq="h"),
            "funding_rate": [0.0001] * 4,
            "funding_rate_absolute": [0.01] * 4,
        }
    )
    mocker.patch.object(
        bot.exchange,
        "refresh_latest_ohlcv",
        return_value={
            (PAIR, "1h", CandleType.FUNDING_RATE): df,
        },
    )
    return bot


@pytest.mark.parametrize("short", [False, True])
def test_dry_entry_add_reduce_close_and_repeated_poll(
    mocker, default_conf_usdt, time_machine, short
):
    time_machine.move_to(START, tick=False)
    bot = make_bot(mocker, default_conf_usdt)
    assert bot.execute_entry(PAIR, 1000.0, ordertype="market", is_short=short)
    trade = Trade.session.scalars(select(Trade)).one()
    assert trade.amount == 10
    assert trade.funding_fees == 0
    sign = 1 if short else -1

    time_machine.move_to(START + timedelta(minutes=30), tick=False)
    bot.update_funding_fees()
    bot.wallets.update()
    assert trade.funding_fees == pytest.approx(sign * 0.05)
    assert bot.wallets.get_total("USDT") == pytest.approx(10000 + sign * 0.05)
    assert bot.execute_entry(PAIR, 1000.0, trade=trade, ordertype="market", is_short=short)
    assert trade.amount == 20
    assert trade.funding_fees == pytest.approx(sign * 0.05)
    addition = trade.orders[-1]
    assert addition.funding_fee == pytest.approx(sign * 0.05)
    bot.update_trade_state(trade, addition.order_id)
    assert addition.funding_fee == pytest.approx(sign * 0.05)

    time_machine.move_to(START + timedelta(minutes=45), tick=False)
    assert bot.execute_trade_exit(
        trade, 100, ExitCheckTuple(ExitType.EXIT_SIGNAL), ordertype="market", sub_trade_amt=10
    )
    assert trade.amount == 10
    assert trade.realized_profit == pytest.approx(sign * 0.1)
    assert trade.funding_fees == pytest.approx(sign * 0.1)
    partial_fill = [
        call.args[0]
        for call in bot.rpc.send_msg.call_args_list
        if call.args[0]["type"] == RPCMessageType.EXIT_FILL and call.args[0]["sub_trade"]
    ][-1]
    assert partial_fill["profit_amount"] == pytest.approx(sign * 0.1)
    assert partial_fill["profit_ratio"] == pytest.approx(sign * 0.0001)
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
