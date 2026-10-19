# pragma pylint: disable=missing-docstring, W0212, line-too-long, C0103, unused-argument

import random
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import ANY, MagicMock, PropertyMock

import numpy as np
import pandas as pd
import pytest

from freqtrade import constants
from freqtrade.commands.optimize_commands import setup_optimize_configuration, start_backtesting
from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.btanalysis import BT_DATA_COLUMNS, evaluate_result_multi
from freqtrade.data.converter import clean_ohlcv_dataframe, ohlcv_fill_up_missing_data
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history import get_timerange
from freqtrade.enums import CandleType, ExitType, RunMode
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.exchange import timeframe_to_next_date, timeframe_to_prev_date
from freqtrade.exchange.exchange_utils import DECIMAL_PLACES, TICK_SIZE
from freqtrade.optimize.backtest_caching import get_backtest_metadata_filename, get_strategy_run_id
from freqtrade.optimize.backtesting import DATE_IDX, HEADERS, Backtesting
from freqtrade.persistence import LocalTrade, Order, Trade
from freqtrade.resolvers import StrategyResolver
from freqtrade.util import dt_now, dt_utc
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    EXMS,
    generate_test_data,
    generate_test_data_raw,
    get_args,
    log_has,
    log_has_re,
    patch_exchange,
    patched_configuration_load_config_file,
)


ORDER_TYPES = [
    {"entry": "limit", "exit": "limit", "stoploss": "limit", "stoploss_on_exchange": False},
    {"entry": "limit", "exit": "limit", "stoploss": "limit", "stoploss_on_exchange": True},
]


def trim_dictlist(dict_list, num):
    new = {}
    for pair, pair_data in dict_list.items():
        new[pair] = pair_data[num:].reset_index()
    return new


def _isolated_funding_backtest(mocker, default_conf_usdt, free_cash=100, fee=0):
    pair = "COIN/USDT:USDT"
    config = default_conf_usdt
    config.update(
        runmode=RunMode.BACKTEST,
        trading_mode="futures",
        margin_mode="isolated",
        candle_type_def=CandleType.FUTURES,
        timeframe="5m",
        timeframe_detail="1m",
        startup_candle_count=0,
        stake_amount=1000,
        dry_run_wallet=1000 + free_cash,
        tradable_balance_ratio=1,
        max_open_trades=1,
        minimal_roi={"0": 100},
        stoploss=-0.99,
        liquidation_buffer=0,
        fee=fee,
    )
    config["exchange"] = {
        "name": "binance",
        "enable_ws": False,
        "pair_whitelist": [pair],
        "pair_blacklist": [],
        "_ft_has_params": {"funding_fee_continuous": True},
    }
    market = {
        "id": "COINUSDT",
        "symbol": pair,
        "base": "COIN",
        "quote": "USDT",
        "settle": "USDT",
        "active": True,
        "swap": True,
        "linear": True,
        "inverse": False,
        "type": "swap",
        "contractSize": 1,
        "precision": {"amount": 8, "price": 8},
    }
    patch_exchange(mocker, exchange="binance", mock_markets={pair: market})
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=1)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_max_leverage", return_value=1)
    mocker.patch(f"{EXMS}.get_maintenance_ratio_and_amt", return_value=(0.01, 0))
    backtesting = Backtesting(config)
    strategy = backtesting.strategylist[0]
    strategy.can_short = True
    backtesting._set_strategy(strategy)
    return pair, backtesting


@pytest.mark.parametrize("is_short", [False, True], ids=["long", "short"])
@pytest.mark.parametrize(
    "free_cash,funding,fee,exit_minute,entry_minute,price_path",
    [
        pytest.param(100, 200, 0, None, -5, None, id="credit"),
        pytest.param(500, -200, 0, None, -5, None, id="cash-funded-debit"),
        pytest.param(100, -200, 0, None, -5, None, id="margin-funded-debit"),
        pytest.param(100, -1095, 0, None, -5, None, id="maintenance-breach"),
        pytest.param(100, -200, 0.001, 0, -5, None, id="fee-full-exit-at-payment"),
        pytest.param(100, -200, 0.001, 1, -5, None, id="fee-full-exit-after-payment"),
        pytest.param(100, -200, 0.001, None, 0, None, id="entry-at-payment"),
        pytest.param(100, -200, 0.001, None, 1, None, id="entry-after-payment"),
        pytest.param(1, -1500, 0.001, None, -5, "credit", id="negative-wallet-credit"),
        pytest.param(1, -1500, 0.001, None, -5, "retrace", id="negative-wallet-liquidation"),
        pytest.param(1, -1500, 0.001, None, -5, "terminal-retrace", id="terminal-liquidation"),
    ],
)
def test_isolated_funding_updates_collateral_at_settlement(
    mocker,
    default_conf_usdt,
    is_short,
    free_cash,
    funding,
    fee,
    exit_minute,
    entry_minute,
    price_path,
):
    """Funding uses available cash before isolated margin, without inventing collateral."""
    from decimal import Decimal

    pair, backtesting = _isolated_funding_backtest(mocker, default_conf_usdt, free_cash, fee)
    strategy = backtesting.strategy
    fees = mocker.spy(backtesting.exchange, "calculate_funding_fees")
    settlement = pd.Timestamp("2024-01-01 08:00", tz="UTC")
    credit_at = settlement + timedelta(hours=8)
    observed_at = settlement + timedelta(minutes=max(0, entry_minute))
    favorable_price = (25.0 if is_short else 300.0) if price_path else 100.0
    valuation_price = favorable_price if price_path else 110.0
    side = "short" if is_short else "long"
    strategy.populate_entry_trend = lambda frame, metadata: frame.assign(
        **{
            f"enter_{side}": (
                frame.date == settlement + timedelta(minutes=min(0, entry_minute) - 5)
            ).astype(int)
        }
    )
    strategy.populate_exit_trend = lambda frame, metadata: frame
    strategy.custom_entry_price = lambda **kwargs: 100.0
    observed = {}
    credited = {}
    observations = {observed_at: observed, credit_at: credited}

    def observe(current_time, **kwargs):
        target = observations.get(current_time)
        if target is not None and LocalTrade.bt_trades_open and not target:
            trade = LocalTrade.bt_trades_open[0]
            if not trade.amount:
                return
            target.update(
                funding=trade.funding_fees,
                free=backtesting.wallets.get_free("USDT"),
                used=backtesting.wallets.get_used("USDT"),
                available=backtesting.wallets.get_available_stake_amount(),
                liquidation=trade.liquidation_price,
                collateral=backtesting.wallets.get_all_positions()[pair].collateral,
                position_value=backtesting.wallets.get_all_positions()[pair].estimate_value(
                    valuation_price
                ),
                total=backtesting.wallets.get_total("USDT"),
            )

    strategy.bot_loop_start = observe

    def custom_exit(current_time, **kwargs):
        observe(current_time)
        if exit_minute is not None and current_time == settlement + timedelta(minutes=exit_minute):
            return "scheduled_exit"

    strategy.custom_exit = custom_exit
    dates = pd.date_range(
        settlement - timedelta(minutes=15),
        (
            settlement + timedelta(minutes=5)
            if price_path == "terminal-retrace"
            else (credit_at if price_path else settlement) + timedelta(minutes=10)
        ),
        freq="1min",
    )
    candles = pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000.0,
        }
    )
    candles.loc[
        (entry_minute == 1) & (candles.date == settlement), ["open", "high", "low", "close"]
    ] = 99.0 if is_short else 101.0
    if price_path:
        candles.loc[candles.date >= settlement, ["open", "high", "low", "close"]] = favorable_price
        if price_path in ("retrace", "terminal-retrace"):
            retrace_at = settlement + timedelta(minutes=5)
            retrace_price = 75.0 if is_short else 125.0
            candles.loc[candles.date >= retrace_at, ["open", "high", "low", "close"]] = (
                retrace_price
            )
            candles.loc[candles.date == retrace_at, ["open", "low" if is_short else "high"]] = (
                favorable_price
            )
    backtesting.detail_data[pair] = candles.copy()
    funding_dates = [settlement, credit_at] if price_path else [settlement]
    payments = [funding, 2000] if price_path else [funding]
    backtesting.futures_data[pair] = pd.DataFrame(
        {
            "date": funding_dates,
            "open_mark": favorable_price,
            "open_fund": [p / (10 * favorable_price) * (1 if is_short else -1) for p in payments],
        }
    )
    result = backtesting.backtest(
        processed={pair: candles.iloc[::5].copy()},
        start_date=dates[0],
        end_date=dates[-1],
    )
    funding = funding if entry_minute <= 0 else 0
    entry_fee = Decimal(1000) * Decimal(str(fee))
    remaining_cash = Decimal(free_cash + funding) - entry_fee
    collateral = Decimal(1000) + min(Decimal(0), remaining_cash)
    expected_price = (
        (Decimal(1000) + collateral) / Decimal("10.1")
        if is_short
        else (Decimal(1000) - collateral) / Decimal("9.9")
    )
    assert observed["funding"] == pytest.approx(funding)
    assert observed["free"] == pytest.approx(float(max(Decimal(0), remaining_cash)))
    assert observed["available"] == pytest.approx(observed["free"])
    assert observed["used"] == pytest.approx(float(collateral))
    assert observed["liquidation"] == pytest.approx(float(expected_price))
    assert observed["collateral"] == pytest.approx(float(collateral))
    assert observed["position_value"] == pytest.approx(
        float(collateral) + (valuation_price - 100) * 10 * (-1 if is_short else 1)
    )
    assert observed["total"] == pytest.approx(1000 + free_cash + funding - float(entry_fee))
    trade = LocalTrade.bt_trades[0]
    assert trade.orders[0].order_filled_utc == settlement + timedelta(minutes=entry_minute)
    close_price = Decimal(100)
    if price_path in ("retrace", "terminal-retrace"):
        assert trade.exit_reason == "liquidation"
        assert trade.close_date_utc == settlement + timedelta(minutes=5)
        close_price = expected_price
        assert trade.close_rate == pytest.approx(float(close_price))
    elif price_path == "credit":
        # Two known settlements in an eight-hour hold must not trigger minute-by-minute sums.
        assert fees.call_count < 20
        assert credited["funding"] == pytest.approx(funding + 2000)
        assert credited["free"] == pytest.approx(2000)
        assert credited["available"] == pytest.approx(2000)
        assert credited["used"] == pytest.approx(float(collateral))
        assert credited["collateral"] == pytest.approx(float(collateral))
        assert credited["liquidation"] == pytest.approx(float(expected_price))
        assert credited["total"] == pytest.approx(2000 + float(collateral))
        funding += 2000
        close_price = Decimal(str(favorable_price))
    elif collateral < Decimal(10):
        assert trade.exit_reason == "liquidation"
        assert trade.close_date_utc == settlement
    elif exit_minute is not None:
        assert trade.close_date_utc == settlement + timedelta(minutes=exit_minute)
    expected_pnl = (close_price - 100) * 10 * (-1 if is_short else 1)
    close_fee = close_price * 10 * Decimal(str(fee))
    assert result["results"].iloc[0]["funding_fees"] == pytest.approx(funding)
    assert result["final_balance"] == pytest.approx(
        float(Decimal(1000 + free_cash + funding) + expected_pnl - entry_fee - close_fee)
    )


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_discrete_funding_idle_checks_keep_native_boundaries(mocker, default_conf_usdt, unit):
    pair, backtesting = _isolated_funding_backtest(mocker, default_conf_usdt)
    backtesting.reset_backtest()
    start = pd.Timestamp("2024-01-01 08:00", tz="UTC")
    if unit == "ns":
        start += pd.Timedelta(1, "ns")
    dates = pd.DatetimeIndex(
        [start, start + timedelta(hours=1), start + timedelta(hours=2)]
    ).as_unit(unit)
    frame = pd.DataFrame({"date": dates, "open_mark": 100.0, "open_fund": [0.01, 0.0, -0.02]})
    backtesting.futures_data[pair] = frame
    opened = start.to_pydatetime(warn=False) - timedelta(minutes=5)
    trade = LocalTrade(
        pair=pair,
        stake_amount=1000,
        amount=10,
        open_rate=100,
        open_date=opened,
        fee_open=0,
        fee_close=0,
        is_open=True,
        is_short=False,
        leverage=1,
        trading_mode="futures",
    )
    entry = Order(
        order_id="entry",
        ft_pair=pair,
        ft_order_side="buy",
        ft_amount=10,
        ft_price=100,
        amount=10,
        filled=10,
        price=100,
        status="closed",
        ft_is_open=False,
        order_date=opened,
        order_filled_date=opened,
        funding_fee=0,
    )
    trade.orders = [entry]
    LocalTrade.add_bt_trade(trade)
    backtesting.wallets.update()
    backtesting._run_funding_fees(trade, start - timedelta(microseconds=1))
    assert trade.funding_fees == 0
    backtesting._run_funding_fees(trade, start, defer_wallet=True)
    pending = trade._funding_fee_payments.copy()
    backtesting._run_funding_fees(trade, start + timedelta(microseconds=1), defer_wallet=True)
    assert trade._funding_fee_payments == pending
    assert backtesting.wallets.get_free("USDT") == 100
    backtesting._run_funding_fees(trade, start + timedelta(microseconds=2))
    assert backtesting.wallets.get_free("USDT") == 90
    assert trade.funding_fees == -10
    backtesting._run_funding_fees(trade, start + timedelta(minutes=1))
    backtesting._run_funding_fees(trade, dates[1])
    assert trade.funding_fee_last_time_utc == dates[1].to_pydatetime(warn=False)
    assert trade.funding_fee_processed == -10
    assert backtesting.wallets.get_free("USDT") == 90
    # A changed past source row must still reach the native allocation refusal.
    frame.loc[0, "open_fund"] = 0.02
    with pytest.raises(OperationalException, match="individual settlements"):
        backtesting._update_continuous_funding(dates[1] + timedelta(minutes=1))
    frame.loc[0, "open_fund"] = 0.01
    backtesting._update_continuous_funding(dates[1] + timedelta(minutes=1))
    revised = frame.copy()
    revised.loc[2, "open_fund"] = -0.03
    backtesting.futures_data[pair] = revised
    backtesting._run_funding_fees(trade, dates[1] + timedelta(minutes=2))
    assert trade.funding_fees == -10
    backtesting._run_funding_fees(trade, dates[2])
    assert trade.funding_fees == 20
    assert backtesting.wallets.get_free("USDT") == 120
    # Native full-fill processing carries prior funding into the filled order.
    exit_time = dates[2].to_pydatetime() + timedelta(minutes=1)
    exit_order = Order(
        order_id="exit",
        ft_pair=pair,
        ft_order_side="sell",
        ft_amount=10,
        ft_price=100,
        amount=10,
        filled=10,
        price=100,
        status="closed",
        ft_is_open=False,
        order_date=exit_time,
        order_filled_date=exit_time,
        funding_fee=20,
    )
    trade.orders.append(exit_order)
    trade.amount = 0
    backtesting._run_funding_fees(trade, exit_time, force=True)
    assert trade.funding_fee_running == 0
    assert trade.funding_fees == 20
    # Empty native input still follows the original no-settlement path.
    backtesting.futures_data[pair] = pd.DataFrame()
    backtesting._run_funding_fees(trade, exit_time + timedelta(minutes=1))
    assert trade.funding_fees == 20


def load_data_test(what, testdatadir):
    timerange = TimeRange.parse_timerange("1510694220-1510700340")
    data = history.load_pair_history(
        pair="UNITTEST/BTC",
        datadir=testdatadir,
        timeframe="1m",
        timerange=timerange,
        drop_incomplete=False,
        fill_up_missing=False,
    )

    base = 0.001
    if what == "raise":
        data.loc[:, "open"] = data.index * base
        data.loc[:, "high"] = data.index * base + 0.0001
        data.loc[:, "low"] = data.index * base - 0.0001
        data.loc[:, "close"] = data.index * base

    if what == "lower":
        data.loc[:, "open"] = 1 - data.index * base
        data.loc[:, "high"] = 1 - data.index * base + 0.0001
        data.loc[:, "low"] = 1 - data.index * base - 0.0001
        data.loc[:, "close"] = 1 - data.index * base

    if what == "sine":
        hz = 0.1  # frequency
        data.loc[:, "open"] = np.sin(data.index * hz) / 1000 + base
        data.loc[:, "high"] = np.sin(data.index * hz) / 1000 + base + 0.0001
        data.loc[:, "low"] = np.sin(data.index * hz) / 1000 + base - 0.0001
        data.loc[:, "close"] = np.sin(data.index * hz) / 1000 + base

    # Prices above are invented; give the fixture finite, ample base volume too.
    data.loc[:, "volume"] = 1000000.0
    return {
        "UNITTEST/BTC": clean_ohlcv_dataframe(
            data, timeframe="1m", pair="UNITTEST/BTC", fill_missing=True, drop_incomplete=True
        )
    }


# FIX: fixturize this?
def _make_backtest_conf(mocker, datadir, conf=None, pair="UNITTEST/BTC"):
    data = history.load_data(datadir=datadir, timeframe="1m", pairs=[pair])
    data = trim_dictlist(data, -201)
    # These alternating-order tests assume sufficient synthetic base quantity.
    data[pair]["volume"] = 1000.0
    patch_exchange(mocker)
    backtesting = Backtesting(conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)
    return {
        "processed": processed,
        "start_date": min_date,
        "end_date": max_date,
    }


def _trend(signals, buy_value, sell_value):
    n = len(signals["low"])
    buy = np.zeros(n)
    sell = np.zeros(n)
    for i in range(len(signals["date"])):
        if random.random() > 0.5:  # Both buy and sell signals at same timeframe
            buy[i] = buy_value
            sell[i] = sell_value
    signals["enter_long"] = buy
    signals["exit_long"] = sell
    signals["enter_short"] = 0
    signals["exit_short"] = 0
    return signals


def _trend_alternate(dataframe=None, metadata=None):
    signals = dataframe
    low = signals["low"]
    n = len(low)
    buy = np.zeros(n)
    sell = np.zeros(n)
    for i in range(len(buy)):
        if i % 2 == 0:
            buy[i] = 1
        else:
            sell[i] = 1
    signals["enter_long"] = buy
    signals["exit_long"] = sell
    signals["enter_short"] = 0
    signals["exit_short"] = 0
    return dataframe


def _volume_backtest(mocker, default_conf_usdt, case, start):
    pair = "COIN/USDT"
    config = default_conf_usdt
    config.update(
        runmode=RunMode.BACKTEST,
        timeframe="1h",
        timeframe_detail="1m",
        startup_candle_count=0,
        stake_amount=1000,
        dry_run_wallet=2000,
        max_open_trades=1,
        minimal_roi={},
        stoploss=-0.99,
        fee=0.01,
        unfilledtimeout={"entry": 1 if case == "timeout" else 10, "exit": 10, "unit": "minutes"},
    )
    config["exchange"].update(pair_whitelist=[pair], pair_blacklist=[])
    if case == "quote-volume":
        config["exchange"]["_ft_has_params"] = {"ohlcv_volume_currency": "quote"}
    market = {
        "id": "COINUSDT",
        "symbol": pair,
        "base": "COIN",
        "quote": "USDT",
        "active": True,
        "spot": True,
        "type": "spot",
        "swap": False,
        "contract": False,
        "precision": {"amount": 3, "price": 2},
    }
    patch_exchange(mocker, mock_markets={pair: market})
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=1)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    backtesting = Backtesting(config)
    strategy = backtesting.strategylist[0]
    backtesting._set_strategy(strategy)
    terminal = case.startswith("terminal") or case == "shared-capacity"
    strategy.populate_entry_trend = lambda frame, metadata: frame.assign(
        enter_long=(frame.date == start).astype(int), enter_short=0
    )
    strategy.populate_exit_trend = lambda frame, metadata: frame.assign(
        exit_long=((frame.date == start + timedelta(hours=1)) & (not terminal)).astype(int),
        exit_short=0,
    )
    if case == "shared-capacity":
        strategy.minimal_roi = {0: -1.0}
    hourly = pd.DataFrame(
        {
            "date": pd.date_range(start, periods=5, freq="h"),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000.0,
        }
    )
    detail = pd.DataFrame(
        {
            "date": pd.date_range(start, periods=300, freq="min"),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000.0,
        }
    )
    for hour in (1, 2):
        mask = detail.date.between(
            start + timedelta(hours=hour), start + timedelta(hours=hour, minutes=3)
        )
        if case in ("invalid-volume", "timeout"):
            detail.loc[mask, "volume"] = [0, np.nan, -1, np.inf]
        elif case == "oversized":
            detail.loc[mask, "volume"] = 1
        elif case == "missing-minutes":
            detail = detail.loc[~mask].copy()
    if case == "shared-capacity":
        detail.loc[detail.date == start + timedelta(hours=1), "volume"] = 15
    if case.startswith("terminal"):
        detail.loc[
            detail.date.between(
                start + timedelta(hours=4),
                start + timedelta(hours=4, minutes=59 if case == "terminal-unfillable" else 3),
            ),
            "volume",
        ] = 1
    backtesting.detail_data[pair] = detail
    fills = []
    strategy.order_filled = lambda trade, order, current_time, **kwargs: fills.append(
        (
            current_time,
            order.ft_order_side,
            order.safe_filled,
            order.safe_price,
            order.order_date_utc,
        )
    )
    return pair, backtesting, hourly, detail, fills


@pytest.mark.parametrize(
    "case",
    [
        "invalid-volume",
        "oversized",
        "missing-minutes",
        "shared-capacity",
        "timeout",
        "terminal-deferred",
        "terminal-unfillable",
        "quote-volume",
    ],
)
def test_native_fills_use_observed_candle_volume(mocker, default_conf_usdt, case):
    """Real native order lifetimes and money accounting obey observed minute capacity."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    if case == "quote-volume":
        with pytest.raises(OperationalException, match="base-asset OHLCV volume"):
            _volume_backtest(mocker, default_conf_usdt, case, start)
        return
    pair, backtesting, hourly, detail, fills = _volume_backtest(
        mocker, default_conf_usdt, case, start
    )
    try:
        if case == "terminal-unfillable":
            with pytest.raises(OperationalException, match="available volume"):
                backtesting.backtest({pair: hourly}, start, start + timedelta(hours=4))
            assert len(fills) == 1
            assert [
                trade.amount for trade in LocalTrade.bt_trades_open if trade.has_open_position
            ] == [10]
            return
        result = backtesting.backtest({pair: hourly}, start, start + timedelta(hours=4))
        expected = [start + timedelta(hours=1, minutes=4), start + timedelta(hours=2, minutes=4)]
        if case == "shared-capacity":
            expected = [start + timedelta(hours=1), start + timedelta(hours=1, minutes=1)]
        elif case == "terminal-deferred":
            expected = [start + timedelta(hours=1), start + timedelta(hours=4, minutes=4)]
        assert [fill[0] for fill in fills] == expected
        assert [fill[1] for fill in fills] == ["buy", "sell"]
        assert all(fill[2:4] == (10, 100) for fill in fills)
        used = defaultdict(float)
        volume = detail.set_index("date").volume
        for timestamp, _, amount, _, _ in fills:
            used[timestamp] += amount
            assert (
                np.isfinite(volume.loc[timestamp]) and 0 < used[timestamp] <= volume.loc[timestamp]
            )
        assert result["final_balance"] == pytest.approx(1980, abs=1e-9, rel=0)
        if case == "timeout":
            assert result["timedout_entry_orders"] == 4
            assert fills[0][-1] == start + timedelta(hours=1, minutes=4)
        elif case in ("invalid-volume", "oversized"):
            assert fills[0][-1] == start + timedelta(hours=1)
    finally:
        backtesting.cleanup()


# Unit tests
def test_setup_optimize_configuration_without_arguments(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--export",
        "none",
    ]

    config = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert "max_open_trades" in config
    assert "stake_currency" in config
    assert "stake_amount" in config
    assert "exchange" in config
    assert "pair_whitelist" in config["exchange"]
    assert "datadir" in config
    assert log_has("Using data directory: {} ...".format(config["datadir"]), caplog)
    assert "timeframe" in config
    assert not log_has_re("Parameter -i/--ticker-interval detected .*", caplog)

    assert "position_stacking" not in config
    assert not log_has("Parameter --enable-position-stacking detected ...", caplog)

    assert "timerange" not in config
    assert "export" in config
    assert config["export"] == "none"
    assert "runmode" in config
    assert config["runmode"] == RunMode.BACKTEST


def test_setup_bt_configuration_with_arguments(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("freqtrade.configuration.configuration.create_datadir", lambda c, x: x)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--datadir",
        "/foo/bar",
        "--timeframe",
        "1m",
        "--enable-position-stacking",
        "--timerange",
        ":100",
        "--export-filename",
        "foo_bar.json",
        "--fee",
        "0",
    ]

    config = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert "max_open_trades" in config
    assert "stake_currency" in config
    assert "stake_amount" in config
    assert "exchange" in config
    assert "pair_whitelist" in config["exchange"]
    assert "datadir" in config
    assert config["runmode"] == RunMode.BACKTEST

    assert log_has("Using data directory: {} ...".format(config["datadir"]), caplog)
    assert "timeframe" in config
    assert log_has("Parameter -i/--timeframe detected ... Using timeframe: 1m ...", caplog)

    assert "position_stacking" in config
    assert log_has("Parameter --enable-position-stacking detected ...", caplog)

    assert "timerange" in config
    assert log_has("Parameter --timerange detected: {} ...".format(config["timerange"]), caplog)

    assert "export" in config
    assert "exportfilename" in config
    assert isinstance(config["exportfilename"], Path)
    assert log_has("Storing backtest results to {} ...".format(config["exportfilename"]), caplog)
    assert log_has_re(
        "DEPRECATED: Using `--export-filename` has no impact when backtesting.*", caplog
    )

    assert "fee" in config
    assert log_has("Parameter --fee detected, setting fee to: {} ...".format(config["fee"]), caplog)


def test_setup_optimize_configuration_stake_amount(mocker, default_conf, caplog) -> None:
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--stake-amount",
        "1",
        "--starting-balance",
        "2",
    ]

    conf = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert isinstance(conf, dict)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--stake-amount",
        "1",
        "--starting-balance",
        "0.5",
    ]
    with pytest.raises(OperationalException, match=r"Starting balance .* smaller .*"):
        setup_optimize_configuration(get_args(args), RunMode.BACKTEST)


def test_start(mocker, fee, default_conf, caplog) -> None:
    start_mock = MagicMock()
    mocker.patch(f"{EXMS}.get_fee", fee)
    patch_exchange(mocker)
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.start", start_mock)
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
    ]
    pargs = get_args(args)
    start_backtesting(pargs)
    assert log_has("Starting freqtrade in Backtesting mode", caplog)
    assert start_mock.call_count == 1


@pytest.mark.parametrize("order_types", ORDER_TYPES)
def test_backtesting_init(mocker, default_conf, order_types) -> None:
    """
    Check that stoploss_on_exchange is set to False while backtesting
    since backtesting assumes a perfect stoploss anyway.
    """
    default_conf["order_types"] = order_types
    patch_exchange(mocker)
    get_fee = mocker.patch(f"{EXMS}.get_fee", MagicMock(return_value=0.5))
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.config == default_conf
    assert backtesting.timeframe == "5m"
    assert callable(backtesting.strategy.advise_all_indicators)
    assert callable(backtesting.strategy.advise_entry)
    assert callable(backtesting.strategy.advise_exit)
    assert isinstance(backtesting.strategy.dp, DataProvider)
    get_fee.assert_called()
    assert backtesting.fee == 0.5
    assert not backtesting.strategy.order_types["stoploss_on_exchange"]
    assert backtesting.strategy.bot_started is True


def test_backtesting_init_no_timeframe(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    del default_conf["timeframe"]
    default_conf["strategy_list"] = [CURRENT_TEST_STRATEGY, "HyperoptableStrategy"]

    mocker.patch(f"{EXMS}.get_fee", MagicMock(return_value=0.5))
    with pytest.raises(
        OperationalException, match=r"Timeframe needs to be set in either configuration"
    ):
        Backtesting(default_conf)


def test_data_with_fee(default_conf, mocker) -> None:
    patch_exchange(mocker)
    default_conf["fee"] = 0.01234

    fee_mock = mocker.patch(f"{EXMS}.get_fee", MagicMock(return_value=0.5))
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.fee == 0.01234
    assert fee_mock.call_count == 0

    default_conf["fee"] = 0.0
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.fee == 0.0
    assert fee_mock.call_count == 0


def test_data_to_dataframe_bt(default_conf, mocker, testdatadir) -> None:
    patch_exchange(mocker)
    timerange = TimeRange.parse_timerange("1510694220-1510700340")
    data = history.load_data(
        testdatadir, "1m", ["UNITTEST/BTC"], timerange=timerange, fill_up_missing=True
    )
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    processed = backtesting.strategy.advise_all_indicators(data)
    assert len(processed["UNITTEST/BTC"]) == 103

    # Load strategy to compare the result between Backtesting function and strategy are the same
    strategy = StrategyResolver.load_strategy(default_conf)

    processed2 = strategy.advise_all_indicators(data)
    assert processed["UNITTEST/BTC"].equals(processed2["UNITTEST/BTC"])


def test_get_pair_precision_bt(default_conf, mocker) -> None:
    patch_exchange(mocker)
    default_conf["timeframe"] = "30m"
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = "UNITTEST/BTC"
    backtesting.pairlists._whitelist = [pair]
    ex_mock = mocker.patch(f"{EXMS}.get_precision_price", return_value=1e-5)
    data, _timerange = backtesting.load_bt_data()
    assert data

    assert backtesting.get_pair_precision(pair, dt_utc(2018, 1, 1)) == (1e-8, TICK_SIZE)
    assert ex_mock.call_count == 0
    assert backtesting.get_pair_precision(pair, dt_utc(2017, 12, 15)) == (1e-8, TICK_SIZE)
    assert ex_mock.call_count == 0

    # Fallback to exchange logic
    assert backtesting.get_pair_precision(pair, dt_utc(2017, 1, 15)) == (1e-5, DECIMAL_PLACES)
    assert ex_mock.call_count == 1
    assert backtesting.get_pair_precision("ETH/BTC", dt_utc(2017, 1, 15)) == (1e-5, DECIMAL_PLACES)
    assert ex_mock.call_count == 2


def test_backtest_abort(default_conf, mocker) -> None:
    patch_exchange(mocker)
    default_conf["runmode"] = RunMode.BACKTEST
    backtesting = Backtesting(default_conf)
    backtesting.check_abort()

    backtesting.abort = True

    with pytest.raises(DependencyException, match="Stop requested"):
        backtesting.check_abort()
    # abort flag resets
    assert backtesting.abort is False
    assert backtesting.progress.tasks[backtesting._progress_task].completed == 0


def test_backtesting_start(default_conf, mocker, caplog) -> None:
    def get_timerange(input1):
        return dt_utc(2017, 11, 14, 21, 17), dt_utc(2017, 11, 14, 22, 59)

    mocker.patch("freqtrade.data.history.get_timerange", get_timerange)
    patch_exchange(mocker)
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest")
    mocker.patch("freqtrade.optimize.backtesting.generate_backtest_stats")
    mocker.patch("freqtrade.optimize.backtesting.show_backtest_results")
    sbs = mocker.patch("freqtrade.optimize.backtesting.store_backtest_results")
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["UNITTEST/BTC"]),
    )

    default_conf["timeframe"] = "1m"
    default_conf["export"] = "signals"
    default_conf["exportfilename"] = "export.txt"
    default_conf["timerange"] = "-1510694220"
    default_conf["runmode"] = RunMode.BACKTEST

    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.bot_loop_start = MagicMock()
    backtesting.strategy.bot_start = MagicMock()
    backtesting.start()
    # check the logs, that will contain the backtest result
    exists = ["Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days)."]
    for line in exists:
        assert log_has(line, caplog)
    assert backtesting.strategy.dp._pairlists is not None
    assert backtesting.strategy.bot_start.call_count == 1
    assert backtesting.strategy.bot_loop_start.call_count == 0
    assert sbs.call_count == 1


def test_backtesting_start_no_data(default_conf, mocker, caplog, testdatadir) -> None:
    def get_timerange(input1):
        return dt_utc(2017, 11, 14, 21, 17), dt_utc(2017, 11, 14, 22, 59)

    mocker.patch(
        "freqtrade.data.history.history_utils.load_pair_history",
        MagicMock(return_value=pd.DataFrame()),
    )
    mocker.patch("freqtrade.data.history.get_timerange", get_timerange)
    patch_exchange(mocker)
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest")
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["UNITTEST/BTC"]),
    )

    default_conf["timeframe"] = "1m"
    default_conf["export"] = "none"
    default_conf["timerange"] = "20180101-20180102"

    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    with pytest.raises(OperationalException, match=r"No data found. Terminating\."):
        backtesting.start()


def test_backtesting_no_pair_left(default_conf, mocker) -> None:
    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))
    mocker.patch(
        "freqtrade.data.history.history_utils.load_pair_history",
        MagicMock(return_value=pd.DataFrame()),
    )
    mocker.patch("freqtrade.data.history.get_timerange", get_timerange)
    patch_exchange(mocker)
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest")
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist", PropertyMock(return_value=[])
    )

    default_conf["timeframe"] = "1m"
    default_conf["export"] = "none"
    default_conf["timerange"] = "20180101-20180102"

    with pytest.raises(OperationalException, match=r"No pair in whitelist\."):
        Backtesting(default_conf)

    default_conf.update(
        {
            "pairlists": [{"method": "StaticPairList"}],
            "timeframe_detail": "1d",
        }
    )

    with pytest.raises(
        OperationalException, match=r"Detail timeframe must be smaller than strategy timeframe\."
    ):
        Backtesting(default_conf)


def test_backtesting_pairlist_list(default_conf, mocker, tickers) -> None:
    mocker.patch(f"{EXMS}.exchange_has", MagicMock(return_value=True))
    mocker.patch(f"{EXMS}.get_tickers", tickers)
    mocker.patch(f"{EXMS}.price_to_precision", lambda s, x, y: y)
    mocker.patch("freqtrade.data.history.get_timerange", get_timerange)
    patch_exchange(mocker)
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest")
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["XRP/BTC"]),
    )
    mocker.patch("freqtrade.plugins.pairlistmanager.PairListManager.refresh_pairlist")

    default_conf["ticker_interval"] = "1m"
    default_conf["export"] = "none"
    # Use stoploss from strategy
    del default_conf["stoploss"]
    default_conf["timerange"] = "20180101-20180102"

    default_conf["pairlists"] = [{"method": "VolumePairList", "number_assets": 5}]
    with pytest.raises(
        OperationalException,
        match=r"VolumePairList not allowed for backtesting\..*StaticPairList.*",
    ):
        Backtesting(default_conf)

    default_conf["pairlists"] = [
        {"method": "StaticPairList"},
        {"method": "PrecisionFilter"},
    ]
    Backtesting(default_conf)

    # Multiple strategies
    default_conf["strategy_list"] = [CURRENT_TEST_STRATEGY, "StrategyTestV2"]
    with pytest.raises(
        OperationalException,
        match=r"PrecisionFilter not allowed for backtesting multiple strategies\.",
    ):
        Backtesting(default_conf)


def test_backtest__enter_trade(default_conf, fee, mocker) -> None:
    default_conf["use_exit_signal"] = False
    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    patch_exchange(mocker)
    default_conf["stake_amount"] = "unlimited"
    default_conf["max_open_trades"] = 2
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = "UNITTEST/BTC"
    row = [
        pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=0),
        1.0,  # Open
        1.0011,  # High
        0.999,  # Low
        1.0,  # Close
        1,  # enter_long
        0,  # exit_long
        0,  # enter_short
        0,  # exit_short
        "",  # Enter tag
        "",  # Exit tag
        5000.0,  # Base volume
    ]
    trade = backtesting._enter_trade(pair, row=row, direction="long")
    assert isinstance(trade, LocalTrade)
    assert trade.stake_amount == 495

    # Fake 2 trades, so there's not enough amount for the next trade left.
    LocalTrade.bt_trades_open.append(trade)
    backtesting.wallets.update()
    trade = backtesting._enter_trade(pair, row=row, direction="long")
    assert trade is None
    LocalTrade.bt_trades_open.pop()
    trade = backtesting._enter_trade(pair, row=row, direction="long")
    assert trade is not None
    LocalTrade.bt_trades_open.pop()

    backtesting.strategy.custom_stake_amount = lambda **kwargs: 123.5
    backtesting.wallets.update()
    trade = backtesting._enter_trade(pair, row=row, direction="long")
    LocalTrade.bt_trades_open.pop()
    assert trade
    assert trade.stake_amount == 123.5

    # In case of error - use proposed stake
    backtesting.strategy.custom_stake_amount = lambda **kwargs: 20 / 0
    trade = backtesting._enter_trade(pair, row=row, direction="long")
    LocalTrade.bt_trades_open.pop()
    assert trade
    assert trade.stake_amount == 495
    assert trade.is_short is False

    trade = backtesting._enter_trade(pair, row=row, direction="short")
    LocalTrade.bt_trades_open.pop()
    assert trade
    assert trade.stake_amount == 495
    assert trade.is_short is True

    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=300.0)
    trade = backtesting._enter_trade(pair, row=row, direction="long")
    LocalTrade.bt_trades_open.pop()
    assert trade
    assert trade.stake_amount == 300.0


def test_backtest__enter_trade_futures(default_conf_usdt, fee, mocker) -> None:
    default_conf_usdt["use_exit_signal"] = False
    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(
        "freqtrade.persistence.trade_model.price_to_precision", lambda p, *args, **kwargs: p
    )
    mocker.patch(f"{EXMS}.get_max_leverage", return_value=100)
    mocker.patch("freqtrade.optimize.backtesting.price_to_precision", lambda p, *args: p)
    patch_exchange(mocker)
    default_conf_usdt["stake_amount"] = 300
    default_conf_usdt["max_open_trades"] = 2
    default_conf_usdt["trading_mode"] = "futures"
    default_conf_usdt["margin_mode"] = "isolated"
    default_conf_usdt["stake_currency"] = "USDT"
    default_conf_usdt["exchange"]["pair_whitelist"] = [".*"]
    backtesting = Backtesting(default_conf_usdt)
    backtesting._set_strategy(backtesting.strategylist[0])
    mocker.patch("freqtrade.optimize.backtesting.Backtesting._run_funding_fees")
    pair = "ETH/USDT:USDT"
    row = [
        pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=0),
        0.1,  # Open
        0.12,  # High
        0.099,  # Low
        0.11,  # Close
        1,  # enter_long
        0,  # exit_long
        1,  # enter_short
        0,  # exit_hsort
        "",  # Long Signal Name
        "",  # Short Signal Name
        1000000.0,  # Base volume
    ]

    backtesting.strategy.leverage = MagicMock(return_value=5.0)
    mocker.patch(f"{EXMS}.get_maintenance_ratio_and_amt", return_value=(0.01, 0.01))

    # leverage = 5
    # ep1(trade.open_rate) = 0.1
    # position(trade.amount) = 15000
    # stake_amount = 300 -> wb = 300 / 5 = 60
    # mmr = 0.01
    # cum_b = 0.01
    # side_1: -1 if is_short else 1
    # liq_buffer = 0.05
    #
    # Binance, Long
    # liquidation_price
    #   = ((wb + cum_b) - (side_1 * position * ep1)) / ((position * mmr_b) - (side_1 * position))
    #   = ((300 + 0.01) - (1 * 15000 * 0.1)) / ((15000 * 0.01) - (1 * 15000))
    #   = 0.0008080740740740741
    # freqtrade_liquidation_price = liq + (abs(open_rate - liq) * liq_buffer * side_1)
    #   = 0.08080740740740741 + ((0.1 - 0.08080740740740741) * 0.05 * 1)
    #   = 0.08176703703703704

    trade = backtesting._enter_trade(pair, row=row, direction="long")
    assert pytest.approx(trade.liquidation_price) == 0.081767037

    # Binance, Short
    # liquidation_price
    #   = ((wb + cum_b) - (side_1 * position * ep1)) / ((position * mmr_b) - (side_1 * position))
    #   = ((300 + 0.01) - ((-1) * 15000 * 0.1)) / ((15000 * 0.01) - ((-1) * 15000))
    #   = 0.0011881254125412541
    # freqtrade_liquidation_price = liq + (abs(open_rate - liq) * liq_buffer * side_1)
    #   = 0.11881254125412541 + (abs(0.1 - 0.11881254125412541) * 0.05 * -1)
    #   = 0.11787191419141915

    trade = backtesting._enter_trade(pair, row=row, direction="short")
    assert pytest.approx(trade.liquidation_price) == 0.11787191
    assert pytest.approx(trade.orders[0].cost) == (
        trade.stake_amount * trade.leverage * (1 + fee.return_value)
    )
    assert pytest.approx(trade.orders[-1].stake_amount) == trade.stake_amount

    # Stake-amount too high!
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=600.0)

    trade = backtesting._enter_trade(pair, row=row, direction="long")
    assert trade is None

    # Stake-amount throwing error
    mocker.patch(
        "freqtrade.wallets.Wallets.get_trade_stake_amount", side_effect=DependencyException
    )

    trade = backtesting._enter_trade(pair, row=row, direction="long")
    assert trade is None


def test_backtest__check_trade_exit(default_conf, mocker) -> None:
    default_conf["use_exit_signal"] = False
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    default_conf["timeframe_detail"] = "1m"
    default_conf["max_open_trades"] = 2
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = "UNITTEST/BTC"
    row = [
        pd.Timestamp(year=2020, month=1, day=1, hour=4, minute=55, tzinfo=UTC),
        200,  # Open
        201.5,  # High
        195,  # Low
        201,  # Close
        1,  # enter_long
        0,  # exit_long
        0,  # enter_short
        0,  # exit_hsort
        "",  # Long Signal Name
        "",  # Short Signal Name
        1000000.0,  # Base volume
    ]

    trade = backtesting._enter_trade(pair, row=row, direction="long")
    assert isinstance(trade, LocalTrade)

    row_sell = [
        pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=0, tzinfo=UTC),
        200,  # Open
        210.5,  # High
        195,  # Low
        201,  # Close
        0,  # enter_long
        0,  # exit_long
        0,  # enter_short
        0,  # exit_short
        "",  # long Signal Name
        "",  # Short Signal Name
        1000000.0,  # Base volume
    ]

    # No data available.
    res = backtesting._check_trade_exit(trade, row_sell, row_sell[0].to_pydatetime())
    assert res is not None
    assert res.exit_reason == ExitType.ROI.value
    assert res.close_date_utc == datetime(2020, 1, 1, 5, 0, tzinfo=UTC)

    # Enter new trade
    trade = backtesting._enter_trade(pair, row=row, direction="long")
    assert isinstance(trade, LocalTrade)
    # Assign empty ... no result.
    backtesting.detail_data[pair] = pd.DataFrame(
        [],
        columns=[
            "date",
            "open",
            "high",
            "low",
            "close",
            "enter_long",
            "exit_long",
            "enter_short",
            "exit_short",
            "long_tag",
            "short_tag",
            "exit_tag",
        ],
    )

    res = backtesting._check_trade_exit(trade, row, row[0].to_pydatetime())
    assert res is None


def test_get_detail_data(default_conf, mocker) -> None:
    # get_detail_data returns the detail candles whose "date" falls in the half-open
    # window (current, current + timeframe_td) of the current main candle, with the
    # signal/tag columns filled from the main row. A boolean mask over the detail
    # frame is used here as an independent oracle for that window, exercised across
    # interior, boundary, empty, and out-of-range inputs.
    patch_exchange(mocker)
    default_conf["timeframe"] = "1h"
    default_conf["timeframe_detail"] = "1m"
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.timeframe_td = timedelta(hours=1)
    pair = "UNITTEST/BTC"

    # 4h of 1m candles starting at 04:00 (04:00 .. 07:59).
    detail = generate_test_data("1m", 240, "2020-01-01 04:00:00+00:00")
    backtesting.detail_data[pair] = detail

    def make_row(ts):
        # date, O, H, L, C, enter_long, exit_long, enter_short, exit_short,
        # enter_tag, exit_tag, base volume
        return [ts, 200.0, 201.5, 195.0, 201.0, 1, 0, 0, 0, "ent", "ext", 1000.0]

    def mask_oracle(ts):
        """Reference window selection via a boolean mask over the detail frame."""
        end = ts + backtesting.timeframe_td
        sub = detail.loc[(detail["date"] >= ts) & (detail["date"] < end)].copy()
        if len(sub) == 0:
            return None
        sub.loc[:, "enter_long"] = 1
        sub.loc[:, "exit_long"] = 0
        sub.loc[:, "enter_short"] = 0
        sub.loc[:, "exit_short"] = 0
        sub.loc[:, "enter_tag"] = "ent"
        sub.loc[:, "exit_tag"] = "ext"
        return {row[0]: row for row in sub[HEADERS].values.tolist()}

    test_times = [
        pd.Timestamp("2020-01-01 03:00", tz="UTC"),  # before all data -> None
        pd.Timestamp("2020-01-01 04:00", tz="UTC"),  # aligned to first detail row
        pd.Timestamp("2020-01-01 05:00", tz="UTC"),  # interior, full 60-row window
        pd.Timestamp("2020-01-01 06:30", tz="UTC"),  # interior, offset from data start
        pd.Timestamp("2020-01-01 07:30", tz="UTC"),  # straddles end of data (30 rows)
        pd.Timestamp("2020-01-01 08:00", tz="UTC"),  # exactly at end -> None
        pd.Timestamp("2020-01-01 09:00", tz="UTC"),  # after all data -> None
    ]
    for ts in test_times:
        assert backtesting.get_detail_data(pair, make_row(ts)) == mask_oracle(ts)

    # Boundary: a candle whose date == exit_candle_end must be excluded.
    res = backtesting.get_detail_data(pair, make_row(pd.Timestamp("2020-01-01 04:00", tz="UTC")))
    assert res is not None
    assert len(res) == 60
    assert list(res)[-1] == pd.Timestamp("2020-01-01 04:59", tz="UTC")  # not 05:00
    # Signal columns are taken from the row, not the detail frame.
    assert res[pd.Timestamp("2020-01-01 04:00", tz="UTC")][HEADERS.index("enter_long")] == 1
    assert res[pd.Timestamp("2020-01-01 04:00", tz="UTC")][HEADERS.index("enter_tag")] == "ent"


def test_backtest_one(default_conf, mocker, testdatadir) -> None:
    default_conf["use_exit_signal"] = False
    default_conf["max_open_trades"] = 10
    default_conf["runmode"] = RunMode.BACKTEST

    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_pair_base_currency", lambda _, x: x.split("/")[0])
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = "UNITTEST/BTC"
    timerange = TimeRange("date", None, 1517227800, 0)
    data = history.load_data(
        datadir=testdatadir, timeframe="5m", pairs=["UNITTEST/BTC"], timerange=timerange
    )
    processed = backtesting.strategy.advise_all_indicators(data)
    backtesting.strategy.order_filled = MagicMock()
    min_date, max_date = get_timerange(processed)

    result = backtesting.backtest(
        processed=deepcopy(processed),
        start_date=min_date,
        end_date=max_date,
    )
    results = result["results"]
    assert not results.empty
    assert len(results) == 2

    expected = pd.DataFrame(
        {
            "pair": [pair, pair],
            "stake_amount": [0.001, 0.001],
            "max_stake_amount": [0.001, 0.001],
            "amount": [0.00957442, 0.0097064],
            "open_date": pd.to_datetime(
                [dt_utc(2018, 1, 29, 18, 40, 0), dt_utc(2018, 1, 30, 3, 30, 0)], utc=True
            ),
            "close_date": pd.to_datetime(
                [dt_utc(2018, 1, 29, 22, 35, 0), dt_utc(2018, 1, 30, 4, 10, 0)], utc=True
            ),
            "open_rate": [0.104445, 0.10302485],
            "close_rate": [0.104969, 0.103541],
            "fee_open": [0.0025, 0.0025],
            "fee_close": [0.0025, 0.0025],
            "trade_duration": [235, 40],
            "profit_ratio": [0.0, 0.0],
            "profit_abs": [0.0, 0.0],
            "exit_reason": [ExitType.ROI.value, ExitType.ROI.value],
            "initial_stop_loss_abs": [0.0940005, 0.09272236],
            "initial_stop_loss_ratio": [-0.1, -0.1],
            "stop_loss_abs": [0.0940005, 0.09272236],
            "stop_loss_ratio": [-0.1, -0.1],
            "min_rate": [0.10370188, 0.10300000000000001],
            "max_rate": [0.10501, 0.1038888],
            "is_open": [False, False],
            "enter_tag": ["", ""],
            "leverage": [1.0, 1.0],
            "is_short": [False, False],
            "open_timestamp": [1517251200000, 1517283000000],
            "close_timestamp": [1517265300000, 1517285400000],
            "orders": [
                [
                    {
                        "amount": 0.00957442,
                        "safe_price": 0.104445,
                        "ft_order_side": "buy",
                        "order_filled_timestamp": 1517251200000,
                        "ft_is_entry": True,
                        "ft_order_tag": "",
                        "cost": ANY,
                    },
                    {
                        "amount": 0.00957442,
                        "safe_price": 0.10496853383458644,
                        "ft_order_side": "sell",
                        "order_filled_timestamp": 1517265300000,
                        "ft_is_entry": False,
                        "ft_order_tag": "roi",
                        "cost": ANY,
                    },
                ],
                [
                    {
                        "amount": 0.0097064,
                        "safe_price": 0.10302485,
                        "ft_order_side": "buy",
                        "order_filled_timestamp": 1517283000000,
                        "ft_is_entry": True,
                        "ft_order_tag": "",
                        "cost": ANY,
                    },
                    {
                        "amount": 0.0097064,
                        "safe_price": 0.10354126528822055,
                        "ft_order_side": "sell",
                        "order_filled_timestamp": 1517285400000,
                        "ft_is_entry": False,
                        "ft_order_tag": "roi",
                        "cost": ANY,
                    },
                ],
            ],
            "funding_fees": [0.0, 0.0],
        }
    )
    # TODO: pandas3 - create correctly above ?!?
    expected["open_date"] = expected["open_date"].astype("datetime64[ms, UTC]")
    expected["close_date"] = expected["close_date"].astype("datetime64[ms, UTC]")
    pd.testing.assert_frame_equal(results, expected)
    assert "orders" in results.columns
    data_pair = processed[pair]
    # Called once per order
    assert backtesting.strategy.order_filled.call_count == 4
    for _, t in results.iterrows():
        assert len(t["orders"]) == 2
        ln = data_pair.loc[data_pair["date"] == t["open_date"]]
        # Check open trade rate aligns to open rate
        assert not ln.empty
        assert round(ln.iloc[0]["open"], 6) == round(t["open_rate"], 6)
        # check close trade rate aligns to close rate or is between high and low
        ln1 = data_pair.loc[data_pair["date"] == t["close_date"]]
        assert round(ln1.iloc[0]["open"], 6) == round(t["close_rate"], 6) or round(
            ln1.iloc[0]["low"], 6
        ) < round(t["close_rate"], 6) < round(ln1.iloc[0]["high"], 6)

    wallet_summary = result["wallet_summary"]
    assert isinstance(wallet_summary, pd.DataFrame)
    assert len(wallet_summary) == 255
    unique_currencies = wallet_summary["currency"].value_counts()
    assert unique_currencies["BTC"] == 200
    assert unique_currencies["UNITTEST"] == 55


@pytest.mark.parametrize("use_detail", [True, False])
def test_backtest_one_detail(default_conf_usdt, mocker, testdatadir, use_detail) -> None:
    default_conf_usdt["use_exit_signal"] = False
    default_conf_usdt["runmode"] = RunMode.BACKTEST
    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_pair_base_currency", lambda _, x: x.split("/")[0])

    default_conf_usdt["unfilledtimeout"] = {
        "entry": 11,
        "exit": 30,
    }
    if use_detail:
        default_conf_usdt["timeframe_detail"] = "1m"

    def advise_entry(df, *args, **kwargs):
        # Mock function to force several entries
        df.loc[(df["rsi"] < 40), "enter_long"] = 1
        return df

    def custom_entry_price(proposed_rate, **kwargs):
        return proposed_rate * 0.997

    default_conf_usdt["max_open_trades"] = 10

    backtesting = Backtesting(default_conf_usdt)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.populate_entry_trend = advise_entry
    backtesting.strategy.ignore_buying_expired_candle_after = 59
    backtesting.strategy.custom_entry_price = custom_entry_price
    pair = "XRP/ETH"
    # Pick a timerange adapted to the pair we use to test
    timerange = TimeRange.parse_timerange("20191010-20191013")
    data = history.load_data(datadir=testdatadir, timeframe="5m", pairs=[pair], timerange=timerange)
    # Invent ample base quantity so the test isolates limit-price timeouts.
    data[pair]["volume"] = 1000000.0
    if use_detail:
        data_1m = history.load_data(
            datadir=testdatadir, timeframe="1m", pairs=[pair], timerange=timerange
        )
        data_1m[pair]["volume"] = 1000000.0
        backtesting.detail_data = data_1m
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)

    result = backtesting.backtest(
        processed=deepcopy(processed),
        start_date=min_date,
        end_date=max_date,
    )
    results = result["results"]
    assert not results.empty
    # Timeout settings from = entry: 11, exit: 30
    assert len(results) == (2 if use_detail else 3)

    assert "orders" in results.columns
    data_pair = processed[pair]

    data_1m_pair = data_1m[pair] if use_detail else pd.DataFrame()
    late_entry = 0
    for _, t in results.iterrows():
        assert len(t["orders"]) == 2

        entryo = t["orders"][0]
        entry_ts = datetime.fromtimestamp(entryo["order_filled_timestamp"] // 1000, tz=UTC)
        if entry_ts > t["open_date"]:
            late_entry += 1

        # Get "entry fill" candle
        ln = (
            data_1m_pair.loc[data_1m_pair["date"] == entry_ts]
            if use_detail
            else data_pair.loc[data_pair["date"] == entry_ts]
        )
        # Check open trade rate aligns to open rate
        assert not ln.empty

        # assert round(ln.iloc[0]["open"], 6) == round(t["open_rate"], 6)
        assert (
            round(ln.iloc[0]["low"], 6) <= round(t["open_rate"], 6) <= round(ln.iloc[0]["high"], 6)
        )
        # check close trade rate aligns to close rate or is between high and low
        ln1 = data_pair.loc[data_pair["date"] == t["close_date"]]
        if use_detail:
            ln1_1m = data_1m_pair.loc[data_1m_pair["date"] == t["close_date"]]
            assert not ln1.empty or not ln1_1m.empty
        else:
            assert not ln1.empty
        ln2 = ln1_1m if ln1.empty else ln1

        assert (
            round(ln2.iloc[0]["low"], 6)
            <= round(t["close_rate"], 6)
            <= round(ln2.iloc[0]["high"], 6)
        )

    assert late_entry > 0
    wallet_summary = result["wallet_summary"]
    assert isinstance(wallet_summary, pd.DataFrame)
    assert len(wallet_summary) == 591 if use_detail else 597
    unique_currencies = wallet_summary["currency"].value_counts()
    assert unique_currencies["USDT"] == 576
    assert unique_currencies["XRP"] == 15 if use_detail else 21


@pytest.mark.parametrize(
    "use_detail,exp_funding_fee, exp_ff_updates",
    [
        (True, -0.0180457882, 15),
        (False, -0.0178000543, 12),
    ],
)
def test_backtest_one_detail_futures(
    default_conf_usdt, mocker, testdatadir, use_detail, exp_funding_fee, exp_ff_updates
) -> None:
    default_conf_usdt["use_exit_signal"] = False
    default_conf_usdt["trading_mode"] = "futures"
    default_conf_usdt["margin_mode"] = "isolated"
    default_conf_usdt["candle_type_def"] = CandleType.FUTURES

    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["XRP/USDT:USDT"]),
    )
    mocker.patch(f"{EXMS}.get_maintenance_ratio_and_amt", return_value=(0.01, 0.01))
    default_conf_usdt["timeframe"] = "1h"
    if use_detail:
        default_conf_usdt["timeframe_detail"] = "5m"

    def advise_entry(df, *args, **kwargs):
        # Mock function to force several entries
        df.loc[(df["rsi"] < 40), "enter_long"] = 1
        return df

    def custom_entry_price(proposed_rate, **kwargs):
        return proposed_rate * 0.997

    default_conf_usdt["max_open_trades"] = 10

    backtesting = Backtesting(default_conf_usdt)
    ff_spy = mocker.spy(backtesting.exchange, "calculate_funding_fees")

    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.populate_entry_trend = advise_entry
    backtesting.strategy.custom_entry_price = custom_entry_price
    pair = "XRP/USDT:USDT"
    # Pick a timerange adapted to the pair we use to test
    timerange = TimeRange.parse_timerange("20211117-20211119")
    data = history.load_data(
        datadir=Path(testdatadir),
        timeframe="1h",
        pairs=[pair],
        timerange=timerange,
        candle_type=CandleType.FUTURES,
    )
    backtesting._load_bt_data_detail()
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)

    result = backtesting.backtest(
        processed=deepcopy(processed),
        start_date=min_date,
        end_date=max_date,
    )
    results = result["results"]
    assert not results.empty
    # Timeout settings from default_conf = entry: 10, exit: 30
    assert len(results) == (4 if use_detail else 2)

    assert "orders" in results.columns
    data_pair = processed[pair]

    data_1m_pair = backtesting.detail_data[pair] if use_detail else pd.DataFrame()
    late_entry = 0
    for _, t in results.iterrows():
        assert len(t["orders"]) == 2

        entryo = t["orders"][0]
        entry_ts = datetime.fromtimestamp(entryo["order_filled_timestamp"] // 1000, tz=UTC)
        if entry_ts > t["open_date"]:
            late_entry += 1

        # Get "entry fill" candle
        ln = (
            data_1m_pair.loc[data_1m_pair["date"] == entry_ts]
            if use_detail
            else data_pair.loc[data_pair["date"] == entry_ts]
        )
        # Check open trade rate aligns to open rate
        assert not ln.empty

        assert (
            round(ln.iloc[0]["low"], 6) <= round(t["open_rate"], 6) <= round(ln.iloc[0]["high"], 6)
        )
        # check close trade rate aligns to close rate or is between high and low
        ln1 = data_pair.loc[data_pair["date"] == t["close_date"]]
        if use_detail:
            ln1_1m = data_1m_pair.loc[data_1m_pair["date"] == t["close_date"]]
            assert not ln1.empty or not ln1_1m.empty
        else:
            assert not ln1.empty
        ln2 = ln1_1m if ln1.empty else ln1

        assert (
            round(ln2.iloc[0]["low"], 6)
            <= round(t["close_rate"], 6)
            <= round(ln2.iloc[0]["high"], 6)
        )
    assert pytest.approx(Trade.bt_trades[1].funding_fees) == exp_funding_fee
    assert ff_spy.call_count == exp_ff_updates
    # assert late_entry > 0


@pytest.mark.parametrize(
    "use_detail,entries,max_stake,ff_updates,expected_ff",
    [
        (True, 50, 3000, 78, -1.17988972),
        (False, 6, 360, 34, -0.14673681),
    ],
)
def test_backtest_one_detail_futures_funding_fees(
    default_conf_usdt,
    fee,
    mocker,
    testdatadir,
    use_detail,
    entries,
    max_stake,
    ff_updates,
    expected_ff,
) -> None:
    """
    Funding fees are expected to differ, as the maximum position size differs.
    """
    default_conf_usdt["use_exit_signal"] = False
    default_conf_usdt["trading_mode"] = "futures"
    default_conf_usdt["margin_mode"] = "isolated"
    default_conf_usdt["candle_type_def"] = CandleType.FUTURES
    default_conf_usdt["minimal_roi"] = {"0": 1}
    default_conf_usdt["dry_run_wallet"] = 100000

    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["XRP/USDT:USDT"]),
    )
    mocker.patch(f"{EXMS}.get_maintenance_ratio_and_amt", return_value=(0.01, 0.01))
    default_conf_usdt["timeframe"] = "1h"
    if use_detail:
        default_conf_usdt["timeframe_detail"] = "5m"
    patch_exchange(mocker)

    def advise_entry(df, *args, **kwargs):
        # Mock function to force several entries
        df.loc[:, "enter_long"] = 1
        return df

    def adjust_trade_position(trade, current_time, **kwargs):
        if current_time > datetime(2021, 11, 18, 2, 0, 0, tzinfo=UTC):
            return None
        return default_conf_usdt["stake_amount"]

    default_conf_usdt["max_open_trades"] = 1

    backtesting = Backtesting(default_conf_usdt)
    ff_spy = mocker.spy(backtesting.exchange, "calculate_funding_fees")
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.populate_entry_trend = advise_entry
    backtesting.strategy.adjust_trade_position = adjust_trade_position
    backtesting.strategy.leverage = lambda **kwargs: 1
    backtesting.strategy.position_adjustment_enable = True
    pair = "XRP/USDT:USDT"
    # Pick a timerange adapted to the pair we use to test
    timerange = TimeRange.parse_timerange("20211117-20211119")
    data = history.load_data(
        datadir=Path(testdatadir),
        timeframe="1h",
        pairs=[pair],
        timerange=timerange,
        candle_type=CandleType.FUTURES,
    )
    backtesting._load_bt_data_detail()
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)

    result = backtesting.backtest(
        processed=deepcopy(processed),
        start_date=min_date,
        end_date=max_date,
    )
    results = result["results"]
    assert not results.empty
    # Only one result - as we're not selling.
    assert len(results) == 1

    assert "orders" in results.columns
    # funding_fees have been calculated for each funding-fee candle
    # the trade is open for 26 hours - hence we expect the 8h fee to apply 4 times.
    # Additional counts will happen due each successful entry, which needs to call this, too.
    assert ff_spy.call_count == ff_updates

    for t in Trade.bt_trades:
        # At least 6 adjustment orders
        assert t.nr_of_successful_entries == entries
        # Funding fees will vary depending on the number of adjustment orders
        # That number is a lot higher with detail data.
        assert t.max_stake_amount == max_stake
        assert pytest.approx(t.funding_fees) == expected_ff


def test_backtest_timedout_entry_orders(default_conf, fee, mocker, testdatadir) -> None:
    # This strategy intentionally places unfillable orders.
    default_conf["strategy"] = "StrategyTestV3CustomEntryPrice"
    default_conf["startup_candle_count"] = 0
    # Cancel unfilled order after 4 minutes on 5m timeframe.
    default_conf["unfilledtimeout"] = {"entry": 4}
    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    patch_exchange(mocker)
    default_conf["max_open_trades"] = 1
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    # Testing dataframe contains 11 candles. Expecting 10 timed out orders.
    timerange = TimeRange("date", "date", 1517227800, 1517231100)
    data = history.load_data(
        datadir=testdatadir, timeframe="5m", pairs=["UNITTEST/BTC"], timerange=timerange
    )
    min_date, max_date = get_timerange(data)

    result = backtesting.backtest(
        processed=deepcopy(data),
        start_date=min_date,
        end_date=max_date,
    )

    assert result["timedout_entry_orders"] == 10


def test_backtest_1min_timeframe(default_conf, fee, mocker, testdatadir) -> None:
    default_conf["timeframe"] = "1m"
    default_conf["use_exit_signal"] = False
    default_conf["max_open_trades"] = 1
    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])

    # Run a backtesting for an exiting 1min timeframe
    timerange = TimeRange.parse_timerange("1510688220-1510700340")
    data = history.load_data(
        datadir=testdatadir, timeframe="1m", pairs=["UNITTEST/BTC"], timerange=timerange
    )
    # This timeframe check uses invented sufficient volume, including the terminal candle.
    data["UNITTEST/BTC"]["volume"] = 1000.0
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)
    results = backtesting.backtest(
        processed=processed,
        start_date=min_date,
        end_date=max_date,
    )
    assert not results["results"].empty
    assert len(results["results"]) == 1


def test_backtest_trim_no_data_left(default_conf, fee, mocker, testdatadir) -> None:
    default_conf["use_exit_signal"] = False
    default_conf["max_open_trades"] = 10

    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    timerange = TimeRange("date", None, 1517227800, 0)
    backtesting.required_startup = 100
    backtesting.timerange = timerange
    data = history.load_data(
        datadir=testdatadir, timeframe="5m", pairs=["UNITTEST/BTC"], timerange=timerange
    )
    df = data["UNITTEST/BTC"]
    df["date"] = df.loc[:, "date"] - timedelta(days=1)
    # Trimming 100 candles, so after 2nd trimming, no candle is left.
    df = df.iloc[:100]
    data["XRP/USDT"] = df
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)

    backtesting.backtest(
        processed=deepcopy(processed),
        start_date=min_date,
        end_date=max_date,
    )


def test_processed(default_conf, mocker, testdatadir) -> None:
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])

    dict_of_tickerrows = load_data_test("raise", testdatadir)
    dataframes = backtesting.strategy.advise_all_indicators(dict_of_tickerrows)
    dataframe = dataframes["UNITTEST/BTC"]
    cols = dataframe.columns
    # assert the dataframe got some of the indicator columns
    for col in ["close", "high", "low", "open", "date", "ema10", "rsi", "fastd", "plus_di"]:
        assert col in cols


def test_backtest_dataprovider_analyzed_df(default_conf, fee, mocker, testdatadir) -> None:
    default_conf["use_exit_signal"] = False
    default_conf["max_open_trades"] = 10
    default_conf["runmode"] = "backtest"
    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=100000)
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    timerange = TimeRange("date", None, 1517227800, 0)
    data = history.load_data(
        datadir=testdatadir, timeframe="5m", pairs=["UNITTEST/BTC"], timerange=timerange
    )
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)

    count = 0

    def tmp_confirm_entry(pair, current_time, **kwargs):
        nonlocal count
        dp = backtesting.strategy.dp
        df, _ = dp.get_analyzed_dataframe(pair, backtesting.strategy.timeframe)
        current_candle = df.iloc[-1].squeeze()
        assert current_candle["enter_long"] == 1

        candle_date = timeframe_to_next_date(backtesting.strategy.timeframe, current_candle["date"])
        assert candle_date == current_time
        # These asserts don't properly raise as they are nested,
        # therefore we increment count and assert for that.
        df = dp.get_pair_dataframe(pair, backtesting.strategy.timeframe)
        prior_time = timeframe_to_prev_date(
            backtesting.strategy.timeframe, candle_date - timedelta(seconds=1)
        )
        assert prior_time == df.iloc[-1].squeeze()["date"]
        assert df.iloc[-1].squeeze()["date"] < current_time

        count += 1

    backtesting.strategy.confirm_trade_entry = tmp_confirm_entry
    backtesting.backtest(
        processed=deepcopy(processed),
        start_date=min_date,
        end_date=max_date,
    )
    assert count == 5


def test_backtest_pricecontours_protections(default_conf, fee, mocker, testdatadir) -> None:
    # While this test IS a copy of test_backtest_pricecontours, it's needed to ensure
    # results do not carry-over to the next run, which is not given by using parametrize.
    patch_exchange(mocker)
    default_conf["_strategy_protections"] = [
        {
            "method": "CooldownPeriod",
            "stop_duration": 3,
        }
    ]

    default_conf["enable_protections"] = True
    default_conf["timeframe"] = "1m"
    default_conf["max_open_trades"] = 1
    mocker.patch(f"{EXMS}.get_fee", fee)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    tests = [
        ["sine", 10],
        ["raise", 11],
        ["lower", 0],
        ["sine", 10],
        ["raise", 11],
    ]
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])

    # While entry-signals are unrealistic, running backtesting
    # over and over again should not cause different results
    for [contour, numres] in tests:
        # Debug output for random test failure
        print(f"{contour}, {numres}")
        data = load_data_test(contour, testdatadir)
        processed = backtesting.strategy.advise_all_indicators(data)
        min_date, max_date = get_timerange(processed)
        assert isinstance(processed, dict)
        results = backtesting.backtest(
            processed=processed,
            start_date=min_date,
            end_date=max_date,
        )
        assert len(results["results"]) == numres


@pytest.mark.parametrize(
    "protections,contour,expected",
    [
        (None, "sine", 35),
        (None, "raise", 19),
        (None, "lower", 0),
        ([{"method": "CooldownPeriod", "stop_duration": 3}], "sine", 10),
        ([{"method": "CooldownPeriod", "stop_duration": 3}], "raise", 11),
        ([{"method": "CooldownPeriod", "stop_duration": 3}], "lower", 0),
    ],
)
def test_backtest_pricecontours(
    default_conf, mocker, testdatadir, protections, contour, expected
) -> None:
    if protections:
        default_conf["_strategy_protections"] = protections
        default_conf["enable_protections"] = True

    patch_exchange(mocker)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    # While entry-signals are unrealistic, running backtesting
    # over and over again should not cause different results

    default_conf["timeframe"] = "1m"
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])

    data = load_data_test(contour, testdatadir)
    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)
    assert isinstance(processed, dict)
    backtesting.strategy.max_open_trades = 1
    backtesting.config.update({"max_open_trades": 1})
    results = backtesting.backtest(
        processed=processed,
        start_date=min_date,
        end_date=max_date,
    )
    assert len(results["results"]) == expected


def test_backtest_clash_buy_sell(mocker, default_conf, testdatadir):
    # Override the default buy trend function in our StrategyTest
    def fun(dataframe=None, pair=None):
        buy_value = 1
        sell_value = 1
        return _trend(dataframe, buy_value, sell_value)

    default_conf["max_open_trades"] = 10
    backtest_conf = _make_backtest_conf(mocker, conf=default_conf, datadir=testdatadir)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.advise_entry = fun  # Override
    backtesting.strategy.advise_exit = fun  # Override
    result = backtesting.backtest(**backtest_conf)
    assert result["results"].empty


def test_backtest_only_sell(mocker, default_conf, testdatadir):
    # Override the default buy trend function in our StrategyTest
    def fun(dataframe=None, pair=None):
        buy_value = 0
        sell_value = 1
        return _trend(dataframe, buy_value, sell_value)

    default_conf["max_open_trades"] = 10
    backtest_conf = _make_backtest_conf(mocker, conf=default_conf, datadir=testdatadir)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.advise_entry = fun  # Override
    backtesting.strategy.advise_exit = fun  # Override
    result = backtesting.backtest(**backtest_conf)
    assert result["results"].empty


def test_backtest_alternate_buy_sell(default_conf, fee, mocker, testdatadir):
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_fee", fee)
    default_conf["max_open_trades"] = 10
    default_conf["runmode"] = "backtest"
    backtest_conf = _make_backtest_conf(
        mocker, conf=default_conf, pair="UNITTEST/BTC", datadir=testdatadir
    )
    default_conf["timeframe"] = "1m"
    backtesting = Backtesting(default_conf)
    backtesting.required_startup = 0
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.advise_entry = _trend_alternate  # Override
    backtesting.strategy.advise_exit = _trend_alternate  # Override
    result = backtesting.backtest(**backtest_conf)
    # 200 candles in backtest data
    # won't buy on first (shifted by 1)
    # 100 buys signals
    results = result["results"]
    assert len(results) == 100
    # Cached data should be 200
    analyzed_df = backtesting.dataprovider.get_analyzed_dataframe("UNITTEST/BTC", "1m")[0]
    assert len(analyzed_df) == 200
    # Expect last candle to be 1 below end date (as the last candle is assumed as "incomplete"
    # during backtesting)
    expected_last_candle_date = backtest_conf["end_date"] - timedelta(minutes=1)
    assert analyzed_df.iloc[-1]["date"].to_pydatetime() == expected_last_candle_date

    # One trade was force-closed at the end
    assert len(results.loc[results["is_open"]]) == 0


@pytest.mark.parametrize("pair", ["ADA/BTC", "LTC/BTC"])
@pytest.mark.parametrize("tres", [0, 20, 30])
def test_backtest_multi_pair(default_conf, fee, mocker, tres, pair, testdatadir):
    def _trend_alternate_hold(dataframe=None, metadata=None):
        """
        Buy every xth candle - sell every other xth -2 (hold on to pairs a bit)
        """
        if metadata["pair"] in ("ETH/BTC", "LTC/BTC"):
            multi = 20
        else:
            multi = 18
        dataframe["enter_long"] = np.where(dataframe.index % multi == 0, 1, 0)
        dataframe["exit_long"] = np.where((dataframe.index + multi - 2) % multi == 0, 1, 0)
        dataframe["enter_short"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    default_conf["runmode"] = "backtest"
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_fee", fee)
    patch_exchange(mocker)

    pairs = ["ADA/BTC", "DASH/BTC", "ETH/BTC", "LTC/BTC", "NXT/BTC"]
    data = history.load_data(datadir=testdatadir, timeframe="5m", pairs=pairs)
    # Only use 500 lines to increase performance
    data = trim_dictlist(data, -500)

    # Remove data for one pair from the beginning of the data
    if tres > 0:
        data[pair] = data[pair][tres:].reset_index()
    default_conf["timeframe"] = "5m"
    default_conf["max_open_trades"] = 3

    backtesting = Backtesting(default_conf)
    vr_spy = mocker.spy(backtesting, "validate_row")
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.bot_loop_start = MagicMock()
    backtesting.strategy.advise_entry = _trend_alternate_hold  # Override
    backtesting.strategy.advise_exit = _trend_alternate_hold  # Override

    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)

    backtest_conf = {
        "processed": deepcopy(processed),
        "start_date": min_date,
        "end_date": max_date,
    }

    results = backtesting.backtest(**backtest_conf)

    # bot_loop_start is called once per candle.
    assert backtesting.strategy.bot_loop_start.call_count == 499
    # Validated row once per candle and pair
    assert vr_spy.call_count == 2495
    # List of calls pair args - in batches of 5 (s)
    calls_per_candle = defaultdict(list)
    for call in vr_spy.call_args_list:
        calls_per_candle[call[0][3]].append(call[0][1])

    all_orients = [x for _, x in calls_per_candle.items()]

    distinct_calls = [list(x) for x in {tuple(x) for x in all_orients}]

    # All calls must be made for the full pairlist
    assert all(len(x) == 5 for x in distinct_calls)

    # order varied - and is not always identical
    assert not all(
        x == ["ADA/BTC", "DASH/BTC", "ETH/BTC", "LTC/BTC", "NXT/BTC"] for x in distinct_calls
    )
    # But some calls should've kept the original ordering
    assert any(
        x == ["ADA/BTC", "DASH/BTC", "ETH/BTC", "LTC/BTC", "NXT/BTC"] for x in distinct_calls
    )
    assert (
        # Ordering can be different, but should be one of the following
        any(x == ["ETH/BTC", "ADA/BTC", "DASH/BTC", "LTC/BTC", "NXT/BTC"] for x in distinct_calls)
        or any(
            x == ["ETH/BTC", "LTC/BTC", "ADA/BTC", "DASH/BTC", "NXT/BTC"] for x in distinct_calls
        )
    )

    # Make sure we have parallel trades
    assert len(evaluate_result_multi(results["results"], "5m", 2)) > 0
    # make sure we don't have trades with more than configured max_open_trades
    assert len(evaluate_result_multi(results["results"], "5m", 3)) == 0

    # Cached data correctly removed amounts
    removed_candles = len(data[pair]) - 1
    assert len(backtesting.dataprovider.get_analyzed_dataframe(pair, "5m")[0]) == removed_candles
    assert (
        len(backtesting.dataprovider.get_analyzed_dataframe("NXT/BTC", "5m")[0])
        == len(data["NXT/BTC"]) - 1
    )

    backtesting.strategy.max_open_trades = 1
    backtesting.config.update({"max_open_trades": 1})
    backtest_conf = {
        "processed": deepcopy(processed),
        "start_date": min_date,
        "end_date": max_date,
    }
    results = backtesting.backtest(**backtest_conf)
    assert len(evaluate_result_multi(results["results"], "5m", 1)) == 0


@pytest.mark.parametrize("use_detail", [True, False])
@pytest.mark.parametrize("pair", ["ADA/USDT", "LTC/USDT"])
@pytest.mark.parametrize("tres", [0, 20, 30])
def test_backtest_multi_pair_detail(
    default_conf_usdt,
    fee,
    mocker,
    tres,
    pair,
    use_detail,
):
    """
    literally the same as test_backtest_multi_pair - but with artificial data
    and detail timeframe.
    """

    def _trend_alternate_hold(dataframe=None, metadata=None):
        """
        Buy every xth candle - sell every other xth -2 (hold on to pairs a bit)
        """
        if metadata["pair"] in ("ETH/USDT", "LTC/USDT"):
            multi = 20
        else:
            multi = 18
        dataframe["enter_long"] = np.where(dataframe.index % multi == 0, 1, 0)
        dataframe["exit_long"] = np.where((dataframe.index + multi - 2) % multi == 0, 1, 0)
        dataframe["enter_short"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    default_conf_usdt.update(
        {
            "runmode": "backtest",
            "stoploss": -1.0,
            "minimal_roi": {"0": 100},
        }
    )

    if use_detail:
        default_conf_usdt["timeframe_detail"] = "1m"

    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_fee", fee)
    patch_exchange(mocker)

    raw_candles_1m = generate_test_data("1m", 1000, "2022-01-03 12:00:00+00:00")
    raw_candles = ohlcv_fill_up_missing_data(raw_candles_1m, "5m", "dummy")

    pairs = ["ADA/USDT", "DASH/USDT", "ETH/USDT", "LTC/USDT", "NXT/USDT"]
    data = {pair: raw_candles for pair in pairs}
    detail_data = {pair: raw_candles_1m for pair in pairs}

    # Only use 500 lines to increase performance
    data = trim_dictlist(data, -200)

    # Remove data for one pair from the beginning of the data
    if tres > 0:
        data[pair] = data[pair][tres:].reset_index()
    default_conf_usdt["timeframe"] = "5m"
    default_conf_usdt["max_open_trades"] = 3

    backtesting = Backtesting(default_conf_usdt)
    vr_spy = mocker.spy(backtesting, "validate_row")
    bl_spy = mocker.spy(backtesting, "backtest_loop")
    backtesting.detail_data = detail_data
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.bot_loop_start = MagicMock()
    backtesting.strategy.advise_entry = _trend_alternate_hold  # Override
    backtesting.strategy.advise_exit = _trend_alternate_hold  # Override

    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)

    backtest_conf = {
        "processed": deepcopy(processed),
        "start_date": min_date,
        "end_date": max_date,
    }

    results = backtesting.backtest(**backtest_conf)

    # bot_loop_start is called once per candle.
    assert backtesting.strategy.bot_loop_start.call_count == 199
    # Validated row once per candle and pair
    assert vr_spy.call_count == 995

    if use_detail:
        # Backtest loop is called once per candle per pair
        # Exact numbers depend on trade state - but should be around 3_800
        assert bl_spy.call_count > 1_220
        assert bl_spy.call_count < 1_300
    else:
        assert bl_spy.call_count < 995

    # Make sure we have parallel trades
    assert len(evaluate_result_multi(results["results"], "5m", 2)) > 0
    # make sure we don't have trades with more than configured max_open_trades
    assert len(evaluate_result_multi(results["results"], "5m", 3)) == 0

    # Cached data correctly removed amounts
    removed_candles = len(data[pair]) - 1
    assert len(backtesting.dataprovider.get_analyzed_dataframe(pair, "5m")[0]) == removed_candles
    assert (
        len(backtesting.dataprovider.get_analyzed_dataframe("NXT/USDT", "5m")[0])
        == len(data["NXT/USDT"]) - 1
    )

    backtesting.strategy.max_open_trades = 1
    backtesting.config.update({"max_open_trades": 1})
    backtest_conf = {
        "processed": deepcopy(processed),
        "start_date": min_date,
        "end_date": max_date,
    }
    results = backtesting.backtest(**backtest_conf)
    assert len(evaluate_result_multi(results["results"], "5m", 1)) == 0


@pytest.mark.parametrize("use_detail", [True, False])
@pytest.mark.parametrize("pair", ["ADA/USDT", "LTC/USDT"])
@pytest.mark.parametrize("tres", [0, 20, 30])
def test_backtest_multi_pair_detail_simplified(
    default_conf_usdt,
    fee,
    mocker,
    tres,
    pair,
    use_detail,
):
    """
    literally the same as test_backtest_multi_pair_detail
    but with an "always enter" strategy, exiting after about half of the candle duration.
    """

    def _always_buy(dataframe, metadata):
        """
        Buy every xth candle - sell every other xth -2 (hold on to pairs a bit)
        """
        dataframe["enter_long"] = 1
        dataframe["enter_short"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def custom_exit(
        trade: Trade,
        current_time: datetime,
        **kwargs,
    ) -> str | bool | None:
        # Exit within the same candle.
        if (trade.open_date_utc + timedelta(minutes=20)) < current_time:
            return "exit after 20 minutes"

    default_conf_usdt.update(
        {
            "runmode": "backtest",
            "stoploss": -1.0,
            "minimal_roi": {"0": 100},
        }
    )

    if use_detail:
        default_conf_usdt["timeframe_detail"] = "5m"

    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_fee", fee)
    patch_exchange(mocker)

    raw_candles_5m = generate_test_data("5m", 1000, "2022-01-03 12:00:00+00:00")
    raw_candles = ohlcv_fill_up_missing_data(raw_candles_5m, "1h", "dummy")

    pairs = ["ADA/USDT", "DASH/USDT", "ETH/USDT", "LTC/USDT", "NXT/USDT"]
    data = {pair: raw_candles for pair in pairs}
    detail_data = {pair: raw_candles_5m for pair in pairs}

    # Only use 500 lines to increase performance
    data = trim_dictlist(data, -200)

    # Remove data for one pair from the beginning of the data
    if tres > 0:
        data[pair] = data[pair][tres:].reset_index()
    default_conf_usdt["timeframe"] = "1h"
    default_conf_usdt["max_open_trades"] = 3

    backtesting = Backtesting(default_conf_usdt)
    vr_spy = mocker.spy(backtesting, "validate_row")
    bl_spy = mocker.spy(backtesting, "backtest_loop")
    backtesting.detail_data = detail_data
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.bot_loop_start = MagicMock()
    backtesting.strategy.advise_entry = _always_buy  # Override
    backtesting.strategy.advise_exit = _always_buy  # Override
    backtesting.strategy.custom_exit = custom_exit  # Override

    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)

    backtest_conf = {
        "processed": deepcopy(processed),
        "start_date": min_date,
        "end_date": max_date,
    }

    results = backtesting.backtest(**backtest_conf)

    # bot_loop_start is called once per candle.
    # assert backtesting.strategy.bot_loop_start.call_count == 83
    # Validated row once per candle and pair
    assert vr_spy.call_count == 415

    if use_detail:
        # Backtest loop is called once per candle per pair
        # Exact numbers depend on trade state - but should be around 2_600
        assert bl_spy.call_count > 2_159
        assert bl_spy.call_count < 2_800
        assert len(evaluate_result_multi(results["results"], "1h", 3)) > 0
    else:
        assert bl_spy.call_count < 995
        assert len(evaluate_result_multi(results["results"], "1h", 3)) == 0

    # Make sure we have parallel trades
    assert len(evaluate_result_multi(results["results"], "1h", 2)) > 0
    assert len(evaluate_result_multi(results["results"], "5m", 2)) > 0
    # make sure we don't have trades with more than configured max_open_trades
    # This must evaluate on detail timeframe - as we can have entries within the candle.
    assert len(evaluate_result_multi(results["results"], "5m", 3)) == 0
    assert len(evaluate_result_multi(results["results"], "1m", 3)) == 0

    # # Cached data correctly removed amounts
    offset = 1
    removed_candles = len(data[pair]) - offset
    assert len(backtesting.dataprovider.get_analyzed_dataframe(pair, "1h")[0]) == removed_candles
    assert (
        len(backtesting.dataprovider.get_analyzed_dataframe("NXT/USDT", "1h")[0])
        == len(data["NXT/USDT"]) - 1
    )

    backtesting.strategy.max_open_trades = 1
    backtesting.config.update({"max_open_trades": 1})
    backtest_conf = {
        "processed": deepcopy(processed),
        "start_date": min_date,
        "end_date": max_date,
    }
    results = backtesting.backtest(**backtest_conf)
    if use_detail:
        assert len(evaluate_result_multi(results["results"], "1h", 1)) > 0
    else:
        assert len(evaluate_result_multi(results["results"], "1h", 1)) == 0
    assert len(evaluate_result_multi(results["results"], "5m", 1)) == 0
    assert len(evaluate_result_multi(results["results"], "1m", 1)) == 0


@pytest.mark.parametrize("use_detail", [True, False])
def test_backtest_multi_pair_long_short_switch(
    default_conf_usdt,
    fee,
    mocker,
    use_detail,
):
    """
    literally the same as test_backtest_multi_pair - but with artificial data
    and detail timeframe.
    """

    def _trend_alternate_hold(dataframe=None, metadata=None):
        """
        Buy every xth candle - sell every other xth -2 (hold on to pairs a bit)
        """
        if metadata["pair"] in ("ETH/USDT", "LTC/USDT"):
            multi = 20
        else:
            multi = 18
        dataframe["enter_long"] = np.where(dataframe.index % multi == 0, 1, 0)
        dataframe["exit_long"] = np.where((dataframe.index + multi - 2) % multi == 0, 1, 0)
        dataframe["enter_short"] = dataframe["exit_long"]
        dataframe["exit_short"] = dataframe["enter_long"]
        return dataframe

    default_conf_usdt.update(
        {
            "runmode": "backtest",
            "timeframe": "5m",
            "max_open_trades": 1,
            "stoploss": -1.0,
            "minimal_roi": {"0": 100},
            "margin_mode": "isolated",
            "trading_mode": "futures",
        }
    )

    if use_detail:
        default_conf_usdt["timeframe_detail"] = "1m"

    mocker.patch(
        "freqtrade.optimize.backtesting.price_to_precision", lambda price, *args, **kwargs: price
    )
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=0.00001)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_fee", fee)
    patch_exchange(mocker)

    raw_candles_1m = generate_test_data("1m", 2500, "2022-01-03 12:00:00+00:00")
    raw_candles = ohlcv_fill_up_missing_data(raw_candles_1m, "5m", "dummy")

    pairs = [
        "ETH/USDT:USDT",
    ]
    default_conf_usdt["exchange"]["pair_whitelist"] = pairs
    # Fake whitelist to avoid some mock data issues
    mocker.patch(f"{EXMS}.get_maintenance_ratio_and_amt", return_value=(0.01, 0.01))

    data = {pair: raw_candles for pair in pairs}
    detail_data = {pair: raw_candles_1m for pair in pairs}

    # Only use 500 lines to increase performance
    data = trim_dictlist(data, -500)

    backtesting = Backtesting(default_conf_usdt)
    vr_spy = mocker.spy(backtesting, "validate_row")
    backtesting.detail_data = detail_data
    backtesting.funding_fee_timeframe_secs = 3600 * 8  # 8h
    backtesting.futures_data = {pair: pd.DataFrame() for pair in pairs}

    backtesting.strategylist[0].can_short = True
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.bot_loop_start = MagicMock()
    backtesting.strategy.advise_entry = _trend_alternate_hold  # Override
    backtesting.strategy.advise_exit = _trend_alternate_hold  # Override

    processed = backtesting.strategy.advise_all_indicators(data)
    min_date, max_date = get_timerange(processed)

    backtest_conf = {
        "processed": deepcopy(processed),
        "start_date": min_date,
        "end_date": max_date,
    }

    results = backtesting.backtest(**backtest_conf)

    # bot_loop_start is called once per candle.
    assert backtesting.strategy.bot_loop_start.call_count == 499
    # Validated row once per candle and pair
    assert vr_spy.call_count == 499

    # Make sure we have parallel trades
    assert len(evaluate_result_multi(results["results"], "5m", 0)) > 0
    # make sure we don't have trades with more than configured max_open_trades
    assert len(evaluate_result_multi(results["results"], "5m", 1)) == 0

    # Expect 26 results initially
    assert len(results["results"]) == 53


def test_backtest_start_timerange(default_conf, mocker, caplog, testdatadir):
    patch_exchange(mocker)
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest")
    mocker.patch("freqtrade.optimize.backtesting.generate_backtest_stats")
    mocker.patch("freqtrade.optimize.backtesting.show_backtest_results")
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["UNITTEST/BTC"]),
    )
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--strategy",
        CURRENT_TEST_STRATEGY,
        "--datadir",
        str(testdatadir),
        "--timeframe",
        "1m",
        "--timerange",
        "1510694220-1510700340",
        "--enable-position-stacking",
    ]
    args = get_args(args)
    start_backtesting(args)
    # check the logs, that will contain the backtest result
    exists = [
        "Parameter -i/--timeframe detected ... Using timeframe: 1m ...",
        "Parameter --timerange detected: 1510694220-1510700340 ...",
        f"Using data directory: {testdatadir} ...",
        "Loading data from 2017-11-14 20:57:00 up to 2017-11-14 22:59:00 (0 days).",
        "Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).",
        "Parameter --enable-position-stacking detected ...",
    ]

    for line in exists:
        assert log_has(line, caplog)


@pytest.mark.filterwarnings("ignore:deprecated")
def test_backtest_start_multi_strat(default_conf, mocker, caplog, testdatadir):
    default_conf.update(
        {
            "use_exit_signal": True,
            "exit_profit_only": False,
            "exit_profit_offset": 0.0,
            "ignore_roi_if_entry_signal": False,
        }
    )
    patch_exchange(mocker)
    backtestmock = MagicMock(
        return_value={
            "results": pd.DataFrame(columns=BT_DATA_COLUMNS),
            "config": default_conf,
            "locks": [],
            "rejected_signals": 20,
            "timedout_entry_orders": 0,
            "timedout_exit_orders": 0,
            "canceled_trade_entries": 0,
            "canceled_entry_orders": 0,
            "replaced_entry_orders": 0,
            "final_balance": 1000,
        }
    )
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["UNITTEST/BTC"]),
    )
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest", backtestmock)
    text_table_mock = MagicMock()
    tag_metrics_mock = MagicMock()
    strattable_mock = MagicMock()
    strat_summary = MagicMock()

    mocker.patch.multiple(
        "freqtrade.optimize.optimize_reports.bt_output",
        text_table_bt_results=text_table_mock,
        text_table_strategy=strattable_mock,
    )
    mocker.patch.multiple(
        "freqtrade.optimize.optimize_reports.optimize_reports",
        generate_pair_metrics=MagicMock(),
        generate_tag_metrics=tag_metrics_mock,
        generate_strategy_comparison=strat_summary,
        generate_daily_stats=MagicMock(),
    )
    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--strategy-path",
        str(Path(__file__).parents[1] / "strategy/strats"),
        "--timeframe",
        "1m",
        "--timerange",
        "1510694220-1510700340",
        "--enable-position-stacking",
        "--strategy-list",
        CURRENT_TEST_STRATEGY,
        "StrategyTestV2",
    ]
    args = get_args(args)
    start_backtesting(args)
    # 2 backtests, 6 tables (entry, exit, mixed - each 2x)
    assert backtestmock.call_count == 2
    assert text_table_mock.call_count == 4
    assert strattable_mock.call_count == 1
    assert tag_metrics_mock.call_count == 6
    assert strat_summary.call_count == 1

    # check the logs, that will contain the backtest result
    exists = [
        "Parameter -i/--timeframe detected ... Using timeframe: 1m ...",
        "Parameter --timerange detected: 1510694220-1510700340 ...",
        f"Using data directory: {testdatadir} ...",
        "Loading data from 2017-11-14 20:57:00 up to 2017-11-14 22:59:00 (0 days).",
        "Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).",
        "Parameter --enable-position-stacking detected ...",
        f"Running backtesting for Strategy {CURRENT_TEST_STRATEGY}",
        "Running backtesting for Strategy StrategyTestV2",
    ]

    for line in exists:
        assert log_has(line, caplog)


def test_backtest_start_multi_strat_nomock(default_conf, mocker, caplog, testdatadir, capsys):
    default_conf.update(
        {
            "use_exit_signal": True,
            "exit_profit_only": False,
            "exit_profit_offset": 0.0,
            "ignore_roi_if_entry_signal": False,
        }
    )
    patch_exchange(mocker)
    result1 = pd.DataFrame(
        {
            "pair": ["XRP/BTC", "LTC/BTC"],
            "profit_ratio": [0.0, 0.0],
            "profit_abs": [0.0, 0.0],
            "open_date": pd.to_datetime(
                [
                    "2018-01-29 18:40:00",
                    "2018-01-30 03:30:00",
                ],
                utc=True,
            ),
            "close_date": pd.to_datetime(
                [
                    "2018-01-29 20:45:00",
                    "2018-01-30 05:35:00",
                ],
                utc=True,
            ),
            "trade_duration": [235, 40],
            "is_open": [False, False],
            "stake_amount": [0.01, 0.01],
            "open_rate": [0.104445, 0.10302485],
            "close_rate": [0.104969, 0.103541],
            "is_short": [False, False],
            "exit_reason": [ExitType.ROI.value, ExitType.ROI.value],
        }
    )
    result2 = pd.DataFrame(
        {
            "pair": ["XRP/BTC", "LTC/BTC", "ETH/BTC"],
            "profit_ratio": [0.03, 0.01, 0.1],
            "profit_abs": [0.01, 0.02, 0.2],
            "open_date": pd.to_datetime(
                ["2018-01-29 18:40:00", "2018-01-30 03:30:00", "2018-01-30 05:30:00"], utc=True
            ),
            "close_date": pd.to_datetime(
                ["2018-01-29 20:45:00", "2018-01-30 05:35:00", "2018-01-30 08:30:00"], utc=True
            ),
            "trade_duration": [47, 40, 20],
            "is_open": [False, False, False],
            "stake_amount": [0.01, 0.01, 0.01],
            "open_rate": [0.104445, 0.10302485, 0.122541],
            "close_rate": [0.104969, 0.103541, 0.123541],
            "is_short": [False, False, False],
            "exit_reason": [ExitType.ROI.value, ExitType.ROI.value, ExitType.STOP_LOSS.value],
        }
    )
    backtestmock = MagicMock(
        side_effect=[
            {
                "results": result1,
                "config": default_conf,
                "locks": [],
                "rejected_signals": 20,
                "timedout_entry_orders": 0,
                "timedout_exit_orders": 0,
                "canceled_trade_entries": 0,
                "canceled_entry_orders": 0,
                "replaced_entry_orders": 0,
                "final_balance": 1000,
            },
            {
                "results": result2,
                "config": default_conf,
                "locks": [],
                "rejected_signals": 20,
                "timedout_entry_orders": 0,
                "timedout_exit_orders": 0,
                "canceled_trade_entries": 0,
                "canceled_entry_orders": 0,
                "replaced_entry_orders": 0,
                "final_balance": 1000,
            },
        ]
    )
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["UNITTEST/BTC"]),
    )
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest", backtestmock)

    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--strategy-path",
        str(Path(__file__).parents[1] / "strategy/strats"),
        "--timeframe",
        "1m",
        "--timerange",
        "1510694220-1510700340",
        "--enable-position-stacking",
        "--breakdown",
        "day",
        "--strategy-list",
        CURRENT_TEST_STRATEGY,
        "StrategyTestV2",
    ]
    args = get_args(args)
    start_backtesting(args)

    # check the logs, that will contain the backtest result
    exists = [
        "Parameter -i/--timeframe detected ... Using timeframe: 1m ...",
        "Parameter --timerange detected: 1510694220-1510700340 ...",
        f"Using data directory: {testdatadir} ...",
        "Loading data from 2017-11-14 20:57:00 up to 2017-11-14 22:59:00 (0 days).",
        "Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).",
        "Parameter --enable-position-stacking detected ...",
        f"Running backtesting for Strategy {CURRENT_TEST_STRATEGY}",
        "Running backtesting for Strategy StrategyTestV2",
    ]

    for line in exists:
        assert log_has(line, caplog)

    captured = capsys.readouterr()
    assert "BACKTESTING REPORT" in captured.out
    assert "EXIT REASON STATS" in captured.out
    assert "DAY BREAKDOWN" in captured.out
    assert "LEFT OPEN TRADES REPORT" in captured.out
    assert "2017-11-14 21:17:00 -> 2017-11-14 22:59:00 | Max open trades : 1" in captured.out
    assert "STRATEGY SUMMARY" in captured.out


@pytest.mark.filterwarnings("ignore:deprecated")
def test_backtest_start_futures_noliq(default_conf_usdt, mocker, caplog, testdatadir, capsys):
    # Tests detail-data loading
    default_conf_usdt.update(
        {
            "trading_mode": "futures",
            "margin_mode": "isolated",
            "use_exit_signal": True,
            "exit_profit_only": False,
            "exit_profit_offset": 0.0,
            "ignore_roi_if_entry_signal": False,
            "strategy": CURRENT_TEST_STRATEGY,
        }
    )
    patch_exchange(mocker)

    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["HULUMULU/USDT", "XRP/USDT:USDT"]),
    )
    # mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)

    patched_configuration_load_config_file(mocker, default_conf_usdt)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--strategy-path",
        str(Path(__file__).parents[1] / "strategy/strats"),
        "--timeframe",
        "1h",
    ]
    args = get_args(args)
    with pytest.raises(OperationalException, match=r"Pairs .* got no leverage tiers available\."):
        start_backtesting(args)


@pytest.mark.filterwarnings("ignore:deprecated")
def test_backtest_start_nomock_futures(default_conf_usdt, mocker, caplog, testdatadir, capsys):
    # Tests detail-data loading
    default_conf_usdt.update(
        {
            "trading_mode": "futures",
            "margin_mode": "isolated",
            "use_exit_signal": True,
            "exit_profit_only": False,
            "exit_profit_offset": 0.0,
            "ignore_roi_if_entry_signal": False,
            "strategy": CURRENT_TEST_STRATEGY,
        }
    )
    patch_exchange(mocker)
    result1 = pd.DataFrame(
        {
            "pair": ["XRP/USDT:USDT", "XRP/USDT:USDT"],
            "profit_ratio": [0.0, 0.0],
            "profit_abs": [0.0, 0.0],
            "open_date": pd.to_datetime(
                [
                    "2021-11-18 18:00:00",
                    "2021-11-18 03:00:00",
                ],
                utc=True,
            ),
            "close_date": pd.to_datetime(
                [
                    "2021-11-18 20:00:00",
                    "2021-11-18 05:00:00",
                ],
                utc=True,
            ),
            "trade_duration": [235, 40],
            "is_open": [False, False],
            "is_short": [False, False],
            "stake_amount": [0.01, 0.01],
            "open_rate": [0.104445, 0.10302485],
            "close_rate": [0.104969, 0.103541],
            "exit_reason": [ExitType.ROI, ExitType.ROI],
        }
    )
    result2 = pd.DataFrame(
        {
            "pair": ["XRP/USDT:USDT", "XRP/USDT:USDT", "XRP/USDT:USDT"],
            "profit_ratio": [0.03, 0.01, 0.1],
            "profit_abs": [0.01, 0.02, 0.2],
            "open_date": pd.to_datetime(
                ["2021-11-19 18:00:00", "2021-11-19 03:00:00", "2021-11-19 05:00:00"], utc=True
            ),
            "close_date": pd.to_datetime(
                ["2021-11-19 20:00:00", "2021-11-19 05:00:00", "2021-11-19 08:00:00"], utc=True
            ),
            "trade_duration": [47, 40, 20],
            "is_open": [False, False, False],
            "is_short": [False, False, False],
            "stake_amount": [0.01, 0.01, 0.01],
            "open_rate": [0.104445, 0.10302485, 0.122541],
            "close_rate": [0.104969, 0.103541, 0.123541],
            "exit_reason": [ExitType.ROI, ExitType.ROI, ExitType.STOP_LOSS],
        }
    )
    backtestmock = MagicMock(
        side_effect=[
            {
                "results": result1,
                "config": default_conf_usdt,
                "locks": [],
                "rejected_signals": 20,
                "timedout_entry_orders": 0,
                "timedout_exit_orders": 0,
                "canceled_trade_entries": 0,
                "canceled_entry_orders": 0,
                "replaced_entry_orders": 0,
                "final_balance": 1000,
            },
            {
                "results": result2,
                "config": default_conf_usdt,
                "locks": [],
                "rejected_signals": 20,
                "timedout_entry_orders": 0,
                "timedout_exit_orders": 0,
                "canceled_trade_entries": 0,
                "canceled_entry_orders": 0,
                "replaced_entry_orders": 0,
                "final_balance": 1000,
            },
        ]
    )
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["XRP/USDT:USDT"]),
    )
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest", backtestmock)

    patched_configuration_load_config_file(mocker, default_conf_usdt)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--strategy-path",
        str(Path(__file__).parents[1] / "strategy/strats"),
        "--timeframe",
        "1h",
    ]
    args = get_args(args)
    start_backtesting(args)

    # check the logs, that will contain the backtest result
    exists = [
        "Parameter -i/--timeframe detected ... Using timeframe: 1h ...",
        f"Using data directory: {testdatadir} ...",
        "Loading data from 2021-11-17 01:00:00 up to 2021-11-21 04:00:00 (4 days).",
        "Backtesting with data from 2021-11-17 21:00:00 up to 2021-11-21 04:00:00 (3 days).",
        "XRP/USDT:USDT, funding_rate, 1h, data starts at 2021-11-18 00:00:00",
        f"Running backtesting for Strategy {CURRENT_TEST_STRATEGY}",
    ]

    for line in exists:
        assert log_has(line, caplog), line

    captured = capsys.readouterr()
    assert "BACKTESTING REPORT" in captured.out
    assert "EXIT REASON STATS" in captured.out
    assert "LEFT OPEN TRADES REPORT" in captured.out


@pytest.mark.filterwarnings("ignore:deprecated")
def test_backtest_start_multi_strat_nomock_detail(
    default_conf, mocker, caplog, testdatadir, capsys
):
    # Tests detail-data loading
    default_conf.update(
        {
            "use_exit_signal": True,
            "exit_profit_only": False,
            "exit_profit_offset": 0.0,
            "ignore_roi_if_entry_signal": False,
        }
    )
    patch_exchange(mocker)
    result1 = pd.DataFrame(
        {
            "pair": ["XRP/BTC", "LTC/BTC"],
            "profit_ratio": [0.0, 0.0],
            "profit_abs": [0.0, 0.0],
            "open_date": pd.to_datetime(
                [
                    "2018-01-29 18:40:00",
                    "2018-01-30 03:30:00",
                ],
                utc=True,
            ),
            "close_date": pd.to_datetime(
                [
                    "2018-01-29 20:45:00",
                    "2018-01-30 05:35:00",
                ],
                utc=True,
            ),
            "trade_duration": [235, 40],
            "is_open": [False, False],
            "is_short": [False, False],
            "stake_amount": [0.01, 0.01],
            "open_rate": [0.104445, 0.10302485],
            "close_rate": [0.104969, 0.103541],
            "exit_reason": [ExitType.ROI, ExitType.ROI],
        }
    )
    result2 = pd.DataFrame(
        {
            "pair": ["XRP/BTC", "LTC/BTC", "ETH/BTC"],
            "profit_ratio": [0.03, 0.01, 0.1],
            "profit_abs": [0.01, 0.02, 0.2],
            "open_date": pd.to_datetime(
                ["2018-01-29 18:40:00", "2018-01-30 03:30:00", "2018-01-30 05:30:00"], utc=True
            ),
            "close_date": pd.to_datetime(
                ["2018-01-29 20:45:00", "2018-01-30 05:35:00", "2018-01-30 08:30:00"], utc=True
            ),
            "trade_duration": [47, 40, 20],
            "is_open": [False, False, False],
            "is_short": [False, False, False],
            "stake_amount": [0.01, 0.01, 0.01],
            "open_rate": [0.104445, 0.10302485, 0.122541],
            "close_rate": [0.104969, 0.103541, 0.123541],
            "exit_reason": [ExitType.ROI, ExitType.ROI, ExitType.STOP_LOSS],
        }
    )
    backtestmock = MagicMock(
        side_effect=[
            {
                "results": result1,
                "config": default_conf,
                "locks": [],
                "rejected_signals": 20,
                "timedout_entry_orders": 0,
                "timedout_exit_orders": 0,
                "canceled_trade_entries": 0,
                "canceled_entry_orders": 0,
                "replaced_entry_orders": 0,
                "final_balance": 1000,
            },
            {
                "results": result2,
                "config": default_conf,
                "locks": [],
                "rejected_signals": 20,
                "timedout_entry_orders": 0,
                "timedout_exit_orders": 0,
                "canceled_trade_entries": 0,
                "canceled_entry_orders": 0,
                "replaced_entry_orders": 0,
                "final_balance": 1000,
            },
        ]
    )
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["XRP/ETH"]),
    )
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest", backtestmock)

    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--strategy-path",
        str(Path(__file__).parents[1] / "strategy/strats"),
        "--timeframe",
        "5m",
        "--timeframe-detail",
        "1m",
        "--strategy-list",
        CURRENT_TEST_STRATEGY,
    ]
    args = get_args(args)
    start_backtesting(args)

    # check the logs, that will contain the backtest result
    exists = [
        "Parameter -i/--timeframe detected ... Using timeframe: 5m ...",
        "Parameter --timeframe-detail detected, using 1m for intra-candle backtesting ...",
        f"Using data directory: {testdatadir} ...",
        "Loading data from 2019-10-11 00:00:00 up to 2019-10-13 11:15:00 (2 days).",
        "Backtesting with data from 2019-10-11 01:40:00 up to 2019-10-13 11:15:00 (2 days).",
        f"Running backtesting for Strategy {CURRENT_TEST_STRATEGY}",
    ]

    for line in exists:
        assert log_has(line, caplog)

    captured = capsys.readouterr()
    assert "BACKTESTING REPORT" in captured.out
    assert "EXIT REASON STATS" in captured.out
    assert "LEFT OPEN TRADES REPORT" in captured.out


@pytest.mark.filterwarnings("ignore:deprecated")
@pytest.mark.parametrize("run_id", ["2", "changed"])
@pytest.mark.parametrize("start_delta", [{"days": 0}, {"days": 1}, {"weeks": 1}, {"weeks": 4}])
@pytest.mark.parametrize("cache", constants.BACKTEST_CACHE_AGE)
def test_backtest_start_multi_strat_caching(
    default_conf, mocker, caplog, testdatadir, run_id, start_delta, cache
):
    default_conf.update(
        {
            "use_exit_signal": True,
            "exit_profit_only": False,
            "exit_profit_offset": 0.0,
            "ignore_roi_if_entry_signal": False,
        }
    )
    patch_exchange(mocker)
    backtestmock = MagicMock(
        return_value={
            "results": pd.DataFrame(columns=BT_DATA_COLUMNS),
            "config": default_conf,
            "locks": [],
            "rejected_signals": 20,
            "timedout_entry_orders": 0,
            "timedout_exit_orders": 0,
            "canceled_trade_entries": 0,
            "canceled_entry_orders": 0,
            "replaced_entry_orders": 0,
            "final_balance": 1000,
        }
    )
    mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["UNITTEST/BTC"]),
    )
    mocker.patch("freqtrade.optimize.backtesting.Backtesting.backtest", backtestmock)
    mocker.patch("freqtrade.optimize.backtesting.show_backtest_results", MagicMock())

    now = min_backtest_date = datetime.now(tz=UTC)
    start_time = now - timedelta(**start_delta) + timedelta(hours=1)
    if cache == "none":
        min_backtest_date = now + timedelta(days=1)
    elif cache == "day":
        min_backtest_date = now - timedelta(days=1)
    elif cache == "week":
        min_backtest_date = now - timedelta(weeks=1)
    elif cache == "month":
        min_backtest_date = now - timedelta(weeks=4)
    load_backtest_metadata = MagicMock(
        return_value={
            "StrategyTestV2": {"run_id": "1", "backtest_start_time": now.timestamp()},
            "StrategyTestV3": {"run_id": run_id, "backtest_start_time": start_time.timestamp()},
        }
    )
    load_backtest_stats = MagicMock(
        side_effect=[
            {
                "metadata": {"StrategyTestV2": {"run_id": "1"}},
                "strategy": {"StrategyTestV2": {}},
                "strategy_comparison": [{"key": "StrategyTestV2"}],
            },
            {
                "metadata": {"StrategyTestV3": {"run_id": "2"}},
                "strategy": {"StrategyTestV3": {}},
                "strategy_comparison": [{"key": "StrategyTestV3"}],
            },
        ]
    )
    mocker.patch(
        "pathlib.Path.glob",
        return_value=[
            Path(datetime.strftime(datetime.now(), "backtest-result-%Y-%m-%d_%H-%M-%S.json"))
        ],
    )
    mocker.patch.multiple(
        "freqtrade.data.btanalysis.bt_fileutils",
        load_backtest_metadata=load_backtest_metadata,
        load_backtest_stats=load_backtest_stats,
    )
    mocker.patch("freqtrade.optimize.backtesting.get_strategy_run_id", side_effect=["1", "2", "2"])

    patched_configuration_load_config_file(mocker, default_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--strategy-path",
        str(Path(__file__).parents[1] / "strategy/strats"),
        "--timeframe",
        "1m",
        "--timerange",
        "1510694220-1510700340",
        "--enable-position-stacking",
        "--cache",
        cache,
        "--strategy-list",
        "StrategyTestV2",
        "StrategyTestV3",
    ]
    args = get_args(args)
    start_backtesting(args)

    # check the logs, that will contain the backtest result
    exists = [
        "Parameter -i/--timeframe detected ... Using timeframe: 1m ...",
        "Parameter --timerange detected: 1510694220-1510700340 ...",
        f"Using data directory: {testdatadir} ...",
        "Loading data from 2017-11-14 20:57:00 up to 2017-11-14 22:59:00 (0 days).",
        "Parameter --enable-position-stacking detected ...",
    ]

    for line in exists:
        assert log_has(line, caplog)

    if cache == "none":
        assert backtestmock.call_count == 2
        exists = [
            "Running backtesting for Strategy StrategyTestV2",
            "Running backtesting for Strategy StrategyTestV3",
            "Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).",
        ]
    elif run_id == "2" and min_backtest_date < start_time:
        assert backtestmock.call_count == 0
        exists = [
            "Reusing result of previous backtest for StrategyTestV2",
            "Reusing result of previous backtest for StrategyTestV3",
        ]
    else:
        exists = [
            "Reusing result of previous backtest for StrategyTestV2",
            "Running backtesting for Strategy StrategyTestV3",
            "Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).",
        ]
        assert backtestmock.call_count == 1

    for line in exists:
        assert log_has(line, caplog)


def test_get_strategy_run_id(default_conf_usdt):
    default_conf_usdt.update({"strategy": "StrategyTestV2", "max_open_trades": float("inf")})
    strategy = StrategyResolver.load_strategy(default_conf_usdt)
    x = get_strategy_run_id(strategy)
    assert isinstance(x, str)


def test_get_backtest_metadata_filename():
    # Test with a file path
    filename = Path("backtest_results.json")
    expected = Path("backtest_results.meta.json")
    assert get_backtest_metadata_filename(filename) == expected

    # Test with a file path with multiple dots in the name
    filename = Path("/path/to/backtest.results.json")
    expected = Path("/path/to/backtest.results.meta.json")
    assert get_backtest_metadata_filename(filename) == expected

    # Test with a file path with no parent directory
    filename = Path("backtest_results.json")
    expected = Path("backtest_results.meta.json")
    assert get_backtest_metadata_filename(filename) == expected

    # Test with a string file path
    filename = "/path/to/backtest_results.json"
    expected = Path("/path/to/backtest_results.meta.json")
    assert get_backtest_metadata_filename(filename) == expected

    # Test with a string file path with no extension
    filename = "/path/to/backtest_results"
    expected = Path("/path/to/backtest_results.meta.json")
    assert get_backtest_metadata_filename(filename) == expected

    # Test with a string file path with multiple dots in the name
    filename = "/path/to/backtest.results.json"
    expected = Path("/path/to/backtest.results.meta.json")
    assert get_backtest_metadata_filename(filename) == expected

    # Test with a string file path with no parent directory
    filename = "backtest_results.json"
    expected = Path("backtest_results.meta.json")
    assert get_backtest_metadata_filename(filename) == expected
    # Test with a string file path with no parent directory

    filename = "backtest_results_zip.zip"
    expected = Path("backtest_results_zip.meta.json")
    assert get_backtest_metadata_filename(filename) == expected


@pytest.mark.parametrize("dynamic_pairlist", [True, False])
def test_time_pair_generator_refresh_pairlist(mocker, default_conf, dynamic_pairlist):
    patch_exchange(mocker)
    default_conf["enable_dynamic_pairlist"] = dynamic_pairlist
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.dynamic_pairlist == dynamic_pairlist

    refresh_mock = mocker.patch(
        "freqtrade.plugins.pairlistmanager.PairListManager.refresh_pairlist"
    )

    # Simulate 2 candles
    start_date = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    end_date = start_date + timedelta(minutes=10)
    pairs = default_conf["exchange"]["pair_whitelist"]
    data = {pair: [] for pair in pairs}

    # Simulate backtest loop
    list(backtesting.time_pair_generator(start_date, end_date, pairs, data))

    if dynamic_pairlist:
        assert refresh_mock.call_count == 2
    else:
        assert refresh_mock.call_count == 0


@pytest.mark.parametrize("dynamic_pairlist", [True, False])
def test_time_pair_generator_open_trades_first(mocker, default_conf, dynamic_pairlist):
    patch_exchange(mocker)
    default_conf["enable_dynamic_pairlist"] = dynamic_pairlist
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.dynamic_pairlist == dynamic_pairlist

    pairs = ["XRP/BTC", "LTC/BTC", "NEO/BTC", "ETH/BTC"]

    # Simulate open trades
    trades = [
        LocalTrade(pair="XRP/BTC", open_date=dt_now(), amount=1, open_rate=1),
        LocalTrade(pair="NEO/BTC", open_date=dt_now(), amount=1, open_rate=1),
    ]
    LocalTrade.bt_trades_open = trades
    LocalTrade.bt_trades_open_pp = {
        "XRP/BTC": [trades[0]],
        "NEO/BTC": [trades[1]],
        "LTC/BTC": [],
        "ETH/BTC": [],
    }

    start_date = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    end_date = start_date + timedelta(minutes=5)
    dummy_row = (end_date, 1.0, 1.1, 0.9, 1.0, 0, 0, 0, 0, None, None, 1000.0)
    data = {pair: [dummy_row] for pair in pairs}

    def mock_refresh(self, **kwargs):
        # Simulate shuffle
        self._whitelist = pairs[::-1]  # ['ETH/BTC', 'NEO/BTC', 'LTC/BTC', 'XRP/BTC']

    mocker.patch("freqtrade.plugins.pairlistmanager.PairListManager.refresh_pairlist", mock_refresh)

    processed_pairs = []
    for _, pair, _, _, _ in backtesting.time_pair_generator(start_date, end_date, pairs, data):
        processed_pairs.append(pair)

    # Open trades first in both cases
    if dynamic_pairlist:
        assert processed_pairs == ["XRP/BTC", "NEO/BTC", "ETH/BTC", "LTC/BTC"]
    else:
        assert processed_pairs == ["XRP/BTC", "NEO/BTC", "LTC/BTC", "ETH/BTC"]


def test_time_pair_generator_dynamic_pairlist_no_stale_replay(mocker, default_conf):
    patch_exchange(mocker)
    default_conf["enable_dynamic_pairlist"] = True
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])

    n = 10
    tf = backtesting.timeframe_td
    start = datetime(2025, 1, 1, tzinfo=UTC)
    # data[pair][0] is dated start + tf
    candles = generate_test_data_raw(backtesting.timeframe, n, start=start)
    rows = [
        [start + tf * (i + 1), c[1], c[2], c[3], c[4], 0, 0, 0, 0, None, None, 1000.0]
        for i, c in enumerate(candles)
    ]
    pairs = ["A/BTC", "B/BTC"]
    data = {p: list(rows) for p in pairs}

    # B/BTC absent for candles 3-5, then re-enters
    absent = {3, 4, 5}
    calls = {"n": -1}

    def mock_refresh(self, **kwargs):
        calls["n"] += 1
        self._whitelist = ["A/BTC"] if calls["n"] in absent else ["A/BTC", "B/BTC"]

    mocker.patch("freqtrade.plugins.pairlistmanager.PairListManager.refresh_pairlist", mock_refresh)

    end = start + tf * n
    a_rows, b_rows = [], []
    for current_time, pair, r, _, _ in backtesting.time_pair_generator(start, end, pairs, data):
        (a_rows if pair == "A/BTC" else b_rows).append((current_time, r[DATE_IDX]))

    # Every row is dated at its candle, and B resumes with no replay on re-entry
    assert all(ct == d for ct, d in a_rows)
    assert len(a_rows) == n
    b_present = [a for a in a_rows if a[0] not in {start + tf * (i + 1) for i in absent}]
    assert b_rows == b_present


def test_time_pair_generator_dynamic_pairlist_real_filters(mocker, default_conf):
    patch_exchange(mocker)
    pairs = ["ETH/BTC", "LTC/BTC", "XRP/BTC", "NEO/BTC", "ADA/BTC", "DASH/BTC"]
    default_conf["exchange"]["pair_whitelist"] = pairs
    default_conf["enable_dynamic_pairlist"] = True
    # Rotates the active 2 pairs each candle, so pairs leave and re-enter (seeded)
    default_conf["pairlists"] = [
        {"method": "StaticPairList"},
        {"method": "ShuffleFilter", "seed": 42, "shuffle_frequency": "iteration"},
        {"method": "OffsetFilter", "number_assets": 2},
    ]
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.available_pairs = list(pairs)

    n = 20
    tf = backtesting.timeframe_td
    start = datetime(2025, 1, 1, tzinfo=UTC)
    grid = [start + tf * (i + 1) for i in range(n)]

    def row(t):
        return [t, 1.0, 1.1, 0.9, 1.0, 0, 0, 0, 0, None, None, 1000.0]

    data = {p: [row(t) for t in grid] for p in pairs}
    end = start + tf * n

    present_at: dict = {}
    for current_time, pair, r, _, _ in backtesting.time_pair_generator(start, end, pairs, data):
        assert r[DATE_IDX] == current_time
        present_at.setdefault(pair, []).append(current_time)

    # Confirm the rotation drops and re-adds a pair, so re-entry was exercised
    candles = sorted({c for cs in present_at.values() for c in cs})
    reentered = [
        p
        for p, cs in present_at.items()
        if any(c not in cs for c in candles[candles.index(cs[0]) : candles.index(cs[-1])])
    ]
    assert reentered
