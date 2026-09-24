from datetime import timedelta

import pandas as pd
import pytest

from freqtrade.enums import CandleType, RunMode
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import LocalTrade
from tests.conftest import EXMS, patch_exchange


def _at(clock: str) -> pd.Timestamp:
    return pd.Timestamp(f"2024-01-01 {clock}", tz="UTC")


def _backtesting(mocker, config, pairs, stake=1000):
    config.update(
        {
            "runmode": RunMode.BACKTEST,
            "max_open_trades": len(pairs),
            "trading_mode": "futures",
            "margin_mode": "isolated",
            "candle_type_def": CandleType.FUTURES,
            "timeframe": "5m",
            "startup_candle_count": 0,
            "stake_amount": stake,
            "stake_currency": "USD",
            "dry_run_wallet": 3000,
            "tradable_balance_ratio": 1,
            "minimal_roi": {"0": 1},
            "stoploss": -0.5,
            "fee": 0,
            "unfilledtimeout": {"entry": 60, "exit": 60},
        }
    )
    config["exchange"] = {
        "name": "krakenfutures",
        "enable_ws": False,
        "pair_whitelist": pairs,
        "pair_blacklist": [],
    }
    markets = {
        pair: {
            "id": f"PF_{pair.split('/')[0]}USD",
            "symbol": pair,
            "base": pair.split("/")[0],
            "quote": "USD",
            "settle": "USD",
            "active": True,
            "swap": True,
            "linear": True,
            "type": "swap",
            "contractSize": 1,
            "precision": {"amount": 8, "price": 8},
        }
        for pair in pairs
    }
    patch_exchange(mocker, exchange="krakenfutures", mock_markets=markets)
    mocker.patch(f"{EXMS}.get_min_pair_stake_amount", return_value=1)
    mocker.patch(f"{EXMS}.get_max_pair_stake_amount", return_value=float("inf"))
    mocker.patch(f"{EXMS}.get_max_leverage", return_value=1)
    mocker.patch(f"{EXMS}.get_liquidation_price", return_value=None)

    return Backtesting(config)


@pytest.mark.parametrize("is_short", [False, True], ids=["long", "short"])
@pytest.mark.parametrize(
    "entry_submit,entry_fill,exit_submit,exit_fill,stake,adjustment,payment",
    [
        pytest.param("10:10", "10:10", "10:40", "10:40", 1000, None, 0.05, id="same-hour"),
        pytest.param("09:45", "10:00", "10:30", "10:30", 1000, None, 0.05, id="delayed-entry"),
        pytest.param("10:00", "10:00", "10:15", "10:45", 1000, None, 0.075, id="delayed-exit"),
        pytest.param("10:00", "10:00", "11:00", "11:00", 1000, 1000, 0.15, id="delayed-add"),
        pytest.param("10:00", "10:00", "11:00", "11:00", 2000, -1000, 0.15, id="reduce"),
    ],
)
def test_kraken_continuous_funding_native_backtest(
    mocker,
    default_conf_usdt,
    is_short,
    entry_submit,
    entry_fill,
    exit_submit,
    exit_fill,
    stake,
    adjustment,
    payment,
):
    """Funding follows filled exposure through the native backtest and profit accounting."""
    pair = "COIN/USD:USD"
    backtesting = _backtesting(mocker, default_conf_usdt, [pair], stake)
    strategy = backtesting.strategylist[0]
    strategy.can_short = True
    backtesting._set_strategy(strategy)
    strategy.custom_entry_price = lambda **kwargs: 100
    strategy.custom_exit_price = lambda **kwargs: 100
    side = "short" if is_short else "long"
    observed_funding = {}
    observed_after_reduction = {}

    def observe_open_trade(trade, current_time, **kwargs):
        observed_funding[current_time] = trade.funding_fees
        if adjustment == -1000 and current_time == _at("10:35"):
            profit = trade.calculate_profit(100)
            observed_after_reduction.update(
                funding=trade.funding_fees,
                realized=trade.realized_profit,
                remaining_profit=profit.profit_abs,
                total_profit=profit.total_profit,
                remaining_ratio=trade.calc_profit_ratio(100),
            )

    strategy.custom_exit = observe_open_trade

    def entry_signal(frame, metadata):
        frame["enter_long"] = 0
        frame["enter_short"] = 0
        frame.loc[frame["date"] == _at(entry_submit) - timedelta(minutes=5), f"enter_{side}"] = 1
        return frame

    def exit_signal(frame, metadata):
        frame["exit_long"] = 0
        frame["exit_short"] = 0
        frame.loc[frame["date"] == _at(exit_submit) - timedelta(minutes=5), f"exit_{side}"] = 1
        return frame

    strategy.populate_entry_trend = entry_signal
    strategy.populate_exit_trend = exit_signal
    if adjustment:
        strategy.position_adjustment_enable = True
        adjustment_submit = _at("10:15" if adjustment > 0 else "10:30")
        strategy.adjust_trade_position = lambda current_time, **kwargs: (
            adjustment if current_time == adjustment_submit else None
        )

    candles = pd.DataFrame(
        {
            "date": pd.date_range(_at("09:35"), _at("11:10"), freq="5min"),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000.0,
        }
    )
    # Orders are deliberately outside these candles until their specified fill time.
    # All actual fills remain at 100, isolating funding from price profit and loss.
    for start, end, blocked_price in [
        (_at(entry_submit), _at(entry_fill), 99 if is_short else 101),
        (_at(exit_submit), _at(exit_fill), 101 if is_short else 99),
    ]:
        candles.loc[
            candles["date"].between(start, end, inclusive="left"), ["open", "high", "low", "close"]
        ] = blocked_price
    if adjustment and adjustment > 0:
        candles.loc[
            candles["date"].between(_at("10:15"), _at("10:30"), inclusive="left"),
            ["open", "high", "low", "close"],
        ] = 99 if is_short else 101

    # Fabricated hourly quote-currency payment per base unit, independent of mark prices.
    backtesting.futures_data[pair] = pd.DataFrame(
        {"date": pd.date_range(_at("09:00"), _at("11:00"), freq="h"), "funding_rate_absolute": 0.01}
    )
    result = backtesting.backtest(
        processed={pair: candles},
        start_date=candles.iloc[0]["date"],
        end_date=candles.iloc[-1]["date"],
    )

    assert len(result["results"]) == 1
    trade = LocalTrade.bt_trades[0]
    expected_fills = [_at(entry_fill)]
    expected_amounts = [stake / 100]
    if adjustment:
        expected_fills.append(_at("10:30"))
        expected_amounts.append(abs(adjustment) / 100)
    expected_fills.append(_at(exit_fill))
    expected_amounts.append((stake + (adjustment or 0)) / 100)
    assert [order.order_filled_utc for order in trade.orders] == expected_fills
    assert [order.safe_filled for order in trade.orders] == expected_amounts
    assert all(order.safe_price == 100 for order in trade.orders)
    assert not trade.is_open

    expected_funding = payment if is_short else -payment
    # The strategy must see accrued funding on an ordinary non-settlement candle,
    # before the eventual close triggers the final funding calculation.
    first_five_minutes = (1 if is_short else -1) * (1 / 120 if stake == 1000 else 1 / 60)
    assert observed_funding[_at(entry_fill) + timedelta(minutes=5)] == pytest.approx(
        first_five_minutes
    )
    if adjustment == -1000:
        direction = 1 if is_short else -1
        assert observed_after_reduction == pytest.approx(
            {
                "funding": direction * (0.1 + 1 / 120),
                "realized": direction * 0.1,
                "remaining_profit": round(direction / 120, 8),
                "total_profit": round(direction * (0.1 + 1 / 120), 8),
                "remaining_ratio": round(direction / 120 / 1000, 8),
            }
        )
    assert trade.funding_fees == pytest.approx(expected_funding)
    assert sum(order.funding_fee or 0 for order in trade.orders) == pytest.approx(expected_funding)
    assert result["results"].iloc[0]["funding_fees"] == pytest.approx(expected_funding)
    assert result["results"].iloc[0]["profit_abs"] == pytest.approx(expected_funding)
    assert result["final_balance"] == pytest.approx(3000 + expected_funding, abs=1e-9, rel=0)


@pytest.mark.parametrize("is_short", [False, True], ids=["long", "short"])
@pytest.mark.parametrize("use_detail", [False, True], ids=["main", "detail"])
def test_kraken_funding_accrues_all_pairs_before_strategy_and_wallet_capture(
    mocker, default_conf_usdt, is_short, use_detail
):
    """The first pair's decision sees the same current funding as the other pair."""
    pairs = ["COIN/USD:USD", "TOKEN/USD:USD"]
    if use_detail:
        default_conf_usdt["timeframe_detail"] = "1m"
    backtesting = _backtesting(mocker, default_conf_usdt, pairs)
    strategy = backtesting.strategylist[0]
    strategy.can_short = True
    backtesting._set_strategy(strategy)
    strategy.position_adjustment_enable = True
    side = "short" if is_short else "long"
    loop_observations = {}
    first_pair_observations = {}

    def snapshot():
        return (
            {trade.pair: trade.funding_fees for trade in LocalTrade.bt_trades_open},
            backtesting.wallets.get_total("USD"),
        )

    def observe_loop(current_time, **kwargs):
        loop_observations[current_time] = snapshot()

    def observe_first_pair(trade, current_time, **kwargs):
        if trade.pair == pairs[0]:
            first_pair_observations[current_time] = snapshot()

    def entry_signal(frame, metadata):
        frame[f"enter_{side}"] = (frame["date"] == _at("09:55")).astype(int)
        return frame

    def exit_signal(frame, metadata):
        frame[f"exit_{side}"] = (frame["date"] == _at("10:05")).astype(int)
        return frame

    strategy.bot_loop_start = observe_loop
    strategy.adjust_trade_position = observe_first_pair
    strategy.populate_entry_trend = entry_signal
    strategy.populate_exit_trend = exit_signal
    candles = pd.DataFrame(
        {
            "date": pd.date_range(_at("09:50"), _at("10:20"), freq="1min"),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000.0,
        }
    )
    for pair, rate in zip(pairs, [0.01, 0.02], strict=True):
        backtesting.futures_data[pair] = pd.DataFrame(
            {"date": [_at("09:00"), _at("10:00")], "funding_rate_absolute": rate}
        )
        if use_detail:
            backtesting.detail_data[pair] = candles.copy()
    result = backtesting.backtest(
        processed={pair: candles.iloc[::5].copy() for pair in pairs},
        start_date=_at("09:50"),
        end_date=_at("10:20"),
    )

    assert len(result["results"]) == 2
    assert all(trade.date_entry_fill_utc == _at("10:00") for trade in LocalTrade.bt_trades)
    direction = 1 if is_short else -1
    expected_main = {pairs[0]: direction / 120, pairs[1]: direction / 60}
    funding, total = loop_observations[_at("10:05")]
    assert funding == pytest.approx(expected_main)
    assert total == pytest.approx(3000 + direction * 0.025, abs=1e-9, rel=0)
    captured = [
        total
        for date, currency, price, total in backtesting.wallet_captures
        if date == _at("10:05") and currency == "USD"
    ]
    assert captured
    assert all(
        value == pytest.approx(3000 + direction * 0.025, abs=1e-9, rel=0) for value in captured
    )

    decision_at = _at("10:01") if use_detail else _at("10:05")
    expected_decision = (
        {pairs[0]: direction / 600, pairs[1]: direction / 300} if use_detail else expected_main
    )
    funding, total = first_pair_observations[decision_at]
    assert funding == pytest.approx(expected_decision)
    assert total == pytest.approx(
        3000 + direction * (0.005 if use_detail else 0.025), abs=1e-9, rel=0
    )


@pytest.mark.parametrize("is_short", [False, True], ids=["long", "short"])
def test_kraken_detail_funding_stops_at_requested_backtest_end(mocker, default_conf_usdt, is_short):
    """An open trade closes at the requested end without reading the next funding hour."""
    pair = "COIN/USD:USD"
    default_conf_usdt["timeframe_detail"] = "1m"
    backtesting = _backtesting(mocker, default_conf_usdt, [pair])
    strategy = backtesting.strategylist[0]
    strategy.can_short = True
    backtesting._set_strategy(strategy)
    side = "short" if is_short else "long"

    def entry_signal(frame, metadata):
        frame[f"enter_{side}"] = (frame["date"] == _at("09:55")).astype(int)
        return frame

    strategy.populate_entry_trend = entry_signal
    strategy.populate_exit_trend = lambda frame, metadata: frame
    candles = pd.DataFrame(
        {
            "date": pd.date_range(_at("09:50"), _at("11:05"), freq="1min"),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000.0,
        }
    )
    backtesting.detail_data[pair] = candles.copy()
    # Coverage deliberately ends at 11:00. Reading even one minute beyond it must fail.
    backtesting.futures_data[pair] = pd.DataFrame(
        {"date": [_at("10:00")], "funding_rate_absolute": [0.01]}
    )
    main_candles = candles.loc[candles["date"] <= _at("11:00")].iloc[::5].copy()
    result = backtesting.backtest(
        processed={pair: main_candles}, start_date=_at("09:50"), end_date=_at("11:00")
    )

    assert len(result["results"]) == 1
    trade = LocalTrade.bt_trades[0]
    assert trade.date_entry_fill_utc == _at("10:00")
    assert trade.amount == 10
    assert trade.close_date_utc == _at("11:00")
    assert trade.exit_reason == "force_exit"
    assert max(date for date, *_ in backtesting.wallet_captures) == _at("11:00")
    expected_funding = 0.1 if is_short else -0.1
    assert trade.funding_fees == pytest.approx(expected_funding)
    assert result["results"].iloc[0]["profit_abs"] == pytest.approx(expected_funding)
    assert result["final_balance"] == pytest.approx(3000 + expected_funding, abs=1e-9, rel=0)
