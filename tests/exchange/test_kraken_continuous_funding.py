"""Synthetic Kraken absolute funding and public-data storage regression tests."""

from datetime import timedelta
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from freqtrade.data.converter import ohlcv_to_dataframe
from freqtrade.data.history.datahandlers import get_datahandler
from freqtrade.enums import CandleType
from freqtrade.exceptions import OperationalException
from freqtrade.util import dt_utc
from tests.conftest import get_patched_exchange


START = dt_utc(2024, 1, 1, 12)
PAIR = "BTC/USD:USD"


def rates():
    return pd.DataFrame(
        {
            "date": pd.date_range(START, periods=4, freq="h"),
            "funding_rate": [0.0001] * 4,
            "funding_rate_absolute": [0.01, -0.02, 0.03, 0.04],
        }
    )


@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize(
    "opened,closed,payment",
    [
        (15, 45, 0.05),
        (59, 61, -0.0016666666666666668),
        (0, 60, 0.1),
        (0, 0, 0),
        (30, 120, -0.15),
    ],
)
def test_continuous_hourly_payment(mocker, default_conf, short, opened, closed, payment):
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")
    frame = ex.combine_funding_and_mark(rates(), pd.DataFrame())
    actual = ex.calculate_funding_fees(
        frame, 10, short, START + timedelta(minutes=opened), START + timedelta(minutes=closed)
    )
    assert actual == pytest.approx(payment if short else -payment)


@pytest.mark.parametrize("problem", ["legacy", "gap", "nan", "duplicate", "late", "unaligned"])
def test_missing_absolute_coverage_refused(mocker, default_conf, problem):
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")
    frame = rates()
    if problem == "legacy":
        frame = frame.drop(columns="funding_rate_absolute")
    elif problem == "gap":
        frame = frame.drop(index=1)
    elif problem == "nan":
        frame.loc[1, "funding_rate_absolute"] = float("nan")
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[[1]]])
    elif problem == "late":
        frame = frame.iloc[1:]
    else:
        frame.loc[1, "date"] += timedelta(minutes=1)
    with pytest.raises(OperationalException, match="Kraken funding"):
        ex.calculate_funding_fees(frame, 10, False, START, START + timedelta(hours=2))


def test_absolute_rate_public_parse(mocker, default_conf):
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")
    ex._api_async.fetch_funding_rate_history = AsyncMock(
        return_value=[
            {
                "timestamp": int(START.timestamp() * 1000),
                "fundingRate": 0.0001,
                "info": {"fundingRate": "0.01"},
            }
        ]
    )
    raw = ex.loop.run_until_complete(ex._fetch_funding_rate_history(PAIR, "1h", 100))
    df = ohlcv_to_dataframe(
        raw,
        "1h",
        PAIR,
        candle_type=CandleType.FUNDING_RATE,
        fill_missing=False,
        drop_incomplete=False,
    )
    assert df.funding_rate.tolist() == [0.0001]
    assert df.open.tolist() == [0.0001]
    assert df.funding_rate_absolute.tolist() == [0.01]


@pytest.mark.parametrize("data_format", ["json", "jsongz", "feather", "parquet"])
def test_absolute_storage_roundtrip(tmp_path, data_format):
    handler = get_datahandler(tmp_path, data_format)
    frame = rates()
    handler.ohlcv_store(PAIR, "1h", frame, CandleType.FUNDING_RATE)
    loaded = handler.ohlcv_load(
        PAIR, "1h", candle_type=CandleType.FUNDING_RATE, fill_missing=False, drop_incomplete=False
    )
    assert loaded.funding_rate.tolist() == frame.funding_rate.tolist()
    assert loaded.funding_rate_absolute.tolist() == frame.funding_rate_absolute.tolist()
    assert loaded.open.tolist() == frame.funding_rate.tolist()


def test_hourly_cache_does_not_refetch_on_each_poll(mocker, default_conf):
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")
    key = (PAIR, "1h", CandleType.FUNDING_RATE)
    fetch = mocker.patch.object(
        ex,
        "refresh_latest_ohlcv",
        side_effect=[
            {key: rates().iloc[:1]},
            {key: rates().iloc[:2]},
        ],
    )
    for minute in (15, 30, 45):
        assert ex._fetch_and_calculate_funding_fees(
            PAIR, 10, True, START, START + timedelta(minutes=minute)
        ) == pytest.approx(0.1 * minute / 60)
    assert fetch.call_count == 1
    ex._fetch_and_calculate_funding_fees(PAIR, 10, True, START, START + timedelta(minutes=61))
    assert fetch.call_count == 2


@pytest.mark.parametrize("problem", ["off_hour", "duplicate", "missing", "nonfinite"])
def test_invalid_public_rate_is_not_cleaned_into_valid_history(mocker, default_conf, problem):
    ex = get_patched_exchange(mocker, default_conf, exchange="krakenfutures")
    row = {
        "timestamp": int(START.timestamp() * 1000),
        "fundingRate": 0.0001,
        "info": {"fundingRate": "0.01"},
    }
    raw = [row]
    if problem == "off_hour":
        row["timestamp"] += 1800000
    elif problem == "duplicate":
        raw.append(row.copy())
    elif problem == "missing":
        row["info"] = {}
    else:
        row["info"] = {"fundingRate": "nan"}
    ex._api_async.fetch_funding_rate_history = AsyncMock(return_value=raw)
    with pytest.raises(OperationalException, match="Kraken funding response"):
        ex.loop.run_until_complete(ex._fetch_funding_rate_history(PAIR, "1h", 100))


@pytest.mark.parametrize("problem", ["off_hour", "conflict"])
def test_absolute_converter_does_not_floor_or_hide_invalid_rates(problem):
    timestamp = int(START.timestamp() * 1000)
    raw = [[timestamp, 0.0001, 0.01]]
    if problem == "off_hour":
        raw[0][0] += 1800000
    else:
        raw.append([timestamp, 0.0001, 0.02])
    with pytest.raises(ValueError, match="funding rates"):
        ohlcv_to_dataframe(
            raw,
            "1h",
            PAIR,
            candle_type=CandleType.FUNDING_RATE,
            fill_missing=False,
            drop_incomplete=False,
        )
