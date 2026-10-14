from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cogalpha_plus.data_sources.baostock import (
    ADJUST_FACTOR_FIELDS,
    BaoStockConfig,
    BaoStockRequest,
    BaoStockResponseError,
    BaoStockSession,
    EQUITY_DAILY_FIELDS,
    HS300_FIELDS,
    INDEX_DAILY_FIELDS,
    INDUSTRY_FIELDS,
    STOCK_BASIC_FIELDS,
    SchemaDriftError,
    TRADE_DATE_FIELDS,
    make_request_key,
    result_to_frame,
)


class FakeResult:
    def __init__(self, fields, rows=(), *, error_code="0", error_msg="success"):
        self.fields = list(fields)
        self.rows = [list(row) for row in rows]
        self.error_code = error_code
        self.error_msg = error_msg
        self._position = -1

    def next(self):
        self._position += 1
        return self._position < len(self.rows)

    def get_row_data(self):
        return self.rows[self._position]


class FakeClient:
    __version__ = "0.9.3-test"

    def __init__(self):
        self.calls = []
        self.fail_history = 0
        self.history_fields = None

    @staticmethod
    def _ok():
        return FakeResult((), (), error_code="0")

    def login(self):
        self.calls.append(("login", {}))
        return self._ok()

    def logout(self):
        self.calls.append(("logout", {}))
        return self._ok()

    def query_trade_dates(self, **kwargs):
        self.calls.append(("trade_dates", kwargs))
        return FakeResult(TRADE_DATE_FIELDS, [(kwargs["start_date"], "1")])

    def query_hs300_stocks(self, **kwargs):
        self.calls.append(("hs300_stocks", kwargs))
        return FakeResult(HS300_FIELDS, [(kwargs["date"], "sh.600000", "浦发银行")])

    def query_history_k_data_plus(self, **kwargs):
        self.calls.append(("history", kwargs))
        if self.fail_history:
            self.fail_history -= 1
            return FakeResult((), (), error_code="10002007", error_msg="temporary transport failure")
        requested = tuple(kwargs["fields"].split(","))
        fields = tuple(self.history_fields or requested)
        values = {
            "date": kwargs["start_date"],
            "code": kwargs["code"],
            "open": "10.0",
            "high": "10.5",
            "low": "9.8",
            "close": "10.2",
            "preclose": "10.0",
            "volume": "10000",
            "amount": "101000",
            "adjustflag": kwargs["adjustflag"],
            "turn": "0.10",
            "tradestatus": "1",
            "pctChg": "2.0",
            "peTTM": "12.0",
            "pbMRQ": "1.2",
            "psTTM": "1.0",
            "pcfNcfTTM": "8.0",
            "isST": "0",
        }
        return FakeResult(fields, [[values.get(field, "") for field in fields]])

    def query_stock_basic(self, **kwargs):
        self.calls.append(("stock_basic", kwargs))
        return FakeResult(
            STOCK_BASIC_FIELDS,
            [(kwargs["code"], "浦发银行", "1999-11-10", "", "1", "1")],
        )

    def query_stock_industry(self, **kwargs):
        self.calls.append(("stock_industry", kwargs))
        return FakeResult(
            INDUSTRY_FIELDS,
            [(kwargs["date"], kwargs["code"], "浦发银行", "银行", "申万一级行业")],
        )

    def query_adjust_factor(self, **kwargs):
        self.calls.append(("adjust_factor", kwargs))
        return FakeResult(
            ADJUST_FACTOR_FIELDS,
            [(kwargs["code"], kwargs["start_date"], "1", "1", "1")],
        )


def test_request_key_is_canonical_and_versioned():
    first = make_request_key(
        "history_k_data",
        {"code": "sh.600000", "range": ["2020-01-01", "2020-12-31"], "flags": {3, 1}},
        provider_version="0.9.3",
    )
    reordered = make_request_key(
        "history_k_data",
        {"flags": {1, 3}, "range": ["2020-01-01", "2020-12-31"], "code": "sh.600000"},
        provider_version="0.9.3",
    )
    different_version = make_request_key(
        "history_k_data",
        {"flags": {1, 3}, "range": ["2020-01-01", "2020-12-31"], "code": "sh.600000"},
        provider_version="0.9.4",
    )
    assert first == reordered
    assert first != different_version
    assert len(first) == 64


def test_result_conversion_fails_on_vendor_error_empty_and_schema_drift():
    with pytest.raises(BaoStockResponseError, match="error_code=1"):
        result_to_frame(
            FakeResult(TRADE_DATE_FIELDS, error_code="1", error_msg="bad"),
            expected_fields=TRADE_DATE_FIELDS,
            operation="trade_dates",
        )
    with pytest.raises(BaoStockResponseError, match="empty"):
        result_to_frame(
            FakeResult(TRADE_DATE_FIELDS),
            expected_fields=TRADE_DATE_FIELDS,
            operation="trade_dates",
        )
    with pytest.raises(SchemaDriftError, match="schema drift"):
        result_to_frame(
            FakeResult(("calendar_date", "renamed"), [("2020-01-01", "1")]),
            expected_fields=TRADE_DATE_FIELDS,
            operation="trade_dates",
        )


def test_success_is_atomically_cached_and_second_call_does_not_query(tmp_path):
    client = FakeClient()
    config = BaoStockConfig(tmp_path, max_attempts=1)
    with BaoStockSession(config, client=client) as session:
        first = session.query_trade_dates("2020-01-01", "2020-12-31")
        second = session.query_trade_dates("2020-01-01", "2020-12-31")
        request_key = first.attrs["baostock_cache"]["request_key"]
        state = session.cache.read_state(request_key)
        object_path = session.cache.object_path("trade_dates", request_key)

    assert first.attrs["baostock_cache"]["cache_hit"] is False
    assert second.attrs["baostock_cache"]["cache_hit"] is True
    assert sum(name == "trade_dates" for name, _ in client.calls) == 1
    assert object_path.is_file()
    assert state["status"] == "success"
    assert state["rows"] == 1
    assert state["provider_version"] == client.__version__
    assert not list(object_path.parent.glob("*.tmp"))


def test_failed_response_has_state_but_no_parquet_then_resumes(tmp_path):
    client = FakeClient()
    client.fail_history = 1
    config = BaoStockConfig(tmp_path, max_attempts=1)
    args = {
        "code": "sh.600000",
        "fields": ",".join(EQUITY_DAILY_FIELDS),
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "frequency": "d",
        "adjustflag": "3",
    }
    request_key = make_request_key("history_k_data", args, provider_version=client.__version__)
    with BaoStockSession(config, client=client) as session:
        with pytest.raises(BaoStockResponseError):
            session.query_history("sh.600000", "2020-01-01", "2020-12-31")
        assert not session.cache.object_path("history_k_data", request_key).exists()
        failed = session.cache.read_state(request_key)
        assert failed["status"] == "failed"
        assert failed["attempt_count"] == 1

        resumed = session.query_history("sh.600000", "2020-01-01", "2020-12-31")
        completed = session.cache.read_state(request_key)

    assert len(resumed) == 1
    assert completed["status"] == "success"
    assert completed["attempt_count"] == 2


def test_long_session_reauthenticates_before_retrying_invalidated_query(tmp_path):
    client = FakeClient()
    client.fail_history = 1
    config = BaoStockConfig(
        tmp_path,
        max_attempts=2,
        initial_backoff_seconds=0,
    )
    with BaoStockSession(config, client=client) as session:
        frame = session.query_history("sh.600000", "2020-01-01", "2020-12-31")

    assert len(frame) == 1
    assert sum(name == "login" for name, _ in client.calls) == 2
    assert sum(name == "logout" for name, _ in client.calls) == 2


def test_schema_drift_never_commits_a_parquet_object(tmp_path):
    client = FakeClient()
    client.history_fields = ("date", "code", "unexpected")
    config = BaoStockConfig(tmp_path, max_attempts=1)
    with BaoStockSession(config, client=client) as session:
        with pytest.raises(SchemaDriftError):
            session.query_history("sh.600000", "2020-01-01", "2020-12-31")
        object_files = list((Path(tmp_path) / "objects").rglob("*.parquet"))
        states = list((Path(tmp_path) / "state").glob("*.json"))
    assert object_files == []
    assert len(states) == 1
    assert json.loads(states[0].read_text())["status"] == "failed"


def test_all_thin_wrappers_forward_exact_provider_arguments(tmp_path):
    client = FakeClient()
    config = BaoStockConfig(tmp_path, max_attempts=1)
    with BaoStockSession(config, client=client) as session:
        assert list(session.query_hs300("2020-01-02").columns) == list(HS300_FIELDS)
        assert list(session.query_basic(code="sh.600000").columns) == list(STOCK_BASIC_FIELDS)
        assert list(session.query_industry(code="sh.600000", date="2020-01-06").columns) == list(INDUSTRY_FIELDS)
        factors = session.query_adjustment_factors(
            "sh.600000", "1990-01-01", "2025-12-31"
        )
        index = session.query_index("2020-01-01", "2020-12-31")

    assert list(factors.columns) == list(ADJUST_FACTOR_FIELDS)
    factor_calls = [args for name, args in client.calls if name == "adjust_factor"]
    assert factor_calls == [
        {
            "code": "sh.600000",
            "start_date": "1990-01-01",
            "end_date": "2025-12-31",
        }
    ]
    assert list(index.columns) == list(INDEX_DAILY_FIELDS)
    history_calls = [args for name, args in client.calls if name == "history"]
    assert history_calls[-1] == {
        "code": "sh.000300",
        "fields": ",".join(INDEX_DAILY_FIELDS),
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "frequency": "d",
        "adjustflag": "3",
    }


def test_resume_skips_committed_requests(tmp_path):
    client = FakeClient()
    requests = [
        BaoStockRequest("trade_dates", {"start_date": "2020-01-01", "end_date": "2020-12-31"}),
        BaoStockRequest("hs300_stocks", {"date": "2020-01-02"}),
    ]
    with BaoStockSession(BaoStockConfig(tmp_path, max_attempts=1), client=client) as session:
        initial = session.resume(requests)
        resumed = session.resume(requests)

    assert len(initial) == len(resumed) == 2
    assert sum(name == "trade_dates" for name, _ in client.calls) == 1
    assert sum(name == "hs300_stocks" for name, _ in client.calls) == 1
    assert all(frame.attrs["baostock_cache"]["cache_hit"] for frame in resumed.values())


def test_login_failure_never_opens_session_or_writes_cache(tmp_path):
    client = FakeClient()
    client.login = lambda: FakeResult((), (), error_code="7", error_msg="denied")
    with pytest.raises(BaoStockResponseError, match="login"):
        with BaoStockSession(BaoStockConfig(tmp_path, max_attempts=1), client=client):
            raise AssertionError("unreachable")
    assert not list((Path(tmp_path) / "objects").rglob("*.parquet"))
