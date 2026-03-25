"""
远程数据读取客户端（二进制 parquet 版本）

默认调用 Flask 二进制接口:
- POST /api/local-data/query/parquet
"""

from __future__ import annotations

import io
import json
import gzip
import warnings
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin
from urllib.request import Request, urlopen, build_opener, ProxyHandler
from urllib.error import URLError, HTTPError

import pandas as pd


class RemoteDataError(RuntimeError):
    """远程取数失败"""


def _normalize_base_url(base_url: str) -> str:
    base = (base_url or "").strip()
    if not base:
        raise ValueError("base_url 不能为空")
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    if not base.endswith("/"):
        base += "/"
    return base


def _to_int_header(headers, key: str, default: int = 0) -> int:
    val = headers.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except Exception:
        return default


def _warn_if_truncated(headers, warn_on_truncated: bool):
    if not warn_on_truncated:
        return
    truncated = headers.get("X-Remote-Data-Truncated", "0") == "1"
    if not truncated:
        return
    total = _to_int_header(headers, "X-Remote-Data-Total", 0)
    returned = _to_int_header(headers, "X-Remote-Data-Returned", 0)
    warnings.warn(
        f"远程数据已截断: returned_records={returned}, total_records={total}。请提高 limit 以获取更多数据。",
        RuntimeWarning,
        stacklevel=2,
    )


def _decode_single_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "date" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    if out.empty:
        return pd.DataFrame()
    out = out.set_index("date").sort_index()
    out.index.name = "date"
    return out


def _decode_single_long(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    required = {"date", "ts_code", "value"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    if out.empty:
        return pd.DataFrame()
    out = out.pivot(index="date", columns="ts_code", values="value").sort_index()
    out.index.name = "date"
    return out


def _decode_multi_long(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    required = {"date", "ts_code", "field", "value"}
    if df is None or df.empty or not required.issubset(df.columns):
        return {}
    out: Dict[str, pd.DataFrame] = {}
    for field_name, part in df.groupby("field", sort=False):
        p = part[["date", "ts_code", "value"]].copy()
        out[str(field_name)] = _decode_single_long(p)
    return out

def _decode_tick_single(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        out = out.dropna(subset=["datetime"])
        out = out.set_index("datetime").sort_index()
        out.index.name = "datetime"
    return out

def _decode_tick_multi(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if df is None or df.empty or "stock_code" not in df.columns:
        return {}
    out: Dict[str, pd.DataFrame] = {}
    for stock_code, part in df.groupby("stock_code", sort=False):
        out[str(stock_code)] = _decode_tick_single(part.copy())
    return out


def get_remote_data(
    data_type: Optional[str] = "daily",
    start: Optional[str] = None,
    end: Optional[str] = None,
    field: Union[str, List[str]] = "close",
    stocks: Optional[Union[List[str], str]] = None,
    base_url: str = "http://127.0.0.1:9999",
    timeout: int = 60,
    limit: int = 10000000,
    output_format: str = "wide",
    parallel: bool = True,
    max_workers: int = 8,
    warn_on_truncated: bool = True,
    disable_proxy: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    从远端 Flask 二进制接口获取数据，返回与 get_local_data 对齐的数据结构。

    Returns:
        单字段: DataFrame（index=date, columns=ts_code）
        多字段: Dict[str, DataFrame]
    """
    base = _normalize_base_url(base_url)
    endpoint = urljoin(base, "api/local-data/query/parquet")

    payload = {
        "data_type": data_type,
        "start": start,
        "end": end,
        "stocks": stocks,
        "format": output_format,
        "limit": int(limit),
        "parallel": bool(parallel),
        "max_workers": int(max_workers),
    }
    if isinstance(field, (list, tuple, set)):
        payload["fields"] = [str(f) for f in field]
    else:
        payload["field"] = str(field)

    req = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip",
        },
        method="POST",
    )

    try:
        if disable_proxy:
            opener = build_opener(ProxyHandler({}))
            resp_ctx = opener.open(req, timeout=timeout)
        else:
            resp_ctx = urlopen(req, timeout=timeout)
        with resp_ctx as resp:
            body = resp.read()
            headers = resp.headers
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        raise RemoteDataError(f"HTTPError {e.code}: {detail}") from e
    except URLError as e:
        raise RemoteDataError(f"无法连接远端服务: {e}") from e

    content_encoding = str(headers.get("Content-Encoding", "")).strip().lower()
    if content_encoding == "gzip":
        try:
            body = gzip.decompress(body)
        except Exception as e:
            raise RemoteDataError(f"gzip 解压失败: {e}") from e

    try:
        df = pd.read_parquet(io.BytesIO(body))
    except Exception as e:
        raise RemoteDataError(f"远端返回内容不是有效 parquet: {e}") from e

    _warn_if_truncated(headers, warn_on_truncated)

    kind = str(headers.get("X-Remote-Data-Kind", "")).strip().lower()
    if kind == "single_wide":
        return _decode_single_wide(df)
    if kind == "single_long":
        return _decode_single_long(df)
    if kind == "multi_long":
        return _decode_multi_long(df)

    # 兜底推断
    if "field" in df.columns:
        return _decode_multi_long(df)
    if {"date", "ts_code", "value"}.issubset(df.columns):
        return _decode_single_long(df)
    return _decode_single_wide(df)


def get_remote_tick_data(
    stock_codes: Union[str, List[str]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    tick_dir: Optional[str] = None,
    short: bool = False,
    base_url: str = "http://127.0.0.1:9999",
    timeout: int = 300,
    disable_proxy: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    从远端 Flask tick 接口获取 tick 数据，返回与 get_tick_data/get_tick_data_short 对齐的数据结构。

    Returns:
        单股票: DataFrame（index=datetime）
        多股票: Dict[str, DataFrame]
    """
    base = _normalize_base_url(base_url)
    endpoint = urljoin(base, "api/tick-data/query/parquet")
    payload = {
        "stock_codes": stock_codes,
        "start_date": start_date,
        "end_date": end_date,
        "short": bool(short),
    }
    if tick_dir:
        payload["tick_dir"] = tick_dir

    req = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip",
        },
        method="POST",
    )

    try:
        if disable_proxy:
            opener = build_opener(ProxyHandler({}))
            resp_ctx = opener.open(req, timeout=timeout)
        else:
            resp_ctx = urlopen(req, timeout=timeout)
        with resp_ctx as resp:
            body = resp.read()
            headers = resp.headers
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        raise RemoteDataError(f"HTTPError {e.code}: {detail}") from e
    except URLError as e:
        raise RemoteDataError(f"无法连接远端服务: {e}") from e

    content_encoding = str(headers.get("Content-Encoding", "")).strip().lower()
    if content_encoding == "gzip":
        try:
            body = gzip.decompress(body)
        except Exception as e:
            raise RemoteDataError(f"gzip 解压失败: {e}") from e

    try:
        df = pd.read_parquet(io.BytesIO(body))
    except Exception as e:
        raise RemoteDataError(f"远端返回内容不是有效 parquet: {e}") from e

    kind = str(headers.get("X-Remote-Tick-Kind", "")).strip().lower()
    if kind == "multi":
        return _decode_tick_multi(df)
    if kind == "single":
        return _decode_tick_single(df)

    if "stock_code" in df.columns:
        unique_codes = df["stock_code"].dropna().astype(str).unique()
        if len(unique_codes) > 1:
            return _decode_tick_multi(df)
    return _decode_tick_single(df)


__all__ = ["get_remote_data", "get_remote_tick_data", "RemoteDataError"]
