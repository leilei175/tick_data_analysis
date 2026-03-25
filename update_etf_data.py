#!/usr/bin/env python3
"""ETF 日频数据自动更新脚本。

功能：
1. 下载 ETF 日线行情（Tushare `fund_daily`）
2. 下载 ETF 净值（优先 Tushare `etf_share_size`，否则回退到 `fund_nav`）
3. 下载 ETF 份额（优先 Tushare `etf_share_size`，否则回退到 `fund_share`）
4. 计算收盘折溢价率
5. 可选合并外部 IOPV 数据，计算收盘相对 IOPV 的折溢价率

输出目录：
daily_data/
├── etf_daily/
├── etf_nav/
├── etf_share/
└── etf_metrics/
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mylib.tushare_client import init_tushare as _init_tushare


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "daily_data"

DATA_DIRS: Dict[str, Path] = {
    "etf_daily": DEFAULT_DATA_DIR / "etf_daily",
    "etf_nav": DEFAULT_DATA_DIR / "etf_nav",
    "etf_share": DEFAULT_DATA_DIR / "etf_share",
    "etf_metrics": DEFAULT_DATA_DIR / "etf_metrics",
}

ETF_DAILY_FIELDS = [
    "ts_code",
    "trade_date",
    "pre_close",
    "open",
    "high",
    "low",
    "close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]

ETF_NAV_FIELDS = [
    "ts_code",
    "ann_date",
    "nav_date",
    "unit_nav",
    "accum_nav",
    "accum_div",
    "net_asset",
    "total_netasset",
    "adj_nav",
]

ETF_SHARE_FIELDS = [
    "ts_code",
    "trade_date",
    "fd_share",
]


def get_etf_universe(pro) -> pd.DataFrame:
    df = pro.fund_basic(market="E")
    if df.empty:
        return df

    out = df.copy()
    out["name"] = out["name"].astype(str)
    out["is_etf"] = out["name"].str.contains("ETF", case=False, na=False)
    out = out[out["is_etf"]].copy()
    return out


def init_tushare():
    config_path = SCRIPT_DIR / "config.py"
    return _init_tushare(config_path=str(config_path))


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y%m%d")


def date_to_str(date_obj: datetime) -> str:
    return date_obj.strftime("%Y%m%d")


def get_today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def is_after_etf_update_time() -> bool:
    now = datetime.now()
    return (now.hour > 19) or (now.hour == 19 and now.minute >= 0)


def get_trade_dates(pro, start_date: str, end_date: str) -> List[str]:
    trade_cal = pro.trade_cal(
        exchange="SSE",
        start_date=start_date,
        end_date=end_date,
        is_open="1",
    )
    if trade_cal.empty:
        return []
    return sorted(trade_cal["cal_date"].astype(str).tolist())


def get_latest_date_from_dir(data_dir: Path, prefix: str) -> Optional[str]:
    if not data_dir.exists():
        return None

    latest_date: Optional[str] = None
    for fp in data_dir.glob("*/*/*.parquet"):
        name = fp.name
        if not name.startswith(f"{prefix}_") or not name.endswith(".parquet"):
            continue
        try:
            date_str = name.split("_")[-1].replace(".parquet", "")
            parse_date(date_str)
        except Exception:
            continue
        if latest_date is None or date_str > latest_date:
            latest_date = date_str
    return latest_date


def save_daily_file(df: pd.DataFrame, output_dir: Path, prefix: str, trade_date: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    year_dir = output_dir / trade_date[:4] / trade_date[4:6]
    year_dir.mkdir(parents=True, exist_ok=True)
    out_file = year_dir / f"{prefix}_{trade_date}.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(out_file))
    return out_file


def normalize_trade_date_column(df: pd.DataFrame, date_col: str, trade_date: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if date_col not in out.columns:
        out[date_col] = trade_date
    out[date_col] = out[date_col].astype(str)
    return out


def fetch_etf_daily(pro, trade_date: str) -> pd.DataFrame:
    df = pro.fund_daily(trade_date=trade_date, fields=",".join(ETF_DAILY_FIELDS))
    return normalize_trade_date_column(df, "trade_date", trade_date)


def fetch_etf_nav(pro, trade_date: str) -> pd.DataFrame:
    df = pro.fund_nav(nav_date=trade_date, market="E", fields=",".join(ETF_NAV_FIELDS))
    return normalize_trade_date_column(df, "nav_date", trade_date)


def fetch_etf_share(pro, trade_date: str) -> pd.DataFrame:
    df = pro.fund_share(trade_date=trade_date, fields=",".join(ETF_SHARE_FIELDS))
    return normalize_trade_date_column(df, "trade_date", trade_date)


def fetch_etf_share_size(pro, trade_date: str) -> pd.DataFrame:
    df = pro.etf_share_size(trade_date=trade_date)
    if df.empty:
        return df

    rename_map = {
        "trade_date": "trade_date",
        "ts_code": "ts_code",
        "name": "name",
        "close": "close",
        "pct_chg": "pct_chg",
        "vol": "vol",
        "amount": "amount",
        "fd_share": "fd_share",
        "total_share": "fd_share",
        "nav": "unit_nav",
        "unit_nav": "unit_nav",
        "exchange": "exchange",
        "market": "exchange",
    }
    keep_cols = [c for c in df.columns if c in rename_map]
    out = df[keep_cols].rename(columns={c: rename_map[c] for c in keep_cols}).copy()
    return normalize_trade_date_column(out, "trade_date", trade_date)


def load_optional_iopv(iopv_dir: Optional[Path], trade_date: str) -> pd.DataFrame:
    if iopv_dir is None:
        return pd.DataFrame()

    candidates = [
        iopv_dir / f"iopv_{trade_date}.parquet",
        iopv_dir / f"{trade_date}.parquet",
        iopv_dir / f"iopv_{trade_date}.csv",
        iopv_dir / f"{trade_date}.csv",
    ]

    src = next((fp for fp in candidates if fp.exists()), None)
    if src is None:
        return pd.DataFrame()

    if src.suffix == ".parquet":
        df = pd.read_parquet(src)
    else:
        df = pd.read_csv(src)

    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "trade_date" not in out.columns:
        out["trade_date"] = trade_date
    out["trade_date"] = out["trade_date"].astype(str)

    if "iopv" not in out.columns:
        for candidate in ["IOPV", "iopv_last", "iopv_close", "last_iopv"]:
            if candidate in out.columns:
                out = out.rename(columns={candidate: "iopv"})
                break

    required = {"ts_code", "trade_date", "iopv"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"IOPV 文件缺少字段: {sorted(missing)}")

    return out[["ts_code", "trade_date", "iopv"]].copy()


def filter_to_etf_universe(df: pd.DataFrame, etf_codes: set[str]) -> pd.DataFrame:
    if df.empty:
        return df
    if "ts_code" not in df.columns:
        return df
    return df[df["ts_code"].isin(etf_codes)].copy()


def enrich_with_etf_basic(metrics_df: pd.DataFrame, etf_basic: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df

    keep_cols = [
        c
        for c in ["ts_code", "name", "fund_type", "invest_type", "type", "management", "market"]
        if c in etf_basic.columns
    ]
    if not keep_cols:
        return metrics_df

    return metrics_df.merge(etf_basic[keep_cols].drop_duplicates(subset=["ts_code"]), on="ts_code", how="left")


def normalize_premium_columns(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df

    metrics = metrics_df.copy()
    if "fund_type" in metrics.columns:
        money_mask = metrics["fund_type"].astype(str).eq("货币市场型")
        metrics.loc[money_mask, "premium_rate_close_nav"] = np.nan
        if "premium_rate_close_iopv" in metrics.columns:
            metrics.loc[money_mask, "premium_rate_close_iopv"] = np.nan
    return metrics


def build_metrics_from_share_size(size_df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    metrics = size_df.copy()
    if metrics.empty:
        return metrics

    metrics["trade_date"] = trade_date
    metrics["premium_rate_close_nav"] = np.where(
        metrics["unit_nav"].astype(float) > 0,
        metrics["close"].astype(float) / metrics["unit_nav"].astype(float) - 1.0,
        np.nan,
    )
    metrics["iopv"] = np.nan
    metrics["premium_rate_close_iopv"] = np.nan
    metrics["data_source"] = "etf_share_size"
    return metrics


def build_metrics_from_components(
    daily_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    share_df: pd.DataFrame,
    trade_date: str,
) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()

    metrics = daily_df.copy()

    if not nav_df.empty:
        nav_keep = [
            c
            for c in [
                "ts_code",
                "nav_date",
                "ann_date",
                "unit_nav",
                "accum_nav",
                "accum_div",
                "net_asset",
                "total_netasset",
                "adj_nav",
            ]
            if c in nav_df.columns
        ]
        metrics = metrics.merge(nav_df[nav_keep], left_on="ts_code", right_on="ts_code", how="left")
    else:
        metrics["nav_date"] = trade_date
        metrics["ann_date"] = np.nan
        metrics["unit_nav"] = np.nan
        metrics["accum_nav"] = np.nan
        metrics["accum_div"] = np.nan
        metrics["net_asset"] = np.nan
        metrics["total_netasset"] = np.nan
        metrics["adj_nav"] = np.nan

    if not share_df.empty:
        share_keep = [c for c in ["ts_code", "trade_date", "fd_share"] if c in share_df.columns]
        metrics = metrics.merge(share_df[share_keep], on=["ts_code", "trade_date"], how="left")
    else:
        metrics["fd_share"] = np.nan

    metrics["premium_rate_close_nav"] = np.where(
        pd.to_numeric(metrics["unit_nav"], errors="coerce") > 0,
        pd.to_numeric(metrics["close"], errors="coerce") / pd.to_numeric(metrics["unit_nav"], errors="coerce") - 1.0,
        np.nan,
    )
    metrics["iopv"] = np.nan
    metrics["premium_rate_close_iopv"] = np.nan
    metrics["data_source"] = "fund_daily+fund_nav+fund_share"
    return metrics


def merge_iopv(metrics_df: pd.DataFrame, iopv_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty or iopv_df.empty:
        return metrics_df

    metrics = metrics_df.drop(columns=["iopv"], errors="ignore").merge(
        iopv_df[["ts_code", "trade_date", "iopv"]],
        on=["ts_code", "trade_date"],
        how="left",
    )
    metrics["premium_rate_close_iopv"] = np.where(
        pd.to_numeric(metrics["iopv"], errors="coerce") > 0,
        pd.to_numeric(metrics["close"], errors="coerce") / pd.to_numeric(metrics["iopv"], errors="coerce") - 1.0,
        np.nan,
    )
    return metrics


def detect_update_range(
    pro,
    start_date: Optional[str],
    end_date: Optional[str],
    include_today: bool,
) -> Tuple[str, str]:
    if start_date is None:
        latest_candidates = [
            get_latest_date_from_dir(DATA_DIRS["etf_metrics"], "etf_metrics"),
            get_latest_date_from_dir(DATA_DIRS["etf_daily"], "etf_daily"),
            get_latest_date_from_dir(DATA_DIRS["etf_nav"], "etf_nav"),
            get_latest_date_from_dir(DATA_DIRS["etf_share"], "etf_share"),
        ]
        latest_candidates = [d for d in latest_candidates if d]
        start_date = min(latest_candidates) if latest_candidates else "20250101"

    if end_date is None:
        today = get_today_str()
        if include_today and is_after_etf_update_time():
            end_date = today
        else:
            window_start = date_to_str(parse_date(today) - timedelta(days=10))
            trade_dates = get_trade_dates(pro, window_start, today)
            if trade_dates:
                end_date = trade_dates[-1]
                if end_date == today and not is_after_etf_update_time():
                    end_date = trade_dates[-2] if len(trade_dates) >= 2 else today
            else:
                end_date = today

    return start_date, end_date


def collect_missing_trade_dates(
    pro,
    start_date: str,
    end_date: str,
    prefix: str,
    output_dir: Path,
) -> List[str]:
    trade_dates = get_trade_dates(pro, start_date, end_date)
    existing_dates = set()

    if output_dir.exists():
        for fp in output_dir.glob("*/*/*.parquet"):
            name = fp.name
            if not name.startswith(f"{prefix}_") or not name.endswith(".parquet"):
                continue
            existing_dates.add(name.split("_")[-1].replace(".parquet", ""))

    return [d for d in trade_dates if d not in existing_dates]


def update_etf_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    include_today: bool = False,
    iopv_dir: Optional[str] = None,
    prefer_share_size: bool = True,
) -> None:
    pro = init_tushare()
    start_date, end_date = detect_update_range(pro, start_date, end_date, include_today)
    iopv_path = Path(iopv_dir).resolve() if iopv_dir else None
    etf_basic = get_etf_universe(pro)
    etf_codes = set(etf_basic["ts_code"].astype(str).unique())

    print("=" * 60)
    print("ETF 数据更新")
    print("=" * 60)
    print(f"当前时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"更新区间: {start_date} ~ {end_date}")
    print(f"优先使用 etf_share_size: {prefer_share_size}")
    print(f"IOPV 目录: {iopv_path if iopv_path else '(未提供)'}")
    print(f"ETF 基金池数量: {len(etf_codes)}")

    trade_dates = collect_missing_trade_dates(
        pro=pro,
        start_date=start_date,
        end_date=end_date,
        prefix="etf_metrics",
        output_dir=DATA_DIRS["etf_metrics"],
    )
    print(f"待更新交易日: {len(trade_dates)}")

    if not trade_dates:
        print("没有需要更新的 ETF 交易日数据。")
        return

    for idx, trade_date in enumerate(trade_dates, 1):
        print(f"\n[{idx}/{len(trade_dates)}] {trade_date}")

        size_df = pd.DataFrame()
        daily_df = pd.DataFrame()
        nav_df = pd.DataFrame()
        share_df = pd.DataFrame()

        if prefer_share_size:
            try:
                size_df = fetch_etf_share_size(pro, trade_date)
                size_df = filter_to_etf_universe(size_df, etf_codes)
                if not size_df.empty:
                    print(f"  etf_share_size: {len(size_df)} 条")
                    try:
                        daily_df = fetch_etf_daily(pro, trade_date)
                        daily_df = filter_to_etf_universe(daily_df, etf_codes)
                        print(f"  fund_daily: {len(daily_df)} 条")
                    except Exception as exc:
                        print(f"  fund_daily 补抓失败，继续使用 share_size 结果: {exc}")
            except Exception as exc:
                print(f"  etf_share_size 获取失败，回退基础接口: {exc}")
                size_df = pd.DataFrame()

        if size_df.empty:
            daily_df = fetch_etf_daily(pro, trade_date)
            nav_df = fetch_etf_nav(pro, trade_date)
            share_df = fetch_etf_share(pro, trade_date)
            daily_df = filter_to_etf_universe(daily_df, etf_codes)
            nav_df = filter_to_etf_universe(nav_df, etf_codes)
            share_df = filter_to_etf_universe(share_df, etf_codes)
            print(f"  fund_daily: {len(daily_df)} 条")
            print(f"  fund_nav: {len(nav_df)} 条")
            print(f"  fund_share: {len(share_df)} 条")

        if not daily_df.empty:
            save_daily_file(daily_df, DATA_DIRS["etf_daily"], "etf_daily", trade_date)
        if not nav_df.empty:
            save_daily_file(nav_df, DATA_DIRS["etf_nav"], "etf_nav", trade_date)
        if not share_df.empty:
            save_daily_file(share_df, DATA_DIRS["etf_share"], "etf_share", trade_date)

        if not size_df.empty:
            nav_stub = size_df.copy()
            nav_stub["nav_date"] = trade_date
            share_stub = size_df.copy()
            share_stub["trade_date"] = trade_date
            save_daily_file(nav_stub, DATA_DIRS["etf_nav"], "etf_nav", trade_date)
            save_daily_file(share_stub, DATA_DIRS["etf_share"], "etf_share", trade_date)
            if not daily_df.empty:
                save_daily_file(daily_df, DATA_DIRS["etf_daily"], "etf_daily", trade_date)
            elif "trade_date" in size_df.columns:
                save_daily_file(size_df, DATA_DIRS["etf_daily"], "etf_daily", trade_date)
            metrics_df = build_metrics_from_share_size(size_df, trade_date)
        else:
            metrics_df = build_metrics_from_components(daily_df, nav_df, share_df, trade_date)

        metrics_df = enrich_with_etf_basic(metrics_df, etf_basic)
        iopv_df = load_optional_iopv(iopv_path, trade_date)
        if not iopv_df.empty:
            print(f"  iopv: {len(iopv_df)} 条")
            metrics_df = merge_iopv(metrics_df, iopv_df)
        metrics_df = normalize_premium_columns(metrics_df)

        save_daily_file(metrics_df, DATA_DIRS["etf_metrics"], "etf_metrics", trade_date)
        print(f"  etf_metrics: {len(metrics_df)} 条")

    print("\nETF 数据更新完成。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ETF 日频数据自动更新脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start", "-s", help="开始日期，格式 YYYYMMDD")
    parser.add_argument("--end", "-e", help="结束日期，格式 YYYYMMDD")
    parser.add_argument(
        "--include-today",
        action="store_true",
        help="19:00 后允许更新到今天；否则默认更新到最近一个已完成交易日",
    )
    parser.add_argument(
        "--iopv-dir",
        help="可选，外部 IOPV 文件目录，支持 parquet/csv",
    )
    parser.add_argument(
        "--no-share-size",
        action="store_true",
        help="不使用 etf_share_size，直接走 fund_daily/fund_nav/fund_share 组合接口",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_etf_data(
        start_date=args.start,
        end_date=args.end,
        include_today=args.include_today,
        iopv_dir=args.iopv_dir,
        prefer_share_size=not args.no_share_size,
    )


if __name__ == "__main__":
    main()
