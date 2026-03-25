#!/usr/bin/env python3
"""提取所有股票 09:24:00~09:25:00 区间每个 tick 时点的成交额代理值。

计算规则：
- latest_price: 与 build_auction_latest_price_924_925.py 相同
- bid_vol: 使用 bidVol 第一档
- latest_amount: latest_price * bid_vol

注意：
- 不使用原始 amount 字段，因为集合竞价期间该字段常为 0
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


WINDOW_START = time(9, 24, 0)
WINDOW_END = time(9, 25, 0)


@dataclass
class Config:
    tick_base: Path = Path("./tick_2026")
    output_file: Path = Path("./factor/high_frequency/auction_latest_amount_924_925/auction_latest_amount_924_925_all.parquet")
    summary_file: Path = Path("./factor/high_frequency/auction_latest_amount_924_925/auction_latest_amount_924_925_summary.json")
    max_workers: int = max(1, (os.cpu_count() or 4) // 2)


def list_tick_dates(tick_base: Path, years: Optional[Sequence[int]] = None) -> List[Tuple[date, Path]]:
    if years is None:
        years = sorted(int(p.name) for p in tick_base.iterdir() if p.is_dir() and p.name.isdigit())
    results: List[Tuple[date, Path]] = []
    for year in years:
        year_dir = tick_base / str(year)
        if not year_dir.exists():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir() or not month_dir.name.isdigit():
                continue
            for day_dir in sorted(month_dir.iterdir()):
                if not day_dir.is_dir() or not day_dir.name.isdigit():
                    continue
                try:
                    trade_date = date(year, int(month_dir.name), int(day_dir.name))
                except ValueError:
                    continue
                if any(day_dir.glob("*.parquet")):
                    results.append((trade_date, day_dir))
    return sorted(results, key=lambda x: x[0])


def filter_by_date_range(
    items: Iterable[Tuple[date, Path]],
    start_date: Optional[date],
    end_date: Optional[date],
) -> List[Tuple[date, Path]]:
    results: List[Tuple[date, Path]] = []
    for trade_date, day_dir in items:
        if start_date and trade_date < start_date:
            continue
        if end_date and trade_date > end_date:
            continue
        results.append((trade_date, day_dir))
    return results


def parse_date_arg(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return pd.to_datetime(value).date()


def parse_years_arg(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    years = [int(token.strip()) for token in value.split(",") if token.strip()]
    return sorted(set(years)) if years else None


def _first_level(value) -> float:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.nan
        return pd.to_numeric(value.flat[0], errors="coerce")
    if isinstance(value, (list, tuple)):
        if not value:
            return np.nan
        return pd.to_numeric(value[0], errors="coerce")
    if isinstance(value, str):
        s = value.strip().strip("[]")
        if not s:
            return np.nan
        return pd.to_numeric(s.split(",")[0].strip(), errors="coerce")
    return pd.to_numeric(value, errors="coerce")


def _extract_one_file(file_path: str, trade_date_str: str) -> Optional[pd.DataFrame]:
    fp = Path(file_path)
    stock_code = fp.stem
    trade_date = pd.Timestamp(trade_date_str)
    try:
        df = pd.read_parquet(
            fp,
            columns=["time", "lastPrice", "lastClose", "askPrice", "bidPrice", "bidVol"],
        )
    except Exception:
        return None

    if df.empty:
        return None

    local_dt = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    time_part = local_dt.dt.time
    mask = (time_part >= WINDOW_START) & (time_part <= WINDOW_END)
    snap = df.loc[mask].copy()
    if snap.empty:
        return None

    snap["datetime"] = local_dt.loc[mask].values
    snap = snap.sort_values("datetime").reset_index(drop=True)
    snap["trade_date"] = trade_date
    snap["stock_code"] = stock_code
    snap["tick_seq"] = np.arange(1, len(snap) + 1)
    snap["tick_time"] = snap["datetime"].dt.strftime("%H:%M:%S")
    snap["ask1"] = snap["askPrice"].apply(_first_level)
    snap["bid1"] = snap["bidPrice"].apply(_first_level)
    snap["bid_vol"] = snap["bidVol"].apply(_first_level)
    snap["lastPrice"] = pd.to_numeric(snap["lastPrice"], errors="coerce")
    snap["lastClose"] = pd.to_numeric(snap["lastClose"], errors="coerce")

    latest_price = snap["lastPrice"].where(snap["lastPrice"] > 0)
    price_source = pd.Series(np.where(latest_price.notna(), "lastPrice", ""), index=snap.index, dtype=object)

    mid_price = ((snap["ask1"] + snap["bid1"]) / 2.0).where((snap["ask1"] > 0) & (snap["bid1"] > 0))
    use_mid = latest_price.isna() & mid_price.notna()
    latest_price = latest_price.where(~use_mid, mid_price)
    price_source = price_source.where(~use_mid, "mid")

    use_ask = latest_price.isna() & snap["ask1"].notna()
    latest_price = latest_price.where(~use_ask, snap["ask1"])
    price_source = price_source.where(~use_ask, "ask1")

    use_bid = latest_price.isna() & snap["bid1"].notna()
    latest_price = latest_price.where(~use_bid, snap["bid1"])
    price_source = price_source.where(~use_bid, "bid1")

    latest_amount = np.where(
        latest_price.notna() & snap["bid_vol"].notna(),
        latest_price * snap["bid_vol"],
        np.nan,
    )

    return pd.DataFrame(
        {
            "trade_date": snap["trade_date"],
            "stock_code": snap["stock_code"],
            "tick_seq": snap["tick_seq"],
            "datetime": snap["datetime"],
            "tick_time": snap["tick_time"],
            "time_ms": pd.to_numeric(snap["time"], errors="coerce").astype("Int64"),
            "latest_price": latest_price,
            "price_source": price_source,
            "bid_vol": snap["bid_vol"],
            "latest_amount": latest_amount,
            "lastPrice": snap["lastPrice"],
            "lastClose": snap["lastClose"],
            "ask1": snap["ask1"],
            "bid1": snap["bid1"],
            "pct_from_last_close": np.where(
                (latest_price > 0) & (snap["lastClose"] > 0),
                latest_price / snap["lastClose"] - 1.0,
                np.nan,
            ),
        }
    )


def extract_day_rows(day_dir: Path, trade_date: date, max_workers: int) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    parquet_files = sorted(day_dir.glob("*.parquet"))
    tasks = [(str(fp), trade_date.isoformat()) for fp in parquet_files]

    if max_workers <= 1:
        for file_path, trade_date_str in tasks:
            out = _extract_one_file(file_path, trade_date_str)
            if out is not None and not out.empty:
                rows.append(out)
    else:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for out in executor.map(
                    _extract_one_file,
                    (t[0] for t in tasks),
                    (t[1] for t in tasks),
                    chunksize=32,
                ):
                    if out is not None and not out.empty:
                        rows.append(out)
        except BrokenProcessPool:
            print(f"警告: {trade_date} 并行处理失败，自动降级为单进程重试。")
            rows = []
            for file_path, trade_date_str in tasks:
                out = _extract_one_file(file_path, trade_date_str)
                if out is not None and not out.empty:
                    rows.append(out)

    if not rows:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "stock_code",
                "tick_seq",
                "datetime",
                "tick_time",
                "time_ms",
                "latest_price",
                "price_source",
                "bid_vol",
                "latest_amount",
                "lastPrice",
                "lastClose",
                "ask1",
                "bid1",
                "pct_from_last_close",
            ]
        )

    return pd.concat(rows, ignore_index=True)


def run(cfg: Config, years: Optional[Sequence[int]], start_date: Optional[date], end_date: Optional[date]) -> None:
    tick_dates = filter_by_date_range(list_tick_dates(cfg.tick_base, years), start_date, end_date)
    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    total_rows = 0
    total_days = 0
    total_stock_days = 0

    try:
        for i, (trade_date, day_dir) in enumerate(tick_dates, start=1):
            day_df = extract_day_rows(day_dir, trade_date, cfg.max_workers)
            if day_df.empty:
                print(f"[{i}/{len(tick_dates)}] {trade_date} -> empty")
                continue

            table = pa.Table.from_pandas(day_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(cfg.output_file), table.schema)
            writer.write_table(table)

            total_rows += len(day_df)
            total_days += 1
            total_stock_days += int(day_df[["trade_date", "stock_code"]].drop_duplicates().shape[0])
            print(
                f"[{i}/{len(tick_dates)}] {trade_date} -> rows={len(day_df)} "
                f"stocks={day_df['stock_code'].nunique()}"
            )
    finally:
        if writer is not None:
            writer.close()

    summary = {
        "output_file": str(cfg.output_file.resolve()),
        "total_rows": int(total_rows),
        "total_trade_days": int(total_days),
        "total_stock_days": int(total_stock_days),
        "window_start": WINDOW_START.strftime("%H:%M:%S"),
        "window_end": WINDOW_END.strftime("%H:%M:%S"),
        "amount_formula": "latest_price * bid_vol",
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "years": list(years) if years is not None else None,
    }
    cfg.summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="提取 09:24~09:25 每个 tick 时点的成交额代理值")
    parser.add_argument("--tick-base", default="./tick_2026", help="tick 根目录")
    parser.add_argument("--start", default="", help="开始日期，支持 YYYY-MM-DD / YYYYMMDD")
    parser.add_argument("--end", default="", help="结束日期，支持 YYYY-MM-DD / YYYYMMDD")
    parser.add_argument("--years", default="", help="限制年份，逗号分隔，如 2025,2026")
    parser.add_argument(
        "--output-file",
        default="./factor/high_frequency/auction_latest_amount_924_925/auction_latest_amount_924_925_all.parquet",
        help="输出 parquet 文件",
    )
    parser.add_argument(
        "--summary-file",
        default="./factor/high_frequency/auction_latest_amount_924_925/auction_latest_amount_924_925_summary.json",
        help="输出 summary 文件",
    )
    parser.add_argument("--max-workers", type=int, default=4, help="并行进程数")
    args = parser.parse_args()

    cfg = Config(
        tick_base=Path(args.tick_base),
        output_file=Path(args.output_file),
        summary_file=Path(args.summary_file),
        max_workers=max(1, args.max_workers),
    )
    if cfg.output_file.exists():
        cfg.output_file.unlink()

    run(
        cfg=cfg,
        years=parse_years_arg(args.years),
        start_date=parse_date_arg(args.start),
        end_date=parse_date_arg(args.end),
    )


if __name__ == "__main__":
    main()
