#!/usr/bin/env python3
"""构建集合竞价盘口快照因子。

保存的 4 个指标都只使用集合竞价阶段的盘口快照，时间窗口定义为
09:15:00 <= tick_time < 09:25:00。这里刻意排除 09:25:00 及之后的记录，
避免把开盘撮合后或 9:25 整点之后的盘口混入集合竞价特征。

保存字段说明:
- auction_last1_ask1_ret:
  9:25 之前最后一个 tick 的卖一价 askPrice1，相对昨收 lastClose 的涨幅，
  计算公式为 askPrice1 / lastClose - 1。
  这个指标反映集合竞价结束前最后时刻的卖盘定价强弱。

- auction_last2_ask1_ret:
  9:25 之前倒数第二个 tick 的卖一价 askPrice1，相对昨收 lastClose 的涨幅，
  计算公式同样为 askPrice1 / lastClose - 1。
  这个指标用来和最后一笔快照对比，观察集合竞价尾部报价是否出现跳变。

- auction_last1_askVol1:
  9:25 之前最后一个 tick 的卖一量 askVol1。
  这个指标反映集合竞价结束前最后时刻卖一档挂单量的大小。

- auction_last2_askVol1:
  9:25 之前倒数第二个 tick 的卖一量 askVol1。
  这个指标和最后一个卖一量配合使用，可以衡量集合竞价尾部挂单量是否明显放大或缩小。

输出:
- 日频明细: factor/high_frequency/call_auction_snapshot/call_auction_snapshot_YYYY_MM_DD.parquet
- 年度宽表: factor/by_factor/<factor_name>_YYYY.parquet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


FACTOR_NAMES = [
    "auction_last1_ask1_ret",
    "auction_last2_ask1_ret",
    "auction_last1_askVol1",
    "auction_last2_askVol1",
]


@dataclass
class Config:
    tick_base: Path = Path("/data1/quant-data/tick_2026")
    daily_output_dir: Path = Path("./factor/high_frequency/call_auction_snapshot")
    wide_output_dir: Path = Path("./factor/by_factor")


def list_tick_dates(tick_base: Path, years: Sequence[int]) -> List[Tuple[date, Path]]:
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
                    d = date(year, int(month_dir.name), int(day_dir.name))
                except ValueError:
                    continue
                results.append((d, day_dir))
    return sorted(results, key=lambda x: x[0])


def filter_by_date_range(
    items: Iterable[Tuple[date, Path]],
    start_date: date | None,
    end_date: date | None,
) -> List[Tuple[date, Path]]:
    results = []
    for d, p in items:
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        results.append((d, p))
    return results


def _auction_mask(local_dt: pd.Series) -> pd.Series:
    hour = local_dt.dt.hour
    minute = local_dt.dt.minute
    return ((hour == 9) & (minute >= 15) & (minute < 25))


def _first_level(value) -> float:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.nan
        return pd.to_numeric(value.flat[0], errors="coerce")
    if isinstance(value, (list, tuple)):
        if not value:
            return np.nan
        return pd.to_numeric(value[0], errors="coerce")
    return pd.to_numeric(value, errors="coerce")


def _calc_ret(price: float, last_close: float) -> float:
    if pd.isna(price) or pd.isna(last_close) or last_close == 0:
        return np.nan
    return float(price / last_close - 1.0)


def compute_day_factor(day_dir: Path, trade_date: date) -> pd.DataFrame:
    """计算单个交易日的 4 个集合竞价盘口快照因子。

    对每只股票:
    1. 读取 time、lastClose、askPrice、askVol 四列。
    2. 将毫秒时间戳转换为 Asia/Shanghai 时区本地时间。
    3. 只保留 09:15:00 <= t < 09:25:00 的集合竞价 tick。
    4. 提取最后一个和倒数第二个 tick 的 askPrice1 / askVol1。
    5. 计算两个价格涨幅指标和两个卖一量指标。

    若某只股票在集合竞价窗口内没有 tick，则该股票当天不输出。
    若只有 1 条 tick，则倒数第二个相关指标记为 NaN。
    """
    rows = []
    parquet_files = sorted(day_dir.glob("*.parquet"))

    for fp in parquet_files:
        code = fp.stem
        try:
            df = pd.read_parquet(fp, columns=["time", "lastClose", "askPrice", "askVol"])
        except Exception:
            continue

        if df.empty:
            continue

        dt = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
        auction_df = df.loc[_auction_mask(dt)].copy()
        if auction_df.empty:
            continue

        auction_df["askPrice1"] = auction_df["askPrice"].apply(_first_level)
        auction_df["askVol1"] = auction_df["askVol"].apply(_first_level)

        last_row = auction_df.iloc[-1]
        prev_row = auction_df.iloc[-2] if len(auction_df) >= 2 else None
        last_close = pd.to_numeric(last_row.get("lastClose"), errors="coerce")

        rows.append(
            {
                "date": trade_date.strftime("%Y-%m-%d"),
                "stock_code": code,
                "auction_last1_ask1_ret": _calc_ret(
                    pd.to_numeric(last_row.get("askPrice1"), errors="coerce"),
                    last_close,
                ),
                "auction_last2_ask1_ret": _calc_ret(
                    pd.to_numeric(prev_row.get("askPrice1"), errors="coerce") if prev_row is not None else np.nan,
                    last_close,
                ),
                "auction_last1_askVol1": pd.to_numeric(last_row.get("askVol1"), errors="coerce"),
                "auction_last2_askVol1": pd.to_numeric(prev_row.get("askVol1"), errors="coerce") if prev_row is not None else np.nan,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["date", "stock_code", *FACTOR_NAMES])
    return result.sort_values("stock_code").reset_index(drop=True)


def save_day_factor(df: pd.DataFrame, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False, engine="pyarrow")


def build_wide_for_year(year: int, cfg: Config) -> List[Path]:
    files = sorted(cfg.daily_output_dir.glob(f"call_auction_snapshot_{year}_*.parquet"))
    if not files:
        return []

    dfs = []
    for f in files:
        try:
            daily_df = pd.read_parquet(f)
        except Exception:
            continue
        if daily_df.empty:
            continue
        dfs.append(daily_df)

    if not dfs:
        return []

    merged = pd.concat(dfs, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])
    out_files: List[Path] = []

    cfg.wide_output_dir.mkdir(parents=True, exist_ok=True)
    for factor_name in FACTOR_NAMES:
        wide = merged.pivot_table(
            index="date",
            columns="stock_code",
            values=factor_name,
            aggfunc="first",
        ).sort_index()
        wide.index = wide.index.strftime("%Y-%m-%d")
        wide.columns = wide.columns.astype(str)

        out_file = cfg.wide_output_dir / f"{factor_name}_{year}.parquet"
        wide.to_parquet(out_file, engine="pyarrow")
        out_files.append(out_file)

    return out_files


def parse_years(years_arg: str) -> List[int]:
    years: List[int] = []
    for token in years_arg.split(","):
        token = token.strip()
        if not token:
            continue
        years.append(int(token))
    if not years:
        raise ValueError("--years 不能为空")
    return sorted(set(years))


def parse_date_arg(value: str | None) -> date | None:
    if not value:
        return None
    return pd.to_datetime(value).date()


def run_backfill(
    cfg: Config,
    years: Sequence[int],
    start_date: date | None,
    end_date: date | None,
    skip_existing: bool,
) -> None:
    tick_dates = filter_by_date_range(list_tick_dates(cfg.tick_base, years), start_date, end_date)
    touched_years: set[int] = set()

    for i, (d, day_dir) in enumerate(tick_dates, 1):
        out_file = cfg.daily_output_dir / f"call_auction_snapshot_{d.strftime('%Y_%m_%d')}.parquet"
        if skip_existing and out_file.exists():
            continue

        day_df = compute_day_factor(day_dir, d)
        save_day_factor(day_df, out_file)
        touched_years.add(d.year)
        print(f"[{i}/{len(tick_dates)}] {d} -> {out_file} ({len(day_df)} rows)")

    for year in sorted(touched_years):
        out_files = build_wide_for_year(year, cfg)
        for out_file in out_files:
            print(f"updated wide: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建集合竞价盘口快照因子")
    parser.add_argument("--tick-base", default="/data1/quant-data/tick_2026")
    parser.add_argument("--daily-output-dir", default="./factor/high_frequency/call_auction_snapshot")
    parser.add_argument("--wide-output-dir", default="./factor/by_factor")
    parser.add_argument("--years", required=True, help="如 2025,2026")
    parser.add_argument("--start-date", default=None, help="如 2026-01-01")
    parser.add_argument("--end-date", default=None, help="如 2026-01-31")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        tick_base=Path(args.tick_base),
        daily_output_dir=Path(args.daily_output_dir),
        wide_output_dir=Path(args.wide_output_dir),
    )
    run_backfill(
        cfg=cfg,
        years=parse_years(args.years),
        start_date=parse_date_arg(args.start_date),
        end_date=parse_date_arg(args.end_date),
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
