#!/usr/bin/env python3
"""构建集合竞价成交额高频因子。

因子定义:
- 09:15:00 <= tick_time < 09:26:00 的成交额增量之和。

数据输入:
- /data1/quant-data/tick_2026/{year}/{month}/{day}/*.parquet

数据输出:
- 日频明细: factor/high_frequency/<factor_name>/<factor_name>_YYYY_MM_DD.parquet
- 因子宽表: factor/by_factor/<factor_name>_YYYY.parquet
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd


FACTOR_NAME = "kzz_call_auction_amount"  # 向后兼容：默认仍指向可转债版本
KZZ_PREFIXES = ("110", "111", "113", "118", "123", "127", "128")
FACTOR_NAME_BY_UNIVERSE = {
    "kzz": "kzz_call_auction_amount",
    "all": "call_auction_amount_all",
}


@dataclass
class Config:
    tick_base: Path = Path("/data1/quant-data/tick_2026")
    daily_output_dir: Path = Path("./factor/high_frequency/kzz_call_auction_amount")
    wide_output_dir: Path = Path("./factor/by_factor")


def is_kzz_code(code: str) -> bool:
    return code.startswith(KZZ_PREFIXES) and code.endswith((".SH", ".SZ"))


def should_include_code(code: str, universe: str) -> bool:
    if universe == "all":
        return code.endswith((".SH", ".SZ"))
    return is_kzz_code(code)


def parse_years(years_arg: str) -> List[int]:
    years: List[int] = []
    for token in years_arg.split(","):
        token = token.strip()
        if not token:
            continue
        y = int(token)
        years.append(y)
    if not years:
        raise ValueError("--years 不能为空")
    return sorted(set(years))


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


def output_file_for_date(d: date, daily_output_dir: Path, factor_name: str = FACTOR_NAME) -> Path:
    return daily_output_dir / f"{factor_name}_{d.strftime('%Y_%m_%d')}.parquet"


def parse_existing_factor_dates(daily_output_dir: Path, factor_name: str = FACTOR_NAME) -> set[date]:
    existing: set[date] = set()
    if not daily_output_dir.exists():
        return existing

    pattern = re.compile(rf"^{factor_name}_(\d{{4}})_(\d{{2}})_(\d{{2}})\.parquet$")
    for f in daily_output_dir.glob(f"{factor_name}_*.parquet"):
        m = pattern.match(f.name)
        if not m:
            continue
        y, mth, d = map(int, m.groups())
        try:
            existing.add(date(y, mth, d))
        except ValueError:
            continue
    return existing


def _auction_mask(local_dt: pd.Series) -> pd.Series:
    # 09:15:00 <= t < 09:26:00
    h = local_dt.dt.hour
    m = local_dt.dt.minute
    return (h == 9) & (m >= 15) & (m <= 25)


def compute_day_factor(
    day_dir: Path,
    trade_date: date,
    universe: str = "kzz",
    factor_name: str = FACTOR_NAME,
) -> pd.DataFrame:
    rows = []
    parquet_files = sorted(day_dir.glob("*.parquet"))

    for fp in parquet_files:
        code = fp.stem
        if not should_include_code(code, universe):
            continue

        try:
            df = pd.read_parquet(fp, columns=["time", "amount"])
        except Exception:
            continue

        if df.empty:
            continue

        dt = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
        amount = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

        delta = amount.diff()
        if len(delta) > 0:
            delta.iloc[0] = amount.iloc[0]
        delta = delta.clip(lower=0.0)

        auction_amount = float(delta[_auction_mask(dt)].sum())

        rows.append(
            {
                "date": trade_date.strftime("%Y-%m-%d"),
                "stock_code": code,
                factor_name: auction_amount,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("stock_code").reset_index(drop=True)
    else:
        result = pd.DataFrame(columns=["date", "stock_code", factor_name])
    return result


def save_day_factor(df: pd.DataFrame, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, index=False, engine="pyarrow")


def build_wide_for_year(year: int, cfg: Config, factor_name: str = FACTOR_NAME) -> Path | None:
    pattern = f"{factor_name}_{year}_*.parquet"
    files = sorted(cfg.daily_output_dir.glob(pattern))
    if not files:
        return None

    dfs = []
    for f in files:
        try:
            daily_df = pd.read_parquet(f, columns=["date", "stock_code", factor_name])
        except Exception:
            continue
        if daily_df.empty:
            continue
        dfs.append(daily_df)

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])

    wide = merged.pivot_table(
        index="date",
        columns="stock_code",
        values=factor_name,
        aggfunc="first",
    ).sort_index()

    wide.index = wide.index.strftime("%Y-%m-%d")
    wide.columns = wide.columns.astype(str)

    cfg.wide_output_dir.mkdir(parents=True, exist_ok=True)
    out_file = cfg.wide_output_dir / f"{factor_name}_{year}.parquet"
    wide.to_parquet(out_file, engine="pyarrow")
    return out_file


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


def run_backfill(
    cfg: Config,
    years: Sequence[int],
    start_date: date | None,
    end_date: date | None,
    skip_existing: bool,
    universe: str = "kzz",
    factor_name: str = FACTOR_NAME,
) -> None:
    tick_dates = list_tick_dates(cfg.tick_base, years)
    tick_dates = filter_by_date_range(tick_dates, start_date, end_date)

    existing_dates = parse_existing_factor_dates(cfg.daily_output_dir, factor_name=factor_name) if skip_existing else set()
    targets = [(d, p) for d, p in tick_dates if d not in existing_dates]

    print(f"tick交易日: {len(tick_dates)}, 待处理: {len(targets)}, 已存在跳过: {len(tick_dates) - len(targets)}")

    touched_years: set[int] = set()
    for i, (d, day_dir) in enumerate(targets, 1):
        out_file = output_file_for_date(d, cfg.daily_output_dir, factor_name=factor_name)
        day_df = compute_day_factor(day_dir, d, universe=universe, factor_name=factor_name)
        save_day_factor(day_df, out_file)
        touched_years.add(d.year)
        if i % 10 == 0 or i == len(targets):
            print(f"[{i}/{len(targets)}] {d} -> rows={len(day_df)}")

    # 回填完成后，重建对应年份宽表
    years_to_build = sorted(set(years) | touched_years)
    for y in years_to_build:
        out = build_wide_for_year(y, cfg, factor_name=factor_name)
        if out:
            print(f"wide[{y}] -> {out}")


def run_update(
    cfg: Config,
    years: Sequence[int],
    today_cutoff: date | None,
    universe: str = "kzz",
    factor_name: str = FACTOR_NAME,
) -> None:
    tick_dates = list_tick_dates(cfg.tick_base, years)
    existing = parse_existing_factor_dates(cfg.daily_output_dir, factor_name=factor_name)

    targets: List[Tuple[date, Path]] = []
    for d, p in tick_dates:
        if d in existing:
            continue
        if today_cutoff and d > today_cutoff:
            continue
        targets.append((d, p))

    print(f"增量更新待处理交易日: {len(targets)}")

    touched_years: set[int] = set()
    for i, (d, day_dir) in enumerate(targets, 1):
        out_file = output_file_for_date(d, cfg.daily_output_dir, factor_name=factor_name)
        day_df = compute_day_factor(day_dir, d, universe=universe, factor_name=factor_name)
        save_day_factor(day_df, out_file)
        touched_years.add(d.year)
        if i % 5 == 0 or i == len(targets):
            print(f"[update {i}/{len(targets)}] {d} -> rows={len(day_df)}")

    if not touched_years:
        print("无新增交易日，无需重建宽表")
        return

    for y in sorted(touched_years):
        out = build_wide_for_year(y, cfg, factor_name=factor_name)
        if out:
            print(f"wide[{y}] -> {out}")


def parse_date_arg(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="构建集合竞价成交额因子")
    parser.add_argument("--mode", choices=["backfill", "update"], default="backfill", help="backfill=批量回填, update=增量更新")
    parser.add_argument("--universe", choices=["all", "kzz"], default="all", help="all=全部股票+转债, kzz=仅可转债")
    parser.add_argument("--years", default="2025,2026", help="年份列表，逗号分隔")
    parser.add_argument("--start-date", default=None, help="起始日期 YYYY-MM-DD，仅backfill生效")
    parser.add_argument("--end-date", default=None, help="结束日期 YYYY-MM-DD，仅backfill生效")
    parser.add_argument("--skip-existing", action="store_true", help="backfill时跳过已存在日文件")
    parser.add_argument("--tick-base", default="/data1/quant-data/tick_2026", help="tick根目录")
    parser.add_argument("--daily-output", default=None, help="日频因子输出目录")
    parser.add_argument("--wide-output", default="./factor/by_factor", help="宽表输出目录")
    parser.add_argument("--today-cutoff", default=None, help="update模式的最大处理日期 YYYY-MM-DD，默认不限制")

    args = parser.parse_args()

    factor_name = FACTOR_NAME_BY_UNIVERSE[args.universe]
    daily_output = (
        Path(args.daily_output)
        if args.daily_output
        else Path(f"./factor/high_frequency/{factor_name}")
    )

    cfg = Config(
        tick_base=Path(args.tick_base),
        daily_output_dir=daily_output,
        wide_output_dir=Path(args.wide_output),
    )
    years = parse_years(args.years)

    if args.mode == "backfill":
        run_backfill(
            cfg=cfg,
            years=years,
            start_date=parse_date_arg(args.start_date),
            end_date=parse_date_arg(args.end_date),
            skip_existing=bool(args.skip_existing),
            universe=args.universe,
            factor_name=factor_name,
        )
    else:
        run_update(
            cfg=cfg,
            years=years,
            today_cutoff=parse_date_arg(args.today_cutoff),
            universe=args.universe,
            factor_name=factor_name,
        )


if __name__ == "__main__":
    main()
