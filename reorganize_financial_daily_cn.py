"""
将财务日频中文数据重组为与 daily/daily_basic 一致的目录结构。

输入(现有):
- daily_data/cashflow_daily/cashflow_daily_YYYYMMDD_cn.parquet
- daily_data/income_daily/income_daily_YYYYMMDD_cn.parquet
- daily_data/balance_daily/balance_daily_YYYYMMDD_cn.parquet

输出(重组后):
- daily_data/cashflow_daily_cn/YYYY/MM/cashflow_daily_cn_YYYYMMDD.parquet
- daily_data/income_daily_cn/YYYY/MM/income_daily_cn_YYYYMMDD.parquet
- daily_data/balance_daily_cn/YYYY/MM/balance_daily_cn_YYYYMMDD.parquet
- daily_data/*_daily_cn/YYYY_all.parquet
- daily_data/*_daily_cn/YYYY_full.parquet
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, Iterable, Optional, Set

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


TABLE_MAP = {
    "cashflow_daily": {
        "src_dir": Path("daily_data/cashflow_daily"),
        "src_prefix": "cashflow_daily",
        "dst_dir": Path("daily_data/cashflow_daily_cn"),
        "dst_prefix": "cashflow_daily_cn",
    },
    "income_daily": {
        "src_dir": Path("daily_data/income_daily"),
        "src_prefix": "income_daily",
        "dst_dir": Path("daily_data/income_daily_cn"),
        "dst_prefix": "income_daily_cn",
    },
    "balance_daily": {
        "src_dir": Path("daily_data/balance_daily"),
        "src_prefix": "balance_daily",
        "dst_dir": Path("daily_data/balance_daily_cn"),
        "dst_prefix": "balance_daily_cn",
    },
}


def _parse_date_from_name(name: str, src_prefix: str) -> str:
    m = re.match(rf"^{src_prefix}_(\d{{8}})_cn\.parquet$", name)
    if not m:
        return ""
    return m.group(1)


def _normalize_years(years: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if years is None:
        return None
    result: Set[str] = set()
    for y in years:
        y = str(y)
        if not re.fullmatch(r"\d{4}", y):
            raise ValueError(f"非法年份: {y}")
        result.add(y)
    return result


def _copy_daily_files(
    src_dir: Path,
    src_prefix: str,
    dst_dir: Path,
    dst_prefix: str,
    years: Optional[Set[str]],
    overwrite_daily: bool
) -> Dict[str, int]:
    copied = 0
    touched_years: Set[str] = set()

    src_files = sorted(src_dir.glob(f"{src_prefix}_*_cn.parquet"))
    for f in src_files:
        date_str = _parse_date_from_name(f.name, src_prefix)
        if not date_str:
            continue

        year = date_str[:4]
        if years is not None and year not in years:
            continue

        month = date_str[4:6]
        ymd_dir = dst_dir / year / month
        ymd_dir.mkdir(parents=True, exist_ok=True)
        out_file = ymd_dir / f"{dst_prefix}_{date_str}.parquet"

        if out_file.exists() and not overwrite_daily:
            touched_years.add(year)
            continue

        # 读写一次，确保输出格式稳定
        df = pq.read_table(f).to_pandas()
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(out_file))
        copied += 1
        touched_years.add(year)

    return {"copied": copied, "touched_years": len(touched_years), "years": touched_years}


def _rebuild_year_merged(dst_dir: Path, dst_prefix: str, year: str) -> int:
    year_dir = dst_dir / year
    if not year_dir.exists():
        return 0

    files = sorted(year_dir.glob(f"*/{dst_prefix}_*.parquet"))
    if not files:
        return 0

    parts = []
    for f in files:
        try:
            parts.append(pq.read_table(f).to_pandas())
        except Exception:
            continue

    if not parts:
        return 0

    year_df = pd.concat(parts, ignore_index=True)
    year_df = year_df.drop_duplicates()
    table = pa.Table.from_pandas(year_df, preserve_index=False)
    pq.write_table(table, str(dst_dir / f"{year}_all.parquet"))
    pq.write_table(table, str(dst_dir / f"{year}_full.parquet"))
    return len(files)


def reorganize_one_table(
    table_key: str,
    years: Optional[Set[str]] = None,
    overwrite_daily: bool = False
) -> Dict[str, int]:
    cfg = TABLE_MAP[table_key]
    src_dir: Path = cfg["src_dir"]
    src_prefix: str = cfg["src_prefix"]
    dst_dir: Path = cfg["dst_dir"]
    dst_prefix: str = cfg["dst_prefix"]
    dst_dir.mkdir(parents=True, exist_ok=True)

    copy_stats = _copy_daily_files(
        src_dir=src_dir,
        src_prefix=src_prefix,
        dst_dir=dst_dir,
        dst_prefix=dst_prefix,
        years=years,
        overwrite_daily=overwrite_daily,
    )

    rebuilt_years = 0
    merged_source_files = 0
    for year in sorted(copy_stats["years"]):
        merged_source_files += _rebuild_year_merged(dst_dir, dst_prefix, year)
        rebuilt_years += 1

    return {
        "daily_files_copied": int(copy_stats["copied"]),
        "years_touched": int(copy_stats["touched_years"]),
        "years_rebuilt": rebuilt_years,
        "merged_source_files": merged_source_files,
    }


def _resolve_tables(args: argparse.Namespace) -> list[str]:
    if args.tables:
        tables = [t.strip() for t in args.tables]
        for t in tables:
            if t not in TABLE_MAP:
                raise ValueError(f"未知表: {t}, 可选: {list(TABLE_MAP.keys())}")
        return tables
    return list(TABLE_MAP.keys())


def main():
    parser = argparse.ArgumentParser(description="重组财务中文日频数据 -> 年/月 + 年合并文件")
    parser.add_argument(
        "--year",
        dest="years",
        action="append",
        help="指定年份，可重复传入，例如 --year 2026 --year 2025",
    )
    parser.add_argument(
        "--current-year",
        action="store_true",
        help="只处理当前年份",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=list(TABLE_MAP.keys()),
        help="指定处理表，默认全部",
    )
    parser.add_argument(
        "--overwrite-daily",
        action="store_true",
        help="覆盖已存在的日文件（默认只补缺）",
    )
    args = parser.parse_args()

    years = set(args.years or [])
    if args.current_year:
        years.add(str(datetime.now().year))
    years_filter = _normalize_years(years) if years else None
    tables = _resolve_tables(args)

    print("=" * 72)
    print("重组财务中文日频数据 -> 年/月 + 年合并文件")
    print(f"tables={tables}")
    print(f"years={sorted(years_filter) if years_filter else 'ALL'}")
    print(f"overwrite_daily={args.overwrite_daily}")
    print("=" * 72)

    results = {}
    for table_key in tables:
        print(f"\n处理 {table_key} ...")
        stats = reorganize_one_table(
            table_key=table_key,
            years=years_filter,
            overwrite_daily=args.overwrite_daily,
        )
        results[table_key] = stats
        print(f"  拷贝日文件: {stats['daily_files_copied']}")
        print(f"  触达年份: {stats['years_touched']}")
        print(f"  重建年文件: {stats['years_rebuilt']}")

    print("\n" + "=" * 72)
    print("完成")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("=" * 72)


if __name__ == "__main__":
    main()
