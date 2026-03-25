#!/usr/bin/env python3
"""分析每日最高价前10转债在次日开盘阶段的价格与成交额分布。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


KZZ_PREFIXES = ("110", "111", "113", "118", "123", "127", "128")
AUCTION_START = "09:15:00"
AUCTION_END = "09:26:00"
OPEN_START = "09:30:00"
OPEN_END = "09:32:00"
TOP_N = 10
BIN_SECONDS = 10


@dataclass
class Config:
    tick_base: Path = Path("/data1/quant-data/tick_2026")
    output_dir: Path = Path("notebook/kzz_top_price_next_open_outputs")
    years: tuple[int, ...] = (2025, 2026)


def is_kzz_code(code: str) -> bool:
    return code.startswith(KZZ_PREFIXES) and code.endswith((".SH", ".SZ"))


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


def parse_years(years_arg: str) -> tuple[int, ...]:
    years = []
    for token in years_arg.split(","):
        token = token.strip()
        if token:
            years.append(int(token))
    if not years:
        raise ValueError("--years 不能为空")
    return tuple(sorted(set(years)))


def to_local_datetime(time_series: pd.Series) -> pd.Series:
    return pd.to_datetime(time_series, unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)


def compute_amount_delta(amount: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(amount, errors="coerce").fillna(0.0)
    delta = numeric.diff()
    if len(delta) > 0:
        delta.iloc[0] = numeric.iloc[0]
    return delta.clip(lower=0.0)


def get_day_top_bonds(day_dir: Path, top_n: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for fp in sorted(day_dir.glob("*.parquet")):
        code = fp.stem
        if not is_kzz_code(code):
            continue
        try:
            df = pd.read_parquet(fp, columns=["high", "lastClose"])
        except Exception:
            continue
        if df.empty:
            continue
        high = pd.to_numeric(df["high"], errors="coerce").dropna()
        if high.empty:
            continue
        last_close = pd.to_numeric(df["lastClose"], errors="coerce").dropna()
        rows.append(
            {
                "code": code,
                "prev_day_high": float(high.max()),
                "prev_day_close_ref": float(last_close.iloc[-1]) if not last_close.empty else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["code", "prev_day_high", "prev_day_close_ref"])
    return pd.DataFrame(rows).sort_values(["prev_day_high", "code"], ascending=[False, True]).head(top_n).reset_index(drop=True)


def analyze_next_day_file(
    parquet_file: Path,
    select_date: date,
    next_date: date,
    prev_day_high: float,
    prev_day_close_ref: float,
) -> tuple[Dict, List[Dict]] | tuple[None, None]:
    code = parquet_file.stem
    try:
        df = pd.read_parquet(parquet_file, columns=["time", "amount", "lastPrice", "open", "lastClose"])
    except Exception:
        return None, None

    if df.empty:
        return None, None

    df = df.sort_values("time").reset_index(drop=True)
    dt = to_local_datetime(df["time"])
    delta_amount = compute_amount_delta(df["amount"])
    trade_price = pd.to_numeric(df["lastPrice"], errors="coerce").where(lambda s: s > 0)
    open_field = pd.to_numeric(df["open"], errors="coerce").where(lambda s: s > 0)
    prev_close = pd.to_numeric(df["lastClose"], errors="coerce").where(lambda s: s > 0).dropna()

    auction_start = pd.Timestamp(f"{next_date} {AUCTION_START}")
    auction_end = pd.Timestamp(f"{next_date} {AUCTION_END}")
    open_start = pd.Timestamp(f"{next_date} {OPEN_START}")
    open_end = pd.Timestamp(f"{next_date} {OPEN_END}")

    auction_mask = (dt >= auction_start) & (dt < auction_end)
    open_mask = (dt >= open_start) & (dt < open_end)

    auction_amount = float(delta_amount[auction_mask].sum())
    open2m_amount = float(delta_amount[open_mask].sum())

    open_slice = trade_price[open_mask].dropna()
    open_price = float(open_field.dropna().iloc[0]) if not open_field.dropna().empty else np.nan
    if np.isnan(open_price) and not open_slice.empty:
        open_price = float(open_slice.iloc[0])

    open2m_last_price = float(open_slice.iloc[-1]) if not open_slice.empty else np.nan
    prev_close_price = float(prev_close.iloc[-1]) if not prev_close.empty else prev_day_close_ref

    price_change_vs_prev_close = (
        open_price / prev_close_price - 1.0
        if pd.notna(open_price) and pd.notna(prev_close_price) and prev_close_price > 0
        else np.nan
    )
    price_change_vs_prev_high = (
        open_price / prev_day_high - 1.0
        if pd.notna(open_price) and pd.notna(prev_day_high) and prev_day_high > 0
        else np.nan
    )
    open2m_return = (
        open2m_last_price / open_price - 1.0
        if pd.notna(open_price) and pd.notna(open2m_last_price) and open_price > 0
        else np.nan
    )

    bin_df = pd.DataFrame(
        {
            "dt": dt[open_mask],
            "delta_amount": delta_amount[open_mask],
        }
    )
    bin_rows: List[Dict] = []
    if not bin_df.empty:
        start = pd.Timestamp(f"{next_date} {OPEN_START}")
        bin_df["seconds_since_open"] = (bin_df["dt"] - start).dt.total_seconds().astype(int)
        bin_df = bin_df[(bin_df["seconds_since_open"] >= 0) & (bin_df["seconds_since_open"] < 120)].copy()
        bin_df["bin_index"] = bin_df["seconds_since_open"] // BIN_SECONDS
        grouped = bin_df.groupby("bin_index", as_index=False)["delta_amount"].sum()
        grouped["bin_start"] = grouped["bin_index"] * BIN_SECONDS
        for row in grouped.itertuples(index=False):
            bin_rows.append(
                {
                    "select_date": select_date.isoformat(),
                    "next_date": next_date.isoformat(),
                    "code": code,
                    "bin_index": int(row.bin_index),
                    "bin_label": build_bin_label(int(row.bin_index)),
                    "amount": float(row.delta_amount),
                }
            )

    sample_row = {
        "select_date": select_date.isoformat(),
        "next_date": next_date.isoformat(),
        "code": code,
        "prev_day_high": prev_day_high,
        "prev_day_close_ref": prev_day_close_ref,
        "next_prev_close": prev_close_price,
        "next_open_price": open_price,
        "next_open2m_last_price": open2m_last_price,
        "open_change_vs_prev_close": price_change_vs_prev_close,
        "open_change_vs_prev_high": price_change_vs_prev_high,
        "open2m_return": open2m_return,
        "auction_amount": auction_amount,
        "open_0930_0932_amount": open2m_amount,
        "open_to_auction_amount_ratio": (open2m_amount / auction_amount) if auction_amount > 0 else np.nan,
    }
    return sample_row, bin_rows


def build_bin_label(bin_index: int) -> str:
    start_seconds = bin_index * BIN_SECONDS
    end_seconds = start_seconds + BIN_SECONDS
    start_total = 9 * 3600 + 30 * 60 + start_seconds
    end_total = 9 * 3600 + 30 * 60 + end_seconds
    sh, sm = divmod(start_total, 3600)
    sm, ss = divmod(sm, 60)
    eh, em = divmod(end_total, 3600)
    em, es = divmod(em, 60)
    return f"{sh:02d}:{sm:02d}:{ss:02d}-{eh:02d}:{em:02d}:{es:02d}"


def summarize_bins(bin_df: pd.DataFrame, detail_df: pd.DataFrame) -> pd.DataFrame:
    all_bins = pd.DataFrame({"bin_index": np.arange(120 // BIN_SECONDS)})
    all_bins["bin_label"] = all_bins["bin_index"].map(build_bin_label)
    if bin_df.empty:
        return all_bins.assign(
            sample_count=0,
            mean_amount=0.0,
            median_amount=0.0,
            p25_amount=0.0,
            p75_amount=0.0,
            positive_ratio=0.0,
            mean_amount_share=0.0,
        )

    sample_base = detail_df[["select_date", "next_date", "code", "open_0930_0932_amount"]].copy()
    sample_base["_key"] = 1
    bin_base = all_bins.copy()
    bin_base["_key"] = 1
    complete = sample_base.merge(bin_base, on="_key", how="inner").drop(columns="_key")
    merged = complete.merge(
        bin_df,
        on=["select_date", "next_date", "code", "bin_index", "bin_label"],
        how="left",
    )
    merged["amount"] = merged["amount"].fillna(0.0)

    stats = (
        merged.groupby(["bin_index", "bin_label"], as_index=False)
        .agg(
            sample_count=("amount", "count"),
            mean_amount=("amount", "mean"),
            median_amount=("amount", "median"),
            p25_amount=("amount", lambda s: s.quantile(0.25)),
            p75_amount=("amount", lambda s: s.quantile(0.75)),
            positive_ratio=("amount", lambda s: float((s > 0).mean())),
        )
    )
    share = merged.copy()
    share["amount_share"] = np.where(
        share["open_0930_0932_amount"] > 0,
        share["amount"] / share["open_0930_0932_amount"],
        np.nan,
    )
    share_stats = share.groupby("bin_index", as_index=False)["amount_share"].mean().rename(columns={"amount_share": "mean_amount_share"})
    stats = stats.merge(share_stats, on="bin_index", how="left")
    stats = all_bins.merge(stats, on=["bin_index", "bin_label"], how="left").fillna(0.0)
    stats["sample_count"] = stats["sample_count"].astype(int)
    return stats.sort_values("bin_index").reset_index(drop=True)


def summarize_samples(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame(columns=["metric", "value"])

    rows = [
        ("sample_count", len(detail_df)),
        ("selection_days", detail_df["select_date"].nunique()),
        ("avg_prev_day_high", detail_df["prev_day_high"].mean()),
        ("avg_next_open_price", detail_df["next_open_price"].mean()),
        ("mean_open_change_vs_prev_close", detail_df["open_change_vs_prev_close"].mean()),
        ("median_open_change_vs_prev_close", detail_df["open_change_vs_prev_close"].median()),
        ("mean_open_change_vs_prev_high", detail_df["open_change_vs_prev_high"].mean()),
        ("median_open_change_vs_prev_high", detail_df["open_change_vs_prev_high"].median()),
        ("mean_open2m_return", detail_df["open2m_return"].mean()),
        ("median_open2m_return", detail_df["open2m_return"].median()),
        ("mean_auction_amount", detail_df["auction_amount"].mean()),
        ("median_auction_amount", detail_df["auction_amount"].median()),
        ("mean_open_0930_0932_amount", detail_df["open_0930_0932_amount"].mean()),
        ("median_open_0930_0932_amount", detail_df["open_0930_0932_amount"].median()),
        ("mean_open_to_auction_amount_ratio", detail_df["open_to_auction_amount_ratio"].mean()),
        ("median_open_to_auction_amount_ratio", detail_df["open_to_auction_amount_ratio"].median()),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_markdown_report(summary_df: pd.DataFrame, bin_summary_df: pd.DataFrame, output_file: Path) -> None:
    lines = [
        "# 每日最高价前10转债次日开盘分析",
        "",
        "## 样本概览",
        summary_df.to_markdown(index=False) if not summary_df.empty else "无有效样本。",
        "",
        "## 9:30-9:32 每10秒成交额分布",
        bin_summary_df.round(4).to_markdown(index=False) if not bin_summary_df.empty else "无有效分箱统计。",
        "",
    ]
    output_file.write_text("\n".join(lines), encoding="utf-8")


def analyze(config: Config, start_date: date | None, end_date: date | None) -> Dict[str, pd.DataFrame]:
    tick_dates = list_tick_dates(config.tick_base, config.years)
    if start_date is not None:
        tick_dates = [(d, p) for d, p in tick_dates if d >= start_date]
    if end_date is not None:
        tick_dates = [(d, p) for d, p in tick_dates if d <= end_date]
    if len(tick_dates) < 2:
        raise RuntimeError("可用交易日不足，至少需要两个交易日。")

    sample_rows: List[Dict] = []
    bin_rows: List[Dict] = []

    for idx in range(len(tick_dates) - 1):
        select_date, select_dir = tick_dates[idx]
        next_date, next_dir = tick_dates[idx + 1]

        top_df = get_day_top_bonds(select_dir, TOP_N)
        if top_df.empty:
            continue

        for row in top_df.itertuples(index=False):
            next_file = next_dir / f"{row.code}.parquet"
            if not next_file.exists():
                continue
            sample_row, sample_bin_rows = analyze_next_day_file(
                next_file,
                select_date=select_date,
                next_date=next_date,
                prev_day_high=float(row.prev_day_high),
                prev_day_close_ref=float(row.prev_day_close_ref),
            )
            if sample_row is None:
                continue
            sample_rows.append(sample_row)
            bin_rows.extend(sample_bin_rows)

        if (idx + 1) % 20 == 0:
            print(f"processed {idx + 1}/{len(tick_dates) - 1} selection days, samples={len(sample_rows)}")

    detail_df = pd.DataFrame(sample_rows)
    bin_df = pd.DataFrame(bin_rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(["select_date", "code"]).reset_index(drop=True)
    if not bin_df.empty:
        bin_df = bin_df.sort_values(["select_date", "code", "bin_index"]).reset_index(drop=True)

    bin_summary_df = summarize_bins(bin_df, detail_df)
    sample_summary_df = summarize_samples(detail_df)
    daily_summary_df = (
        detail_df.groupby(["select_date", "next_date"], as_index=False)
        .agg(
            sample_count=("code", "count"),
            mean_open_change_vs_prev_close=("open_change_vs_prev_close", "mean"),
            mean_open_change_vs_prev_high=("open_change_vs_prev_high", "mean"),
            mean_open2m_return=("open2m_return", "mean"),
            mean_auction_amount=("auction_amount", "mean"),
            mean_open_0930_0932_amount=("open_0930_0932_amount", "mean"),
        )
        if not detail_df.empty
        else pd.DataFrame()
    )

    return {
        "detail": detail_df,
        "bin_detail": bin_df,
        "bin_summary": bin_summary_df,
        "sample_summary": sample_summary_df,
        "daily_summary": daily_summary_df,
    }


def save_outputs(output_dir: Path, results: Dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results["detail"].to_csv(output_dir / "kzz_top_price_next_open_detail.csv", index=False)
    results["bin_detail"].to_csv(output_dir / "kzz_top_price_next_open_bin_detail.csv", index=False)
    results["bin_summary"].to_csv(output_dir / "kzz_top_price_next_open_bin_summary.csv", index=False)
    results["sample_summary"].to_csv(output_dir / "kzz_top_price_next_open_sample_summary.csv", index=False)
    results["daily_summary"].to_csv(output_dir / "kzz_top_price_next_open_daily_summary.csv", index=False)
    build_markdown_report(
        results["sample_summary"],
        results["bin_summary"],
        output_dir / "kzz_top_price_next_open_report.md",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="分析每日最高价前10转债在次日开盘阶段的成交额与价格变化")
    parser.add_argument("--years", default="2025,2026", help="逗号分隔的年份列表，例如 2025,2026")
    parser.add_argument("--start-date", default=None, help="起始交易日，格式 YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="结束交易日，格式 YYYY-MM-DD")
    parser.add_argument("--output-dir", default="notebook/kzz_top_price_next_open_outputs", help="输出目录")
    args = parser.parse_args()

    config = Config(
        output_dir=Path(args.output_dir),
        years=parse_years(args.years),
    )
    start_date = pd.to_datetime(args.start_date).date() if args.start_date else None
    end_date = pd.to_datetime(args.end_date).date() if args.end_date else None

    results = analyze(config, start_date=start_date, end_date=end_date)
    save_outputs(config.output_dir, results)

    print(f"detail rows: {len(results['detail'])}")
    print(f"bin rows   : {len(results['bin_detail'])}")
    print(f"outputs    : {config.output_dir}")


if __name__ == "__main__":
    main()
