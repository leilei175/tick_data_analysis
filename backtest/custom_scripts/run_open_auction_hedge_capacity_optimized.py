#!/usr/bin/env python3
"""开盘集合竞价涨跌幅对冲策略 - 容量优化版。

主要优化:
1. hold_count: 50 -> 200 (底仓分散)
2. trade_count: 10 -> 40 (日内调仓分散)
3. min_total_mv: 10亿 -> 30亿 (提升选股池流动性)
4. entry_price: 09:30 Open -> 09:30-09:35 VWAP (分散执行)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_BACKTEST_DIR = _THIS_DIR.parent
_REPO_ROOT = _BACKTEST_DIR.parent
if str(_BACKTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKTEST_DIR))

from data_source import load_combined_panel
from performance_metrics import compute_performance_metrics


AUCTION_CUTOFF = time(9, 24, 50)
VWAP_START = time(9, 30, 0)
VWAP_END = time(9, 35, 0)


@dataclass(frozen=True)
class StrategyConfig:
    min_amount: float = 10_000.0
    min_total_mv: float = 300_000.0
    hold_count: int = 200
    trade_count: int = 40
    buy_fee: float = 0.0002
    sell_fee: float = 0.0007
    slippage: float = 0.0003
    cash: float = 10_000_000.0


def _ensure_ts(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _float_or_nan(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    if not math.isfinite(out):
        return np.nan
    return out


def _parse_level1(value: object) -> float:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.nan
        return _float_or_nan(value.flat[0])
    if isinstance(value, (list, tuple)):
        if not value:
            return np.nan
        return _float_or_nan(value[0])
    if isinstance(value, str):
        s = value.strip().strip("[]")
        if not s:
            return np.nan
        first = s.split(",")[0].strip()
        return _float_or_nan(first)
    return _float_or_nan(value)


def list_tick_dates(tick_root: Path, start_year: int, end_year: int) -> List[pd.Timestamp]:
    out: List[pd.Timestamp] = []
    for year in range(start_year, end_year + 1):
        year_dir = tick_root / f"{year:04d}"
        if not year_dir.exists():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            for day_dir in sorted(month_dir.iterdir()):
                if not day_dir.is_dir():
                    continue
                if any(day_dir.glob("*.parquet")):
                    out.append(pd.Timestamp(f"{year:04d}-{month_dir.name}-{day_dir.name}").normalize())
    return sorted(set(out))


def tick_file_path(tick_root: Path, trade_date: pd.Timestamp, ts_code: str) -> Path:
    return (
        tick_root
        / f"{trade_date.year:04d}"
        / f"{trade_date.month:02d}"
        / f"{trade_date.day:02d}"
        / f"{ts_code}.parquet"
    )


def pick_weekly_universe(prev_panel: pd.DataFrame, config: StrategyConfig) -> List[str]:
    if prev_panel.empty:
        return []
    use = prev_panel.copy()
    use = use[use["ts_code"].str.endswith((".SH", ".SZ"))]
    code_prefix = use["ts_code"].str.split(".", n=1).str[0]
    use = use[~code_prefix.str.startswith(("200", "900"))]
    use = use.dropna(subset=["amount", "total_mv"])
    use = use[(use["amount"] >= config.min_amount) & (use["total_mv"] >= config.min_total_mv)]
    if use.empty:
        return []
    use = use.sort_values(["total_mv", "ts_code"], ascending=[True, True])
    use = use.drop_duplicates(subset=["ts_code"], keep="first")
    return use.head(config.hold_count)["ts_code"].tolist()


def build_weekly_holding_map(
    panel: pd.DataFrame,
    trade_dates: Sequence[pd.Timestamp],
    config: StrategyConfig,
) -> Tuple[Dict[pd.Timestamp, List[str]], pd.DataFrame]:
    by_date: Dict[pd.Timestamp, pd.DataFrame] = {
        _ensure_ts(d): df.drop_duplicates(subset=["ts_code"], keep="first").reset_index(drop=True)
        for d, df in panel.groupby("trade_date")
    }
    ordered_dates = sorted(by_date)
    prev_date_map = {ordered_dates[i]: ordered_dates[i - 1] for i in range(1, len(ordered_dates))}

    holding_map: Dict[pd.Timestamp, List[str]] = {}
    log_rows: List[Dict[str, object]] = []
    current_holdings: List[str] = []
    current_rebalance_date: Optional[pd.Timestamp] = None
    last_week_key: Optional[Tuple[int, int]] = None

    for trade_date in trade_dates:
        trade_date = _ensure_ts(trade_date)
        week_key = trade_date.isocalendar()[:2]
        need_rebalance = week_key != last_week_key
        if need_rebalance and trade_date in prev_date_map:
            prev_trade_date = prev_date_map[trade_date]
            current_holdings = pick_weekly_universe(by_date.get(prev_trade_date, pd.DataFrame()), config)
            current_rebalance_date = trade_date
            last_week_key = week_key
            log_rows.append(
                {
                    "trade_date": trade_date,
                    "prev_trade_date": prev_trade_date,
                    "hold_count": len(current_holdings),
                    "symbols_preview": ",".join(current_holdings[:10]),
                }
            )
        elif need_rebalance:
            last_week_key = week_key

        holding_map[trade_date] = list(current_holdings)

    log_df = pd.DataFrame(log_rows)
    if not log_df.empty:
        log_df["trade_date"] = pd.to_datetime(log_df["trade_date"])
        log_df["prev_trade_date"] = pd.to_datetime(log_df["prev_trade_date"])
    return holding_map, log_df


def extract_auction_and_vwap(file_path: Path) -> Dict[str, float]:
    if not file_path.exists():
        return {}
    try:
        df = pd.read_parquet(file_path, columns=["time", "lastPrice", "lastClose", "askPrice", "bidPrice", "amount", "volume"])
    except Exception:
        return {}
    if df.empty:
        return {}

    local_dt = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    df = df.assign(dt=local_dt)
    
    # 1. Auction signal (9:24:50)
    auction = df[(df["dt"].dt.time >= time(9, 15)) & (df["dt"].dt.time <= AUCTION_CUTOFF)].copy()
    snap_price = np.nan
    snap_return = np.nan
    
    if not auction.empty:
        auction["ask1"] = auction["askPrice"].apply(_parse_level1)
        auction["bid1"] = auction["bidPrice"].apply(_parse_level1)
        auction["lastPrice"] = pd.to_numeric(auction["lastPrice"], errors="coerce")
        auction["lastClose"] = pd.to_numeric(auction["lastClose"], errors="coerce")

        price_s = auction["lastPrice"].where(auction["lastPrice"] > 0)
        mid_s = ((auction["ask1"] + auction["bid1"]) / 2.0).where(
            (auction["ask1"] > 0) & (auction["bid1"] > 0)
        )
        ref_price = price_s.fillna(mid_s).fillna(auction["ask1"]).fillna(auction["bid1"])
        last_valid_idx = ref_price.last_valid_index()
        if last_valid_idx is not None:
            snap_price = _float_or_nan(ref_price.loc[last_valid_idx])
            last_close = _float_or_nan(auction.loc[last_valid_idx, "lastClose"])
            snap_return = snap_price / last_close - 1.0 if snap_price > 0 and last_close > 0 else np.nan

    # 2. VWAP (9:30-9:35)
    # Start: first row >= 9:30:00 (usually the opening snapshot)
    # End: last row <= 9:35:00
    v_start = df[df["dt"].dt.time >= VWAP_START].head(1)
    v_end = df[df["dt"].dt.time <= VWAP_END].tail(1)
    
    vwap_price = np.nan
    if not v_start.empty and not v_end.empty:
        amt_diff = _float_or_nan(v_end["amount"].iloc[0]) - _float_or_nan(v_start["amount"].iloc[0])
        vol_diff = _float_or_nan(v_end["volume"].iloc[0]) - _float_or_nan(v_start["volume"].iloc[0])
        if vol_diff > 0:
            # Volume unit is 'lot' (100 shares)
            vwap_price = (amt_diff / vol_diff) / 100.0
        else:
            # If no trade in 5 mins, fallback to the snapshot at 9:30 or 9:35
            vwap_price = _float_or_nan(v_start["lastPrice"].iloc[0])

    return {
        "auction_price_92450": snap_price,
        "auction_return_92450": snap_return,
        "vwap_0935": vwap_price,
    }


def build_stock_level_panel(
    panel: pd.DataFrame,
    tick_root: Path,
    trade_dates: Sequence[pd.Timestamp],
    holding_map: Dict[pd.Timestamp, List[str]],
    min_coverage_ratio: float,
) -> pd.DataFrame:
    by_date: Dict[pd.Timestamp, pd.DataFrame] = {
        _ensure_ts(d): df.drop_duplicates(subset=["ts_code"], keep="first").reset_index(drop=True)
        for d, df in panel.groupby("trade_date")
    }
    rows: List[Dict[str, object]] = []

    for i, trade_date in enumerate(trade_dates, start=1):
        trade_date = _ensure_ts(trade_date)
        holdings = holding_map.get(trade_date, [])
        today = by_date.get(trade_date, pd.DataFrame())
        if not holdings or today.empty:
            continue
        today = today.set_index("ts_code", drop=False)
        day_rows: List[Dict[str, object]] = []
        for ts_code in holdings:
            if ts_code not in today.index:
                continue
            r = today.loc[ts_code]
            pre_close = _float_or_nan(r.get("pre_close"))
            open_price = _float_or_nan(r.get("open"))
            close_price = _float_or_nan(r.get("close"))
            base_return = (
                close_price / pre_close - 1.0
                if pre_close > 0 and close_price > 0
                else np.nan
            )
            snap = extract_auction_and_vwap(tick_file_path(tick_root, trade_date, ts_code))
            day_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": ts_code,
                    "pre_close": pre_close,
                    "open": open_price,
                    "close": close_price,
                    "base_return": base_return,
                    **snap,
                }
            )
        coverage = len(day_rows) / len(holdings) if holdings else 0.0
        if coverage < min_coverage_ratio:
            print(f"[{i}/{len(trade_dates)}] {trade_date.date()} skip: low coverage {len(day_rows)}/{len(holdings)} ({coverage:.1%})")
            continue
        rows.extend(day_rows)
        print(f"[{i}/{len(trade_dates)}] {trade_date.date()} holdings={len(holdings)} rows={len(day_rows)} coverage={coverage:.1%}")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _long_extra_return(entry_price: float, close_price: float, pre_close: float, cfg: StrategyConfig) -> float:
    if entry_price <= 0 or close_price <= 0 or pre_close <= 0:
        return 0.0
    buy_cost = entry_price * (1.0 + cfg.slippage) * (1.0 + cfg.buy_fee)
    sell_proceed = close_price * (1.0 - cfg.slippage) * (1.0 - cfg.sell_fee)
    return (sell_proceed - buy_cost) / pre_close


def _short_extra_return(entry_price: float, close_price: float, pre_close: float, cfg: StrategyConfig) -> float:
    if entry_price <= 0 or close_price <= 0 or pre_close <= 0:
        return 0.0
    sell_proceed = entry_price * (1.0 - cfg.slippage) * (1.0 - cfg.sell_fee)
    buy_cost = close_price * (1.0 + cfg.slippage) * (1.0 + cfg.buy_fee)
    return (sell_proceed - buy_cost) / pre_close


def simulate_strategy(stock_df: pd.DataFrame, config: StrategyConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[pd.DataFrame] = []
    daily_rows: List[Dict[str, object]] = []

    for trade_date, day_df in stock_df.groupby("trade_date", sort=True):
        day = day_df.copy().sort_values(["auction_return_92450", "ts_code"], ascending=[True, True])
        day["long_signal"] = 0
        day["short_signal"] = 0
        day["long_extra_return"] = 0.0
        day["short_extra_return"] = 0.0
        day["extra_return"] = 0.0

        # Use VWAP as entry if available, else Open
        day["entry_price"] = day["vwap_0935"].fillna(day["open"])

        valid = day.dropna(subset=["auction_return_92450", "entry_price", "close", "pre_close"]).copy()
        valid = valid[(valid["entry_price"] > 0) & (valid["close"] > 0) & (valid["pre_close"] > 0)]
        pair_count = min(config.trade_count, len(valid) // 2)

        buy_codes: List[str] = []
        sell_codes: List[str] = []
        if pair_count > 0:
            buy_codes = (
                valid.sort_values(["auction_return_92450", "ts_code"], ascending=[True, True])
                .head(pair_count)["ts_code"]
                .tolist()
            )
            remain = valid[~valid["ts_code"].isin(buy_codes)]
            sell_codes = (
                remain.sort_values(["auction_return_92450", "ts_code"], ascending=[False, True])
                .head(pair_count)["ts_code"]
                .tolist()
            )

        if buy_codes:
            mask = day["ts_code"].isin(buy_codes)
            day.loc[mask, "long_signal"] = 1
            day.loc[mask, "long_extra_return"] = day.loc[mask].apply(
                lambda r: _long_extra_return(r["entry_price"], r["close"], r["pre_close"], config),
                axis=1,
            )
        if sell_codes:
            mask = day["ts_code"].isin(sell_codes)
            day.loc[mask, "short_signal"] = 1
            day.loc[mask, "short_extra_return"] = day.loc[mask].apply(
                lambda r: _short_extra_return(r["entry_price"], r["close"], r["pre_close"], config),
                axis=1,
            )

        day["extra_return"] = day["long_extra_return"] + day["short_extra_return"]
        day["strategy_return"] = day["base_return"].fillna(0.0) + day["extra_return"]
        day["signal_rank"] = day["auction_return_92450"].rank(method="first")
        rows.append(day)

        daily_rows.append(
            {
                "trade_date": trade_date,
                "hold_count": int(len(day)),
                "valid_signal_count": int(len(valid)),
                "buy_count": int(len(buy_codes)),
                "sell_count": int(len(sell_codes)),
                "base_return": float(day["base_return"].fillna(0.0).mean()),
                "extra_return": float(day["extra_return"].mean()),
                "strategy_return": float(day["strategy_return"].mean()),
            }
        )

    stock_result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    daily_result = pd.DataFrame(daily_rows).sort_values("trade_date").reset_index(drop=True)
    return stock_result, daily_result


def save_nav_plot(daily_df: pd.DataFrame, output_png: Path) -> None:
    if daily_df.empty:
        return
    nav_df = daily_df.copy()
    nav_df["trade_date"] = pd.to_datetime(nav_df["trade_date"])
    nav_df["baseline_nav"] = (1.0 + nav_df["base_return"]).cumprod()
    nav_df["strategy_nav"] = (1.0 + nav_df["strategy_return"]).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(nav_df["trade_date"], nav_df["baseline_nav"], label="Baseline (Small-cap 200)", linewidth=1.8)
    plt.plot(nav_df["trade_date"], nav_df["strategy_nav"], label="Capacity-Optimized Strategy", linewidth=1.8)
    plt.xlabel("Trade Date")
    plt.ylabel("NAV")
    plt.title("Capacity-Optimized Auction Hedge Strategy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def build_metrics(daily_df: pd.DataFrame, config: StrategyConfig) -> Dict[str, object]:
    base_ret = pd.Series(daily_df["base_return"].values, index=pd.to_datetime(daily_df["trade_date"]), dtype=float)
    strategy_ret = pd.Series(daily_df["strategy_return"].values, index=pd.to_datetime(daily_df["trade_date"]), dtype=float)
    extra_ret = strategy_ret - base_ret
    metrics_base = compute_performance_metrics(strategy_returns=base_ret, initial_cash=config.cash)
    metrics_strategy = compute_performance_metrics(strategy_returns=strategy_ret, initial_cash=config.cash, benchmark_returns=base_ret)
    return {
        "config": asdict(config),
        "trade_days": int(len(daily_df)),
        "annual_extra_return": float(extra_ret.mean() * 252.0) if not extra_ret.empty else None,
        "baseline": metrics_base,
        "strategy": metrics_strategy,
        "delta_total_return": metrics_strategy["total_return"] - metrics_base["total_return"] if "total_return" in metrics_strategy and "total_return" in metrics_base else None,
        "delta_sharpe": metrics_strategy["sharpe"] - metrics_base["sharpe"] if "sharpe" in metrics_strategy and "sharpe" in metrics_base else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="容量优化版开盘集合竞价涨跌幅对冲策略回测")
    parser.add_argument("--start", default="20260105", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="20260320", help="结束日期 YYYYMMDD")
    parser.add_argument("--daily-dir", default="daily_data/daily", help="日线目录")
    parser.add_argument("--daily-basic-dir", default="daily_data/daily_basic", help="daily_basic目录")
    parser.add_argument("--min-total-mv", type=float, default=300_000.0, help="最小总市值(万元)")
    parser.add_argument("--hold-count", type=int, default=200, help="底仓持股数")
    parser.add_argument("--trade-count", type=int, default=40, help="每日对冲交易股数(总数)")
    parser.add_argument("--output-dir", default="backtest/output_capacity_optimized", help="输出目录")
    args = parser.parse_args()

    config = StrategyConfig(
        min_total_mv=args.min_total_mv,
        hold_count=args.hold_count,
        trade_count=args.trade_count // 2, # Each side (long/short)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_start = (pd.Timestamp(args.start) - pd.Timedelta(days=14)).strftime("%Y%m%d")

    print(f"Loading panel with min_total_mv >= {config.min_total_mv}...")
    panel = load_combined_panel(
        daily_dir=args.daily_dir,
        daily_basic_dir=args.daily_basic_dir,
        start=data_start,
        end=args.end
    )
    if panel.empty: raise ValueError("未加载到数据")
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    
    tick_root = Path("tick_2026")
    tick_dates = set(list_tick_dates(tick_root, 2026, 2026))
    trade_dates = [d for d in sorted(panel["trade_date"].unique()) if d in tick_dates and d >= pd.Timestamp(args.start)]
    
    holding_map, _ = build_weekly_holding_map(panel, trade_dates, config)
    stock_df = build_stock_level_panel(panel, tick_root, trade_dates, holding_map, 0.7)
    stock_result, daily_df = simulate_strategy(stock_df, config)
    
    stock_result.to_parquet(output_dir / "stock_level_results.parquet", index=False)
    daily_df.to_csv(output_dir / "daily_portfolio.csv", index=False)
    
    metrics = build_metrics(daily_df, config)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    save_nav_plot(daily_df, output_dir / "nav_curve.png")

    print("\nDone.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
