#!/usr/bin/env python3
"""开盘集合竞价涨跌幅对冲策略 - 平衡版 (Balanced Capacity & Alpha)。

主要配置:
1. hold_count: 150 (底仓分散)
2. trade_count: 30 (日内调仓分散)
3. min_total_mv: 15亿
4. entry_price: 09:30 开盘价
5. slippage: 0.05% (5bps)
6. 优化条件: 09:24:50 竞价撮合额 (bidPrice[0] * bidVol[0] * 100) >= 40万元
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

@dataclass(frozen=True)
class StrategyConfig:
    min_amount: float = 10_000.0
    min_total_mv: float = 150_000.0
    hold_count: int = 150
    trade_count: int = 15 # Each side
    buy_fee: float = 0.0002
    sell_fee: float = 0.0007
    slippage: float = 0.0005
    cash: float = 10_000_000.0

def _ensure_ts(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()

def _float_or_nan(value: object) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else np.nan
    except: return np.nan

def _parse_level1(value: object) -> float:
    if isinstance(value, np.ndarray): return _float_or_nan(value.flat[0]) if value.size > 0 else np.nan
    if isinstance(value, (list, tuple)): return _float_or_nan(value[0]) if value else np.nan
    if isinstance(value, str):
        s = value.strip().strip("[]").split(",")[0].strip()
        return _float_or_nan(s) if s else np.nan
    return _float_or_nan(value)

def list_tick_dates(tick_root: Path, start_year: int, end_year: int) -> List[pd.Timestamp]:
    out = []
    for y in range(start_year, end_year + 1):
        y_dir = tick_root / str(y)
        if not y_dir.exists(): continue
        for m_dir in sorted(y_dir.iterdir()):
            if not m_dir.is_dir(): continue
            for d_dir in sorted(m_dir.iterdir()):
                if not d_dir.is_dir(): continue
                if any(d_dir.glob("*.parquet")):
                    out.append(pd.Timestamp(f"{y:04d}-{m_dir.name}-{d_dir.name}").normalize())
    return sorted(set(out))

def tick_file_path(tick_root: Path, trade_date: pd.Timestamp, ts_code: str) -> Path:
    return tick_root / str(trade_date.year) / f"{trade_date.month:02d}" / f"{trade_date.day:02d}" / f"{ts_code}.parquet"

def pick_weekly_universe(prev_panel: pd.DataFrame, config: StrategyConfig) -> List[str]:
    if prev_panel.empty: return []
    use = prev_panel.copy()
    use = use[use["ts_code"].str.endswith((".SH", ".SZ"))]
    code_p = use["ts_code"].str.split(".", n=1).str[0]
    use = use[~code_p.str.startswith(("200", "900"))]
    use = use.dropna(subset=["amount", "total_mv"])
    use = use[(use["amount"] >= config.min_amount) & (use["total_mv"] >= config.min_total_mv)]
    if use.empty: return []
    use = use.sort_values(["total_mv", "ts_code"])
    return use.head(config.hold_count)["ts_code"].tolist()

def build_weekly_holding_map(panel: pd.DataFrame, trade_dates: Sequence[pd.Timestamp], config: StrategyConfig) -> Dict[pd.Timestamp, List[str]]:
    by_date = { _ensure_ts(d): df.drop_duplicates(subset=["ts_code"]) for d, df in panel.groupby("trade_date") }
    ordered = sorted(by_date)
    prev_map = {ordered[i]: ordered[i-1] for i in range(1, len(ordered))}
    holding_map = {}
    current = []
    last_week = None
    for d in trade_dates:
        d = _ensure_ts(d)
        week = d.isocalendar()[:2]
        if week != last_week and d in prev_map:
            current = pick_weekly_universe(by_date.get(prev_map[d], pd.DataFrame()), config)
            last_week = week
        holding_map[d] = list(current)
    return holding_map

def extract_auction_snapshot(file_path: Path) -> Dict[str, float]:
    if not file_path.exists(): return {}
    try:
        df = pd.read_parquet(file_path, columns=["time", "lastPrice", "lastClose", "bidPrice", "bidVol", "askPrice", "askVol"])
    except: return {}
    if df.empty: return {}
    df["dt"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    auction = df[(df["dt"].dt.time >= time(9, 15)) & (df["dt"].dt.time <= AUCTION_CUTOFF)].copy()
    if auction.empty: return {}
    
    auction["bp1"] = auction["bidPrice"].apply(_parse_level1)
    auction["bv1"] = auction["bidVol"].apply(_parse_level1)
    auction["lp"] = pd.to_numeric(auction["lastPrice"], errors="coerce")
    auction["lc"] = pd.to_numeric(auction["lastClose"], errors="coerce")
    
    # 竞价活跃度指标: 买一价 * 买一量 * 100
    # 注意: bidVol 单位是手
    auction["est_amt"] = auction["bp1"] * auction["bv1"] * 100.0
    
    last_idx = auction["est_amt"].last_valid_index()
    if last_idx is None: return {}
    
    row = auction.loc[last_idx]
    return { 
        "auction_return_92450": (row["bp1"] / row["lc"] - 1.0) if row["bp1"] > 0 and row["lc"] > 0 else np.nan,
        "auction_amount_92450": row["est_amt"]
    }

def build_stock_level_panel(panel: pd.DataFrame, tick_root: Path, trade_dates: Sequence[pd.Timestamp], holding_map: Dict[pd.Timestamp, List[str]]) -> pd.DataFrame:
    by_date = { _ensure_ts(d): df.set_index("ts_code") for d, df in panel.groupby("trade_date") }
    rows = []
    for i, d in enumerate(trade_dates, 1):
        holdings = holding_map.get(d, [])
        today = by_date.get(d)
        if not holdings or today is None: continue
        for code in holdings:
            if code not in today.index: continue
            r = today.loc[code]
            snap = extract_auction_snapshot(tick_file_path(tick_root, d, code))
            rows.append({ "trade_date": d, "ts_code": code, "pre_close": _float_or_nan(r.get("pre_close")), "open": _float_or_nan(r.get("open")), "close": _float_or_nan(r.get("close")), **snap })
        print(f"[{i}/{len(trade_dates)}] {d.date()} data ready")
    df = pd.DataFrame(rows)
    df["base_return"] = df["close"] / df["pre_close"] - 1.0
    return df

def _calc_ret(entry: float, exit: float, pre: float, cfg: StrategyConfig, is_long: bool) -> float:
    if entry <= 0 or exit <= 0 or pre <= 0: return 0.0
    if is_long: return (exit * (1 - cfg.slippage - cfg.sell_fee) - entry * (1 + cfg.slippage + cfg.buy_fee)) / pre
    else: return (entry * (1 - cfg.slippage - cfg.sell_fee) - exit * (1 + cfg.slippage + cfg.buy_fee)) / pre

def simulate_strategy(stock_df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    res = []
    MIN_AMT = 400000.0 # 40万过滤 (使用估计成交额)
    for d, day in stock_df.groupby("trade_date"):
        valid = day.dropna(subset=["auction_return_92450", "open", "close", "auction_amount_92450"]).copy()
        valid = valid[(valid["open"] > 0) & (valid["auction_amount_92450"] >= MIN_AMT)]
        n = min(config.trade_count, len(valid) // 2)
        extra = 0.0
        if n > 0:
            lc = valid.sort_values("auction_return_92450").head(n)["ts_code"].tolist()
            sc = valid.sort_values("auction_return_92450", ascending=False).head(n)["ts_code"].tolist()
            lr = day[day["ts_code"].isin(lc)].apply(lambda r: _calc_ret(r["open"], r["close"], r["pre_close"], config, True), axis=1).sum()
            sr = day[day["ts_code"].isin(sc)].apply(lambda r: _calc_ret(r["open"], r["close"], r["pre_close"], config, False), axis=1).sum()
            extra = (lr + sr) / len(day)
        res.append({ "trade_date": d, "base_return": day["base_return"].mean(), "extra_return": extra, "strategy_return": day["base_return"].mean() + extra, "valid_signals": len(valid) })
    return pd.DataFrame(res)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20250801")
    parser.add_argument("--end", default="20260320")
    args = parser.parse_args()
    config = StrategyConfig()
    panel = load_combined_panel(daily_dir="daily_data/daily", daily_basic_dir="daily_data/daily_basic", start="20250715", end=args.end)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    tick_root = Path("tick_2026")
    trade_dates = [d for d in sorted(panel["trade_date"].unique()) if d >= pd.Timestamp(args.start)]
    holding_map = build_weekly_holding_map(panel, trade_dates, config)
    stock_df = build_stock_level_panel(panel, tick_root, trade_dates, holding_map)
    daily_df = simulate_strategy(stock_df, config)
    
    m_base = compute_performance_metrics(daily_df["base_return"], initial_cash=config.cash)
    m_strat = compute_performance_metrics(daily_df["strategy_return"], initial_cash=config.cash, benchmark_returns=daily_df["base_return"])
    
    print("\n" + "="*40 + "\nDETAILED PERFORMANCE (400k est_amt Filter)\n" + "="*40)
    print(f"Extra Return (Ann): {daily_df['extra_return'].mean()*252:.2%}")
    print(f"Strategy Sharpe: {m_strat['sharpe']:.2f}")
    ir = m_strat.get('information_ratio')
    print(f"Info Ratio: {ir:.2f}" if ir is not None else "Info Ratio: N/A")
    daily_df['m'] = daily_df['trade_date'].dt.to_period('M')
    monthly = daily_df.groupby('m')['extra_return'].sum()
    print("-" * 40 + "\nMonthly Extra:")
    for m, v in monthly.items(): print(f"  {m}: {v:.2%}")

if __name__ == "__main__":
    main()
