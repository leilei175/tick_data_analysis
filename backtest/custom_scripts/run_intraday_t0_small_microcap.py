#!/usr/bin/env python3
"""
A股中小微盘日内"T+0"（先卖后买）策略研究脚本。

策略框架：
1. 以前一交易日流通市值(circ_mv)+成交额(amount)筛选中小微盘股票池。
2. 当日基线收益：等权持有股票池，按 pre_close->close 计收益。
3. 日内T策略：
   - 若 9:30-10:30 出现"冲高"（相对开盘价涨幅超过阈值），10:30 卖出底仓的一部分；
   - 午后若回落达到阈值则买回，否则尾盘买回，保持收盘持仓不变。
4. 训练集做参数网格搜索，测试集做样本外验证。

输出：
- features.parquet
- daily_returns.csv
- parameter_grid.csv
- selected_params.json
- metrics_{train,test,full}.json
- nav_curve.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_BACKTEST_DIR = _THIS_DIR.parent
if str(_BACKTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKTEST_DIR))

from data_source import load_combined_panel
from performance_metrics import compute_performance_metrics


@dataclass(frozen=True)
class StrategyParams:
    up_threshold: float
    retrace_threshold: float
    t_fraction: float


def _ensure_timestamp(d: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()


def _float_or_none(val: object) -> Optional[float]:
    try:
        out = float(val)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def list_tick_dates(tick_root: Path, year: int = 2026) -> List[pd.Timestamp]:
    year_dir = tick_root / f"{year:04d}"
    if not year_dir.exists():
        return []
    out: List[pd.Timestamp] = []
    for month_dir in sorted(year_dir.iterdir()):
        if not month_dir.is_dir():
            continue
        for day_dir in sorted(month_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            if not any(day_dir.glob("*.parquet")):
                continue
            out.append(pd.Timestamp(f"{year:04d}-{month_dir.name}-{day_dir.name}").normalize())
    return sorted(set(out))


def list_tick_symbols_for_date(tick_root: Path, trade_date: pd.Timestamp) -> Set[str]:
    date_dir = (
        tick_root
        / f"{trade_date.year:04d}"
        / f"{trade_date.month:02d}"
        / f"{trade_date.day:02d}"
    )
    if not date_dir.exists():
        return set()
    return {p.stem for p in date_dir.glob("*.parquet")}


def tick_file_path(tick_root: Path, trade_date: pd.Timestamp, ts_code: str) -> Path:
    return (
        tick_root
        / f"{trade_date.year:04d}"
        / f"{trade_date.month:02d}"
        / f"{trade_date.day:02d}"
        / f"{ts_code}.parquet"
    )


def _parse_tick_datetime_ms_utc(df: pd.DataFrame) -> pd.Series:
    dt = pd.to_datetime(df["time"], unit="ms", utc=True)
    return dt.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)


def _between_session(ts: pd.Series) -> pd.Series:
    t = ts.dt.time
    morning = (t >= time(9, 30)) & (t <= time(11, 30))
    afternoon = (t >= time(13, 0)) & (t <= time(15, 0))
    return morning | afternoon


def _last_price_before_or_at(series: pd.Series, target: pd.Timestamp) -> float:
    sub = series[series.index <= target]
    if sub.empty:
        return np.nan
    return float(sub.iloc[-1])


def extract_intraday_features(file_path: Path) -> Dict[str, float]:
    if not file_path.exists():
        return {}
    try:
        df = pd.read_parquet(file_path, columns=["time", "lastPrice"])
    except Exception:
        return {}

    if df.empty:
        return {}

    df = df[df["lastPrice"] > 0].copy()
    if df.empty:
        return {}

    df["dt"] = _parse_tick_datetime_ms_utc(df)
    df = df[_between_session(df["dt"])].copy()
    if df.empty:
        return {}

    minute = (
        df.assign(minute=df["dt"].dt.floor("min"))
        .sort_values("dt")
        .groupby("minute", sort=True)["lastPrice"]
        .last()
        .sort_index()
    )
    if minute.empty:
        return {}

    d0 = minute.index[0].normalize()
    t_open = d0 + pd.Timedelta(hours=9, minutes=30)
    t_1030 = d0 + pd.Timedelta(hours=10, minutes=30)
    t_1300 = d0 + pd.Timedelta(hours=13, minutes=0)
    t_1457 = d0 + pd.Timedelta(hours=14, minutes=57)
    t_1500 = d0 + pd.Timedelta(hours=15, minutes=0)

    after_open = minute[minute.index >= t_open]
    if after_open.empty:
        return {}
    open_price = float(after_open.iloc[0])

    morning = minute[(minute.index >= t_open) & (minute.index <= t_1030)]
    if morning.empty:
        return {}
    morning_high = float(morning.max())
    sell_price_1030 = _last_price_before_or_at(morning, t_1030)

    afternoon = minute[(minute.index >= t_1300) & (minute.index <= t_1457)]
    afternoon_min = float(afternoon.min()) if not afternoon.empty else np.nan

    close_slice = minute[(minute.index >= t_1457) & (minute.index <= t_1500)]
    close_intraday = float(close_slice.iloc[-1]) if not close_slice.empty else float(minute.iloc[-1])

    return {
        "open_price": open_price,
        "morning_high": morning_high,
        "sell_price_1030": float(sell_price_1030),
        "afternoon_min": afternoon_min,
        "close_intraday": close_intraday,
    }


def pick_universe(
    prev_panel: pd.DataFrame,
    today_tick_symbols: Set[str],
    min_amount: float,
    topn: int,
) -> List[str]:
    if prev_panel.empty or not today_tick_symbols:
        return []

    use = prev_panel.copy()
    use = use[use["ts_code"].isin(today_tick_symbols)]
    use = use[use["ts_code"].str.endswith((".SH", ".SZ"))]

    code_prefix = use["ts_code"].str.split(".", n=1).str[0]
    use = use[~code_prefix.str.startswith(("200", "900"))]  # B股前缀

    use = use[(use["amount"] >= min_amount) & (use["circ_mv"] > 0)]
    use = use.dropna(subset=["circ_mv", "amount"])
    if use.empty:
        return []

    use = use.sort_values("circ_mv", ascending=True)
    use = use.drop_duplicates(subset=["ts_code"], keep="first")
    return use.head(topn)["ts_code"].tolist()


def build_feature_panel(
    panel: pd.DataFrame,
    tick_root: Path,
    trade_dates: Sequence[pd.Timestamp],
    min_amount: float,
    topn: int,
    min_coverage_ratio: float,
) -> pd.DataFrame:
    by_date: Dict[pd.Timestamp, pd.DataFrame] = {
        _ensure_timestamp(d): df.drop_duplicates(subset=["ts_code"], keep="first").reset_index(drop=True)
        for d, df in panel.groupby(panel["trade_date"])
    }

    date_index = sorted(by_date.keys())
    prev_date_map: Dict[pd.Timestamp, pd.Timestamp] = {
        date_index[i]: date_index[i - 1] for i in range(1, len(date_index))
    }

    rows: List[Dict[str, object]] = []
    for i, d in enumerate(trade_dates, start=1):
        d = _ensure_timestamp(d)
        if d not in prev_date_map:
            continue
        prev_d = prev_date_map[d]

        today_symbols = list_tick_symbols_for_date(tick_root, d)
        universe = pick_universe(
            prev_panel=by_date.get(prev_d, pd.DataFrame()),
            today_tick_symbols=today_symbols,
            min_amount=min_amount,
            topn=topn,
        )
        if not universe:
            print(f"[{i}/{len(trade_dates)}] {d.date()} skip: no universe")
            continue

        today_panel = by_date.get(d, pd.DataFrame())
        if today_panel.empty:
            print(f"[{i}/{len(trade_dates)}] {d.date()} skip: no daily panel")
            continue
        today_panel = today_panel.set_index("ts_code", drop=False)

        day_buffer: List[Dict[str, object]] = []
        for ts_code in universe:
            if ts_code not in today_panel.index:
                continue
            r = today_panel.loc[ts_code]
            pre_close = _float_or_none(r.get("pre_close"))
            close = _float_or_none(r.get("close"))
            base_return = (
                (close - pre_close) / pre_close
                if pre_close is not None and close is not None and pre_close > 0
                else np.nan
            )

            tick_path = tick_file_path(tick_root, d, ts_code)
            feat = extract_intraday_features(tick_path)
            day_buffer.append(
                {
                    "trade_date": d,
                    "prev_trade_date": prev_d,
                    "ts_code": ts_code,
                    "pre_close": pre_close,
                    "close": close,
                    "base_return": base_return,
                    "universe_size": len(universe),
                    **feat,
                }
            )
        day_rows = len(day_buffer)
        coverage = day_rows / len(universe) if universe else 0.0
        if coverage < min_coverage_ratio:
            print(
                f"[{i}/{len(trade_dates)}] {d.date()} skip: "
                f"low coverage {day_rows}/{len(universe)} ({coverage:.1%})"
            )
            continue

        rows.extend(day_buffer)
        print(
            f"[{i}/{len(trade_dates)}] {d.date()} "
            f"universe={len(universe)} rows={day_rows} coverage={coverage:.1%}"
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    return out


def simulate_daily_returns(
    feature_df: pd.DataFrame,
    params: StrategyParams,
    buy_fee: float,
    sell_fee: float,
    slippage: float,
) -> pd.DataFrame:
    sim = simulate_stock_level(
        feature_df=feature_df,
        params=params,
        buy_fee=buy_fee,
        sell_fee=sell_fee,
        slippage=slippage,
    )
    daily = (
        sim.groupby("trade_date", as_index=False)
        .agg(
            base_return=("base_return", "mean"),
            extra_return=("extra_return", "mean"),
            t0_return=("t0_return", "mean"),
            trade_count=("trigger", "sum"),
            retrace_hit_count=("retrace_hit", "sum"),
            universe_size=("ts_code", "count"),
        )
        .sort_values("trade_date")
        .reset_index(drop=True)
    )
    return daily


def simulate_stock_level(
    feature_df: pd.DataFrame,
    params: StrategyParams,
    buy_fee: float,
    sell_fee: float,
    slippage: float,
) -> pd.DataFrame:
    df = feature_df.copy()

    for col in [
        "pre_close",
        "base_return",
        "open_price",
        "morning_high",
        "sell_price_1030",
        "afternoon_min",
        "close_intraday",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    valid = (
        (df["pre_close"] > 0)
        & (df["open_price"] > 0)
        & (df["morning_high"] > 0)
        & (df["sell_price_1030"] > 0)
        & (df["close_intraday"] > 0)
    )
    morning_spike = df["morning_high"] / df["open_price"] - 1.0
    trigger = valid & (morning_spike >= params.up_threshold)

    sell_exec = df["sell_price_1030"] * (1.0 - slippage)
    target_buy = sell_exec * (1.0 - params.retrace_threshold)
    retrace_hit = (df["afternoon_min"] > 0) & (df["afternoon_min"] <= target_buy)

    buy_raw = np.where(retrace_hit, target_buy, df["close_intraday"].values)
    buy_exec = buy_raw * (1.0 + slippage)

    extra_return = np.where(
        trigger,
        params.t_fraction
        * (sell_exec * (1.0 - sell_fee) - buy_exec * (1.0 + buy_fee))
        / df["pre_close"].values,
        0.0,
    )

    sim = df[["trade_date", "ts_code", "base_return", "universe_size"]].copy()
    sim["morning_spike"] = morning_spike
    sim["extra_return"] = np.where(np.isfinite(extra_return), extra_return, 0.0)
    sim["trigger"] = trigger.astype(int)
    sim["retrace_hit"] = (trigger & retrace_hit).astype(int)
    sim["t0_return"] = sim["base_return"].fillna(0.0) + sim["extra_return"]
    return sim


def _annualized_sharpe(ret: pd.Series) -> float:
    s = ret.std(ddof=0)
    if s <= 0 or not np.isfinite(s):
        return np.nan
    return float(ret.mean() / s * np.sqrt(252.0))


def evaluate_param_grid(
    feature_df: pd.DataFrame,
    train_dates: Set[pd.Timestamp],
    param_grid: Iterable[StrategyParams],
    buy_fee: float,
    sell_fee: float,
    slippage: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for p in param_grid:
        daily = simulate_daily_returns(
            feature_df=feature_df,
            params=p,
            buy_fee=buy_fee,
            sell_fee=sell_fee,
            slippage=slippage,
        )
        train = daily[daily["trade_date"].isin(train_dates)].copy()
        if train.empty:
            continue

        ex_ret = train["t0_return"] - train["base_return"]
        ex_sharpe = _annualized_sharpe(ex_ret)
        annual_excess = float(ex_ret.mean() * 252.0)

        trades = int(train["trade_count"].sum())
        retrace_hits = int(train["retrace_hit_count"].sum())
        hit_rate = retrace_hits / trades if trades > 0 else np.nan

        score = ex_sharpe
        if not np.isfinite(score):
            score = -1e9
        if trades < 10:
            score -= 1.0

        rows.append(
            {
                "up_threshold": p.up_threshold,
                "retrace_threshold": p.retrace_threshold,
                "t_fraction": p.t_fraction,
                "train_days": len(train),
                "train_trades": trades,
                "train_retrace_hit_rate": hit_rate,
                "train_annual_excess": annual_excess,
                "train_excess_sharpe": ex_sharpe,
                "score": score,
            }
        )

    if not rows:
        return pd.DataFrame()

    grid = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return grid


def make_metrics_payload(
    daily: pd.DataFrame,
    cash: float,
    label: str,
    params: StrategyParams,
) -> Dict[str, object]:
    base_ret = pd.Series(
        daily["base_return"].values, index=pd.to_datetime(daily["trade_date"]), dtype=float
    )
    t0_ret = pd.Series(
        daily["t0_return"].values, index=pd.to_datetime(daily["trade_date"]), dtype=float
    )
    ex_ret = t0_ret - base_ret

    metrics_base = compute_performance_metrics(strategy_returns=base_ret, initial_cash=cash)
    metrics_t0 = compute_performance_metrics(
        strategy_returns=t0_ret, initial_cash=cash, benchmark_returns=base_ret
    )

    payload = {
        "label": label,
        "period_start": daily["trade_date"].min().strftime("%Y-%m-%d") if not daily.empty else None,
        "period_end": daily["trade_date"].max().strftime("%Y-%m-%d") if not daily.empty else None,
        "trade_days": int(len(daily)),
        "strategy_params": asdict(params),
        "trade_count_total": int(daily["trade_count"].sum()) if not daily.empty else 0,
        "retrace_hit_total": int(daily["retrace_hit_count"].sum()) if not daily.empty else 0,
        "retrace_hit_rate": (
            float(daily["retrace_hit_count"].sum() / daily["trade_count"].sum())
            if not daily.empty and daily["trade_count"].sum() > 0
            else None
        ),
        "annual_excess_return": float(ex_ret.mean() * 252.0) if not ex_ret.empty else None,
        "excess_sharpe": _annualized_sharpe(ex_ret) if not ex_ret.empty else None,
        "metrics_baseline": metrics_base,
        "metrics_t0": metrics_t0,
        "delta_total_return": (
            _float_or_none(metrics_t0.get("total_return")) - _float_or_none(metrics_base.get("total_return"))
            if _float_or_none(metrics_t0.get("total_return")) is not None
            and _float_or_none(metrics_base.get("total_return")) is not None
            else None
        ),
        "delta_annual_return": (
            _float_or_none(metrics_t0.get("annual_return")) - _float_or_none(metrics_base.get("annual_return"))
            if _float_or_none(metrics_t0.get("annual_return")) is not None
            and _float_or_none(metrics_base.get("annual_return")) is not None
            else None
        ),
        "delta_sharpe": (
            _float_or_none(metrics_t0.get("sharpe")) - _float_or_none(metrics_base.get("sharpe"))
            if _float_or_none(metrics_t0.get("sharpe")) is not None
            and _float_or_none(metrics_base.get("sharpe")) is not None
            else None
        ),
        "delta_max_drawdown_pct": (
            _float_or_none(metrics_t0.get("max_drawdown_pct"))
            - _float_or_none(metrics_base.get("max_drawdown_pct"))
            if _float_or_none(metrics_t0.get("max_drawdown_pct")) is not None
            and _float_or_none(metrics_base.get("max_drawdown_pct")) is not None
            else None
        ),
    }
    return payload


def save_nav_plot(daily: pd.DataFrame, output_png: Path) -> None:
    if daily.empty:
        return
    nav = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(daily["trade_date"]),
            "baseline_nav": (1.0 + daily["base_return"]).cumprod(),
            "t0_nav": (1.0 + daily["t0_return"]).cumprod(),
        }
    )
    plt.figure(figsize=(10, 5))
    plt.plot(nav["trade_date"], nav["baseline_nav"], label="Baseline", linewidth=1.8)
    plt.plot(nav["trade_date"], nav["t0_nav"], label="T+0 Overlay", linewidth=1.8)
    plt.title("Small/Micro-cap Baseline vs Intraday T+0 Overlay")
    plt.xlabel("Trade Date")
    plt.ylabel("NAV")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def save_trade_level_analysis(stock_level: pd.DataFrame, output_dir: Path) -> None:
    trigger = stock_level[stock_level["trigger"] == 1].copy()
    trigger["extra_bp"] = trigger["extra_return"] * 10000.0

    summary = pd.DataFrame(
        [
            {
                "trigger_count": int(len(trigger)),
                "retrace_hit_count": int(trigger["retrace_hit"].sum()) if not trigger.empty else 0,
                "retrace_hit_rate": (
                    float(trigger["retrace_hit"].mean()) if not trigger.empty else np.nan
                ),
                "mean_extra_bp": float(trigger["extra_bp"].mean()) if not trigger.empty else np.nan,
                "median_extra_bp": (
                    float(trigger["extra_bp"].median()) if not trigger.empty else np.nan
                ),
                "win_rate_extra": (
                    float((trigger["extra_return"] > 0).mean()) if not trigger.empty else np.nan
                ),
                "p10_extra_bp": (
                    float(trigger["extra_bp"].quantile(0.1)) if not trigger.empty else np.nan
                ),
                "p90_extra_bp": (
                    float(trigger["extra_bp"].quantile(0.9)) if not trigger.empty else np.nan
                ),
            }
        ]
    )
    summary.to_csv(output_dir / "trade_level_summary.csv", index=False)

    if trigger.empty:
        pd.DataFrame(columns=["retrace_hit", "count", "mean", "median"]).to_csv(
            output_dir / "trade_level_by_retrace_hit.csv",
            index=False,
        )
        pd.DataFrame(columns=["spike_bin", "trades", "mean_extra_bp", "win_rate"]).to_csv(
            output_dir / "trade_level_by_spike_quantile.csv",
            index=False,
        )
        return

    by_hit = trigger.groupby("retrace_hit", as_index=False).agg(
        count=("extra_bp", "count"),
        mean=("extra_bp", "mean"),
        median=("extra_bp", "median"),
    )
    by_hit.to_csv(output_dir / "trade_level_by_retrace_hit.csv", index=False)

    trigger["spike_bin"] = pd.qcut(trigger["morning_spike"], q=5, duplicates="drop")
    by_spike = (
        trigger.groupby("spike_bin", observed=True)
        .agg(
            trades=("extra_bp", "count"),
            mean_extra_bp=("extra_bp", "mean"),
            win_rate=("extra_return", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )
    by_spike.to_csv(output_dir / "trade_level_by_spike_quantile.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="A股中小微盘日内T+0策略研究")
    parser.add_argument("--start", default="20260105", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="20260210", help="结束日期 YYYYMMDD")
    parser.add_argument("--daily-dir", default="daily_data/daily", help="日线目录")
    parser.add_argument("--daily-basic-dir", default="daily_data/daily_basic", help="daily_basic目录")
    parser.add_argument("--tick-dir", default="tick_2026", help="tick根目录")
    parser.add_argument("--min-amount", type=float, default=30000.0, help="上一日最小成交额(千元)")
    parser.add_argument("--topn", type=int, default=80, help="中小微盘选股数量")
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=0.7,
        help="单日有效股票覆盖率下限，低于该值则跳过该交易日",
    )
    parser.add_argument("--buy-fee", type=float, default=0.0002, help="买入费率(不含滑点)")
    parser.add_argument("--sell-fee", type=float, default=0.0007, help="卖出费率(含印花税假设)")
    parser.add_argument("--slippage", type=float, default=0.0003, help="单边滑点")
    parser.add_argument("--cash", type=float, default=10_000_000.0, help="初始资金")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="训练集比例")
    parser.add_argument(
        "--output-dir",
        default="backtest/output_intraday_t0_small_microcap",
        help="输出目录",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading combined panel...")
    panel = load_combined_panel(
        daily_dir=args.daily_dir,
        daily_basic_dir=args.daily_basic_dir,
        start=args.start,
        end=args.end,
    )
    if panel.empty:
        raise ValueError("未加载到日线+市值数据")
    panel = panel.drop_duplicates(subset=["trade_date", "ts_code"], keep="first").copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    tick_root = Path(args.tick_dir)
    tick_dates = set(list_tick_dates(tick_root=tick_root, year=2026))
    panel_dates = sorted(panel["trade_date"].unique())
    use_dates = [d for d in panel_dates if d in tick_dates]
    if len(use_dates) < 8:
        raise ValueError(f"可回测日期过少: {len(use_dates)}")

    print(f"Panel dates={len(panel_dates)}, tick dates(intersection)={len(use_dates)}")
    print(f"Date range: {use_dates[0].date()} -> {use_dates[-1].date()}")

    feature_df = build_feature_panel(
        panel=panel,
        tick_root=tick_root,
        trade_dates=use_dates,
        min_amount=args.min_amount,
        topn=args.topn,
        min_coverage_ratio=args.min_coverage_ratio,
    )
    if feature_df.empty:
        raise ValueError("未生成特征数据，无法回测")
    feature_df.to_parquet(output_dir / "features.parquet", index=False)

    all_dates = sorted(pd.to_datetime(feature_df["trade_date"]).dt.normalize().unique())
    split_idx = max(5, min(len(all_dates) - 1, int(len(all_dates) * args.train_ratio)))
    train_dates = set(all_dates[:split_idx])
    test_dates = set(all_dates[split_idx:])
    print(f"Train/Test split: {len(train_dates)} / {len(test_dates)}")

    up_grid = [0.010, 0.015, 0.020, 0.025, 0.030]
    retrace_grid = [0.003, 0.005, 0.008, 0.010]
    t_frac_grid = [0.2, 0.3, 0.4]
    param_grid = [
        StrategyParams(u, r, f) for u in up_grid for r in retrace_grid for f in t_frac_grid
    ]

    grid_df = evaluate_param_grid(
        feature_df=feature_df,
        train_dates=train_dates,
        param_grid=param_grid,
        buy_fee=args.buy_fee,
        sell_fee=args.sell_fee,
        slippage=args.slippage,
    )
    if grid_df.empty:
        raise ValueError("参数网格评估失败")
    grid_df.to_csv(output_dir / "parameter_grid.csv", index=False)

    best = grid_df.iloc[0]
    best_params = StrategyParams(
        up_threshold=float(best["up_threshold"]),
        retrace_threshold=float(best["retrace_threshold"]),
        t_fraction=float(best["t_fraction"]),
    )
    (output_dir / "selected_params.json").write_text(
        json.dumps(asdict(best_params), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Best params: {best_params}")

    daily = simulate_daily_returns(
        feature_df=feature_df,
        params=best_params,
        buy_fee=args.buy_fee,
        sell_fee=args.sell_fee,
        slippage=args.slippage,
    )
    daily = daily.sort_values("trade_date").reset_index(drop=True)
    daily.to_csv(output_dir / "daily_returns.csv", index=False)

    stock_level = simulate_stock_level(
        feature_df=feature_df,
        params=best_params,
        buy_fee=args.buy_fee,
        sell_fee=args.sell_fee,
        slippage=args.slippage,
    )
    stock_level.to_parquet(output_dir / "stock_level_simulation.parquet", index=False)
    save_trade_level_analysis(stock_level=stock_level, output_dir=output_dir)

    save_nav_plot(daily, output_dir / "nav_curve.png")

    train_daily = daily[daily["trade_date"].isin(train_dates)].copy()
    test_daily = daily[daily["trade_date"].isin(test_dates)].copy()

    metrics_train = make_metrics_payload(
        daily=train_daily,
        cash=args.cash,
        label="train",
        params=best_params,
    )
    metrics_test = make_metrics_payload(
        daily=test_daily,
        cash=args.cash,
        label="test",
        params=best_params,
    )
    metrics_full = make_metrics_payload(
        daily=daily,
        cash=args.cash,
        label="full",
        params=best_params,
    )

    (output_dir / "metrics_train.json").write_text(
        json.dumps(metrics_train, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics_test.json").write_text(
        json.dumps(metrics_test, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics_full.json").write_text(
        json.dumps(metrics_full, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Output dir: {output_dir}")
    print(
        f"Full period total return (baseline / t0): "
        f"{metrics_full['metrics_baseline'].get('total_return')} / "
        f"{metrics_full['metrics_t0'].get('total_return')}"
    )
    print(
        f"Full period sharpe (baseline / t0): "
        f"{metrics_full['metrics_baseline'].get('sharpe')} / "
        f"{metrics_full['metrics_t0'].get('sharpe')}"
    )
    print(f"Trade count total: {metrics_full.get('trade_count_total')}")


if __name__ == "__main__":
    main()
