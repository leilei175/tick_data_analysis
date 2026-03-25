#!/usr/bin/env python3
"""开盘集合竞价涨跌幅对冲策略回测。

策略规则:
1. 每周第一个交易日，根据前一交易日数据筛选:
   - amount >= 10000（千元，对应 1000 万元）
   - total_mv >= 100000（万元，对应 10 亿元）
   - 按 total_mv 升序取最小 50 只作为底仓
2. 底仓在一周内保持不变。
3. 每个交易日根据 09:24:50 之前最后一个集合竞价快照价格，相对昨收计算涨跌幅:
   - 开盘买入涨跌幅最小的 10 只（加仓一份底仓权重）
   - 开盘卖出涨跌幅最大的 10 只（卖出一份底仓权重）
   - 收盘卖出/买回，恢复到底仓

输出:
- weekly_rebalance_log.csv
- stock_level_results.parquet
- daily_portfolio.csv
- metrics.json
- nav_curve.png
- notebook/open_auction_hedge_strategy_research.ipynb
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
REBALANCE_WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]


@dataclass(frozen=True)
class StrategyConfig:
    min_amount: float = 10_000.0
    min_total_mv: float = 100_000.0
    hold_count: int = 50
    trade_count: int = 10
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


def extract_auction_snapshot(file_path: Path, cutoff: time = AUCTION_CUTOFF) -> Dict[str, float]:
    if not file_path.exists():
        return {}
    try:
        df = pd.read_parquet(file_path, columns=["time", "lastPrice", "lastClose", "askPrice", "bidPrice"])
    except Exception:
        return {}
    if df.empty:
        return {}

    local_dt = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    df = df.assign(dt=local_dt)
    auction = df[(df["dt"].dt.time >= time(9, 15)) & (df["dt"].dt.time <= cutoff)].copy()
    if auction.empty:
        return {}

    auction["ask1"] = auction["askPrice"].apply(_parse_level1)
    auction["bid1"] = auction["bidPrice"].apply(_parse_level1)
    auction["lastPrice"] = pd.to_numeric(auction["lastPrice"], errors="coerce")
    auction["lastClose"] = pd.to_numeric(auction["lastClose"], errors="coerce")

    price = auction["lastPrice"].where(auction["lastPrice"] > 0)
    mid = ((auction["ask1"] + auction["bid1"]) / 2.0).where(
        (auction["ask1"] > 0) & (auction["bid1"] > 0)
    )
    ref_price = price.fillna(mid).fillna(auction["ask1"]).fillna(auction["bid1"])
    last_valid_idx = ref_price.last_valid_index()
    if last_valid_idx is None:
        return {}

    row = auction.loc[last_valid_idx]
    snap_price = _float_or_nan(ref_price.loc[last_valid_idx])
    last_close = _float_or_nan(row.get("lastClose"))
    snap_return = snap_price / last_close - 1.0 if snap_price > 0 and last_close > 0 else np.nan
    return {
        "auction_dt": row["dt"],
        "auction_price_92450": snap_price,
        "auction_return_92450": snap_return,
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
            snap = extract_auction_snapshot(tick_file_path(tick_root, trade_date, ts_code))
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
            print(
                f"[{i}/{len(trade_dates)}] {trade_date.date()} skip: "
                f"low coverage {len(day_rows)}/{len(holdings)} ({coverage:.1%})"
            )
            continue
        rows.extend(day_rows)
        print(
            f"[{i}/{len(trade_dates)}] {trade_date.date()} "
            f"holdings={len(holdings)} rows={len(day_rows)} coverage={coverage:.1%}"
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _long_extra_return(open_price: float, close_price: float, pre_close: float, cfg: StrategyConfig) -> float:
    if open_price <= 0 or close_price <= 0 or pre_close <= 0:
        return 0.0
    buy_cost = open_price * (1.0 + cfg.slippage) * (1.0 + cfg.buy_fee)
    sell_proceed = close_price * (1.0 - cfg.slippage) * (1.0 - cfg.sell_fee)
    return (sell_proceed - buy_cost) / pre_close


def _short_extra_return(open_price: float, close_price: float, pre_close: float, cfg: StrategyConfig) -> float:
    if open_price <= 0 or close_price <= 0 or pre_close <= 0:
        return 0.0
    sell_proceed = open_price * (1.0 - cfg.slippage) * (1.0 - cfg.sell_fee)
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

        valid = day.dropna(subset=["auction_return_92450", "open", "close", "pre_close"]).copy()
        valid = valid[(valid["open"] > 0) & (valid["close"] > 0) & (valid["pre_close"] > 0)]
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
                lambda r: _long_extra_return(r["open"], r["close"], r["pre_close"], config),
                axis=1,
            )
        if sell_codes:
            mask = day["ts_code"].isin(sell_codes)
            day.loc[mask, "short_signal"] = 1
            day.loc[mask, "short_extra_return"] = day.loc[mask].apply(
                lambda r: _short_extra_return(r["open"], r["close"], r["pre_close"], config),
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
                "avg_buy_signal_return": (
                    float(day.loc[day["long_signal"] == 1, "auction_return_92450"].mean())
                    if buy_codes
                    else np.nan
                ),
                "avg_sell_signal_return": (
                    float(day.loc[day["short_signal"] == 1, "auction_return_92450"].mean())
                    if sell_codes
                    else np.nan
                ),
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
    plt.plot(nav_df["trade_date"], nav_df["baseline_nav"], label="Baseline", linewidth=1.8)
    plt.plot(nav_df["trade_date"], nav_df["strategy_nav"], label="Auction Hedge", linewidth=1.8)
    plt.xlabel("Trade Date")
    plt.ylabel("NAV")
    plt.title("Weekly Small-cap Base + Open Auction Hedge")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close()


def build_metrics(daily_df: pd.DataFrame, config: StrategyConfig) -> Dict[str, object]:
    base_ret = pd.Series(
        daily_df["base_return"].values,
        index=pd.to_datetime(daily_df["trade_date"]),
        dtype=float,
    )
    strategy_ret = pd.Series(
        daily_df["strategy_return"].values,
        index=pd.to_datetime(daily_df["trade_date"]),
        dtype=float,
    )
    extra_ret = strategy_ret - base_ret
    metrics_base = compute_performance_metrics(strategy_returns=base_ret, initial_cash=config.cash)
    metrics_strategy = compute_performance_metrics(
        strategy_returns=strategy_ret,
        initial_cash=config.cash,
        benchmark_returns=base_ret,
    )
    return {
        "config": asdict(config),
        "trade_days": int(len(daily_df)),
        "avg_hold_count": float(daily_df["hold_count"].mean()) if not daily_df.empty else None,
        "avg_valid_signal_count": (
            float(daily_df["valid_signal_count"].mean()) if not daily_df.empty else None
        ),
        "buy_trades_total": int(daily_df["buy_count"].sum()) if not daily_df.empty else 0,
        "sell_trades_total": int(daily_df["sell_count"].sum()) if not daily_df.empty else 0,
        "annual_extra_return": float(extra_ret.mean() * 252.0) if not extra_ret.empty else None,
        "baseline": metrics_base,
        "strategy": metrics_strategy,
        "delta_total_return": (
            metrics_strategy["total_return"] - metrics_base["total_return"]
            if metrics_strategy.get("total_return") is not None and metrics_base.get("total_return") is not None
            else None
        ),
        "delta_sharpe": (
            metrics_strategy["sharpe"] - metrics_base["sharpe"]
            if metrics_strategy.get("sharpe") is not None and metrics_base.get("sharpe") is not None
            else None
        ),
    }


def build_notebook(
    notebook_path: Path,
    output_dir: Path,
    script_path: Path,
    config: StrategyConfig,
    metrics: Dict[str, object],
    rebalance_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> None:
    import nbformat as nbf

    recent_daily = daily_df.tail(20).copy()
    if not recent_daily.empty:
        recent_daily["trade_date"] = pd.to_datetime(recent_daily["trade_date"]).dt.strftime("%Y-%m-%d")
        recent_daily_md = recent_daily.round(6).to_markdown(index=False)
    else:
        recent_daily_md = "无结果。"

    rebalance_preview = rebalance_df.tail(10).copy()
    if not rebalance_preview.empty:
        rebalance_preview["trade_date"] = pd.to_datetime(rebalance_preview["trade_date"]).dt.strftime("%Y-%m-%d")
        rebalance_preview["prev_trade_date"] = pd.to_datetime(rebalance_preview["prev_trade_date"]).dt.strftime("%Y-%m-%d")
        rebalance_md = rebalance_preview.to_markdown(index=False)
    else:
        rebalance_md = "无调仓记录。"

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 开盘买入卖出策略回测\n\n"
            "策略流程:\n"
            "1. 每周第一个交易日，使用前一交易日 `amount >= 1000万`、`total_mv >= 10亿` 筛选股票。\n"
            "2. 从候选中按市值升序选最小的 50 只作为底仓。\n"
            "3. 每个交易日读取 `09:24:50` 之前最后一个集合竞价快照，计算相对昨收涨跌幅。\n"
            "4. 开盘买入涨跌幅最小的 10 只，同时卖出涨跌幅最大的 10 只。\n"
            "5. 收盘反向平仓，恢复到底仓。\n"
        ),
        nbf.v4.new_markdown_cell(
            "## 参数\n\n"
            f"- `min_amount={config.min_amount}`\n"
            f"- `min_total_mv={config.min_total_mv}`\n"
            f"- `hold_count={config.hold_count}`\n"
            f"- `trade_count={config.trade_count}`\n"
            f"- `buy_fee={config.buy_fee}`\n"
            f"- `sell_fee={config.sell_fee}`\n"
            f"- `slippage={config.slippage}`\n"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from IPython.display import Image, display\n\n"
            f"repo_root = Path(r'{str(_REPO_ROOT)}')\n"
            f"outdir = Path(r'{str(output_dir)}')\n"
            f"script_path = Path(r'{str(script_path)}')\n"
            "metrics = json.loads((outdir / 'metrics.json').read_text(encoding='utf-8'))\n"
            "daily = pd.read_csv(outdir / 'daily_portfolio.csv', parse_dates=['trade_date'])\n"
            "rebalance = pd.read_csv(outdir / 'weekly_rebalance_log.csv', parse_dates=['trade_date', 'prev_trade_date'])\n"
            "stock = pd.read_parquet(outdir / 'stock_level_results.parquet')\n"
            "print(metrics)\n"
            "daily.head()\n"
        ),
        nbf.v4.new_markdown_cell(
            "## 最近 20 个交易日组合结果\n\n"
            f"{recent_daily_md}"
        ),
        nbf.v4.new_markdown_cell(
            "## 最近 10 次周调仓记录\n\n"
            f"{rebalance_md}"
        ),
        nbf.v4.new_code_cell(
            "nav = daily[['trade_date', 'base_return', 'strategy_return']].copy()\n"
            "nav['baseline_nav'] = (1 + nav['base_return']).cumprod()\n"
            "nav['strategy_nav'] = (1 + nav['strategy_return']).cumprod()\n"
            "ax = nav.plot(x='trade_date', y=['baseline_nav', 'strategy_nav'], figsize=(10, 5), grid=True)\n"
            "ax.set_title('Baseline vs Auction Hedge')\n"
            "ax.set_ylabel('NAV')\n"
            "plt.show()\n"
        ),
        nbf.v4.new_code_cell(
            "signal_summary = {\n"
            "    'trade_days': len(daily),\n"
            "    'avg_hold_count': daily['hold_count'].mean(),\n"
            "    'avg_valid_signal_count': daily['valid_signal_count'].mean(),\n"
            "    'buy_trades_total': daily['buy_count'].sum(),\n"
            "    'sell_trades_total': daily['sell_count'].sum(),\n"
            "    'avg_extra_return_bp': daily['extra_return'].mean() * 10000,\n"
            "}\n"
            "signal_summary\n"
        ),
        nbf.v4.new_markdown_cell(
            "## 核心绩效\n\n"
            f"- 基线总收益: `{metrics.get('baseline', {}).get('total_return')}`\n"
            f"- 策略总收益: `{metrics.get('strategy', {}).get('total_return')}`\n"
            f"- 超额总收益: `{metrics.get('delta_total_return')}`\n"
            f"- 基线 Sharpe: `{metrics.get('baseline', {}).get('sharpe')}`\n"
            f"- 策略 Sharpe: `{metrics.get('strategy', {}).get('sharpe')}`\n"
            f"- Sharpe 差值: `{metrics.get('delta_sharpe')}`\n"
            f"- 年化超额收益: `{metrics.get('annual_extra_return')}`\n"
        ),
        nbf.v4.new_code_cell(
            "display(Image(filename=str(outdir / 'nav_curve.png')))\n"
        ),
    ]
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, notebook_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="开盘集合竞价涨跌幅对冲策略回测")
    parser.add_argument("--start", default="20260105", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="20260320", help="结束日期 YYYYMMDD")
    parser.add_argument("--daily-dir", default="daily_data/daily", help="日线目录")
    parser.add_argument("--daily-basic-dir", default="daily_data/daily_basic", help="daily_basic目录")
    parser.add_argument("--tick-dir", default="tick_2026", help="tick 根目录")
    parser.add_argument("--min-amount", type=float, default=10_000.0, help="上一交易日最小成交额(千元)")
    parser.add_argument("--min-total-mv", type=float, default=100_000.0, help="最小总市值(万元)")
    parser.add_argument("--hold-count", type=int, default=50, help="底仓持股数")
    parser.add_argument("--trade-count", type=int, default=10, help="每日开盘加减仓股票数")
    parser.add_argument("--buy-fee", type=float, default=0.0002, help="买入费率")
    parser.add_argument("--sell-fee", type=float, default=0.0007, help="卖出费率")
    parser.add_argument("--slippage", type=float, default=0.0003, help="单边滑点")
    parser.add_argument("--cash", type=float, default=10_000_000.0, help="初始资金")
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=0.7,
        help="单日有效覆盖率下限",
    )
    parser.add_argument(
        "--output-dir",
        default="backtest/output_open_auction_hedge",
        help="输出目录",
    )
    parser.add_argument(
        "--notebook-path",
        default="notebook/open_auction_hedge_strategy_research.ipynb",
        help="生成 notebook 路径",
    )
    args = parser.parse_args()

    config = StrategyConfig(
        min_amount=args.min_amount,
        min_total_mv=args.min_total_mv,
        hold_count=args.hold_count,
        trade_count=args.trade_count,
        buy_fee=args.buy_fee,
        sell_fee=args.sell_fee,
        slippage=args.slippage,
        cash=args.cash,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_start = (pd.Timestamp(args.start) - pd.Timedelta(days=14)).strftime("%Y%m%d")

    print("Loading daily + daily_basic panel...")
    panel = load_combined_panel(
        daily_dir=args.daily_dir,
        daily_basic_dir=args.daily_basic_dir,
        start=data_start,
        end=args.end,
    )
    if panel.empty:
        raise ValueError("未加载到日线与市值数据")
    panel = panel.drop_duplicates(subset=["trade_date", "ts_code"], keep="first").copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    tick_root = Path(args.tick_dir)
    tick_dates = set(
        list_tick_dates(
            tick_root=tick_root,
            start_year=int(args.start[:4]),
            end_year=int(args.end[:4]),
        )
    )
    user_start = pd.Timestamp(args.start).normalize()
    trade_dates = [
        d for d in sorted(panel["trade_date"].unique()) if d in tick_dates and pd.Timestamp(d) >= user_start
    ]
    if len(trade_dates) < 5:
        raise ValueError("可用交易日过少")

    holding_map, rebalance_df = build_weekly_holding_map(panel=panel, trade_dates=trade_dates, config=config)
    if rebalance_df.empty:
        raise ValueError("未生成周调仓结果")
    rebalance_df.to_csv(output_dir / "weekly_rebalance_log.csv", index=False)

    stock_df = build_stock_level_panel(
        panel=panel,
        tick_root=tick_root,
        trade_dates=trade_dates,
        holding_map=holding_map,
        min_coverage_ratio=args.min_coverage_ratio,
    )
    if stock_df.empty:
        raise ValueError("未生成股票层面回测面板")

    stock_result, daily_df = simulate_strategy(stock_df=stock_df, config=config)
    if daily_df.empty:
        raise ValueError("未生成组合日收益")

    stock_result.to_parquet(output_dir / "stock_level_results.parquet", index=False)
    daily_df.to_csv(output_dir / "daily_portfolio.csv", index=False)

    metrics = build_metrics(daily_df=daily_df, config=config)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_nav_plot(daily_df=daily_df, output_png=output_dir / "nav_curve.png")
    build_notebook(
        notebook_path=Path(args.notebook_path),
        output_dir=output_dir.resolve(),
        script_path=Path(__file__).resolve(),
        config=config,
        metrics=metrics,
        rebalance_df=rebalance_df,
        daily_df=daily_df,
    )

    print("\nDone.")
    print(f"Output dir: {output_dir}")
    print(f"Notebook: {args.notebook_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
