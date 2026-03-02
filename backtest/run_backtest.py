import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import backtrader as bt
import pandas as pd

from data_source import load_daily_panel, split_symbol_frames
from performance_metrics import (
    build_equal_weight_benchmark_returns,
    compute_performance_metrics,
)


class TusharePandasData(bt.feeds.PandasData):
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", "openinterest"),
    )


class EqualWeightSmaCross(bt.Strategy):
    params = (
        ("short_window", 10),
        ("long_window", 30),
        ("rebalance_days", 5),
    )

    def __init__(self) -> None:
        self.short_sma = {d: bt.ind.SMA(d.close, period=self.p.short_window) for d in self.datas}
        self.long_sma = {d: bt.ind.SMA(d.close, period=self.p.long_window) for d in self.datas}

    def next(self) -> None:
        if self.p.rebalance_days > 1 and (len(self) % self.p.rebalance_days != 0):
            return

        bullish = []
        for d in self.datas:
            if len(d) < self.p.long_window:
                continue
            if self.short_sma[d][0] > self.long_sma[d][0] and d.close[0] > 0:
                bullish.append(d)

        target = 1.0 / len(bullish) if bullish else 0.0
        bullish_names = {d._name for d in bullish}

        for d in self.datas:
            if d._name in bullish_names:
                self.order_target_percent(data=d, target=target)
            else:
                self.order_target_percent(data=d, target=0.0)


def _select_symbols(panel: pd.DataFrame, symbol_limit: int) -> List[str]:
    ranked = (
        panel.groupby("ts_code", as_index=False)["close"]
        .count()
        .rename(columns={"close": "bars"})
        .sort_values(["bars", "ts_code"], ascending=[False, True])
    )
    return ranked["ts_code"].head(symbol_limit).tolist()


def run_backtest(
    daily_dir: str,
    start: str,
    end: str,
    symbols: Optional[List[str]],
    symbol_limit: int,
    cash: float,
    commission: float,
    short_window: int,
    long_window: int,
    rebalance_days: int,
    output_dir: str,
) -> Dict[str, float]:
    panel = load_daily_panel(daily_dir=daily_dir, start=start, end=end, symbols=symbols)
    if panel.empty:
        raise ValueError("No data loaded. Please check date range or symbol list.")

    selected_symbols = symbols or _select_symbols(panel, symbol_limit)
    selected_set = set(selected_symbols)
    panel = panel[panel["ts_code"].isin(selected_set)].copy()
    if panel.empty:
        raise ValueError("No symbols left after selection.")

    symbol_frames = split_symbol_frames(panel)
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    for code in selected_symbols:
        frame = symbol_frames.get(code)
        if frame is None or frame.empty:
            continue
        feed = TusharePandasData(dataname=frame)
        cerebro.adddata(feed, name=code)

    if not cerebro.datas:
        raise ValueError("No valid data feed added to backtrader.")

    cerebro.addstrategy(
        EqualWeightSmaCross,
        short_window=short_window,
        long_window=long_window,
        rebalance_days=rebalance_days,
    )
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")

    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()

    time_return = strat.analyzers.time_return.get_analysis()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    strategy_returns = pd.Series(time_return).sort_index()
    if not strategy_returns.empty:
        equity_curve = (1.0 + strategy_returns).cumprod()
        equity_curve.to_csv(out_path / "equity_curve.csv", header=["nav"])

    benchmark_returns = build_equal_weight_benchmark_returns(panel=panel, symbols=selected_symbols)
    metrics = compute_performance_metrics(
        strategy_returns=strategy_returns,
        initial_cash=cash,
        final_value=float(final_value),
        benchmark_returns=benchmark_returns,
    )
    metrics["symbols_count"] = int(len(cerebro.datas))
    metrics["short_window"] = int(short_window)
    metrics["long_window"] = int(long_window)
    metrics["rebalance_days"] = int(rebalance_days)

    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local-daily-data backtest with Backtrader.")
    parser.add_argument("--daily-dir", default="daily_data/daily", help="Path to daily parquet directory")
    parser.add_argument("--start", default="20220101", help="Start date YYYYMMDD")
    parser.add_argument("--end", default="20221231", help="End date YYYYMMDD")
    parser.add_argument("--symbols", default="", help="Comma-separated ts_code list, e.g. 000001.SZ,600000.SH")
    parser.add_argument("--symbol-limit", type=int, default=20, help="Used when --symbols is empty")
    parser.add_argument("--cash", type=float, default=1_000_000.0, help="Initial cash")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission ratio")
    parser.add_argument("--short-window", type=int, default=10, help="Short SMA window")
    parser.add_argument("--long-window", type=int, default=30, help="Long SMA window")
    parser.add_argument("--rebalance-days", type=int, default=5, help="Rebalance every N bars")
    parser.add_argument("--output-dir", default="backtest/output", help="Output folder")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    metrics = run_backtest(
        daily_dir=args.daily_dir,
        start=args.start,
        end=args.end,
        symbols=symbols if symbols else None,
        symbol_limit=args.symbol_limit,
        cash=args.cash,
        commission=args.commission,
        short_window=args.short_window,
        long_window=args.long_window,
        rebalance_days=args.rebalance_days,
        output_dir=args.output_dir,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
