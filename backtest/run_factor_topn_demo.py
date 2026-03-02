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


class FactorScoreTopN(bt.Strategy):
    params = (
        ("lookback", 20),
        ("topn", 10),
        ("rebalance_days", 5),
    )

    def __init__(self) -> None:
        self.rebalance_records: List[Dict[str, object]] = []

    def _calc_score(self, data: bt.feed.DataBase) -> Optional[float]:
        if len(data) <= self.p.lookback:
            return None
        old_price = float(data.close[-self.p.lookback])
        new_price = float(data.close[0])
        if old_price <= 0 or new_price <= 0:
            return None
        return new_price / old_price - 1.0

    def next(self) -> None:
        if self.p.rebalance_days > 1 and (len(self) % self.p.rebalance_days != 0):
            return

        scored: List[tuple] = []
        for d in self.datas:
            score = self._calc_score(d)
            if score is None:
                continue
            scored.append((d, score))

        if not scored:
            for d in self.datas:
                self.order_target_percent(data=d, target=0.0)
            return

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [item[0] for item in scored[: self.p.topn]]
        selected_names = {d._name for d in selected}
        target = 1.0 / len(selected) if selected else 0.0

        current_date = self.datas[0].datetime.date(0).isoformat()
        self.rebalance_records.append(
            {
                "date": current_date,
                "selected_count": len(selected),
                "selected_symbols": ",".join([d._name for d in selected]),
            }
        )

        for d in self.datas:
            if d._name in selected_names:
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


def run_factor_topn_demo(
    daily_dir: str,
    start: str,
    end: str,
    symbols: Optional[List[str]],
    symbol_limit: int,
    cash: float,
    commission: float,
    lookback: int,
    topn: int,
    rebalance_days: int,
    output_dir: str,
) -> Dict[str, float]:
    panel = load_daily_panel(daily_dir=daily_dir, start=start, end=end, symbols=symbols)
    if panel.empty:
        raise ValueError("未加载到行情数据，请检查日期范围或股票列表。")

    selected_symbols = symbols or _select_symbols(panel, symbol_limit)
    selected_set = set(selected_symbols)
    panel = panel[panel["ts_code"].isin(selected_set)].copy()
    if panel.empty:
        raise ValueError("选股后没有可用数据。")

    symbol_frames = split_symbol_frames(panel)
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    for code in selected_symbols:
        frame = symbol_frames.get(code)
        if frame is None or frame.empty:
            continue
        cerebro.adddata(TusharePandasData(dataname=frame), name=code)

    if not cerebro.datas:
        raise ValueError("没有可用的 Backtrader 数据源。")

    cerebro.addstrategy(
        FactorScoreTopN,
        lookback=lookback,
        topn=topn,
        rebalance_days=rebalance_days,
    )
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")

    results = cerebro.run()
    strat: FactorScoreTopN = results[0]
    final_value = cerebro.broker.getvalue()

    time_return = strat.analyzers.time_return.get_analysis()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    strategy_returns = pd.Series(time_return).sort_index()
    if not strategy_returns.empty:
        equity_curve = (1.0 + strategy_returns).cumprod()
        equity_curve.to_csv(out_path / "equity_curve.csv", header=["nav"])

    if strat.rebalance_records:
        pd.DataFrame(strat.rebalance_records).to_csv(out_path / "rebalance_log.csv", index=False)

    benchmark_returns = build_equal_weight_benchmark_returns(panel=panel, symbols=selected_symbols)
    metrics = compute_performance_metrics(
        strategy_returns=strategy_returns,
        initial_cash=cash,
        final_value=float(final_value),
        benchmark_returns=benchmark_returns,
    )
    metrics["symbols_count"] = int(len(cerebro.datas))
    metrics["lookback"] = int(lookback)
    metrics["topn"] = int(topn)
    metrics["rebalance_days"] = int(rebalance_days)

    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="因子打分 + TopN 调仓 demo（基于本地日线数据）")
    parser.add_argument("--daily-dir", default="daily_data/daily", help="日线 parquet 目录")
    parser.add_argument("--start", default="20220101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="20221231", help="结束日期 YYYYMMDD")
    parser.add_argument("--symbols", default="", help="逗号分隔 ts_code，如 000001.SZ,600000.SH")
    parser.add_argument("--symbol-limit", type=int, default=50, help="未指定 symbols 时自动选取的数据覆盖最多标的数量")
    parser.add_argument("--cash", type=float, default=1_000_000.0, help="初始资金")
    parser.add_argument("--commission", type=float, default=0.001, help="手续费比例")
    parser.add_argument("--lookback", type=int, default=20, help="因子回看窗口（天）")
    parser.add_argument("--topn", type=int, default=10, help="每次调仓持有 TopN")
    parser.add_argument("--rebalance-days", type=int, default=5, help="每隔 N 个交易日调仓")
    parser.add_argument("--output-dir", default="backtest/output_factor_topn", help="输出目录")
    args = parser.parse_args()

    if args.topn <= 0:
        raise ValueError("--topn 必须大于 0")
    if args.lookback <= 0:
        raise ValueError("--lookback 必须大于 0")
    if args.rebalance_days <= 0:
        raise ValueError("--rebalance-days 必须大于 0")

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    metrics = run_factor_topn_demo(
        daily_dir=args.daily_dir,
        start=args.start,
        end=args.end,
        symbols=symbols if symbols else None,
        symbol_limit=args.symbol_limit,
        cash=args.cash,
        commission=args.commission,
        lookback=args.lookback,
        topn=args.topn,
        rebalance_days=args.rebalance_days,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
