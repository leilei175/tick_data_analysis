import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_BACKTEST_DIR = _THIS_DIR.parent
if str(_BACKTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKTEST_DIR))

from data_source import load_daily_panel
from performance_metrics import compute_performance_metrics


def _pick_symbols(panel: pd.DataFrame, limit: int) -> List[str]:
    ranked = (
        panel.groupby("ts_code", as_index=False)["close"]
        .count()
        .rename(columns={"close": "bars"})
        .sort_values(["bars", "ts_code"], ascending=[False, True])
    )
    return ranked["ts_code"].head(limit).tolist()


def run_demo(
    daily_dir: str,
    start: str,
    end: str,
    symbols: Optional[List[str]],
    symbol_limit: int,
    cash: float,
    commission: float,
    output_dir: str,
) -> Dict[str, float]:
    panel = load_daily_panel(daily_dir=daily_dir, start=start, end=end, symbols=symbols)
    if panel.empty:
        raise ValueError("未读取到行情数据")

    selected_symbols = symbols or _pick_symbols(panel, symbol_limit)
    panel = panel[panel["ts_code"].isin(set(selected_symbols))].copy()
    if panel.empty:
        raise ValueError("筛选后无可用数据")

    panel = panel.sort_values(["trade_date", "ts_code"])
    pivot = panel.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    returns = pivot.pct_change().fillna(0.0)
    eq_daily_ret = returns.mean(axis=1)

    # 扣除一个简单手续费近似：每天固定扣减 commission / 10
    eq_daily_ret = eq_daily_ret - (commission / 10.0)
    nav = (1.0 + eq_daily_ret).cumprod()
    if nav.empty:
        raise ValueError("净值序列为空")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eq_df = nav.reset_index()
    eq_df.columns = ["date", "nav"]
    eq_df.to_csv(out / "equity_curve.csv", index=False)

    benchmark_returns = returns.mean(axis=1)
    metrics = compute_performance_metrics(
        strategy_returns=eq_daily_ret,
        initial_cash=cash,
        final_value=float(cash * nav.iloc[-1]),
        benchmark_returns=benchmark_returns,
    )
    metrics["symbols_count"] = int(len(selected_symbols))
    metrics["note"] = "自定义脚本示例：等权日收益组合"
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="自定义回测脚本示例")
    parser.add_argument("--start", default="20220101")
    parser.add_argument("--end", default="20221231")
    parser.add_argument("--daily-dir", default="daily_data/daily")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbol-limit", type=int, default=50)
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--output-dir", default="backtest/output_custom_demo")
    args, _unknown = parser.parse_known_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    metrics = run_demo(
        daily_dir=args.daily_dir,
        start=args.start,
        end=args.end,
        symbols=symbols if symbols else None,
        symbol_limit=args.symbol_limit,
        cash=args.cash,
        commission=args.commission,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
