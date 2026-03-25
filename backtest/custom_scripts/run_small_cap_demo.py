"""
小市值Demo策略回测

策略逻辑：
1. 每日筛选条件：
   - 成交额 >= 1000万
   - 排除ST/*ST股票
   - 排除涨跌停股票
2. 选股规则：从满足条件的股票中，取市值最小的50只
3. 调仓频率：每日调仓
4. 持仓权重：等权配置
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_BACKTEST_DIR = _THIS_DIR.parent
if str(_BACKTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKTEST_DIR))

from data_source import load_combined_panel
from performance_metrics import compute_performance_metrics


def _is_st_stock(ts_code: str) -> bool:
    """判断是否为ST股票"""
    # ST股票代码通常包含 ST、*ST、PT
    code_part = ts_code.split(".")[0]
    return "ST" in code_part or "*ST" in code_part or "PT" in code_part


def _filter_st_stocks(panel: pd.DataFrame) -> pd.DataFrame:
    """过滤ST股票"""
    return panel[~panel["ts_code"].apply(_is_st_stock)]


def _filter_by_amount(panel: pd.DataFrame, min_amount: float) -> pd.DataFrame:
    """按成交额过滤（单位：万元）"""
    return panel[panel["amount"] >= min_amount]


def _filter_limit_up_down(panel: pd.DataFrame) -> pd.DataFrame:
    """
    过滤涨跌停股票
    涨停: (close - pre_close) / pre_close >= 9.5% (近似10%，考虑误差)
    跌停: (close - pre_close) / pre_close <= -9.5%
    """
    if "pre_close" not in panel.columns:
        # 尝试从数据中计算
        panel = panel.copy()
        panel["pre_close"] = panel.groupby("ts_code")["close"].shift(1)

    panel = panel.dropna(subset=["pre_close", "close"])
    pct_change = (panel["close"] - panel["pre_close"]) / panel["pre_close"]

    # 排除涨停(>=9.5%)和跌停(<= -9.5%)
    return panel[(pct_change < 0.095) & (pct_change > -0.095)]


def _compute_daily_signals(
    panel: pd.DataFrame,
    min_amount: float,
    max_stocks: int,
    date: pd.Timestamp,
) -> List[str]:
    """
    计算每日选股信号

    Args:
        panel: 当日所有股票数据
        min_amount: 最小成交额（万元）
        max_stocks: 最大持仓数量
        date: 交易日期

    Returns:
        选中的股票代码列表
    """
    # 筛选当日有数据的股票
    daily_data = panel[panel["trade_date"] == date].copy()

    if daily_data.empty:
        return []

    # 过滤ST股票
    daily_data = _filter_st_stocks(daily_data)

    # 过滤成交额不足的股票
    daily_data = _filter_by_amount(daily_data, min_amount)

    # 过滤涨跌停股票
    daily_data = _filter_limit_up_down(daily_data)

    if daily_data.empty:
        return []

    # 按市值排序，选取最小的N只
    daily_data = daily_data.sort_values("total_mv", ascending=True)
    selected = daily_data.head(max_stocks)

    return selected["ts_code"].tolist()


def run_small_cap_backtest(
    daily_dir: str,
    daily_basic_dir: str,
    start: str,
    end: str,
    min_amount: float,
    max_stocks: int,
    cash: float,
    commission: float,
    output_dir: str,
) -> Dict[str, float]:
    """
    运行小市值策略回测

    Args:
        daily_dir: 日线数据目录
        daily_basic_dir: 每日基本面数据目录
        start: 开始日期 YYYYMMDD
        end: 结束日期 YYYYMMDD
        min_amount: 最小成交额（万元）
        max_stocks: 最大持仓数量
        cash: 初始资金
        commission: 手续费率
        output_dir: 输出目录

    Returns:
        性能指标字典
    """
    # 加载数据
    print(f"Loading data from {start} to {end}...")
    panel = load_combined_panel(
        daily_dir=daily_dir,
        daily_basic_dir=daily_basic_dir,
        start=start,
        end=end,
    )

    if panel.empty:
        raise ValueError("未读取到行情数据")

    print(f"Loaded {len(panel)} records, {panel['ts_code'].nunique()} symbols")

    # 获取所有交易日期
    trade_dates = sorted(panel["trade_date"].unique())
    print(f"Trading days: {len(trade_dates)}")

    # 每日持仓和收益
    holdings: Set[str] = set()
    nav = 1.0
    daily_returns: List[float] = []
    dates: List[pd.Timestamp] = []
    rebalance_log: List[Dict] = []  # 调仓日志

    # 资金分配
    position_value = cash

    for i, date in enumerate(trade_dates):
        # 每天计算新的持仓
        signals = _compute_daily_signals(panel, min_amount, max_stocks, date)

        # 获取当日收盘价
        daily_data = panel[panel["trade_date"] == date][["ts_code", "close"]]
        price_map = dict(zip(daily_data["ts_code"], daily_data["close"]))

        if i > 0:
            # 计算昨日持仓在今天的收益
            prev_date = trade_dates[i - 1]
            prev_daily = panel[panel["trade_date"] == prev_date][["ts_code", "close"]]
            prev_price_map = dict(zip(prev_daily["ts_code"], prev_daily["close"]))

            if holdings:
                # 计算持仓收益
                stock_returns = []
                for symbol in holdings:
                    if symbol in prev_price_map and symbol in price_map:
                        prev_price = prev_price_map[symbol]
                        curr_price = price_map[symbol]
                        if prev_price > 0:
                            ret = (curr_price - prev_price) / prev_price
                            stock_returns.append(ret)

                if stock_returns:
                    # 等权平均收益
                    avg_ret = sum(stock_returns) / len(stock_returns)
                    # 扣除手续费
                    net_ret = avg_ret - commission
                    daily_returns.append(net_ret)
                    nav *= (1 + net_ret)
                else:
                    daily_returns.append(0.0)
            else:
                daily_returns.append(0.0)

        # 更新持仓为今日信号
        # 记录调仓日志
        if signals:
            rebalance_log.append({
                "date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                "action": "rebalance",
                "symbols_count": len(signals),
                "symbols": ",".join(sorted(signals)[:10]) + ("..." if len(signals) > 10 else ""),
            })
        holdings = set(signals)
        dates.append(date)

        # 跳过第一天（没有前一日持仓收益）
        if i == 0:
            daily_returns.append(0.0)

        # 定期打印进度
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(trade_dates)} days, nav: {nav:.4f}")

    # 最后一天也需要计算收益（如果需要）
    # 这里已经包含了最后一天之前的收益

    # 构建收益序列
    strategy_returns = pd.Series(daily_returns, index=dates)
    strategy_returns.index.name = "date"

    # 计算基准收益（等权市场收益）
    print("Computing benchmark returns...")
    pivot = panel.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    returns = pivot.pct_change().fillna(0.0)
    # 跳过第一天（没有前一天的数据）
    benchmark_returns = returns.mean(axis=1).iloc[1:]
    # 对齐日期
    benchmark_returns = benchmark_returns.reindex(dates).fillna(0.0)

    # 保存结果
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 净值曲线
    equity_curve = (1.0 + strategy_returns).cumprod()
    eq_df = equity_curve.reset_index()
    eq_df.columns = ["date", "nav"]
    eq_df.to_csv(out / "equity_curve.csv", index=False)

    # 调仓日志
    if rebalance_log:
        rebalance_df = pd.DataFrame(rebalance_log)
        rebalance_df.to_csv(out / "rebalance_log.csv", index=False)

    # 性能指标
    final_value = cash * nav
    metrics = compute_performance_metrics(
        strategy_returns=strategy_returns,
        initial_cash=cash,
        final_value=final_value,
        benchmark_returns=benchmark_returns,
    )

    metrics["min_amount"] = min_amount
    metrics["max_stocks"] = max_stocks
    metrics["note"] = "小市值策略：每日选取成交额>=1000万股票中市值最小的50只"

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nBacktest completed!")
    print(f"  Final NAV: {nav:.4f}")
    print(f"  Total Return: {(nav - 1) * 100:.2f}%")
    print(f"  Results saved to: {output_dir}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="小市值Demo策略回测")
    parser.add_argument("--start", default="20220101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="20231231", help="结束日期 YYYYMMDD")
    parser.add_argument("--daily-dir", default="daily_data/daily", help="日线数据目录")
    parser.add_argument("--daily-basic-dir", default="daily_data/daily_basic", help="每日基本面数据目录")
    parser.add_argument("--min-amount", type=float, default=1000.0, help="最小成交额（万元）")
    parser.add_argument("--max-stocks", type=int, default=50, help="最大持仓数量")
    parser.add_argument("--cash", type=float, default=10_000_000.0, help="初始资金")
    parser.add_argument("--commission", type=float, default=0.0001, help="手续费率")
    parser.add_argument("--output-dir", default="backtest/output_smallcap", help="输出目录")
    args, _unknown = parser.parse_known_args()

    metrics = run_small_cap_backtest(
        daily_dir=args.daily_dir,
        daily_basic_dir=args.daily_basic_dir,
        start=args.start,
        end=args.end,
        min_amount=args.min_amount,
        max_stocks=args.max_stocks,
        cash=args.cash,
        commission=args.commission,
        output_dir=args.output_dir,
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
