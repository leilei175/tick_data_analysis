"""
生成基本面因子并保存到 factor/fundamental 目录。

默认使用中文财务日频数据：
- income_daily_cn
- cashflow_daily_cn
- balance_daily_cn

以及 daily_basic 的市值数据。
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import gc

from mylib.get_local_data import get_local_data


def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)


def _load_zz1000_from_local() -> List[str]:
    candidates = [
        Path("factor/daily/zz1000_factors_20251231.parquet"),
        Path("factor/daily/zz1000_factors_20251230.parquet"),
        Path("factor/daily/zz1000_all_factors.parquet"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        schema = pq.read_schema(str(path)).names
        if "stock_code" in schema:
            df = pq.read_table(str(path), columns=["stock_code"]).to_pandas()
            sec = (
                df["stock_code"]
                .astype(str)
                .str.strip()
                .str.upper()
                .dropna()
                .unique()
                .tolist()
            )
            if sec:
                print(f"[pool] 使用本地股票池: {path} ({len(sec)} 只)")
                return sec
    raise FileNotFoundError("未找到本地中证1000股票池文件")


def _load_field(sec_list: Optional[List[str]], start: str, end: str, field: str, data_type: str) -> pd.DataFrame:
    df = get_local_data(
        sec_list=sec_list,
        start=start,
        end=end,
        filed=field,
        data_type=data_type,
    )
    if df.empty:
        print(f"[warn] 字段为空: {data_type}.{field}")
    return df


def _save_factor(df: pd.DataFrame, output_dir: Path, prefix: str, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{prefix}_{name}.parquet"
    df.to_parquet(out_file)
    print(f"[save] {out_file} shape={df.shape}")


def build_and_save_factors(
    sec_list: Optional[List[str]],
    start: str,
    end: str,
    output_dir: Path,
    prefix: str,
) -> None:
    print("[load] 流式模式：逐因子加载与保存")
    # ROE
    net_income = _load_field(sec_list, start, end, "净利润(不含少数股东损益)", "income_daily_cn")
    equity = _load_field(sec_list, start, end, "股东权益合计(不含少数股东权益)", "balance_daily_cn")
    _save_factor(_safe_div(net_income, equity), output_dir, prefix, "roe")
    del equity
    gc.collect()

    # ROA
    assets = _load_field(sec_list, start, end, "资产总计", "balance_daily_cn")
    _save_factor(_safe_div(net_income, assets), output_dir, prefix, "roa")

    # GP/TA
    revenue = _load_field(sec_list, start, end, "营业收入", "income_daily_cn")
    cogs = _load_field(sec_list, start, end, "减:营业成本", "income_daily_cn")
    _save_factor(_safe_div(revenue - cogs, assets), output_dir, prefix, "gp_ta")

    # CFO quality
    cfo = _load_field(sec_list, start, end, "经营活动产生的现金流量净额", "cashflow_daily_cn")
    _save_factor(_safe_div(cfo, net_income), output_dir, prefix, "cfo_ratio")
    _save_factor(_safe_div(net_income - cfo, assets), output_dir, prefix, "accruals")

    # FCF yield
    capex = _load_field(sec_list, start, end, "购建固定资产、无形资产和其他长期资产支付的现金", "cashflow_daily_cn")
    market_cap = _load_field(sec_list, start, end, "total_mv", "daily_basic")
    _save_factor(_safe_div(cfo - capex, market_cap), output_dir, prefix, "fcf_yield")

    # Book-to-market
    equity = _load_field(sec_list, start, end, "股东权益合计(不含少数股东权益)", "balance_daily_cn")
    _save_factor(_safe_div(equity, market_cap), output_dir, prefix, "book_to_market")

    # EBITDA/EV
    ebitda = _load_field(sec_list, start, end, "息税折旧摊销前利润", "income_daily_cn")
    cash = _load_field(sec_list, start, end, "货币资金", "balance_daily_cn")
    st_debt = _load_field(sec_list, start, end, "短期借款", "balance_daily_cn")
    lt_debt = _load_field(sec_list, start, end, "长期借款", "balance_daily_cn")
    ev = market_cap + st_debt.fillna(0) + lt_debt.fillna(0) - cash.fillna(0)
    _save_factor(_safe_div(ebitda, ev), output_dir, prefix, "ebitda_ev")

    # Growth
    _save_factor(net_income.pct_change(periods=252).replace([np.inf, -np.inf], np.nan), output_dir, prefix, "profit_growth_yoy")
    _save_factor(revenue.pct_change(periods=252).replace([np.inf, -np.inf], np.nan), output_dir, prefix, "revenue_growth_yoy")

    # Leverage
    liabilities = _load_field(sec_list, start, end, "负债合计", "balance_daily_cn")
    _save_factor(_safe_div(liabilities, assets), output_dir, prefix, "leverage")

    # Interest coverage
    ebit = _load_field(sec_list, start, end, "息税前利润", "income_daily_cn")
    interest_exp = _load_field(sec_list, start, end, "财务费用:利息费用", "income_daily_cn")
    _save_factor(_safe_div(ebit, interest_exp), output_dir, prefix, "interest_coverage")

    # Earnings yield
    _save_factor(_safe_div(net_income, market_cap), output_dir, prefix, "earnings_yield")

    del net_income, assets, revenue, cogs, cfo, capex, market_cap, equity
    del ebitda, cash, st_debt, lt_debt, ev, liabilities, ebit, interest_exp
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="生成基本面因子并保存")
    parser.add_argument("--start", default="20200101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="20261031", help="结束日期 YYYYMMDD")
    parser.add_argument(
        "--output-dir",
        default="factor/fundamental",
        help="输出目录",
    )
    parser.add_argument(
        "--prefix",
        default="zz1000",
        help="输出文件名前缀",
    )
    parser.add_argument(
        "--all-stocks",
        action="store_true",
        help="使用全市场而非本地中证1000股票池",
    )
    args = parser.parse_args()

    sec_list = None if args.all_stocks else _load_zz1000_from_local()
    build_and_save_factors(
        sec_list=sec_list,
        start=args.start,
        end=args.end,
        output_dir=Path(args.output_dir),
        prefix=args.prefix,
    )
    print("[done] 基本面因子生成完成")


if __name__ == "__main__":
    main()
