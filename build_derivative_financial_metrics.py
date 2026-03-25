#!/usr/bin/env python3
"""
构建衍生财务指标（日频）并保存到 daily_data/derivative 目录。

指标：
- roe: 净资产收益率
- roa: 总资产收益率
- gross_margin: 销售毛利率
- roic: 投资资本回报率
- enterprise_value: 企业价值
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from mylib.get_local_data import get_all_data, list_data_files


OUTPUT_DIR = Path("daily_data") / "derivative"
OUTPUT_PREFIX = "derivative"
METRIC_FIELDS = ["roe", "roa", "gross_margin", "roic", "enterprise_value"]


def _safe_div(numer: pd.DataFrame, denom: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """安全除法：分母接近0时置 NaN。"""
    numer, denom = numer.align(denom, join="outer")
    out = numer / denom
    out = out.where(denom.abs() > eps)
    return out.replace([np.inf, -np.inf], np.nan)


def _avg_capital(df: pd.DataFrame) -> pd.DataFrame:
    """按时间计算期初期末平均值（近似日频平均资本）。"""
    return (df + df.shift(1)) / 2.0


def _normalize_tax_rate(income_tax: pd.DataFrame, total_profit: pd.DataFrame) -> pd.DataFrame:
    """估算有效税率并做稳健处理。"""
    tax_rate = _safe_div(income_tax, total_profit)
    tax_rate = tax_rate.clip(lower=0.0, upper=1.0)
    daily_median = tax_rate.median(axis=1, skipna=True)
    tax_rate = tax_rate.T.fillna(daily_median).T.fillna(0.25)
    return tax_rate


def _load_source_data(start: str, end: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    income_fields = [
        "净利润(不含少数股东损益)",
        "营业收入",
        "营业总收入",
        "减:营业成本",
        "营业总成本",
        "息税前利润",
        "所得税费用",
        "利润总额",
    ]
    balance_fields = [
        "资产总计",
        "股东权益合计(不含少数股东权益)",
        "短期借款",
        "长期借款",
        "应付债券",
        "货币资金",
    ]
    daily_basic_fields = ["total_mv"]

    income = get_all_data(
        data_type="income_daily_cn",
        start=start,
        end=end,
        fields=income_fields,
        parallel=True,
    )
    balance = get_all_data(
        data_type="balance_daily_cn",
        start=start,
        end=end,
        fields=balance_fields,
        parallel=True,
    )
    daily_basic = get_all_data(
        data_type="daily_basic",
        start=start,
        end=end,
        fields=daily_basic_fields,
        parallel=True,
    )
    return {"income": income, "balance": balance, "daily_basic": daily_basic}


def _build_metrics(source: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    income = source["income"]
    balance = source["balance"]
    daily_basic = source["daily_basic"]

    n_income = income["净利润(不含少数股东损益)"]
    equity = balance["股东权益合计(不含少数股东权益)"]
    total_assets = balance["资产总计"]

    roe = _safe_div(n_income, _avg_capital(equity))
    roa = _safe_div(n_income, _avg_capital(total_assets))

    revenue = income["营业收入"].combine_first(income["营业总收入"])
    cost = income["减:营业成本"].combine_first(income["营业总成本"])
    gross_margin = _safe_div(revenue - cost, revenue)

    ebit = income["息税前利润"].combine_first(income["利润总额"])
    tax_rate = _normalize_tax_rate(income["所得税费用"], income["利润总额"])
    nopat = ebit * (1.0 - tax_rate)

    debt = (
        balance["短期借款"].fillna(0.0)
        + balance["长期借款"].fillna(0.0)
        + balance["应付债券"].fillna(0.0)
    )
    cash = balance["货币资金"].fillna(0.0)
    invested_capital = equity + debt - cash
    roic = _safe_div(nopat, _avg_capital(invested_capital))
    enterprise_value = daily_basic["total_mv"] + debt - cash

    metrics = {
        "roe": roe,
        "roa": roa,
        "gross_margin": gross_margin,
        "roic": roic,
        "enterprise_value": enterprise_value,
    }

    for name, df in metrics.items():
        metrics[name] = df.replace([np.inf, -np.inf], np.nan).sort_index()
    return metrics


def _write_daily_files(metrics: Dict[str, pd.DataFrame]) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dates = set()
    for df in metrics.values():
        if df is not None and not df.empty:
            all_dates.update(df.index.tolist())
    if not all_dates:
        return 0

    write_count = 0
    for dt in sorted(all_dates):
        parts: List[pd.Series] = []
        for field in METRIC_FIELDS:
            df = metrics[field]
            if dt in df.index:
                parts.append(df.loc[dt].rename(field))
        if not parts:
            continue

        day_df = pd.concat(parts, axis=1)
        day_df.index.name = "ts_code"
        day_df = day_df.reset_index()
        day_df = day_df.dropna(subset=METRIC_FIELDS, how="all")
        if day_df.empty:
            continue

        trade_date = int(pd.Timestamp(dt).strftime("%Y%m%d"))
        day_df["trade_date"] = trade_date
        day_df = day_df[["ts_code", "trade_date", *METRIC_FIELDS]]

        year = f"{trade_date:08d}"[:4]
        month = f"{trade_date:08d}"[4:6]
        out_dir = OUTPUT_DIR / year / month
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{OUTPUT_PREFIX}_{trade_date:08d}.parquet"
        day_df.to_parquet(out_file, index=False)
        write_count += 1

    return write_count


def _build_yearly_full_files(start: str, end: str):
    for year in range(int(start[:4]), int(end[:4]) + 1):
        year_files = sorted((OUTPUT_DIR / str(year)).glob("*/derivative_*.parquet"))
        if not year_files:
            continue
        frames = [pd.read_parquet(fp) for fp in year_files]
        year_df = pd.concat(frames, ignore_index=True)
        year_df.to_parquet(OUTPUT_DIR / f"{year}_full.parquet", index=False)


def main():
    required_types = ["income_daily_cn", "balance_daily_cn", "daily_basic"]
    date_ranges = {}
    for data_type in required_types:
        files = list_data_files(data_type=data_type)
        if not files:
            raise RuntimeError(f"未找到 {data_type} 数据文件，无法确定日期范围")
        date_ranges[data_type] = (str(files[0][0]), str(files[-1][0]))

    start = max(v[0] for v in date_ranges.values())
    end = min(v[1] for v in date_ranges.values())
    print(f"[derivative] 构建范围: {start} ~ {end}")

    source = _load_source_data(start=start, end=end)
    metrics = _build_metrics(source)
    file_count = _write_daily_files(metrics)
    _build_yearly_full_files(start=start, end=end)
    print(f"[derivative] 写入日文件: {file_count}")
    print("[derivative] 完成")


if __name__ == "__main__":
    main()
