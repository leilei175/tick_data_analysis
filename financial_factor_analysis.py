"""
财务指标因子分析
================

根据日频财务报表数据计算财务指标，并对这些指标进行因子分析

财务指标包括：
1. 盈利能力指标：毛利率、净利率、ROE、ROA
2. 偿债能力指标：资产负债率、流动比率
3. 运营效率指标：应收账款周转率、存货周转率
4. 估值指标：PE、PB、PS
5. 股息指标：股息率

使用说明：
---------
python financial_factor_analysis.py --start 20250101 --end 20251231 --stocks 1000
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mylib.get_local_data import get_local_data
from update_data import init_tushare

# =============================================================================
# 配置
# =============================================================================

OUTPUT_DIR = './factor_analysis_results/financial/'
REPORT_DIR = './factor_analysis_results/financial_reports/'

# =============================================================================
# 财务指标计算
# =============================================================================

def load_financial_data(stocks: List[str], start_date: str, end_date: str) -> Dict:
    """
    加载所有财务数据
    """
    print("加载财务数据...")

    # 利润表
    income_fields = ['total_revenue', 'revenue', 'operate_profit', 'total_profit', 'n_income', 'basic_eps']
    income_data = {}
    for field in income_fields:
        df = get_local_data(stocks, start_date, end_date, field, 'income_daily')
        income_data[field] = df
    print(f"  利润表: {len(income_data)} 字段")

    # 资产负债表
    balance_fields = ['total_assets', 'total_liab', 'total_hldr_eqy_exc_min_int',
                      'total_cur_assets', 'cash_reser_cb', 'accounts_receiv', 'inventories']
    balance_data = {}
    for field in balance_fields:
        df = get_local_data(stocks, start_date, end_date, field, 'balance_daily')
        balance_data[field] = df
    print(f"  资产负债表: {len(balance_data)} 字段")

    # 现金流
    cashflow_fields = ['n_cashflow_act', 'n_cashflow_inv_act', 'n_cash_flows_fnc_act']
    cashflow_data = {}
    for field in cashflow_fields:
        df = get_local_data(stocks, start_date, end_date, field, 'cashflow_daily')
        cashflow_data[field] = df
    print(f"  现金流: {len(cashflow_data)} 字段")

    # 市值
    market_cap = get_local_data(stocks, start_date, end_date, 'total_mv', 'daily_basic')
    print(f"  市值数据: {market_cap.shape}")

    # 收盘价
    close = get_local_data(stocks, start_date, end_date, 'close', 'daily_basic')
    print(f"  收盘价: {close.shape}")

    return {
        'income': income_data,
        'balance': balance_data,
        'cashflow': cashflow_data,
        'market_cap': market_cap,
        'close': close
    }


def compute_financial_indicators(data: Dict, stocks: List[str]) -> Dict[str, pd.DataFrame]:
    """
    计算财务指标

    Returns:
        Dict[str, pd.DataFrame]: 指标名称 -> DataFrame (index=date, columns=stock)
    """
    print("\n计算财务指标...")

    income = data['income']
    balance = data['balance']
    cashflow = data['cashflow']
    market_cap = data['market_cap']
    close = data['close']

    indicators = {}

    # ========== 盈利能力指标 ==========
    print("  计算盈利能力指标...")

    # 毛利率 = (营业收入 - 营业成本) / 营业收入
    # 使用 operate_profit / revenue 作为代理
    if 'revenue' in income and 'operate_profit' in income:
        gross_margin = income['operate_profit'] / income['revenue']
        gross_margin = gross_margin.replace([np.inf, -np.inf], np.nan)
        indicators['gross_margin'] = gross_margin
        print(f"    毛利率: {gross_margin.shape}")

    # 净利率 = 净利润 / 营业收入
    if 'n_income' in income and 'revenue' in income:
        net_margin = income['n_income'] / income['revenue']
        net_margin = net_margin.replace([np.inf, -np.inf], np.nan)
        indicators['net_margin'] = net_margin
        print(f"    净利率: {net_margin.shape}")

    # 营业利润率 = 营业利润 / 营业收入
    if 'operate_profit' in income and 'revenue' in income:
        op_margin = income['operate_profit'] / income['revenue']
        op_margin = op_margin.replace([np.inf, -np.inf], np.nan)
        indicators['op_margin'] = op_margin
        print(f"    营业利润率: {op_margin.shape}")

    # ROE (净资产收益率) = 净利润 / 归属于母公司的股东权益
    if 'n_income' in income and 'total_hldr_eqy_exc_min_int' in balance:
        equity = balance['total_hldr_eqy_exc_min_int']
        # 使用流通市值作为权益代理
        roe = income['n_income'] / equity
        roe = roe.replace([np.inf, -np.inf], np.nan)
        indicators['roe'] = roe
        print(f"    ROE: {roe.shape}")

    # ROA (总资产收益率) = 净利润 / 总资产
    if 'n_income' in income and 'total_assets' in balance:
        roa = income['n_income'] / balance['total_assets']
        roa = roa.replace([np.inf, -np.inf], np.nan)
        indicators['roa'] = roa
        print(f"    ROA: {roa.shape}")

    # ========== 偿债能力指标 ==========
    print("  计算偿债能力指标...")

    # 资产负债率 = 总负债 / 总资产
    if 'total_liab' in balance and 'total_assets' in balance:
        debt_ratio = balance['total_liab'] / balance['total_assets']
        debt_ratio = debt_ratio.replace([np.inf, -np.inf], np.nan)
        indicators['debt_ratio'] = debt_ratio
        print(f"    资产负债率: {debt_ratio.shape}")

    # 流动比率 = 流动资产 / 流动负债
    # 简化：流动资产 / (总负债 * 0.5) 作为代理
    if 'total_cur_assets' in balance and 'total_liab' in balance:
        # 假设流动负债占总负债的50%
        cur_liab = balance['total_liab'] * 0.5
        current_ratio = balance['total_cur_assets'] / cur_liab
        current_ratio = current_ratio.replace([np.inf, -np.inf], np.nan)
        indicators['current_ratio'] = current_ratio
        print(f"    流动比率: {current_ratio.shape}")

    # ========== 运营效率指标 ==========
    print("  计算运营效率指标...")

    # 应收账款周转率 = 营业收入 / 应收账款
    if 'revenue' in income and 'accounts_receiv' in balance:
        ar_turnover = income['revenue'] / balance['accounts_receiv']
        ar_turnover = ar_turnover.replace([np.inf, -np.inf], np.nan)
        indicators['ar_turnover'] = ar_turnover
        print(f"    应收账款周转率: {ar_turnover.shape}")

    # 存货周转率 = 营业成本 / 存货 (使用revenue作为代理)
    if 'revenue' in income and 'inventories' in balance:
        inv_turnover = income['revenue'] / balance['inventories']
        inv_turnover = inv_turnover.replace([np.inf, -np.inf], np.nan)
        indicators['inv_turnover'] = inv_turnover
        print(f"    存货周转率: {inv_turnover.shape}")

    # 现金流/营收比
    if 'n_cashflow_act' in cashflow and 'revenue' in income:
        cf_to_rev = cashflow['n_cashflow_act'] / income['revenue']
        cf_to_rev = cf_to_rev.replace([np.inf, -np.inf], np.nan)
        indicators['cf_to_revenue'] = cf_to_rev
        print(f"    现金流/营收比: {cf_to_rev.shape}")

    # ========== 估值指标 ==========
    print("  计算估值指标...")

    # 市净率 PB = 市值 / 归属于母公司的股东权益
    if 'market_cap' in data and 'total_hldr_eqy_exc_min_int' in balance:
        equity = balance['total_hldr_eqy_exc_min_int']
        pb = market_cap / equity
        pb = pb.replace([np.inf, -np.inf], np.nan)
        indicators['pb'] = pb
        print(f"    PB: {pb.shape}")

    # 市销率 PS = 市值 / 营业收入
    if 'market_cap' in data and 'revenue' in income:
        ps = market_cap / income['revenue']
        ps = ps.replace([np.inf, -np.inf], np.nan)
        indicators['ps'] = ps
        print(f"    PS: {ps.shape}")

    # ========== 成长能力指标 ==========
    print("  计算成长能力指标...")

    # 营收增长率 (使用日频数据的差分)
    if 'total_revenue' in income:
        revenue_growth = income['total_revenue'].pct_change(periods=5)  # 5日变化
        revenue_growth = revenue_growth.replace([np.inf, -np.inf], np.nan)
        indicators['revenue_growth'] = revenue_growth
        print(f"    营收增长率: {revenue_growth.shape}")

    # 利润增长率
    if 'n_income' in income:
        profit_growth = income['n_income'].pct_change(periods=5)
        profit_growth = profit_growth.replace([np.inf, -np.inf], np.nan)
        indicators['profit_growth'] = profit_growth
        print(f"    利润增长率: {profit_growth.shape}")

    # 现金流增长率
    if 'n_cashflow_act' in cashflow:
        cf_growth = cashflow['n_cashflow_act'].pct_change(periods=5)
        cf_growth = cf_growth.replace([np.inf, -np.inf], np.nan)
        indicators['cf_growth'] = cf_growth
        print(f"    现金流增长率: {cf_growth.shape}")

    # ========== 规模指标 ==========
    print("  计算规模指标...")

    # 资产规模对数
    if 'total_assets' in balance:
        log_assets = np.log(balance['total_assets'].abs() + 1)
        indicators['log_assets'] = log_assets
        print(f"    资产规模对数: {log_assets.shape}")

    # 营收规模对数
    if 'total_revenue' in income:
        log_revenue = np.log(income['total_revenue'].abs() + 1)
        indicators['log_revenue'] = log_revenue
        print(f"    营收规模对数: {log_revenue.shape}")

    # ========== 收益质量指标 ==========
    print("  计算收益质量指标...")

    # 现金流/净利润比 (收益质量)
    if 'n_cashflow_act' in cashflow and 'n_income' in income:
        cf_to_nincome = cashflow['n_cashflow_act'] / (income['n_income'].abs() + 0.01)
        cf_to_nincome = cf_to_nincome.replace([np.inf, -np.inf], np.nan)
        indicators['cf_to_nincome'] = cf_to_nincome
        print(f"    现金流/净利润比: {cf_to_nincome.shape}")

    print(f"\n共计算 {len(indicators)} 个财务指标")

    return indicators


def save_indicators(indicators: Dict[str, pd.DataFrame], output_dir: str):
    """保存财务指标数据"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 保存每个指标
    for name, df in indicators.items():
        filepath = Path(output_dir) / f"{name}.parquet"
        df.to_parquet(filepath)

    print(f"\n财务指标已保存到: {output_dir}")


# =============================================================================
# 因子分析函数
# =============================================================================

def get_zz1000_stocks() -> List[str]:
    """获取中证1000成分股列表"""
    try:
        pro = init_tushare()
        df = pro.index_weight(index_code='000852.CSI')
        if not df.empty:
            latest_date = df['trade_date'].max()
            df_latest = df[df['trade_date'] == latest_date]
            return df_latest['con_code'].tolist()[:1000]
    except Exception as e:
        print(f"获取成分股失败: {e}")
    return []


def load_returns(stocks: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """加载收益率数据"""
    close_df = get_local_data(stocks, start_date, end_date, 'close', 'daily_basic')
    if close_df.empty:
        return pd.DataFrame()
    returns_df = close_df.pct_change().iloc[1:]
    return returns_df


def compute_ic(factor_df: pd.DataFrame, returns_df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    """计算 IC"""
    common_dates = factor_df.index.intersection(returns_df.index)
    if len(common_dates) < 10:
        return np.array({})

    factor_aligned = factor_df.loc[common_dates]
    returns_aligned = returns_df.loc[common_dates]

    common_stocks = factor_aligned.columns.intersection(returns_aligned.columns)
    if len(common_stocks) < 10:
        return np.array({})

    factor_aligned = factor_aligned[common_stocks]
    returns_aligned = returns_aligned[common_stocks]

    ic_series = []
    for date in common_dates:
        factor_vals = factor_aligned.loc[date].values
        return_vals = returns_aligned.loc[date].values

        mask = ~(np.isnan(factor_vals) | np.isnan(return_vals))
        if mask.sum() > 10:
            ic, _ = stats.spearmanr(factor_vals[mask], return_vals[mask])
            if not np.isnan(ic):
                ic_series.append(ic)

    ic_series = np.array(ic_series)
    if len(ic_series) == 0:
        return ic_series, {}

    ic_stats = {
        'ic_mean': float(np.mean(ic_series)),
        'ic_std': float(np.std(ic_series)),
        'ic_ir': float(np.mean(ic_series) / np.std(ic_series)) if np.std(ic_series) > 0 else 0,
        'ic_positive_ratio': float((ic_series > 0).mean()),
        'ic_t_stat': float(np.mean(ic_series) / (np.std(ic_series) / np.sqrt(len(ic_series)))) if np.std(ic_series) > 0 else 0,
        'ic_count': len(ic_series)
    }

    return ic_series, ic_stats


def compute_quantile_returns(factor_df: pd.DataFrame, returns_df: pd.DataFrame, quantiles: int = 5) -> Dict:
    """计算分层收益（使用rank分组，避免极端值问题）"""
    common_dates = factor_df.index.intersection(returns_df.index)
    if len(common_dates) < 10:
        return {}

    factor_aligned = factor_df.loc[common_dates]
    returns_aligned = returns_df.loc[common_dates]

    common_stocks = factor_aligned.columns.intersection(returns_aligned.columns)
    if len(common_stocks) < 10:
        return {}

    factor_aligned = factor_aligned[common_stocks]
    returns_aligned = returns_aligned[common_stocks]

    # 转换为长格式
    factor_long = factor_aligned.stack().reset_index()
    factor_long.columns = ['date', 'stock', 'factor']
    returns_long = returns_aligned.stack().reset_index()
    returns_long.columns = ['date', 'stock', 'return']

    merged = factor_long.merge(returns_long, on=['date', 'stock']).dropna()
    if len(merged) < 100:
        return {}

    # 使用rank分组（百分比rank），避免qcut的边界问题
    merged['quantile'] = merged.groupby('date')['factor'].transform(
        lambda x: pd.qcut(x.rank(method='first'), quantiles, labels=False, duplicates='drop') + 1
    )

    results = {}
    for q in range(1, quantiles + 1):
        q_data = merged[merged['quantile'] == q]
        if len(q_data) > 0:
            results[f'Q{q}'] = {
                'mean_return': float(q_data['return'].mean()),
                'std_return': float(q_data['return'].std()),
                'count': int(len(q_data))
            }

    if 'Q1' in results and f'Q{quantiles}' in results:
        top = results[f'Q{quantiles}']['mean_return']
        bottom = results['Q1']['mean_return']
        ls_return = top - bottom
        results['Long_Short'] = {
            'mean_return': float(ls_return),
        }

    return results


def analyze_indicator(indicator_name: str, indicator_df: pd.DataFrame,
                      stocks: List[str], start_date: str, end_date: str,
                      limit_up_threshold: float = 9.5) -> Dict:
    """分析单个财务指标"""
    print(f"分析指标: {indicator_name}")

    if indicator_df.empty:
        return None

    # 加载收益率
    returns_df = load_returns(stocks, start_date, end_date)
    if returns_df.empty:
        return None

    # 对齐数据
    common_dates = indicator_df.index.intersection(returns_df.index)
    if len(common_dates) < 10:
        return None

    factor_df = indicator_df.loc[common_dates]
    returns_df = returns_df.loc[common_dates]

    # 找到共同的股票
    common_stocks = factor_df.columns.intersection(returns_df.columns)
    factor_df = factor_df[common_stocks]
    returns_df = returns_df[common_stocks]

    # 过滤涨停和成交不足
    try:
        pct_chg_df = get_local_data(stocks, start_date, end_date, 'pct_chg', 'daily')
        amount_df = get_local_data(stocks, start_date, end_date, 'amount', 'daily')

        pct_chg_df = pct_chg_df.loc[common_dates]
        amount_df = amount_df.loc[common_dates]

        common_stocks2 = pct_chg_df.columns.intersection(amount_df.columns)
        factor_df = factor_df[common_stocks2]
        returns_df = returns_df[common_stocks2]

        limit_up_mask = pct_chg_df >= limit_up_threshold
        low_amount_mask = amount_df < 100
        filter_mask = limit_up_mask | low_amount_mask

        factor_df_filtered = factor_df.copy()
        factor_df_filtered[filter_mask] = np.nan

        returns_df_filtered = returns_df.copy()
        returns_df_filtered[filter_mask] = np.nan

        total = factor_df.size
        filtered = filter_mask.sum().sum()
        print(f"  过滤: {filtered}/{total} ({filtered/total*100:.1f}%) - 涨停/成交不足")

    except Exception as e:
        print(f"  警告: 过滤失败 ({e})")
        factor_df_filtered = factor_df
        returns_df_filtered = returns_df

    # 因子统计
    flat_values = factor_df_filtered.values.flatten()
    flat_values = flat_values[~np.isnan(flat_values)]

    factor_stats = {
        'count': len(flat_values),
        'mean': float(np.mean(flat_values)),
        'std': float(np.std(flat_values)),
        'min': float(np.min(flat_values)),
        'max': float(np.max(flat_values)),
        'skew': float(pd.Series(flat_values).skew()),
        'kurtosis': float(pd.Series(flat_values).kurtosis())
    }

    # 计算 IC
    ic_series, ic_stats = compute_ic(factor_df_filtered, returns_df_filtered)

    # 计算分层收益
    quantile_returns = compute_quantile_returns(factor_df_filtered, returns_df_filtered)

    result = {
        'indicator': indicator_name,
        'start_date': start_date,
        'end_date': end_date,
        'stock_count': len(factor_df.columns),
        'date_count': len(factor_df),
        'factor_stats': factor_stats,
        'ic_stats': ic_stats,
        'quantile_returns': quantile_returns,
        'ic_series': ic_series.tolist() if len(ic_series) > 0 else []
    }

    ic_mean = ic_stats.get('ic_mean', 'N/A')
    print(f"  完成: IC均值={ic_mean:.4f}" if isinstance(ic_mean, float) else f"  完成: IC均值={ic_mean}")

    return result


def generate_summary_report(results: List[Dict], output_path: str):
    """生成汇总报告"""
    lines = []

    lines.append("# 财务指标因子分析报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        lines.append("没有有效的分析结果")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return

    sorted_results = sorted(
        valid_results,
        key=lambda x: x.get('ic_stats', {}).get('ic_ir', 0),
        reverse=True
    )

    # 统计
    total = len(sorted_results)
    good = sum(1 for r in sorted_results if r.get('ic_stats', {}).get('ic_ir', 0) > 0.1)
    neutral = sum(1 for r in sorted_results if 0 <= r.get('ic_stats', {}).get('ic_ir', 0) <= 0.1)
    bad = sum(1 for r in sorted_results if r.get('ic_stats', {}).get('ic_ir', 0) < 0)

    lines.append("## 整体统计")
    lines.append(f"- 分析指标总数: {total}")
    lines.append(f"- 有效指标 (IC IR > 0.1): {good} ({good/total:.1%})")
    lines.append(f"- 中性指标 (0 ≤ IC IR ≤ 0.1): {neutral} ({neutral/total:.1%})")
    lines.append(f"- 无效指标 (IC IR < 0): {bad} ({bad/total:.1%})")
    lines.append("")

    # TOP 10
    lines.append("## TOP 10 最佳指标")
    lines.append("")
    for i, r in enumerate(sorted_results[:10], 1):
        ic_ir = r.get('ic_stats', {}).get('ic_ir', 0)
        ic_mean = r.get('ic_stats', {}).get('ic_mean', 0)
        lines.append(f"{i}. **{r['indicator']}**")
        lines.append(f"   IC IR: {ic_ir:.4f}, IC均值: {ic_mean:.4f}")
    lines.append("")

    # 最差
    lines.append("## TOP 10 最差指标")
    lines.append("")
    for i, r in enumerate(sorted_results[-10:][::-1], 1):
        ic_ir = r.get('ic_stats', {}).get('ic_ir', 0)
        ic_mean = r.get('ic_stats', {}).get('ic_mean', 0)
        lines.append(f"{i}. **{r['indicator']}**")
        lines.append(f"   IC IR: {ic_ir:.4f}, IC均值: {ic_mean:.4f}")
    lines.append("")

    # 结论
    lines.append("## 结论与建议")
    lines.append("")
    lines.append("**推荐指标**:")
    for r in sorted_results[:5]:
        lines.append(f"- {r['indicator']}")
    lines.append("")
    lines.append("**不推荐指标**:")
    for r in sorted_results[-5:]:
        lines.append(f"- {r['indicator']}")
    lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n汇总报告: {output_path}")


def generate_indicator_report(result: Dict, output_path: str):
    """生成单个指标详细报告"""
    lines = []

    lines.append(f"## {result['indicator']}")
    lines.append("")
    lines.append(f"**数据范围**: {result['start_date']} ~ {result['end_date']}")
    lines.append(f"**股票数**: {result['stock_count']}")
    lines.append(f"**交易日数**: {result['date_count']}")
    lines.append("")

    # 因子统计
    fs = result['factor_stats']
    lines.append("### 因子统计")
    lines.append(f"- 均值: {fs['mean']:.4f}")
    lines.append(f"- 标准差: {fs['std']:.4f}")
    lines.append(f"- 最小值: {fs['min']:.4f}")
    lines.append(f"- 最大值: {fs['max']:.4f}")
    lines.append("")

    # IC
    ic = result.get('ic_stats', {})
    if ic:
        lines.append("### IC 分析")
        lines.append(f"- IC 均值: {ic.get('ic_mean', 0):.4f}")
        lines.append(f"- IC 标准差: {ic.get('ic_std', 0):.4f}")
        lines.append(f"- IC IR: {ic.get('ic_ir', 0):.4f}")
        lines.append(f"- 正 IC 占比: {ic.get('ic_positive_ratio', 0):.2%}")
        lines.append(f"- IC T统计量: {ic.get('ic_t_stat', 0):.4f}")
        lines.append("")

    # 分层收益
    qr = result.get('quantile_returns', {})
    if qr:
        lines.append("### 分层收益")
        for key, value in qr.items():
            if isinstance(value, dict) and 'mean_return' in value:
                lines.append(f"- **{key}**: 均值={value['mean_return']:.4%}")
        lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='财务指标因子分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--start', '-s', default='20250101', help='开始日期')
    parser.add_argument('--end', '-e', default='20251231', help='结束日期')
    parser.add_argument('--stocks', '-n', type=int, default=1000, help='股票数量')
    parser.add_argument('--output', '-o', default='./factor_analysis_results/financial/', help='输出目录')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("财务指标因子分析")
    print("=" * 60)
    print(f"时间范围: {args.start} ~ {args.end}")
    print(f"分析股票数: {args.stocks}")

    # 创建输出目录
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

    # 获取股票列表
    print("\n获取中证1000成分股...")
    all_stocks = get_zz1000_stocks()
    if not all_stocks:
        print("无法获取成分股，使用默认样本")
        all_stocks = []
    stocks = all_stocks[:args.stocks]
    print(f"使用 {len(stocks)} 只股票")

    # 加载财务数据
    data = load_financial_data(stocks, args.start, args.end)

    # 计算财务指标
    indicators = compute_financial_indicators(data, stocks)

    # 保存指标数据
    save_indicators(indicators, args.output)

    # 分析每个指标
    print("\n" + "=" * 60)
    print("开始因子分析")
    print("=" * 60)

    results = []
    for indicator_name, indicator_df in indicators.items():
        result = analyze_indicator(indicator_name, indicator_df, stocks, args.start, args.end)
        if result:
            results.append(result)

    # 保存结果
    results_file = os.path.join(args.output, 'indicator_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n分析结果: {results_file}")

    # 生成汇总报告
    summary_path = os.path.join(args.output, 'summary_report.md')
    generate_summary_report(results, summary_path)

    # 生成详细报告
    for result in results:
        report_path = os.path.join(REPORT_DIR, f"{result['indicator']}.md")
        generate_indicator_report(result, report_path)

    print(f"详细报告: {REPORT_DIR}")

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
