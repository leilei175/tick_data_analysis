"""
中证1000因子分析报告生成器
=========================

功能：
- 使用 get_local_data 获取行情数据计算收益率
- 使用 daily_data 中的各类数据作为因子来源
- 对中证1000股票进行因子分析
- 生成因子分析报告

使用说明：
---------

# 分析 2024 年至今的所有因子
python zz1000_factor_analysis.py --start 20240101 --end 20251231

# 只分析部分因子
python zz1000_factor_analysis.py --factors close turnover_rate pe pb

# 只生成汇总报告
python zz1000_factor_analysis.py --summary-only

# 查看帮助
python zz1000_factor_analysis.py --help
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

from mylib.get_local_data import get_local_data, list_data_files

# =============================================================================
# 配置
# =============================================================================

# 可用作因子的数据类型
FACTOR_TYPES = {
    # 日线数据
    'daily': {
        'data_type': 'daily',
        'fields': ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount'],
        'name': '日线行情'
    },
    # 每日基本面
    'daily_basic': {
        'data_type': 'daily_basic',
        'fields': ['close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio',
                   'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm',
                   'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share',
                   'total_mv', 'circ_mv'],
        'name': '每日基本面'
    },
    # 每日现金流
    'cashflow_daily': {
        'data_type': 'cashflow_daily',
        'fields': ['n_cashflow_act', 'n_cashflow_inv_act', 'n_cash_flows_fnc_act'],
        'name': '每日现金流'
    },
    # 每日利润
    'income_daily': {
        'data_type': 'income_daily',
        'fields': ['total_revenue', 'revenue', 'operate_profit', 'total_profit',
                   'income_tax', 'n_income', 'basic_eps'],
        'name': '每日利润'
    },
    # 每日资产负债表
    'balance_daily': {
        'data_type': 'balance_daily',
        'fields': ['total_assets', 'total_liab', 'total_hldr_eqy_exc_min_int',
                   'total_cur_assets', 'cash_reser_cb', 'accounts_receiv', 'inventories'],
        'name': '每日资产负债表'
    }
}

# 中证1000成分股（部分样本）
ZZ1000_SAMPLE = [
    '000001.SZ', '000002.SZ', '000004.SZ', '000005.SZ', '000006.SZ',
    '000007.SZ', '000008.SZ', '000009.SZ', '000010.SZ', '000011.SZ',
    # 更多股票在实际分析时会自动获取
]

# 输出目录
OUTPUT_DIR = './factor_analysis_results/'
REPORT_DIR = './factor_analysis_results/reports/'

# =============================================================================
# 因子分析核心函数
# =============================================================================

def get_zz1000_stocks() -> List[str]:
    """获取中证1000成分股列表（从Tushare获取最新成分股）"""
    try:
        # 尝试从Tushare获取中证1000成分股
        from update_data import init_tushare
        pro = init_tushare()

        # 获取中证1000指数成分股（最新权重）
        df = pro.index_weight(index_code='000852.CSI')
        if not df.empty:
            # 获取最新的成分股列表
            latest_date = df['trade_date'].max()
            df_latest = df[df['trade_date'] == latest_date]
            stocks = df_latest['con_code'].tolist()
            print(f"从Tushare获取到 {len(stocks)} 只中证1000成分股 (更新日期: {latest_date})")
            return stocks
    except Exception as e:
        print(f"从Tushare获取成分股失败: {e}")

    # 备选：从 daily 数据中获取所有股票作为候选
    try:
        df = get_local_data(
            sec_list=None,
            start='20250201',
            end='20250210',
            filed='close',
            data_type='daily'
        )
        stocks = df.columns.tolist()
        print(f"从本地数据获取到 {len(stocks)} 只股票")
        return stocks[:1000]
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return []


def load_factor_data(
    factor: str,
    data_type: str,
    start_date: str,
    end_date: str,
    stocks: List[str] = None
) -> pd.DataFrame:
    """
    加载因子数据

    Args:
        factor: 因子字段名
        data_type: 数据类型
        start_date: 开始日期
        end_date: 结束日期
        stocks: 股票列表

    Returns:
        宽格式 DataFrame: index=date, columns=stock_code
    """
    try:
        df = get_local_data(
            sec_list=stocks,
            start=start_date,
            end=end_date,
            filed=factor,
            data_type=data_type
        )
        return df
    except Exception as e:
        print(f"  加载 {data_type}.{factor} 失败: {e}")
        return pd.DataFrame()


def load_returns(
    stocks: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    计算收益率数据

    Returns:
        宽格式 DataFrame: index=date, columns=stock_code
    """
    # 获取收盘价
    close_df = get_local_data(
        sec_list=stocks,
        start=start_date,
        end=end_date,
        filed='close',
        data_type='daily'
    )

    if close_df.empty:
        return pd.DataFrame()

    # 计算收益率
    returns_df = close_df.pct_change()
    returns_df = returns_df.iloc[1:]  # 去掉第一行 NaN

    return returns_df


def compute_ic(
    factor_df: pd.DataFrame,
    returns_df: pd.DataFrame
) -> Tuple[np.ndarray, Dict]:
    """
    计算 IC (Information Coefficient)

    Args:
        factor_df: 因子值 DataFrame
        returns_df: 收益率 DataFrame

    Returns:
        IC 序列, IC 统计信息
    """
    # 对齐日期
    common_dates = factor_df.index.intersection(returns_df.index)
    if len(common_dates) < 10:
        return np.array([]), {}

    factor_aligned = factor_df.loc[common_dates]
    returns_aligned = returns_df.loc[common_dates]

    # 找到共同的股票
    common_stocks = factor_aligned.columns.intersection(returns_aligned.columns)
    if len(common_stocks) < 10:
        return np.array([]), {}

    factor_aligned = factor_aligned[common_stocks]
    returns_aligned = returns_aligned[common_stocks]

    # 计算每日 IC
    ic_series = []
    for date in common_dates:
        factor_vals = factor_aligned.loc[date].values
        return_vals = returns_aligned.loc[date].values

        # 过滤 NaN
        mask = ~(np.isnan(factor_vals) | np.isnan(return_vals))
        if mask.sum() > 10:
            ic, _ = stats.spearmanr(factor_vals[mask], return_vals[mask])
            if not np.isnan(ic):
                ic_series.append(ic)

    ic_series = np.array(ic_series)

    if len(ic_series) == 0:
        return ic_series, {}

    # IC 统计
    ic_stats = {
        'ic_mean': float(np.mean(ic_series)),
        'ic_std': float(np.std(ic_series)),
        'ic_ir': float(np.mean(ic_series) / np.std(ic_series)) if np.std(ic_series) > 0 else 0,
        'ic_positive_ratio': float((ic_series > 0).mean()),
        'ic_t_stat': float(np.mean(ic_series) / (np.std(ic_series) / np.sqrt(len(ic_series)))) if np.std(ic_series) > 0 else 0,
        'ic_count': len(ic_series)
    }

    return ic_series, ic_stats


def compute_quantile_returns(
    factor_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    quantiles: int = 5
) -> Dict:
    """
    计算分层组合收益

    Args:
        factor_df: 因子值 DataFrame
        returns_df: 收益率 DataFrame
        quantiles: 分位数数量

    Returns:
        各层收益统计
    """
    # 对齐日期
    common_dates = factor_df.index.intersection(returns_df.index)
    if len(common_dates) < 10:
        return {}

    factor_aligned = factor_df.loc[common_dates]
    returns_aligned = returns_df.loc[common_dates]

    # 找到共同的股票
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

    # 合并
    merged = factor_long.merge(returns_long, on=['date', 'stock']).dropna()
    if len(merged) < 100:
        return {}

    # 计算每日分位数
    merged['quantile'] = merged.groupby('date')['factor'].transform(
        lambda x: pd.qcut(x, quantiles, labels=False, duplicates='drop') + 1
    )

    # 按分位数统计
    results = {}
    for q in range(1, quantiles + 1):
        q_data = merged[merged['quantile'] == q]
        if len(q_data) > 0:
            results[f'Q{q}'] = {
                'mean_return': float(q_data['return'].mean()),
                'std_return': float(q_data['return'].std()),
                'count': int(len(q_data))
            }

    # 多空组合
    if 'Q1' in results and f'Q{quantiles}' in results:
        top = results[f'Q{quantiles}']['mean_return']
        bottom = results['Q1']['mean_return']
        ls_return = top - bottom
        results['Long_Short'] = {
            'mean_return': float(ls_return),
            'win_rate': float((top > bottom).mean()) if isinstance(top, pd.Series) else (top > bottom)
        }

    return results


def analyze_factor(
    factor: str,
    data_type: str,
    stocks: List[str],
    start_date: str,
    end_date: str,
    limit_up_threshold: float = 9.5,  # 涨停阈值（%）
    min_amount: float = 100  # 最小成交金额（万），即100万=100
) -> Dict:
    """
    分析单个因子

    Args:
        factor: 因子字段名
        data_type: 数据类型
        stocks: 股票列表
        start_date: 开始日期
        end_date: 结束日期
        limit_up_threshold: 涨停阈值（%），超过此值视为涨停
        min_amount: 最小成交金额，低于此值会被过滤

    Returns:
        因子分析结果
    """
    print(f"分析因子: {data_type}.{factor}")

    # 加载因子数据
    factor_df = load_factor_data(factor, data_type, start_date, end_date, stocks)
    if factor_df.empty:
        print(f"  因子数据为空，跳过")
        return None

    # 加载收益率数据
    returns_df = load_returns(stocks, start_date, end_date)
    if returns_df.empty:
        print(f"  收益率数据为空，跳过")
        return None

    # 对齐数据
    common_dates = factor_df.index.intersection(returns_df.index)
    if len(common_dates) < 10:
        print(f"  日期重叠不足，跳过")
        return None

    factor_df = factor_df.loc[common_dates]
    returns_df = returns_df.loc[common_dates]

    # 加载涨跌幅和成交金额数据，用于过滤
    try:
        pct_chg_df = get_local_data(
            sec_list=stocks,
            start=start_date,
            end=end_date,
            filed='pct_chg',
            data_type='daily'
        )
        pct_chg_df = pct_chg_df.loc[common_dates]

        amount_df = get_local_data(
            sec_list=stocks,
            start=start_date,
            end=end_date,
            filed='amount',
            data_type='daily'
        )
        amount_df = amount_df.loc[common_dates]

        # 找到共同的股票
        common_stocks = factor_df.columns.intersection(pct_chg_df.columns).intersection(amount_df.columns)
        factor_df = factor_df[common_stocks]
        pct_chg_df = pct_chg_df[common_stocks]
        amount_df = amount_df[common_stocks]

        # 识别涨停股票（涨跌幅 >= 阈值）
        limit_up_mask = pct_chg_df >= limit_up_threshold

        # 识别成交金额低于阈值的股票
        low_amount_mask = amount_df < min_amount

        # 合并过滤条件
        filter_mask = pct_chg_df >= limit_up_threshold
        low_amount_mask = amount_df < min_amount
        filter_mask = filter_mask | low_amount_mask

        # 将过滤股票的因子值和收益率都设为 NaN
        factor_df_filtered = factor_df.copy()
        factor_df_filtered[filter_mask] = np.nan

        returns_df_filtered = returns_df.copy()
        returns_df_filtered[filter_mask] = np.nan

        # 统计过滤情况
        total = factor_df.size
        filtered = filter_mask.sum().sum()
        print(f"  过滤: {filtered}/{total} ({filtered/total*100:.1f}%) - 涨停/成交不足")

    except Exception as e:
        print(f"  警告: 无法加载涨跌幅/成交金额数据 ({e})，不进行过滤")
        factor_df_filtered = factor_df
        returns_df_filtered = returns_df

    # 因子统计（使用过滤后的数据）
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

    # 计算 IC（使用过滤后的数据）
    ic_series, ic_stats = compute_ic(factor_df_filtered, returns_df_filtered)

    # 计算分层收益（使用过滤后的数据）
    quantile_returns = compute_quantile_returns(factor_df_filtered, returns_df_filtered)

    # 构建结果
    result = {
        'factor': factor,
        'data_type': data_type,
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


def list_available_factors() -> List[Dict]:
    """列出所有可用的因子"""
    factors = []

    for data_type, config in FACTOR_TYPES.items():
        for field in config['fields']:
            factors.append({
                'factor': field,
                'data_type': data_type,
                'name': f"{config['name']}.{field}",
                'category': config['name']
            })

    return factors


def generate_factor_report(result: Dict) -> str:
    """生成单个因子的分析报告"""
    lines = []

    lines.append(f"## {result['data_type']}.{result['factor']}")
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
    lines.append(f"- 偏度: {fs['skew']:.4f}")
    lines.append(f"- 峰度: {fs['kurtosis']:.4f}")
    lines.append("")

    # IC 统计
    ic = result.get('ic_stats', {})
    if ic:
        lines.append("### IC 分析")
        lines.append(f"- IC 均值: {ic.get('ic_mean', 0):.4f}")
        lines.append(f"- IC 标准差: {ic.get('ic_std', 0):.4f}")
        lines.append(f"- IC IR: {ic.get('ic_ir', 0):.4f}")
        lines.append(f"- 正 IC 占比: {ic.get('ic_positive_ratio', 0):.2%}")
        lines.append(f"- IC T统计量: {ic.get('ic_t_stat', 0):.4f}")
        lines.append(f"- 交易日数: {ic.get('ic_count', 0)}")
        lines.append("")

    # 分层收益
    qr = result.get('quantile_returns', {})
    if qr:
        lines.append("### 分层收益")
        for key, value in qr.items():
            if isinstance(value, dict) and 'mean_return' in value:
                lines.append(f"- **{key}**: 均值={value['mean_return']:.4%}, 标准差={value.get('std_return', 0):.4%}")
        lines.append("")

    # 评分
    score = 0
    if ic.get('ic_ir', 0) > 0.3:
        score = 5
    elif ic.get('ic_ir', 0) > 0.2:
        score = 4
    elif ic.get('ic_ir', 0) > 0.1:
        score = 3
    elif ic.get('ic_ir', 0) > 0:
        score = 2
    else:
        score = 1

    lines.append("### 综合评分")
    lines.append(f"**{score}** / 5")
    lines.append("")

    return "\n".join(lines)


def generate_summary_report(results: List[Dict]) -> str:
    """生成汇总报告"""
    lines = []

    lines.append("# 中证1000因子分析汇总报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 过滤有效结果
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        lines.append("没有有效的因子分析结果")
        return "\n".join(lines)

    # 按 IC IR 排序
    sorted_results = sorted(
        valid_results,
        key=lambda x: x.get('ic_stats', {}).get('ic_ir', 0),
        reverse=True
    )

    # 统计信息
    total = len(sorted_results)
    good = sum(1 for r in sorted_results if r.get('ic_stats', {}).get('ic_ir', 0) > 0.1)
    neutral = sum(1 for r in sorted_results if 0 <= r.get('ic_stats', {}).get('ic_ir', 0) <= 0.1)
    bad = sum(1 for r in sorted_results if r.get('ic_stats', {}).get('ic_ir', 0) < 0)

    lines.append("## 整体统计")
    lines.append(f"- 分析因子总数: {total}")
    lines.append(f"- 有效因子 (IC IR > 0.1): {good} ({good/total:.1%})")
    lines.append(f"- 中性因子 (0 ≤ IC IR ≤ 0.1): {neutral} ({neutral/total:.1%})")
    lines.append(f"- 无效因子 (IC IR < 0): {bad} ({bad/total:.1%})")
    lines.append("")

    # 最佳因子
    lines.append("## TOP 10 最佳因子")
    lines.append("")
    for i, r in enumerate(sorted_results[:10], 1):
        ic_ir = r.get('ic_stats', {}).get('ic_ir', 0)
        ic_mean = r.get('ic_stats', {}).get('ic_mean', 0)
        lines.append(f"{i}. **{r['data_type']}.{r['factor']}**")
        lines.append(f"   IC IR: {ic_ir:.4f}, IC均值: {ic_mean:.4f}")
    lines.append("")

    # 最差因子
    lines.append("## TOP 10 最差因子")
    lines.append("")
    for i, r in enumerate(sorted_results[-10:][::-1], 1):
        ic_ir = r.get('ic_stats', {}).get('ic_ir', 0)
        ic_mean = r.get('ic_stats', {}).get('ic_mean', 0)
        lines.append(f"{i}. **{r['data_type']}.{r['factor']}**")
        lines.append(f"   IC IR: {ic_ir:.4f}, IC均值: {ic_mean:.4f}")
    lines.append("")

    # 按类别统计
    lines.append("## 按类别统计")
    lines.append("")
    categories = {}
    for r in sorted_results:
        cat = r.get('data_type', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r.get('ic_stats', {}).get('ic_ir', 0))

    for cat, irs in sorted(categories.items(), key=lambda x: np.mean(x[1]), reverse=True):
        mean_ir = np.mean(irs)
        good_count = sum(1 for ir in irs if ir > 0.1)
        lines.append(f"- **{cat}**: 平均IC IR={mean_ir:.4f}, 有效因子={good_count}/{len(irs)}")
    lines.append("")

    # 结论
    lines.append("## 结论与建议")
    lines.append("")

    best_factors = [f"{r['data_type']}.{r['factor']}" for r in sorted_results[:5]]
    worst_factors = [f"{r['data_type']}.{r['factor']}" for r in sorted_results[-5:]]

    lines.append("**推荐因子**:")
    for f in best_factors:
        lines.append(f"- {f}")
    lines.append("")

    lines.append("**不推荐因子**:")
    for f in worst_factors:
        lines.append(f"- {f}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='中证1000因子分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--start', '-s',
        default='20240101',
        help='开始日期 (默认: 20240101)'
    )

    parser.add_argument(
        '--end', '-e',
        default='20251231',
        help='结束日期 (默认: 20251231)'
    )

    parser.add_argument(
        '--factors', '-f',
        nargs='+',
        help='指定分析的因子 (如: close turnover_rate pe)'
    )

    parser.add_argument(
        '--data-types', '-d',
        nargs='+',
        help='指定数据类型 (如: daily daily_basic)'
    )

    parser.add_argument(
        '--stocks', '-n',
        type=int,
        default=1000,
        help='分析使用的股票数量 (默认: 1000)'
    )

    parser.add_argument(
        '--output', '-o',
        default='./factor_analysis_results/',
        help='输出目录'
    )

    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='只生成汇总报告'
    )

    parser.add_argument(
        '--parallel', '-p',
        type=int,
        default=1,
        help='并行进程数 (默认: 1)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("中证1000因子分析")
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
        print("无法获取股票列表，使用默认样本")
        all_stocks = ZZ1000_SAMPLE

    stocks = all_stocks[:args.stocks]
    print(f"使用 {len(stocks)} 只股票进行分析")

    # 确定要分析的因子
    if args.factors:
        factors_to_analyze = []
        for f in args.factors:
            # 查找因子的数据类型
            for dt, config in FACTOR_TYPES.items():
                if f in config['fields']:
                    factors_to_analyze.append({'factor': f, 'data_type': dt})
    elif args.data_types:
        factors_to_analyze = []
        for dt in args.data_types:
            if dt in FACTOR_TYPES:
                for f in FACTOR_TYPES[dt]['fields']:
                    factors_to_analyze.append({'factor': f, 'data_type': dt})
    else:
        factors_to_analyze = []
        for dt, config in FACTOR_TYPES.items():
            for f in config['fields']:
                factors_to_analyze.append({'factor': f, 'data_type': dt})

    print(f"需要分析 {len(factors_to_analyze)} 个因子")

    if args.summary_only:
        # 加载现有结果
        results_file = os.path.join(args.output, 'analysis_results.json')
        if os.path.exists(results_file):
            with open(results_file) as f:
                results = json.load(f)
            print(f"加载了 {len(results)} 个已有结果")
        else:
            print("没有找到现有结果，请先运行完整分析")
            return
    else:
        # 并行分析
        results = []

        if args.parallel > 1:
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = []
                for item in factors_to_analyze:
                    future = executor.submit(
                        analyze_factor,
                        item['factor'],
                        item['data_type'],
                        stocks,
                        args.start,
                        args.end
                    )
                    futures.append(future)

                for future in futures:
                    result = future.result()
                    if result:
                        results.append(result)
        else:
            for item in factors_to_analyze:
                result = analyze_factor(
                    item['factor'],
                    item['data_type'],
                    stocks,
                    args.start,
                    args.end
                )
                if result:
                    results.append(result)

        # 保存结果
        results_file = os.path.join(args.output, 'analysis_results.json')
        with open(results_file, 'w') as f:
            # 转换 numpy 类型为 Python 类型
            json_results = []
            for r in results:
                json_r = {}
                for k, v in r.items():
                    if isinstance(v, (np.integer,)):
                        json_r[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        json_r[k] = float(v)
                    elif isinstance(v, (np.ndarray,)):
                        json_r[k] = v.tolist()
                    else:
                        json_r[k] = v
                json_results.append(json_r)
            json.dump(json_results, f, ensure_ascii=False, indent=2)

        print(f"\n分析完成! 保存 {len(results)} 个结果到 {results_file}")

    # 生成汇总报告
    print("\n生成汇总报告...")

    # 转换回 Dict
    results = []
    with open(results_file) as f:
        for r in json.load(f):
            results.append(r)

    # 生成汇总报告
    summary_report = generate_summary_report(results)
    summary_file = os.path.join(args.output, 'summary_report.md')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_report)

    print(f"汇总报告: {summary_file}")

    # 生成每个因子的详细报告
    print("生成因子详细报告...")
    for result in results:
        if result is None:
            continue
        report = generate_factor_report(result)
        filename = f"{result['data_type']}_{result['factor']}.md"
        report_file = os.path.join(REPORT_DIR, filename)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

    print(f"因子报告目录: {REPORT_DIR}")

    # 输出最佳因子
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)

    # 打印TOP5
    valid = [r for r in results if r is not None]
    sorted_results = sorted(valid, key=lambda x: x.get('ic_stats', {}).get('ic_ir', 0), reverse=True)

    print("\nTOP 5 最佳因子:")
    for i, r in enumerate(sorted_results[:5], 1):
        ic_ir = r.get('ic_stats', {}).get('ic_ir', 0)
        ic_mean = r.get('ic_stats', {}).get('ic_mean', 0)
        print(f"  {i}. {r['data_type']}.{r['factor']}: IC IR={ic_ir:.4f}, IC均值={ic_mean:.4f}")


if __name__ == '__main__':
    main()
