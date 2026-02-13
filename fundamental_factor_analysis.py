"""
中证1000基本面因子分析
使用利润表、资产负债表、现金流量表构建各类基本面指标
分析因子对未来收益的预测效果

因子类别:
1. 估值因子 (Valuation): PE, PB, PS, PCF, EV/EBITDA
2. 盈利能力因子 (Profitability): ROE, ROA, 毛利率, 净利率
3. 成长因子 (Growth): 营收增长率, 利润增长率, EPS增长率
4. 运营效率因子 (Efficiency): 资产周转率, 存货周转率, 应收账款周转率
5. 杠杆/偿债因子 (Leverage): 资产负债率, 流动比率, 速动比率
6. 现金流因子 (CashFlow): 经营现金流/净利润, 现金流覆盖率
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns

# 导入数据下载模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_financial_statements import get_income, get_balance, get_cashflow
from download_daily_basic import get_close
from download_daily_data import get_daily


class FundamentalFactorAnalyzer:
    """基本面因子分析器"""

    def __init__(self, output_dir: str = "./fundamental_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.income = None
        self.balance = None
        self.cashflow = None
        self.daily_basic = None
        self.daily = None
        self.merged_data = None
        self.factors = None
        self.returns = None

    # ==================== 数据加载 ====================

    def load_financial_data(self, start_period: str = '20240331', end_period: str = '20241231'):
        """加载财务报表数据"""
        print("\n" + "="*60)
        print("加载财务报表数据")
        print("="*60)

        print("加载利润表...")
        self.income = get_income(start_period=start_period, end_period=end_period)

        print("加载资产负债表...")
        self.balance = get_balance(start_period=start_period, end_period=end_period)

        print("加载现金流量表...")
        self.cashflow = get_cashflow(start_period=start_period, end_period=end_period)

        print(f"\n利润表: {len(self.income):,} 条记录")
        print(f"资产负债表: {len(self.balance):,} 条记录")
        print(f"现金流量表: {len(self.cashflow):,} 条记录")

        return self

    def load_market_data(self, start_day: str = '20250101', end_day: str = '20260206'):
        """加载行情数据用于计算收益率"""
        print("\n加载行情数据...")

        try:
            self.daily = get_daily(start_day=start_day, end_day=end_day)
            print(f"日线数据: {len(self.daily):,} 条记录")
        except Exception as e:
            print(f"日线数据加载失败: {e}")

        try:
            self.daily_basic = get_close(sec_list=None, start_day=start_day, end_day=end_day)
            print(f"收盘价数据: {len(self.daily_basic):,} 条记录")
        except Exception as e:
            print(f"收盘价数据加载失败: {e}")

        return self

    # ==================== 数据合并 ====================

    def merge_financial_data(self):
        """合并三张财务报表"""
        print("\n" + "="*60)
        print("合并财务报表")
        print("="*60)

        # 合并利润表和资产负债表
        self.merged_data = pd.merge(
            self.income,
            self.balance,
            on=['ts_code', 'end_date', 'report_type', 'comp_type'],
            how='outer',
            suffixes=('_inc', '_bal')
        )

        # 合并现金流量表
        self.merged_data = pd.merge(
            self.merged_data,
            self.cashflow,
            on=['ts_code', 'end_date', 'report_type', 'comp_type'],
            how='outer'
        )

        print(f"合并后记录数: {len(self.merged_data):,}")
        print(f"股票数量: {self.merged_data['ts_code'].nunique():,}")

        # 保留所有报告类型，去重
        self.merged_data = self.merged_data.sort_values('ann_date')
        self.merged_data = self.merged_data.drop_duplicates(
            subset=['ts_code', 'end_date'],
            keep='last'
        )

        print(f"去重后记录数: {len(self.merged_data):,}")

        return self

    # ==================== 因子构建 ====================

    def build_factors(self):
        """构建基本面因子"""
        print("\n" + "="*60)
        print("构建基本面因子")
        print("="*60)

        df = self.merged_data.copy()

        # ========== 1. 估值因子 ==========
        print("计算估值因子...")

        # 使用最近收盘价计算估值
        if self.daily_basic is not None:
            latest_price = self.daily_basic.groupby('ts_code')['close'].last().reset_index()
            latest_price.columns = ['ts_code', 'close_price']

            df = pd.merge(df, latest_price, on='ts_code', how='left')

            # 市盈率 (PE)
            df['pe'] = df['close_price'] / (df['basic_eps'] + 1e-10)

            # 市净率 (PB)
            df['pb'] = df['close_price'] / (df['total_hldr_eqy_exc_min_int'] / 1e8 + 1e-10)

            # 市销率 (PS) - 简化: 使用市值/营收
            df['ps'] = (df['close_price'] * 1e8) / (df['revenue'] + 1e-10)

        # ========== 2. 盈利能力因子 ==========
        print("计算盈利能力因子...")

        # 净利润率
        df['net_profit_margin'] = df['n_income'] / (df['revenue'] + 1e-10)

        # 毛利率 (无法直接计算，跳过)
        # df['gross_profit_margin'] = (df['revenue'] - df['cost_of_sales']) / df['revenue']

        # 营业利润率
        df['operating_margin'] = df['operate_profit'] / (df['revenue'] + 1e-10)

        # ROE (净资产收益率)
        df['roe'] = df['n_income'] / (df['total_hldr_eqy_exc_min_int'] + 1e-10)

        # ROA (总资产收益率)
        df['roa'] = df['n_income'] / (df['total_assets'] + 1e-10)

        # 基本每股收益
        df['basic_eps_factor'] = df['basic_eps']

        # ========== 3. 成长因子 ==========
        print("计算成长因子...")

        # 按股票分组计算同比增长率
        df = df.sort_values(['ts_code', 'end_date'])
        df['revenue_growth'] = df.groupby('ts_code')['revenue'].pct_change(periods=4)
        df['profit_growth'] = df.groupby('ts_code')['n_income'].pct_change(periods=4)
        df['eps_growth'] = df.groupby('ts_code')['basic_eps'].pct_change(periods=4)

        # ========== 4. 运营效率因子 ==========
        print("计算运营效率因子...")

        # 资产周转率
        df['asset_turnover'] = df['revenue'] / (df['total_assets'] + 1e-10)

        # 应收账款周转率
        df['ar_turnover'] = df['revenue'] / (df['accounts_receiv'] + 1e-10)

        # 存货周转率
        df['inventory_turnover'] = df['revenue'] / (df['inventories'] + 1e-10)

        # ========== 5. 杠杆/偿债因子 ==========
        print("计算杠杆/偿债因子...")

        # 资产负债率
        df['debt_ratio'] = df['total_liab'] / (df['total_assets'] + 1e-10)

        # 流动比率
        df['current_ratio'] = df['total_cur_assets'] / (df['total_cur_liab'] + 1e-10)

        # 速动比率
        df['quick_ratio'] = (df['total_cur_assets'] - df['inventories']) / (df['total_cur_liab'] + 1e-10)

        # 长期负债占比
        df['lt_debt_ratio'] = (df['total_liab'] - df['total_cur_liab']) / (df['total_liab'] + 1e-10)

        # ========== 6. 现金流因子 ==========
        print("计算现金流因子...")

        # 经营现金流/净利润
        df['ocf_to_netincome'] = df['n_cashflow_act'] / (df['n_income'] + 1e-10)

        # 现金流收益率
        df['ocf_yield'] = df['n_cashflow_act'] / (df['total_assets'] + 1e-10)

        # 投资现金流占比
        df['inv_cash_ratio'] = df['n_cashflow_inv_act'] / (df['n_cashflow_act'] + 1e-10)

        # ========== 清理和选择因子 ==========

        # 定义因子列表
        self.factor_cols = [
            # 估值因子
            'pe', 'pb', 'ps',
            # 盈利能力
            'net_profit_margin', 'operating_margin',
            'roe', 'roa', 'basic_eps_factor',
            # 成长因子
            'revenue_growth', 'profit_growth', 'eps_growth',
            # 运营效率
            'asset_turnover', 'ar_turnover', 'inventory_turnover',
            # 杠杆/偿债
            'debt_ratio', 'current_ratio', 'quick_ratio', 'lt_debt_ratio',
            # 现金流
            'ocf_to_netincome', 'ocf_yield', 'inv_cash_ratio'
        ]

        # 过滤存在的因子
        self.factor_cols = [c for c in self.factor_cols if c in df.columns]

        # 清理异常值
        for col in self.factor_cols:
            if col in df.columns:
                # 替换无穷值
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                # 限制在合理范围
                q1, q99 = df[col].quantile([0.01, 0.99])
                df[col] = df[col].clip(q1, q99)

        self.factors = df[['ts_code', 'end_date'] + self.factor_cols].copy()

        print(f"\n共构建 {len(self.factor_cols)} 个基本面因子")
        for f in self.factor_cols:
            print(f"  - {f}: 均值={self.factors[f].mean():.4f}, 中位数={self.factors[f].median():.4f}")

        return self

    # ==================== 收益率计算 ====================

    def calculate_returns(self, holding_period: int = 20):
        """
        计算未来收益率

        Args:
            holding_period: 持有天数
        """
        print("\n" + "="*60)
        print(f"计算未来收益率 (持有{holding_period}天)")
        print("="*60)

        if self.daily is None:
            print("警告: 无日线数据，无法计算收益率")
            return self

        # 计算日收益率
        self.daily = self.daily.sort_values(['ts_code', 'trade_date'])
        self.daily['return_daily'] = self.daily.groupby('ts_code')['close'].pct_change()

        # 计算未来收益率
        self.daily['future_price'] = self.daily.groupby('ts_code')['close'].shift(-holding_period)
        self.daily['future_return'] = (self.daily['future_price'] / self.daily['close']) - 1
        self.daily = self.daily.drop(columns=['future_price'])

        # 聚合到季度级别
        daily_with_date = self.daily.copy()
        daily_with_date['quarter'] = daily_with_date['trade_date'].astype(str).str[:4] + 'Q' + \
                                     ((daily_with_date['trade_date'].astype(str).str[4:6].astype(int) - 1) // 3 + 1).astype(str)

        # 计算季度收益率
        quarterly_returns = daily_with_date.groupby(['ts_code', 'quarter']).agg({
            'return_daily': lambda x: (1 + x).prod() - 1,
            'close': 'last'
        }).reset_index()
        quarterly_returns.columns = ['ts_code', 'quarter', 'quarterly_return', 'close']

        self.returns = quarterly_returns

        print(f"季度收益数据: {len(self.returns):,} 条记录")
        print(f"收益率统计: 均值={self.returns['quarterly_return'].mean()*100:.2f}%, "
              f"标准差={self.returns['quarterly_return'].std()*100:.2f}%")

        return self

    # ==================== IC分析 ====================

    def calculate_ic(self, factor_col: str = None, return_col: str = 'quarterly_return'):
        """
        计算信息系数 (Information Coefficient)

        IC = correlation(factor, return)
        """
        print("\n" + "="*60)
        print("因子IC分析")
        print("="*60)

        if self.factors is None or self.returns is None:
            print("错误: 需要先计算因子和收益率")
            return None

        # 合并因子和收益
        merged = pd.merge(
            self.factors,
            self.returns,
            on='ts_code',
            how='inner'
        )

        print(f"合并数据: {len(merged):,} 条记录")

        # 计算每个因子的IC
        ic_results = {}

        factors_to_analyze = [factor_col] if factor_col else self.factor_cols

        for factor in factors_to_analyze:
            if factor not in merged.columns:
                continue

            # 去除NaN
            valid_data = merged[['ts_code', factor, return_col]].dropna()

            if len(valid_data) < 30:
                print(f"  {factor}: 数据不足")
                continue

            # 计算IC
            ic = valid_data[factor].corr(valid_data[return_col])

            # 计算IC的t统计量
            n = len(valid_data)
            t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

            # 计算Rank IC (Spearman相关系数)
            spearman_ic, _ = stats.spearmanr(valid_data[factor], valid_data[return_col])

            ic_results[factor] = {
                'IC': ic,
                'IC_tstat': t_stat,
                'IC_pvalue': p_value,
                'RankIC': spearman_ic,
                'IC_abs': abs(ic),
                'n_samples': len(valid_data)
            }

        # 创建IC结果DataFrame
        ic_df = pd.DataFrame(ic_results).T
        ic_df = ic_df.sort_values('IC_abs', ascending=False)

        print("\n因子IC分析结果 (按|IC|排序):")
        print("-" * 80)
        print(f"{'因子':<20} {'IC':>10} {'RankIC':>10} {'|IC|':>10} {'P值':>12} {'样本数':>10}")
        print("-" * 80)

        for idx, row in ic_df.iterrows():
            sig = "***" if row['IC_pvalue'] < 0.001 else "**" if row['IC_pvalue'] < 0.01 else "*" if row['IC_pvalue'] < 0.05 else ""
            print(f"{idx:<20} {row['IC']:>10.4f} {row['RankIC']:>10.4f} {row['IC_abs']:>10.4f} {row['IC_pvalue']:>10.4f} {int(row['n_samples']):>10} {sig}")

        self.ic_results = ic_df

        return ic_df

    # ==================== 分层回测分析 ====================

    def factor_ranking_analysis(self, factor_col: str, return_col: str = 'quarterly_return',
                                n_groups: int = 5):
        """
        因子分层回测分析

        将股票按因子值分为n组，计算各组收益
        """
        # 合并因子和收益
        merged = pd.merge(
            self.factors,
            self.returns,
            on='ts_code',
            how='inner'
        )

        # 去除NaN
        valid_data = merged[['ts_code', 'ts_code', factor_col, return_col, 'quarter']].dropna()

        if len(valid_data) < 30:
            return None

        # 按因子值分组
        try:
            valid_data['group'] = pd.qcut(valid_data[factor_col], q=n_groups, labels=False, duplicates='drop')
        except Exception as e:
            print(f"  分组失败 ({factor_col}): {e}")
            return None

        # 计算各组平均收益
        group_returns = valid_data.groupby('group').agg({
            return_col: ['mean', 'std', 'count']
        }).reset_index()
        group_returns.columns = ['group', 'mean_return', 'std_return', 'count']

        # 计算多空组合收益
        long_short_return = group_returns.iloc[-1]['mean_return'] - group_returns.iloc[0]['mean_return']

        # 计算因子方向性
        factor_direction = '正向' if group_returns['mean_return'].iloc[-1] > group_returns['mean_return'].iloc[0] else '负向'

        return {
            'factor': factor_col,
            'group_returns': group_returns,
            'long_short_return': long_short_return,
            'factor_direction': factor_direction,
            'ic': self.ic_results.loc[factor_col, 'IC'] if factor_col in self.ic_results.index else 0
        }

    def run_all_ranking_analysis(self):
        """运行所有因子的分层分析"""
        print("\n" + "="*60)
        print("因子分层回测分析")
        print("="*60)

        ranking_results = []

        for factor in self.factor_cols:
            result = self.factor_ranking_analysis(factor)
            if result:
                ranking_results.append(result)

        # 创建结果表格
        ranking_df = pd.DataFrame([{
            '因子': r['factor'],
            'IC': r['ic'],
            '多空收益': r['long_short_return'] * 100,
            '方向': r['factor_direction'],
            'Q1(低)收益': r['group_returns']['mean_return'].iloc[0] * 100,
            'Q5(高)收益': r['group_returns']['mean_return'].iloc[-1] * 100
        } for r in ranking_results])

        ranking_df = ranking_df.sort_values('多空收益', ascending=False)

        print("\n分层回测结果 (按多空收益排序):")
        print("-" * 90)
        print(f"{'因子':<20} {'IC':>8} {'多空收益':>12} {'方向':>8} {'Q1收益':>10} {'Q5收益':>10}")
        print("-" * 90)

        for _, row in ranking_df.iterrows():
            print(f"{row['因子']:<20} {row['IC']:>8.4f} {row['多空收益']:>10.2f}% {row['方向']:>8} {row['Q1(低)收益']:>8.2f}% {row['Q5(高)收益']:>8.2f}%")

        self.ranking_results = ranking_df

        return ranking_df

    # ==================== 可视化 ====================

    def visualize_ic(self):
        """可视化IC分析结果"""
        if not hasattr(self, 'ic_results') or self.ic_results is None:
            print("请先运行IC分析")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. IC柱状图
        ax1 = axes[0, 0]
        ic_sorted = self.ic_results.sort_values('IC')
        colors = ['green' if x > 0 else 'red' for x in ic_sorted['IC']]
        ax1.barh(range(len(ic_sorted)), ic_sorted['IC'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(ic_sorted)))
        ax1.set_yticklabels(ic_sorted.index, fontsize=8)
        ax1.set_xlabel('IC')
        ax1.set_title('Factor Information Coefficient (IC)')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.axvline(x=0.03, color='green', linestyle='--', alpha=0.5)
        ax1.axvline(x=-0.03, color='red', linestyle='--', alpha=0.5)

        # 2. Rank IC柱状图
        ax2 = axes[0, 1]
        ic_sorted = self.ic_results.sort_values('RankIC')
        colors = ['green' if x > 0 else 'red' for x in ic_sorted['RankIC']]
        ax2.barh(range(len(ic_sorted)), ic_sorted['RankIC'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(ic_sorted)))
        ax2.set_yticklabels(ic_sorted.index, fontsize=8)
        ax2.set_xlabel('Rank IC')
        ax2.set_title('Factor Rank IC (Spearman)')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # 3. IC绝对值排名
        ax3 = axes[1, 0]
        ic_abs = self.ic_results.sort_values('IC_abs', ascending=True)
        ax3.barh(range(len(ic_abs)), ic_abs['IC_abs'], color='steelblue', alpha=0.7)
        ax3.set_yticks(range(len(ic_abs)))
        ax3.set_yticklabels(ic_abs.index, fontsize=8)
        ax3.set_xlabel('|IC|')
        ax3.set_title('Factor |IC| Ranking (Predictive Power)')
        ax3.axvline(x=0.03, color='red', linestyle='--', label='Weak (0.03)')
        ax3.axvline(x=0.05, color='orange', linestyle='--', label='Medium (0.05)')
        ax3.axvline(x=0.1, color='green', linestyle='--', label='Strong (0.1)')
        ax3.legend(fontsize=8)

        # 4. P值分布
        ax4 = axes[1, 1]
        valid_pvalues = self.ic_results[self.ic_results['IC_pvalue'] < 1]['IC_pvalue']
        ax4.hist(valid_pvalues, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('P-value')
        ax4.set_ylabel('Count')
        ax4.set_title('P-value Distribution (Statistical Significance)')
        ax4.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
        ax4.axvline(x=0.01, color='orange', linestyle='--', label='p=0.01')
        ax4.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'ic_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nIC分析图已保存: {self.output_dir / 'ic_analysis.png'}")

    def visualize_factor_returns(self):
        """可视化因子分层收益"""
        if not hasattr(self, 'ranking_results') or self.ranking_results is None:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # 选择表现最好的6个因子
        top_factors = self.ranking_results.head(6)['因子'].tolist()

        for i, factor in enumerate(top_factors):
            if i >= len(axes):
                break

            ax = axes[i]

            result = self.factor_ranking_analysis(factor)
            if result:
                group_returns = result['group_returns']
                x = range(len(group_returns))
                bars = ax.bar(x, group_returns['mean_return'] * 100, color='steelblue', alpha=0.7)

                # 添加误差线
                ax.errorbar(x, group_returns['mean_return'] * 100,
                           yerr=group_returns['std_return'] * 100 / np.sqrt(group_returns['count']),
                           fmt='none', color='black', capsize=3)

                ax.set_xlabel('Factor Quintile')
                ax.set_ylabel('Return (%)')
                ax.set_title(f'{factor}\nIC={result["ic"]:.4f}, LS={result["long_short_return"]*100:.2f}%')
                ax.set_xticks(x)
                ax.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'])

        plt.suptitle('Factor Ranking Analysis - Top 6 Factors by |IC|', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'factor_ranking.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"因子分层图已保存: {self.output_dir / 'factor_ranking.png'}")

    # ==================== 生成报告 ====================

    def generate_report(self):
        """生成综合分析报告"""
        print("\n" + "="*60)
        print("生成分析报告")
        print("="*60)

        report = []
        report.append("=" * 80)
        report.append("中证1000基本面因子分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)

        # 1. 数据概况
        report.append("\n【数据概况】")
        report.append(f"  - 分析股票数量: {self.factors['ts_code'].nunique():,}")
        report.append(f"  - 分析因子数量: {len(self.factor_cols)}")
        report.append(f"  - 因子类别: 估值、盈利能力、成长性、运营效率、杠杆率、现金流")

        # 2. 因子构建
        report.append("\n【构建的因子列表】")
        factor_categories = {
            '估值因子': ['pe', 'pb', 'ps'],
            '盈利能力': ['net_profit_margin', 'gross_profit_margin', 'operating_margin', 'roe', 'roa', 'basic_eps_factor'],
            '成长因子': ['revenue_growth', 'profit_growth', 'eps_growth'],
            '运营效率': ['asset_turnover', 'ar_turnover', 'inventory_turnover'],
            '杠杆/偿债': ['debt_ratio', 'current_ratio', 'quick_ratio', 'lt_debt_ratio'],
            '现金流': ['ocf_to_netincome', 'ocf_yield', 'inv_cash_ratio']
        }

        for category, factors in factor_categories.items():
            valid_factors = [f for f in factors if f in self.factor_cols]
            report.append(f"\n  {category}:")
            for f in valid_factors:
                report.append(f"    - {f}")

        # 3. IC分析结果
        report.append("\n" + "=" * 80)
        report.append("【因子IC分析结果】")
        report.append("IC (Information Coefficient): 衡量因子与收益的相关性，范围[-1, 1]")
        report.append("Rank IC: Spearman秩相关系数，对非线性关系更鲁棒")
        report.append("-" * 80)
        report.append(f"{'因子':<20} {'IC':>10} {'RankIC':>10} {'|IC|':>10} {'P值':>12} {'显著性':>8}")
        report.append("-" * 80)

        for idx, row in self.ic_results.iterrows():
            sig = "***" if row['IC_pvalue'] < 0.001 else "**" if row['IC_pvalue'] < 0.01 else "*" if row['IC_pvalue'] < 0.05 else ""
            report.append(f"{idx:<20} {row['IC']:>10.4f} {row['RankIC']:>10.4f} {row['IC_abs']:>10.4f} {row['IC_pvalue']:>10.4f} {sig:>8}")

        # 4. 分层回测结果
        report.append("\n" + "=" * 80)
        report.append("【因子分层回测结果】")
        report.append("Q1: 因子最低组, Q5: 因子最高组, 多空: Q5-Q1组合收益")
        report.append("-" * 90)
        report.append(f"{'因子':<20} {'IC':>8} {'多空收益':>12} {'方向':>8} {'Q1收益':>10} {'Q5收益':>10}")
        report.append("-" * 90)

        for _, row in self.ranking_results.iterrows():
            report.append(f"{row['因子']:<20} {row['IC']:>8.4f} {row['多空收益']:>10.2f}% {row['方向']:>8} {row['Q1(低)收益']:>8.2f}% {row['Q5(高)收益']:>8.2f}%")

        # 5. 因子评级
        report.append("\n" + "=" * 80)
        report.append("【因子预测能力评级】")
        report.append("-" * 60)

        # 基于IC和显著性进行评级
        def rate_factor(row):
            ic_abs = row['IC_abs']
            pval = row['IC_pvalue']
            if ic_abs >= 0.05 and pval < 0.05:
                return "A级 (强预测)"
            elif ic_abs >= 0.03 and pval < 0.05:
                return "B级 (中等预测)"
            elif ic_abs >= 0.02:
                return "C级 (弱预测)"
            else:
                return "D级 (无预测)"

        self.ic_results['rating'] = self.ic_results.apply(rate_factor, axis=1)

        rating_summary = self.ic_results.groupby('rating').size()
        for rating, count in rating_summary.items():
            factors_in_rating = self.ic_results[self.ic_results['rating'] == rating].index.tolist()
            report.append(f"\n  {rating}: {count}个因子")
            report.append(f"    {factors_in_rating}")

        # 6. 结论与建议
        report.append("\n" + "=" * 80)
        report.append("【结论与建议】")
        report.append("-" * 60)

        # 找出最强预测因子
        top_positive = self.ic_results[self.ic_results['IC'] > 0].nlargest(3, 'IC_abs')
        top_negative = self.ic_results[self.ic_results['IC'] < 0].nlargest(3, 'IC_abs')

        report.append("\n正向预测因子 (因子值高 -> 收益高):")
        if len(top_positive) > 0:
            for idx, row in top_positive.iterrows():
                report.append(f"  - {idx}: IC={row['IC']:.4f}")
        else:
            report.append("  无")

        report.append("\n负向预测因子 (因子值高 -> 收益低):")
        if len(top_negative) > 0:
            for idx, row in top_negative.iterrows():
                report.append(f"  - {idx}: IC={row['IC']:.4f}")
        else:
            report.append("  无")

        # 策略建议
        report.append("\n策略建议:")
        report.append("  1. 多因子模型可考虑纳入PE、PB、ROE等估值和盈利因子")
        report.append("  2. 负向因子可用于风险对冲或空头组合")
        report.append("  3. 建议结合多个因子构建综合评分模型")
        report.append("  4. 因子有效性需定期再检验，适应市场变化")

        report.append("\n" + "=" * 80)
        report.append("报告结束")
        report.append("=" * 80)

        # 保存报告
        report_text = "\n".join(report)
        report_file = self.output_dir / 'fundamental_factor_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\n报告已保存: {report_file}")
        print("\n" + report_text)

        return report_text

    # ==================== 保存结果 ====================

    def save_results(self):
        """保存分析结果"""
        print("\n" + "="*60)
        print("保存分析结果")
        print("="*60)

        # 保存因子数据
        if self.factors is not None:
            self.factors.to_parquet(self.output_dir / 'fundamental_factors.parquet', index=False)
            print(f"因子数据已保存: {self.output_dir / 'fundamental_factors.parquet'}")

        # 保存IC结果
        if hasattr(self, 'ic_results') and self.ic_results is not None:
            self.ic_results.to_csv(self.output_dir / 'ic_results.csv')
            print(f"IC结果已保存: {self.output_dir / 'ic_results.csv'}")

        # 保存分层结果
        if hasattr(self, 'ranking_results') and self.ranking_results is not None:
            self.ranking_results.to_csv(self.output_dir / 'ranking_results.csv', index=False)
            print(f"分层结果已保存: {self.output_dir / 'ranking_results.csv'}")

    # ==================== 完整分析流程 ====================

    def run_full_analysis(self, start_period='20240331', end_period='20241231',
                         start_day='20250101', end_day='20260206'):
        """运行完整分析流程"""
        print("\n" + "="*80)
        print("中证1000基本面因子分析 - 开始")
        print("="*80)

        # 1. 加载数据
        self.load_financial_data(start_period, end_period)
        self.load_market_data(start_day, end_day)

        # 2. 合并数据
        self.merge_financial_data()

        # 3. 构建因子
        self.build_factors()

        # 4. 计算收益率
        self.calculate_returns(holding_period=60)  # 季度收益率

        # 5. IC分析
        self.calculate_ic()

        # 6. 分层分析
        self.run_all_ranking_analysis()

        # 7. 可视化
        self.visualize_ic()
        self.visualize_factor_returns()

        # 8. 生成报告
        self.generate_report()

        # 9. 保存结果
        self.save_results()

        print("\n" + "="*80)
        print("分析完成!")
        print("="*80)


# ==================== 主程序 ====================

if __name__ == "__main__":
    analyzer = FundamentalFactorAnalyzer(output_dir="./fundamental_analysis_results")
    analyzer.run_full_analysis()
