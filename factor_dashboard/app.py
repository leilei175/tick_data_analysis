"""
高频因子分析平台
专业金融风格Web仪表盘
支持按因子存储的宽格式数据
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Optional, List, Dict

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from flask import Flask, render_template, jsonify, request, redirect, url_for

# 添加父目录到路径（确保正确导入 update_data）
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from factor_analysis import FactorAnalysis

# 导入 update_data 模块
import update_data as _update_data_module
get_all_latest_dates = _update_data_module.get_all_latest_dates
init_tushare = _update_data_module.init_tushare
update_daily_data = _update_data_module.download_daily_data
update_financial_data = _update_data_module.update_financial_data
is_after_market_close = _update_data_module.is_after_market_close
get_today_str = _update_data_module.get_today_str
parse_date = _update_data_module.parse_date
date_to_str = _update_data_module.date_to_str

from datetime import datetime, timedelta

# ==================== 配置 ====================
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'hf-factor-secret-key-2026')
    PORT = 9999
    DEBUG = False

# ==================== Flask应用 ====================
app = Flask(__name__)
app.config.from_object(Config)

# 缓存因子数据
_factor_data_cache = {}

def get_by_factor_dir() -> Path:
    """获取按因子存储的目录"""
    return Path(__file__).resolve().parent.parent / "factor" / "by_factor"

def load_factor_data(factor_name: str, year: int = None) -> Optional[pd.DataFrame]:
    """懒加载因子数据"""
    key = f"{factor_name}_{year}" if year else factor_name

    if key not in _factor_data_cache:
        # 优先加载不带年份的完整文件
        if year is None:
            filepath = get_by_factor_dir() / f"zz1000_{factor_name}.parquet"
            if not filepath.exists():
                # 回退到年份文件
                year_files = sorted(get_by_factor_dir().glob(f"zz1000_{factor_name}_*.parquet"))
                if year_files:
                    dfs = [pd.read_parquet(f) for f in year_files]
                    _factor_data_cache[key] = pd.concat(dfs).sort_index()
                    return _factor_data_cache[key]
                return None
        else:
            filepath = get_by_factor_dir() / f"zz1000_{factor_name}_{year}.parquet"

        if filepath.exists():
            _factor_data_cache[key] = pd.read_parquet(filepath)
        else:
            return None

    return _factor_data_cache[key]

def list_available_factors() -> pd.DataFrame:
    """列出所有可用的因子文件"""
    data_path = get_by_factor_dir()
    if not data_path.exists():
        return pd.DataFrame()

    files = list(data_path.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    info = []
    for f in files:
        # 只处理zz1000_开头的因子文件
        if not f.stem.startswith('zz1000_'):
            continue
        # 解析文件名: zz1000_factor_name_year.parquet
        parts = f.stem.split('_')
        if len(parts) < 4:
            continue
        prefix = parts[0]
        factor = '_'.join(parts[1:-1])  # 处理因子名中可能的下划线
        year_str = parts[-1]
        # 跳过非年份文件（如 return_1d）
        if not year_str.isdigit():
            continue
        info.append({
            'filename': f.name,
            'prefix': prefix,
            'factor': factor,
            'year': int(year_str),
            'file_size_mb': float(round(f.stat().st_size / (1024 * 1024), 2)),
            'file_path': str(f)
        })

    return pd.DataFrame(info) if info else pd.DataFrame()


def list_all_factor_names() -> list:
    """列出所有可用的因子名称（包括不带年份的合并文件）"""
    data_path = get_by_factor_dir()
    if not data_path.exists():
        return []

    files = list(data_path.glob("zz1000_*.parquet"))
    factors = set()
    for f in files:
        stem = f.stem
        if stem.startswith('zz1000_'):
            # 去掉zz1000_前缀
            rest = stem[7:]  # len('zz1000_') == 7
            # 如果剩余部分以数字年份结尾，去掉它
            parts = rest.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                factor = parts[0]
            else:
                factor = rest
            factors.add(factor)

    return sorted(list(factors))

# 缓存收益率数据
_returns_cache = {}

def load_returns(period: int = 1) -> Optional[pd.DataFrame]:
    """加载收益率数据"""
    if period not in _returns_cache:
        filepath = get_by_factor_dir() / f"return_{period}d.parquet"
        if filepath.exists():
            _returns_cache[period] = pd.read_parquet(filepath)
        else:
            return None
    return _returns_cache[period]

# ==================== 路由 ====================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html', active_page='dashboard')

@app.route('/dashboard')
def dashboard():
    """仪表盘页面"""
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/correlation')
def correlation():
    """相关性分析页面"""
    return render_template('correlation.html', active_page='correlation')

@app.route('/ic-analysis')
def ic_analysis():
    """IC分析页面"""
    return render_template('ic_analysis.html', active_page='ic')

@app.route('/quantile')
def quantile():
    """分层分析页面"""
    return render_template('quantile.html', active_page='quantile')

@app.route('/long-short')
def long_short():
    """多空组合页面"""
    return render_template('long_short.html', active_page='longshort')

@app.route('/data-manager')
def data_manager():
    """数据管理页面"""
    return render_template('data_manager.html', active_page='data-manager')

# ==================== 新版 API（按因子宽格式数据） ====================

@app.route('/api/factors/list')
def api_factors_list():
    """API: 列出所有可用的因子文件"""
    try:
        # 获取所有因子名称
        factors = list_all_factor_names()
        if not factors:
            return jsonify({'status': 'error', 'message': '没有找到因子文件'})

        # 获取年份信息
        df = list_available_factors()
        years = sorted([int(y) for y in df['year'].unique()]) if not df.empty else []

        # 构建文件信息
        files = []
        if not df.empty:
            for _, row in df.iterrows():
                files.append({
                    'filename': str(row['filename']),
                    'prefix': str(row['prefix']),
                    'factor': str(row['factor']),
                    'year': int(row['year']),
                    'file_size_mb': float(row['file_size_mb']),
                    'file_path': str(row['file_path'])
                })

        return jsonify({
            'status': 'success',
            'data': {
                'factors': factors,
                'years': years,
                'files': files
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/factor/info')
def api_factor_info():
    """API: 获取因子数据汇总信息"""
    try:
        df = list_available_factors()
        if df.empty:
            return jsonify({'status': 'error', 'message': '没有找到因子文件'})

        # 按因子统计
        factor_summary = []
        for factor in df['factor'].unique():
            factor_files = df[df['factor'] == factor]
            years = [int(y) for y in factor_files['year'].tolist()]
            total_size = float(factor_files['file_size_mb'].sum())

            factor_summary.append({
                'factor': str(factor),
                'years': years,
                'file_count': len(factor_files),
                'total_size_mb': total_size
            })

        # 按年份统计
        year_summary = []
        for year in sorted(df['year'].unique()):
            year_files = df[df['year'] == year]
            year_summary.append({
                'year': int(year),
                'file_count': len(year_files),
                'factors': [str(f) for f in year_files['factor'].tolist()]
            })

        return jsonify({
            'status': 'success',
            'data': {
                'data_source': str(get_by_factor_dir()),
                'factor_summary': factor_summary,
                'year_summary': year_summary
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/factor/data')
def api_factor_data():
    """API: 获取因子数据（宽格式）"""
    try:
        factor = request.args.get('factor')
        year = request.args.get('year', type=int)

        if not factor:
            return jsonify({'status': 'error', 'message': '缺少因子名称参数'})

        # 加载因子数据
        if year:
            df = load_factor_data(factor, year)
            desc = f"{factor} - {year}年"
        else:
            # 加载所有年份
            df_list = []
            for f in get_by_factor_dir().glob(f"zz1000_{factor}_*.parquet"):
                df_list.append(pd.read_parquet(f))
            if df_list:
                df = pd.concat(df_list).sort_index()
                desc = f"{factor} - 所有年份"
            else:
                return jsonify({'status': 'error', 'message': '未找到因子数据'})

        if df is None or df.empty:
            return jsonify({'status': 'error', 'message': '因子数据为空'})

        # 返回数据信息
        return jsonify({
            'status': 'success',
            'data': {
                'factor': factor,
                'year': year or 'all',
                'description': desc,
                'shape': {'rows': len(df), 'cols': len(df.columns)},
                'date_range': {
                    'start': str(df.index.min()),
                    'end': str(df.index.max())
                },
                'stock_count': len(df.columns),
                'date_count': len(df)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/factor/sample')
def api_factor_sample():
    """API: 获取因子数据样例"""
    try:
        factor = request.args.get('factor')
        year = request.args.get('year', type=int)

        # 加载因子数据
        if year:
            df = load_factor_data(factor, year)
        else:
            files = sorted(get_by_factor_dir().glob(f"zz1000_{factor}_*.parquet"))
            if not files:
                return jsonify({'status': 'error', 'message': '未找到因子数据'})
            df = pd.read_parquet(files[0])

        if df is None or df.empty:
            return jsonify({'status': 'error', 'message': '因子数据为空'})

        # 转换为长格式样例
        sample = df.head(10)

        # 构建返回数据
        records = []
        for date_idx in sample.index:
            for stock in sample.columns[:20]:  # 只返回前20只股票
                value = sample.loc[date_idx, stock]
                records.append({
                    'date': str(date_idx),
                    'stock': stock,
                    'factor': round(float(value), 6) if pd.notna(value) else None
                })

        return jsonify({
            'status': 'success',
            'data': {
                'factor': factor,
                'filename': f"zz1000_{factor}_{year or df.index.year.min()}.parquet",
                'sample_data': records,
                'total_rows': len(df),
                'total_stocks': len(df.columns)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/factor/stats')
def api_factor_stats():
    """API: 计算因子统计信息"""
    try:
        factor = request.args.get('factor')
        year = request.args.get('year', type=int)

        # 加载因子数据
        if year:
            df = load_factor_data(factor, year)
        else:
            files = sorted(get_by_factor_dir().glob(f"zz1000_{factor}_*.parquet"))
            if not files:
                return jsonify({'status': 'error', 'message': '未找到因子数据'})
            df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()

        if df is None or df.empty:
            return jsonify({'status': 'error', 'message': '因子数据为空'})

        # 计算统计信息
        flat_values = df.values.flatten()
        flat_values = flat_values[~np.isnan(flat_values)]

        stats = {
            'factor': factor,
            'count': len(flat_values),
            'mean': float(np.mean(flat_values)),
            'std': float(np.std(flat_values)),
            'min': float(np.min(flat_values)),
            'max': float(np.max(flat_values)),
            'median': float(np.median(flat_values)),
            'skew': float(pd.Series(flat_values).skew()),
            'kurtosis': float(pd.Series(flat_values).kurtosis())
        }

        # 计算每日的均值和标准差
        daily_means = df.mean(axis=1)
        daily_stds = df.std(axis=1)
        stats['daily_mean_std'] = {
            'mean': float(daily_means.mean()),
            'std': float(daily_means.std())
        }

        return jsonify({'status': 'success', 'data': stats})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/factor/ic')
def api_factor_ic():
    """API: 计算因子IC"""
    try:
        factor = request.args.get('factor')
        year = request.args.get('year', type=int)
        period = request.args.get('period', default=1, type=int)

        # 加载因子数据 - 宽格式
        factor_df = load_factor_data(factor, year)
        if factor_df is None or factor_df.empty:
            return jsonify({'status': 'error', 'message': f'因子文件不存在: {factor}'})
        if factor_df.empty:
            return jsonify({'status': 'error', 'message': '因子数据为空'})

        # 加载收益率数据
        returns_df = load_returns(period)
        if returns_df is None or returns_df.empty:
            return jsonify({'status': 'error', 'message': '收益率数据未找到'})

        # 宽格式转长格式
        factor_long = factor_df.stack().reset_index()
        factor_long.columns = ['date', 'stock_code', 'factor_value']
        # 转换日期为统一的字符串格式 YYYY-MM-DD
        factor_long['date'] = pd.to_datetime(factor_long['date']).dt.strftime('%Y-%m-%d')

        # 标准化收益率日期格式
        returns_df['date'] = pd.to_datetime(returns_df['date']).dt.strftime('%Y-%m-%d')

        returns_df = returns_df.rename(columns={'return_1d': f'return_{period}d'})
        returns_long = returns_df[['date', 'stock_code', f'return_{period}d']].copy()

        merged = factor_long.merge(returns_long, on=['date', 'stock_code']).dropna()

        if len(merged) == 0:
            return jsonify({'status': 'error', 'message': '没有可用的重叠数据'})

        # 计算IC
        ic_series = merged.groupby('date').apply(
            lambda x: x['factor_value'].corr(x[f'return_{period}d'])
        )

        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std())
        ic_ir = ic_mean / ic_std if ic_std != 0 else 0
        ic_positive_ratio = float((ic_series > 0).mean())

        return jsonify({
            'status': 'success',
            'data': {
                'factor': factor,
                'period': period,
                'ic_series': [{'date': str(d), 'ic': float(v)} for d, v in ic_series.items()],
                'ic_stats': {
                    'ic_mean': ic_mean,
                    'ic_std': ic_std,
                    'ic_ir': ic_ir,
                    'ic_positive_ratio': ic_positive_ratio
                }
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/factor/cross-correlation')
def api_factor_cross_correlation():
    """API: 计算因子之间的相关性"""
    try:
        factors = request.args.getlist('factors')
        year = request.args.get('year', type=int)

        if len(factors) < 2:
            return jsonify({'status': 'error', 'message': '至少需要两个因子'})

        # 加载因子数据
        dfs = {}
        for factor in factors:
            if year:
                df = load_factor_data(factor, year)
            else:
                files = sorted(get_by_factor_dir().glob(f"zz1000_{factor}_*.parquet"))
                if files:
                    df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
                else:
                    continue
            if df is not None:
                dfs[factor] = df

        if len(dfs) < 2:
            return jsonify({'status': 'error', 'message': '无法加载足够的因子数据'})

        # 合并数据并计算相关性
        common_dates = set.intersection(*[set(df.index) for df in dfs.values()])

        corr_data = {}
        for f1 in factors:
            if f1 not in dfs:
                continue
            corr_data[f1] = {}
            for f2 in factors:
                if f2 not in dfs:
                    continue
                # 计算截面相关性
                common_df = pd.DataFrame({f1: dfs[f1].loc[common_dates].mean(axis=1),
                                         f2: dfs[f2].loc[common_dates].mean(axis=1)}).dropna()
                if len(common_df) > 10:
                    corr_data[f1][f2] = round(common_df[f1].corr(common_df[f2]), 4)
                else:
                    corr_data[f1][f2] = None

        return jsonify({
            'status': 'success',
            'data': {
                'factors': factors,
                'year': year,
                'correlation_matrix': corr_data
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/quantile')
def api_quantile():
    """API: 分层分析"""
    try:
        factor = request.args.get('factor')
        period = request.args.get('period', default=1, type=int)
        quantiles = request.args.get('quantiles', default=5, type=int)

        if not factor:
            return jsonify({'status': 'error', 'message': '缺少因子参数'})

        # 加载因子数据 - 宽格式: index=date, columns=stock_code
        factor_df = load_factor_data(factor)
        if factor_df is None or factor_df.empty:
            return jsonify({'status': 'error', 'message': f'因子文件不存在: {factor}'})
        if factor_df.empty:
            return jsonify({'status': 'error', 'message': '因子数据为空'})

        # 加载收益率数据
        returns_df = load_returns(period)
        if returns_df is None or returns_df.empty:
            return jsonify({'status': 'error', 'message': '收益率数据为空'})

        # 合并数据 - 宽格式转长格式
        factor_long = factor_df.stack().reset_index()
        factor_long.columns = ['date', 'stock_code', 'factor_value']
        # 转换日期为统一的字符串格式 YYYY-MM-DD
        factor_long['date'] = pd.to_datetime(factor_long['date']).dt.strftime('%Y-%m-%d')

        # 标准化收益率日期格式
        returns_df['date'] = pd.to_datetime(returns_df['date']).dt.strftime('%Y-%m-%d')

        returns_df = returns_df.rename(columns={'return_1d': f'return_{period}d'})
        returns_long = returns_df[['date', 'stock_code', f'return_{period}d']].copy()

        merged = factor_long.merge(returns_long, on=['date', 'stock_code']).dropna()

        if len(merged) == 0:
            return jsonify({'status': 'error', 'message': '没有可用的重叠数据'})

        # 计算每日因子分位数
        merged['quantile'] = merged.groupby('date')['factor_value'].transform(
            lambda x: pd.qcut(x, quantiles, labels=False, duplicates='drop') + 1
        )

        # 按分位数统计
        result = []
        for q in range(1, quantiles + 1):
            q_data = merged[merged['quantile'] == q]
            result.append({
                'quantile': q,
                'factor_mean': float(q_data['factor_value'].mean()),
                'return_mean': float(q_data[f'return_{period}d'].mean()),
                'return_std': float(q_data[f'return_{period}d'].std()),
                'count': int(len(q_data)),
                'ic': float(q_data['factor_value'].corr(q_data[f'return_{period}d']))
            })

        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/long-short')
def api_long_short():
    """API: 多空组合分析"""
    try:
        factor = request.args.get('factor')
        period = request.args.get('period', default=1, type=int)

        if not factor:
            return jsonify({'status': 'error', 'message': '缺少因子参数'})

        # 加载因子数据 - 宽格式
        factor_df = load_factor_data(factor)
        if factor_df is None or factor_df.empty:
            return jsonify({'status': 'error', 'message': f'因子文件不存在: {factor}'})
        if factor_df.empty:
            return jsonify({'status': 'error', 'message': '因子数据为空'})

        # 加载收益率数据
        returns_df = load_returns(period)
        if returns_df is None or returns_df.empty:
            return jsonify({'status': 'error', 'message': '收益率数据为空'})

        # 宽格式转长格式
        factor_long = factor_df.stack().reset_index()
        factor_long.columns = ['date', 'stock_code', 'factor_value']
        # 转换日期为统一的字符串格式 YYYY-MM-DD
        factor_long['date'] = pd.to_datetime(factor_long['date']).dt.strftime('%Y-%m-%d')

        # 标准化收益率日期格式
        returns_df['date'] = pd.to_datetime(returns_df['date']).dt.strftime('%Y-%m-%d')

        returns_df = returns_df.rename(columns={'return_1d': f'return_{period}d'})
        returns_long = returns_df[['date', 'stock_code', f'return_{period}d']].copy()

        merged = factor_long.merge(returns_long, on=['date', 'stock_code']).dropna()

        if len(merged) == 0:
            return jsonify({'status': 'error', 'message': '没有可用的重叠数据'})

        # 计算每日因子分位数
        merged['quantile'] = merged.groupby('date')['factor_value'].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1
        )

        # 计算多空收益
        top_returns = merged[merged['quantile'] == 5].groupby('date')[f'return_{period}d'].mean()
        bottom_returns = merged[merged['quantile'] == 1].groupby('date')[f'return_{period}d'].mean()
        ls_returns = top_returns - bottom_returns

        # 统计指标
        total_return = (1 + ls_returns).prod() - 1
        mean_daily = ls_returns.mean()
        std_daily = ls_returns.std()
        sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily != 0 else 0
        win_rate = (ls_returns > 0).mean()

        # 计算最大回撤
        cumulative = (1 + ls_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return jsonify({
            'status': 'success',
            'data': {
                'factor': factor,
                'period': period,
                'total_return': float(total_return),
                'mean_daily': float(mean_daily),
                'std_daily': float(std_daily),
                'sharpe': float(sharpe),
                'win_rate': float(win_rate),
                'max_drawdown': float(max_drawdown)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ==================== 原有 API（保持兼容性） ====================

@app.route('/api/summary')
def api_summary():
    """API: 获取汇总数据"""
    try:
        # 尝试使用新版因子列表
        factors_df = list_available_factors()
        if not factors_df.empty:
            years = sorted([int(y) for y in factors_df['year'].unique()])

            summary = {
                'total_files': int(len(factors_df)),
                'factor_count': int(len(factors_df['factor'].unique())),
                'factors': [str(f) for f in factors_df['factor'].unique()],
                'years': years,
                'data_source': str(get_by_factor_dir()),
                'date_range': {
                    'start': f"{years[0]}-01-01" if years else None,
                    'end': f"{years[-1]}-12-31" if years else None
                }
            }
            return jsonify({'status': 'success', 'data': summary})

        # 回退到原有方式
        analyzer = None
        global factor_analyzer
        if factor_analyzer is None:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            factor_analyzer = FactorAnalysis(os.path.join(base_path, "factor/daily"))
            try:
                factor_analyzer.load_all_factors()
                factor_analyzer.compute_returns(periods=[1, 5, 10])
            except:
                pass

        if analyzer and analyzer.factors_df is not None:
            summary = {
                'total_records': int(len(analyzer.factors_df)),
                'total_stocks': int(analyzer.factors_df['stock_code'].nunique()),
                'total_dates': int(analyzer.factors_df['date'].nunique()),
                'factor_count': len(analyzer.factor_cols),
                'date_range': {
                    'start': str(analyzer.factors_df['date'].min()),
                    'end': str(analyzer.factors_df['date'].max())
                },
                'factors': analyzer.factor_cols
            }
            return jsonify({'status': 'success', 'data': summary})

        return jsonify({'status': 'error', 'message': '无法加载数据'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/descriptive-stats')
def api_descriptive_stats():
    """API: 描述性统计"""
    try:
        analyzer = get_analyzer()
        stats = analyzer.descriptive_stats()
        return jsonify({'status': 'success', 'data': stats.to_dict(orient='records')})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/correlation')
def api_correlation():
    """API: 相关性矩阵"""
    try:
        analyzer = get_analyzer()
        corr = analyzer.factor_correlation(method='spearman')
        return jsonify({'status': 'success', 'data': corr.to_dict()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/ic-stats')
def api_ic_stats():
    """API: IC统计"""
    try:
        analyzer = get_analyzer()
        ic_result = analyzer.ic_analysis(return_col='return_1', by='date')
        stats = ic_result['ic_stats'].reset_index()
        stats.columns = ['factor', 'ic_mean', 'ic_std', 'ic_ir', 'ic_positive_ratio', 'ic_t_stat']
        return jsonify({'status': 'success', 'data': stats.to_dict(orient='records')})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/ic-series')
def api_ic_series():
    """API: IC时间序列"""
    try:
        analyzer = get_analyzer()
        ic_result = analyzer.ic_analysis(return_col='return_1', by='date')

        series_data = {}
        for factor, ic_vals in ic_result['ic_series'].items():
            series_data[factor] = [
                {'date': str(d), 'ic': float(v)} for d, v in zip(ic_result['ic_series'][factor].index, ic_vals)
            ]

        return jsonify({'status': 'success', 'data': series_data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/ic-decay')
def api_ic_decay():
    """API: IC衰减"""
    try:
        analyzer = get_analyzer()
        decay = analyzer.ic_decay_analysis(max_lag=20)
        return jsonify({'status': 'success', 'data': decay.to_dict(orient='records')})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/quantile/<factor>')
def api_quantile_by_factor(factor):
    """API: 分层分析（带因子参数）"""
    try:
        period = request.args.get('period', default=1, type=int)
        quantiles = request.args.get('quantiles', default=5, type=int)

        # 加载因子数据 - 宽格式
        factor_df = load_factor_data(factor)
        if factor_df is None or factor_df.empty:
            return jsonify({'status': 'error', 'message': f'因子文件不存在: {factor}'})
        if factor_df.empty:
            return jsonify({'status': 'error', 'message': '因子数据为空'})

        # 加载收益率数据
        returns_df = load_returns(period)
        if returns_df is None or returns_df.empty:
            return jsonify({'status': 'error', 'message': '收益率数据为空'})

        # 宽格式转长格式
        factor_long = factor_df.stack().reset_index()
        factor_long.columns = ['date', 'stock_code', 'factor_value']
        # 转换日期为统一的字符串格式 YYYY-MM-DD
        factor_long['date'] = pd.to_datetime(factor_long['date']).dt.strftime('%Y-%m-%d')

        # 标准化收益率日期格式
        returns_df['date'] = pd.to_datetime(returns_df['date']).dt.strftime('%Y-%m-%d')

        returns_df = returns_df.rename(columns={'return_1d': f'return_{period}d'})
        returns_long = returns_df[['date', 'stock_code', f'return_{period}d']].copy()

        merged = factor_long.merge(returns_long, on=['date', 'stock_code']).dropna()

        if len(merged) == 0:
            return jsonify({'status': 'error', 'message': '没有可用的重叠数据'})

        # 计算每日因子分位数
        merged['quantile'] = merged.groupby('date')['factor_value'].transform(
            lambda x: pd.qcut(x, quantiles, labels=False, duplicates='drop') + 1
        )

        # 按分位数统计
        result = []
        for q in range(1, quantiles + 1):
            q_data = merged[merged['quantile'] == q]
            result.append({
                'quantile': q,
                'factor_mean': float(q_data['factor_value'].mean()),
                'return_mean': float(q_data[f'return_{period}d'].mean()),
                'return_std': float(q_data[f'return_{period}d'].std()),
                'count': int(len(q_data)),
                'ic': float(q_data['factor_value'].corr(q_data[f'return_{period}d']))
            })

        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ==================== 页面路由 ====================

@app.route('/factor-info')
def factor_info():
    """因子信息查看页面"""
    return render_template('factor_info.html', active_page='factor-info')

@app.route('/factor-viewer')
def factor_viewer():
    """因子数据查看器页面"""
    return render_template('factor_viewer.html', active_page='factor-viewer')

# ==================== 数据管理 API ====================

@app.route('/api/data/status')
def api_data_status():
    """API: 获取daily_data目录状态"""
    try:
        # 获取各类数据的最新日期
        latest_dates = get_all_latest_dates()

        # 获取交易日信息
        try:
            pro = init_tushare()
            today = get_today_str()
            trade_dates = pro.trade_cal(
                exchange='SSE',
                start_date=today,
                end_date=today,
                is_open='1'
            )
            is_trading_day = len(trade_dates) > 0
        except:
            is_trading_day = False

        # 数据表配置
        data_tables = {
            'daily': {
                'name': '日线行情',
                'description': 'OHLCV行情数据',
                'fields': ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'],
                'directory': 'daily_data/daily/'
            },
            'daily_basic': {
                'name': '每日基本面',
                'description': '每日基本面指标',
                'fields': ['ts_code', 'trade_date', 'close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv'],
                'directory': 'daily_data/daily_basic/'
            },
            'cashflow_daily': {
                'name': '现金流量表',
                'description': '每日推算的现金流量数据',
                'fields': ['ts_code', 'trade_date', 'n_cashflow_act', 'n_cashflow_inv_act', 'n_cash_flows_fnc_act', 'c_fr_sale_sg', 'c_paid_goods_s', 'net_profit'],
                'directory': 'daily_data/cashflow_daily/'
            },
            'income_daily': {
                'name': '利润表',
                'description': '每日推算的利润数据',
                'fields': ['ts_code', 'trade_date', 'total_revenue', 'revenue', 'oper_cost', 'operate_profit', 'total_profit', 'n_income', 'basic_eps'],
                'directory': 'daily_data/income_daily/'
            },
            'balance_daily': {
                'name': '资产负债表',
                'description': '每日推算的资产负债数据',
                'fields': ['ts_code', 'trade_date', 'total_assets', 'total_liab', 'total_hldr_eqy_exc_min_int', 'total_cur_assets', 'total_cur_liab', 'cash_reser_cb'],
                'directory': 'daily_data/balance_daily/'
            }
        }

        # 构建返回数据
        tables_info = []
        for table_id, config in data_tables.items():
            latest = latest_dates.get(table_id)
            tables_info.append({
                'id': table_id,
                'name': config['name'],
                'description': config['description'],
                'fields': config['fields'],
                'latest_date': latest,
                'directory': config['directory'],
                'status': 'updated' if latest else 'empty'
            })

        return jsonify({
            'status': 'success',
            'data': {
                'tables': tables_info,
                'is_trading_day': is_trading_day,
                'is_after_market_close': is_after_market_close(),
                'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'today': get_today_str()
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/data/update', methods=['POST'])
def api_data_update():
    """API: 执行数据更新"""
    try:
        import threading
        import subprocess

        data = request.get_json() or {}
        update_type = data.get('type', 'all')
        include_today = data.get('include_today', is_after_market_close())

        # 启动后台任务执行更新
        def run_update():
            try:
                cmd = [sys.executable, 'update_data.py']
                if update_type == 'daily':
                    cmd.extend(['--daily'])
                elif update_type == 'daily_basic':
                    cmd.extend(['--daily-basic'])
                elif update_type == 'financial':
                    cmd.extend(['--financial'])

                if include_today:
                    cmd.append('--include-today')

                subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            except Exception as e:
                print(f"数据更新错误: {e}")

        # 后台执行
        thread = threading.Thread(target=run_update)
        thread.start()

        return jsonify({
            'status': 'success',
            'message': '数据更新任务已启动',
            'data': {
                'update_type': update_type,
                'include_today': include_today
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/data/update/sync', methods=['POST'])
def api_data_update_sync():
    """API: 同步执行数据更新（返回进度）"""
    try:
        data = request.get_json() or {}
        update_type = data.get('type', 'all')
        include_today = data.get('include_today', is_after_market_close())

        # 执行更新
        try:
            pro = init_tushare()

            # 确定更新范围
            latest_dates = get_all_latest_dates()
            latest = latest_dates.get('daily', '20250101')

            # 获取需要更新的交易日
            today = get_today_str()
            trade_dates = pro.trade_cal(
                exchange='SSE',
                start_date=latest,
                end_date=today,
                is_open='1'
            )['cal_date'].tolist()

            if not trade_dates:
                return jsonify({
                    'status': 'success',
                    'message': '没有需要更新的交易日',
                    'data': {'updated_count': 0}
                })

            # 如果不包含今天，过滤掉
            if not include_today and not is_after_market_close():
                trade_dates = [d for d in trade_dates if d != today]

            # 限制为最新5个交易日
            trade_dates = trade_dates[-5:]

            updated_count = 0

            if update_type in ('all', 'daily'):
                from update_data import download_daily_data
                download_daily_data(pro, trade_dates[0], trade_dates[-1])

            if update_type in ('all', 'daily_basic'):
                from update_data import download_daily_basic_data
                download_daily_basic_data(pro, trade_dates[0], trade_dates[-1])

            return jsonify({
                'status': 'success',
                'message': f'数据更新完成，更新了 {len(trade_dates)} 个交易日',
                'data': {
                    'updated_count': len(trade_dates),
                    'trade_dates': trade_dates,
                    'update_type': update_type
                }
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'更新失败: {str(e)}'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/data/refresh', methods=['POST'])
def api_data_refresh():
    """API: 刷新数据状态"""
    try:
        # 重新获取最新数据状态
        latest_dates = get_all_latest_dates()

        return jsonify({
            'status': 'success',
            'message': '数据状态已刷新',
            'data': {
                'latest_dates': latest_dates,
                'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# ==================== 启动 ====================

def get_analyzer():
    """懒加载分析器（保持兼容性）"""
    global factor_analyzer
    if factor_analyzer is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        factor_analyzer = FactorAnalysis(os.path.join(base_path, "factor/daily"))
        try:
            factor_analyzer.load_all_factors()
            factor_analyzer.compute_returns(periods=[1, 5, 10])
            print(f"Loaded {len(factor_analyzer.factors_df)} records")
        except Exception as e:
            print(f"Warning: Could not load data: {e}")
    return factor_analyzer

if __name__ == '__main__':
    print("=" * 60)
    print("高频因子分析平台")
    print("=" * 60)
    print(f"\n因子数据目录: {get_by_factor_dir()}")
    print(f"启动服务器: http://localhost:{Config.PORT}")
    print(f"API文档: http://localhost:{Config.PORT}/api/factors/list")
    print("\n按 Ctrl+C 停止服务器")
    print("=" * 60)

    app.run(host='0.0.0.0', port=Config.PORT, debug=Config.DEBUG)
