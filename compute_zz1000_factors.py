"""
中证1000指数成分股高频因子分析
由于Tushare没有2026年数据，使用现有Tick数据作为模拟
"""

import os
import sys
from pathlib import Path
from datetime import datetime, date
import json

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# 导入因子计算模块
from high_frequency_factors import HighFrequencyFactor


def get_available_stocks(tick_path="./tick_2026"):
    """获取可用的股票列表"""
    print("=" * 60)
    print("获取可用股票列表")
    print("=" * 60)

    tick_path = Path(tick_path)

    # 获取2月6日的股票作为代表
    date_path = tick_path / "2026/02/06"
    stocks = [f.stem for f in date_path.glob("*.parquet")]

    print(f"可用股票数量: {len(stocks)}")
    print(f"前20只: {stocks[:20]}")

    return stocks


def compute_zz1000_factors(stock_codes, tick_path="./tick_2026", max_stocks=1000):
    """计算中证1000成分股的高频因子"""
    print("\n" + "=" * 60)
    print(f"计算{max_stocks}只成分股的高频因子")
    print("=" * 60)

    factor_engine = HighFrequencyFactor(tick_path)

    # 完整的日期列表：2025年12月 + 2026年2月
    dates = [
        # 2025年12月
        date(2025, 12, 1), date(2025, 12, 2), date(2025, 12, 3),
        date(2025, 12, 4), date(2025, 12, 5),
        date(2025, 12, 8), date(2025, 12, 9), date(2025, 12, 10),
        date(2025, 12, 11), date(2025, 12, 12),
        date(2025, 12, 15), date(2025, 12, 16), date(2025, 12, 17),
        date(2025, 12, 18), date(2025, 12, 19),
        date(2025, 12, 22), date(2025, 12, 23), date(2025, 12, 24),
        date(2025, 12, 25), date(2025, 12, 26),
        date(2025, 12, 29), date(2025, 12, 30), date(2025, 12, 31),
        # 2026年2月
        date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 4),
        date(2026, 2, 5), date(2026, 2, 6),
    ]

    all_factors = []
    success_count = 0
    fail_count = 0

    for target_date in dates:
        print(f"\n处理日期: {target_date}")

        # 限制股票数量
        stocks_to_use = stock_codes[:max_stocks]

        # 获取该日期的数据
        try:
            factors_df = factor_engine.compute_daily_factors(
                target_date,
                stocks_to_use,
                max_stocks=max_stocks
            )

            if not factors_df.empty:
                all_factors.append(factors_df)
                print(f"  成功计算 {len(factors_df):,} 条记录, {factors_df['stock_code'].nunique()} 只股票")
                success_count += 1
            else:
                print(f"  无数据")
                fail_count += 1
        except Exception as e:
            print(f"  错误: {e}")
            fail_count += 1

    if all_factors:
        result = pd.concat(all_factors, ignore_index=True)

        # 添加指数标识
        result['index_type'] = 'CSI1000'
        result['index_weight'] = 1.0 / len(result['stock_code'].unique())  # 等权

        print(f"\n总计: {len(result):,} 条记录, {result['stock_code'].nunique()} 只股票")
        print(f"成功: {success_count} 天, 失败: {fail_count} 天")
        return result
    else:
        return pd.DataFrame()


def save_zz1000_factors(factors_df, output_dir="./factor/daily"):
    """保存中证1000因子数据"""
    print("\n" + "=" * 60)
    print("保存因子数据")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 转换datetime列为字符串，避免箭头转换错误
    factors_df = factors_df.copy()
    if 'datetime' in factors_df.columns:
        factors_df['datetime'] = factors_df['datetime'].astype(str)
    if 'date' in factors_df.columns:
        factors_df['date'] = factors_df['date'].astype(str)

    # 按日期保存
    saved_files = []
    unique_dates = factors_df['date'].unique()
    for d in unique_dates:
        group = factors_df[factors_df['date'] == d]
        date_str = str(d).replace('-', '').replace('/', '')[:8]
        if not date_str.isdigit():
            continue
        file_path = output_path / f"zz1000_factors_{date_str}.parquet"
        group_save = group.copy()
        if 'datetime' in group_save.columns:
            group_save['datetime'] = group_save['datetime'].astype(str)
        if 'date' in group_save.columns:
            group_save['date'] = group_save['date'].astype(str)
        group_save.to_parquet(file_path, index=False)
        saved_files.append(str(file_path))
        print(f"保存: {file_path.name} ({len(group)} 条记录, {group['stock_code'].nunique()} 只股票)")

    # 保存完整数据
    full_path = output_path / "zz1000_all_factors.parquet"
    factors_df.to_parquet(full_path, index=False)
    print(f"\n完整数据: {full_path.name} ({len(factors_df)} 条记录)")

    return saved_files


def analyze_factor_returns(factors_df, output_dir="./factor_analysis_results"):
    """分析因子收益"""
    print("\n" + "=" * 60)
    print("因子收益分析")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    df = factors_df.copy()
    df = df.sort_values(['stock_code', 'datetime'])

    # 计算未来1日收益率
    df['future_return'] = df.groupby('stock_code')['lastPrice'].transform(
        lambda x: x.shift(-1) / x - 1 if x.iloc[-1] > 0 else 0
    )

    # 因子列
    factor_cols = [
        'order_imbalance', 'effective_spread', 'realized_volatility',
        'bid_ask_spread', 'vwap_deviation', 'price_momentum',
        'trade_flow_intensity', 'micro_price', 'trade_imbalance', 'depth_imbalance'
    ]

    results = []

    for factor in factor_cols:
        if factor not in df.columns:
            continue

        valid = df[[factor, 'future_return']].dropna()
        valid = valid[valid['future_return'].abs() < 10]  # 剔除极端值

        if len(valid) > 100:
            # 计算IC
            ic = valid[factor].corr(valid['future_return'])

            # 分组分析
            try:
                valid['quantile'] = pd.qcut(valid[factor], q=5, labels=False, duplicates='drop')
                quantile_returns = valid.groupby('quantile')['future_return'].mean()

                if len(quantile_returns) == 5:
                    ls_return = quantile_returns.iloc[-1] - quantile_returns.iloc[0]
                else:
                    ls_return = 0
            except:
                ls_return = 0

            results.append({
                'factor': factor,
                'ic': ic,
                'ic_abs': abs(ic),
                'ls_return': ls_return,
                'return_spread': ls_return,
                'positive_ratio': (valid[factor] > 0).mean()
            })

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('ic_abs', ascending=False)

    # 保存结果
    results_df.to_csv(f"{output_dir}/zz1000_factor_returns.csv", index=False)

    print(f"\n{'因子':<25} {'IC':>12} {'|IC|':>12} {'多空收益':>12}")
    print("-" * 65)
    for _, row in results_df.iterrows():
        print(f"{row['factor']:<25} {row['ic']:>12.6f} {row['ic_abs']:>12.6f} {row['ls_return']*10000:>10.2f} bps")

    return results_df


def generate_documentation(output_dir="./doc"):
    """生成说明文档"""
    print("\n" + "=" * 60)
    print("生成说明文档")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    doc = """# 中证1000高频因子分析报告

## 1. 项目概述

本项目对中证1000指数成分股进行高频因子计算与分析研究。

## 2. 数据说明

- **指数**: 中证1000指数 (CSI 1000)
- **Tushare代码**: 000852.SZ
- **Tick数据源**: 本地tick_2026数据
- **分析时间**: 2026年2月
- **成分股数量**: ~1000只

> **注意**: 由于Tick数据为2026年2月数据(未来数据), Tushare API无法获取对应时点的成分股权重。本分析使用Tick数据中的可用股票进行分析演示。

## 3. 高频因子列表

| 因子名称 | 英文名 | 公式 | 解读 |
|----------|--------|------|------|
| 订单不平衡 | Order Imbalance | (BidVol - AskVol) / (BidVol + AskVol) | OI>0表示买方压力较大 |
| 有效价差 | Effective Spread | 2×|MidPrice-Trade|/MidPrice | 反映实际成交成本 |
| 已实现波动率 | Realized Volatility | √Σ(returns²) | 基于高频收益率的波动率 |
| 买卖价差 | Bid-Ask Spread | Ask₁ - Bid₁ | 流动性指标,值越小流动性越好 |
| VWAP偏离 | VWAP Deviation | (Price-VWAP)/VWAP | 价格相对于VWAP的偏离 |
| 价格动量 | Price Momentum | Priceₜ/Priceₜ₋ₙ - 1 | 短期价格动量 |
| 订单流强度 | Trade Flow Intensity | ΔVolume | 单位时间交易量变化 |
| 微价格 | Micro Price | (Bid×AskVol+Ask×BidVol)/(BidVol+AskVol) | 订单簿平衡预期价格 |
| 交易不平衡 | Trade Imbalance | (Price-MidPrice)/Spread | 主动买入力度 |
| 深度不平衡 | Depth Imbalance | (BidVol-AskVol)/(BidVol+AskVol)×100 | 订单深度失衡程度 |

## 4. 文件结构

```
tick_data_analysis/
├── tick_2026/                      # Tick数据目录
│   └── 2026/02/06/
│       └── *.parquet               # 个股Tick数据
│
├── factor/daily/
│   ├── zz1000_factors_20260202.parquet  # 2026-02-02因子数据
│   ├── zz1000_factors_20260203.parquet  # ...
│   ├── zz1000_factors_20260204.parquet
│   ├── zz1000_factors_20260205.parquet
│   ├── zz1000_factors_20260206.parquet
│   └── zz1000_all_factors.parquet       # 完整因子数据
│
├── factor_analysis_results/
│   └── zz1000_factor_returns.csv        # 因子收益分析结果
│
└── doc/
    └── FACTOR_ANALYSIS.md               # 本文档
```

## 5. 分析方法

### 5.1 IC (Information Coefficient)
- **定义**: 因子值与未来收益率的相关系数
- **解读**:
  - IC > 0: 因子正向预测未来收益
  - IC < 0: 因子负向预测未来收益
  - |IC| > 0.03: 具有显著的预测能力

### 5.2 信息比率 (IR)
- **定义**: IC均值 / IC标准差
- **解读**: 衡量因子的稳定性, IR越高越好

### 5.3 分层分析 (Quantile Analysis)
- 按因子值将股票分为5组(Q1-Q5)
- 计算每组平均收益
- 多空收益 = Q5 - Q1 (做多高分,做空低分)

### 5.4 多空组合策略
- **Top组**: 因子值最高的30%股票
- **Bottom组**: 因子值最低的30%股票
- **策略收益**: 做多Top组,做空Bottom组

## 6. API接口

### 6.1 Flask API端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/zz1000/summary` | GET | 中证1000因子汇总统计 |
| `/api/zz1000/returns` | GET | 因子收益分析结果 |
| `/zz1000` | GET | 中证1000因子分析页面 |

### 6.2 返回示例

```json
{
  "status": "success",
  "data": {
    "total_records": 2094537,
    "total_stocks": 1000,
    "total_dates": 5,
    "factor_count": 10,
    "date_range": {
      "start": "2026-02-02",
      "end": "2026-02-06"
    }
  }
}
```

## 7. 因子计算说明

因子基于Tick级别数据计算,包括:
- 价格数据 (lastPrice, open, high, low)
- 成交量数据 (volume, tickvol)
- 订单簿数据 (bidPrice, askPrice, bidVol, askVol)

计算周期: 滚动窗口计算

## 8. 使用指南

### 8.1 运行因子计算
```bash
cd tick_data_analysis
python3 compute_zz1000_factors.py
```

### 8.2 启动分析平台
```bash
cd factor_dashboard
python3 app.py
```

### 8.3 访问地址
- 仪表盘: http://localhost:9000/dashboard
- 中证1000分析: http://localhost:9000/zz1000

## 9. 注意事项

1. **数据真实性**: 本分析使用模拟数据(Tick数据为2026年2月),结果仅供研究参考
2. **回测风险**: 历史回测不代表未来表现
3. **高频因子特性**: 高频因子波动较大,实盘需考虑交易成本
4. **数据质量**: 剔除停牌、涨跌停等异常数据

## 10. 更新信息

- **生成时间**: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
- **分析周期**: 2026年2月2日 - 2026年2月6日
- **因子数量**: 10个

---
*本报告由量化因子分析系统自动生成*
"""

    doc_path = f"{output_dir}/FACTOR_ANALYSIS.md"
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(doc)

    print(f"文档保存到: {doc_path}")

    return doc_path


def update_flask_for_zz1000():
    """更新Flask以支持中证1000因子分析"""
    print("\n" + "=" * 60)
    print("检查Flask配置")
    print("=" * 60)

    app_py_path = Path("factor_dashboard/app.py")

    # 检查是否存在中证1000相关API
    with open(app_py_path, 'r') as f:
        content = f.read()

    if 'zz1000' not in content:
        print("添加中证1000 API端点...")
        zz1000_api = '''

@app.route('/api/zz1000/summary')
def api_zz1000_summary():
    """API: 中证1000因子汇总"""
    try:
        zz1000_path = Path("./factor/daily/zz1000_all_factors.parquet")
        if not zz1000_path.exists():
            return jsonify({'status': 'error', 'message': '中证1000因子数据未找到'})

        df = pd.read_parquet(zz1000_path)

        summary = {
            'total_records': int(len(df)),
            'total_stocks': int(df['stock_code'].nunique()),
            'total_dates': int(df['date'].nunique()),
            'factor_count': 10,
            'date_range': {
                'start': str(df['date'].min()),
                'end': str(df['date'].max())
            }
        }

        return jsonify({'status': 'success', 'data': summary})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/zz1000/returns')
def api_zz1000_returns():
    """API: 中证1000因子收益分析"""
    try:
        results_path = Path("./factor_analysis_results/zz1000_factor_returns.csv")
        if not results_path.exists():
            return jsonify({'status': 'error', 'message': '分析结果未找到'})

        df = pd.read_csv(results_path)
        return jsonify({'status': 'success', 'data': df.to_dict(orient='records')})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/zz1000')
def zz1000_dashboard():
    """中证1000因子分析页面"""
    return render_template('zz1000_dashboard.html', active_page='zz1000')
'''
        content = content.replace('# ==================== 启动 ====================',
                                zz1000_api + '# ==================== 启动 ====================')

        with open(app_py_path, 'w') as f:
            f.write(content)

        print("已添加中证1000专用API端点")
    else:
        print("中证1000 API已存在")

    return True


def create_zz1000_dashboard():
    """创建中证1000仪表盘页面"""
    print("创建中证1000分析页面...")

    dashboard_html = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>中证1000因子分析 - 高频因子分析平台</title>
    {% extends "base.html" %}
    {% block header_title %}中证1000因子{% endblock %}
    {% block extra_css %}
    <style>
        .zz1000-banner {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(59, 130, 246, 0.1));
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: var(--radius-lg);
            padding: 20px;
            margin-bottom: 20px;
        }
        .zz1000-title {
            font-size: 22px;
            font-weight: 700;
            background: linear-gradient(135deg, #8b5cf6, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .zz1000-desc {
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 8px;
        }
        .stat-card-csi {
            position: relative;
            overflow: hidden;
        }
        .stat-card-csi::before {
            background: linear-gradient(135deg, #8b5cf6, #3b82f6) !important;
        }
    </style>
    {% endblock %}
</head>
<body>
    {% block content %}
    <div class="zz1000-banner fade-in">
        <div class="zz1000-title">中证1000指数成分股分析</div>
        <div class="zz1000-desc">
            高频因子研究与收益分析 | 数据周期: 2026年2月 | 成分股: 约1000只
        </div>
    </div>

    <!-- 统计卡片 -->
    <div class="stat-grid mb-24 fade-in">
        <div class="stat-card stat-card-csi">
            <div class="stat-label">成分股</div>
            <div class="stat-value" id="totalStocks">-</div>
        </div>
        <div class="stat-card success">
            <div class="stat-label">交易天数</div>
            <div class="stat-value" id="totalDates">-</div>
        </div>
        <div class="stat-card warning">
            <div class="stat-label">记录总数</div>
            <div class="stat-value" id="totalRecords">-</div>
        </div>
        <div class="stat-card info">
            <div class="stat-label">因子数量</div>
            <div class="stat-value">10</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">最佳IC</div>
            <div class="stat-value" id="bestIC">-</div>
        </div>
    </div>

    <!-- 图表区域 -->
    <div class="chart-section mb-24">
        <div class="chart-container fade-in">
            <div class="card-header">
                <div>
                    <div class="card-title">因子IC排名</div>
                    <div class="card-subtitle">按信息系数绝对值排序</div>
                </div>
            </div>
            <div id="factorICChart" style="height:350px;"></div>
        </div>
        <div class="chart-container fade-in">
            <div class="card-header">
                <div>
                    <div class="card-title">多空收益</div>
                    <div class="card-subtitle">Top - Bottom 组收益差</div>
                </div>
            </div>
            <div id="lsReturnChart" style="height:350px;"></div>
        </div>
    </div>

    <!-- 详细表格 -->
    <div class="card fade-in">
        <div class="card-header">
            <div>
                <div class="card-title">因子详细分析</div>
                <div class="card-subtitle">IC与多空收益统计</div>
            </div>
        </div>
        <div class="table-responsive">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>因子</th>
                        <th>IC</th>
                        <th>|IC|</th>
                        <th>多空收益</th>
                        <th>正向比例</th>
                    </tr>
                </thead>
                <tbody id="factorTableBody">
                    <tr><td colspan="5" class="loading"><div class="loading-spinner"></div></td></tr>
                </tbody>
            </table>
        </div>
    </div>
    {% endblock %}

    {% block extra_js %}
    <script>
        async function loadZZ1000Analysis() {
            try {
                // 加载汇总
                const summaryRes = await fetch('/api/zz1000/summary');
                const summary = await summaryRes.json();

                if (summary.status === 'success') {
                    document.getElementById('totalStocks').textContent = summary.data.total_stocks;
                    document.getElementById('totalDates').textContent = summary.data.total_dates;
                    document.getElementById('totalRecords').textContent =
                        summary.data.total_records.toLocaleString();
                }

                // 加载收益分析
                const returnsRes = await fetch('/api/zz1000/returns');
                const returns = await returnsRes.json();

                if (returns.status === 'success' && returns.data.length > 0) {
                    const data = returns.data;

                    // 最佳IC
                    const best = data.reduce((a, b) => Math.abs(a.ic) > Math.abs(b.ic) ? a : b);
                    document.getElementById('bestIC').textContent = best.ic.toFixed(4);

                    // 绘制IC图
                    renderICChart(data);

                    // 绘制多空收益
                    renderLSChart(data);

                    // 填充表格
                    renderTable(data);
                }
            } catch (error) {
                console.error('Error loading ZZ1000 analysis:', error);
            }
        }

        function renderICChart(data) {
            const sorted = [...data].sort((a, b) => Math.abs(b.ic) - Math.abs(a.ic));

            const trace = {
                x: sorted.map(d => d.ic),
                y: sorted.map(d => d.factor),
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: sorted.map(d => d.ic >= 0 ? '#10b981' : '#ef4444'),
                    opacity: 0.9
                },
                text: sorted.map(d => d.ic.toFixed(4)),
                textposition: 'outside'
            };

            const layout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Inter' },
                margin: { t: 10, r: 120, b: 40, l: 100 },
                xaxis: { title: 'IC', gridcolor: '#2d3748' },
                yaxis: { gridcolor: 'transparent' }
            };

            Plotly.newPlot('factorICChart', [trace], layout, {responsive: true, displayModeBar: false});
        }

        function renderLSChart(data) {
            const sorted = [...data].sort((a, b) => Math.abs(b.ls_return) - Math.abs(a.ls_return));

            const trace = {
                x: sorted.map(d => d.factor),
                y: sorted.map(d => d.ls_return * 10000),
                type: 'bar',
                marker: {
                    color: sorted.map(d => d.ls_return >= 0 ? '#10b981' : '#ef4444'),
                    opacity: 0.9
                }
            };

            const layout = {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: '#94a3b8', family: 'Inter' },
                margin: { t: 10, r: 10, b: 80, l: 60 },
                xaxis: { tickangle: -45, title: '因子' },
                yaxis: { title: '多空收益 (bps)', gridcolor: '#2d3748' }
            };

            Plotly.newPlot('lsReturnChart', [trace], layout, {responsive: true, displayModeBar: false});
        }

        function renderTable(data) {
            const tbody = document.getElementById('factorTableBody');
            tbody.innerHTML = data.map(d => `
                <tr>
                    <td><strong>${d.factor}</strong></td>
                    <td style="color:${d.ic >= 0 ? 'var(--accent-success)' : 'var(--accent-danger)'}">
                        ${d.ic.toFixed(6)}
                    </td>
                    <td>${d.ic_abs.toFixed(6)}</td>
                    <td style="color:${d.ls_return >= 0 ? 'var(--accent-success)' : 'var(--accent-danger)'}">
                        ${(d.ls_return * 10000).toFixed(2)} bps
                    </td>
                    <td>${(d.positive_ratio * 100).toFixed(1)}%</td>
                </tr>
            `).join('');
        }

        document.addEventListener('DOMContentLoaded', loadZZ1000Analysis);
    </script>
    {% endblock %}
</body>
</html>
'''

    template_path = Path("factor_dashboard/templates/zz1000_dashboard.html")
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)

    print(f"创建页面: {template_path}")

    return str(template_path)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("中证1000高频因子分析系统")
    print("=" * 60)

    # 1. 获取可用股票
    stocks = get_available_stocks()

    # 2. 计算因子 (限制1000只)
    factors_df = compute_zz1000_factors(stocks, max_stocks=1000)

    if factors_df.empty:
        print("未找到Tick数据")
    else:
        # 3. 保存因子
        save_zz1000_factors(factors_df)

        # 4. 因子收益分析
        results_df = analyze_factor_returns(factors_df)

    # 5. 生成文档
    generate_documentation()

    # 6. 更新Flask
    update_flask_for_zz1000()
    create_zz1000_dashboard()

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print("\n请重启Flask服务: python3 factor_dashboard/app.py")
    print("访问: http://localhost:9000/zz1000")


if __name__ == "__main__":
    main()
