# 中证1000高频因子分析报告

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

- **生成时间**: 2026-02-08 12:25:06
- **分析周期**: 2026年2月2日 - 2026年2月6日
- **因子数量**: 10个

---
*本报告由量化因子分析系统自动生成*
