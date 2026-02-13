# 高频因子计算模块使用文档

## 概述

本模块提供基于Tick数据计算10个高频因子的功能，支持按日期批量处理所有股票。

## 安装依赖

```bash
pip install pandas numpy pyarrow
```

## 使用方法

### 1. 命令行模式

```bash
# 计算单日所有股票
python high_frequency_factors.py --date 2025-12-01

# 计算单只股票
python high_frequency_factors.py --date 2025-12-01 --stock 000001.SZ

# 计算多只股票
python high_frequency_factors.py --date 2025-12-01 --stock "000001.SZ,000002.SZ"

# 批量计算日期范围
python high_frequency_factors.py --start 2025-12-01 --end 2025-12-31

# 指定输出目录
python high_frequency_factors.py --date 2025-12-01 --output /path/to/output
```

### 2. Python API模式

```python
from high_frequency_factors import calc_high_frequency, calc_date_range

# 计算单日所有股票
df = calc_high_frequency('2025-12-01', 'all')

# 计算单只股票
df = calc_high_frequency('2025-12-01', '000001.SZ')

# 计算多只股票
df = calc_high_frequency('2025-12-01', ['000001.SZ', '000002.SZ'])

# 批量计算日期范围
results = calc_date_range('2025-12-01', '2025-12-31', 'all')
```

## 函数说明

### calc_high_frequency()

计算指定日期的高频因子数据

**参数：**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| date | str | 必填 | 日期，格式 'YYYY-MM-DD' 或 'YYYY/MM/DD' 或 'YYYYMMDD' |
| stock_code | str/list | 'all' | 股票代码，'all'=所有股票 |
| base_dir | str | /data1/quant-data/tick_2026 | tick数据根目录 |
| output_dir | str | factor/high_frequency | 输出目录 |

**返回值：**
- DataFrame: 包含股票代码和10个因子

### calc_date_range()

计算日期范围内所有交易日的高频因子

**参数：**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| start_date | str | 必填 | 起始日期 (YYYY-MM-DD) |
| end_date | str | 必填 | 结束日期 (YYYY-MM-DD) |
| stock_code | str/list | 'all' | 股票代码 |
| base_dir | str | /data1/quant-data/tick_2026 | tick数据根目录 |
| output_dir | str | factor/high_frequency | 输出目录 |
| verbose | bool | True | 是否打印进度 |

**返回值：**
- dict: {日期: DataFrame}

## 输出文件格式

文件保存路径: `/factor/high_frequency/{year}_{month}_{day}.parquet`

**数据格式：**

```
date         stock_code    bid_ask_spread  vwap_deviation  ...  trade_flow_intensity
2025-12-01   000001.SZ    0.0112         -0.9897        ...  45.23
2025-12-01   000002.SZ    0.0105         -0.9898        ...  32.15
...
```

**列说明：**

| 列名 | 类型 | 说明 |
|------|------|------|
| date | string | 交易日期 (YYYY-MM-DD) |
| stock_code | string | 股票代码 (如 000001.SZ) |
| bid_ask_spread | float | 买卖价差 |
| vwap_deviation | float | VWAP偏离度 |
| trade_imbalance | float | 交易失衡度 |
| order_imbalance | float | 订单失衡度 |
| depth_imbalance | float | 深度失衡度 |
| realized_volatility | float | 已实现波动率 |
| effective_spread | float | 有效价差 |
| micro_price | float | 微价格 |
| price_momentum | float | 价格动量 |
| trade_flow_intensity | float | 交易流强度 |

## 因子定义与计算公式

### 1. bid_ask_spread (买卖价差)

**公式:** `Spread = Ask1 - Bid1`

**解读:**
- 反映市场流动性和交易成本
- 值越小表示流动性越好

### 2. vwap_deviation (VWAP偏离度)

**公式:** `Dev = (Price - VWAP) / VWAP`

**解读:**
- 反映价格相对于均值的偏离程度
- 正值表示价格高于VWAP，负值表示低于VWAP

### 3. trade_imbalance (交易失衡度)

**公式:** 基于价格变动判断主动买入力度

**解读:**
- Price > MicroPrice: 主动买入
- Price < MicroPrice: 主动卖出

### 4. order_imbalance (订单失衡度)

**公式:** `OI = (BidVol - AskVol) / (BidVol + AskVol)`

**解读:**
- OI > 0: 买方压力较大
- OI < 0: 卖方压力较大

### 5. depth_imbalance (深度失衡度)

**公式:** `Depth_Imb = (BidVol1 - AskVol1) / (BidVol1 + AskVol1)`

**解读:**
- 衡量买卖盘深度的平衡程度

### 6. realized_volatility (已实现波动率)

**公式:** `RV = sqrt(sum(returns^2))`

**解读:**
- 基于高频收益率计算的波动率
- 值越大表示短期波动越剧烈

### 7. effective_spread (有效价差)

**公式:** `ES = 2 * |MidPrice - TradePrice| / MidPrice`

**解读:**
- 有效价差反映实际成交成本
- 值越大表示成交成本越高

### 8. micro_price (微价格)

**公式:** `MicroPrice = (Bid1 * AskVol1 + Ask1 * BidVol1) / (BidVol1 + AskVol1)`

**解读:**
- 考虑订单簿平衡的预期价格
- 比简单中间价更能反映真实价格走向

### 9. price_momentum (价格动量)

**公式:** `Mom = Price_t / Price_{t-n} - 1`

**解读:**
- 正值表示上涨动量，负值表示下跌动量

### 10. trade_flow_intensity (交易流强度)

**公式:** `TFI = tick_vol的滚动均值`

**解读:**
- 衡量单位时间内的交易活跃程度
- 值越大表示交易越活跃

## 数据源说明

**Tick数据目录结构:**
```
/data1/quant-data/tick_2026/
├── 2025/
│   ├── 12/
│   │   ├── 01/
│   │   │   ├── 000001.SZ.parquet
│   │   │   ├── 000002.SZ.parquet
│   │   │   └── ...
```

## 示例完整代码

```python
from high_frequency_factors import calc_high_frequency, calc_date_range
import pandas as pd

# 示例1: 计算单日所有股票
print("=" * 60)
print("计算 2025-12-01 所有股票的高频因子")
print("=" * 60)

df = calc_high_frequency('2025-12-01', 'all')
print(f"\n结果形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

# 查看因子统计
print("\n因子统计:")
print(df.describe())

# 示例2: 计算单只股票
print("\n" + "=" * 60)
print("计算 000001.SZ 在 2025-12-01 的高频因子")
print("=" * 60)

df_stock = calc_high_frequency('2025-12-01', '000001.SZ')
print(df_stock)

# 示例3: 批量计算
print("\n" + "=" * 60)
print("批量计算 2025-12-01 到 2025-12-03")
print("=" * 60)

results = calc_date_range('2025-12-01', '2025-12-03', 'all', verbose=True)
for date, df in results.items():
    print(f"\n{date}: {len(df)} 股票")
```

## 输出文件位置

计算结果保存在以下目录：

| 日期 | 文件路径 |
|------|----------|
| 2025-12-01 | `/factor/high_frequency/2025_12_01.parquet` |
| 2025-12-02 | `/factor/high_frequency/2025_12_02.parquet` |
| ... | ... |

## 注意事项

1. **数据量**: 每天约处理5000-6000只股票
2. **处理时间**: 单日处理约需3-5分钟
3. **内存占用**: 建议至少8GB内存
4. **错误处理**: 遇到重复标签等错误会自动跳过

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-02-09 | 初始版本，支持10个高频因子 |
