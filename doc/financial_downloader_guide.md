# Tushare 财务报表数据下载

从 Tushare 下载现金流量表、资产负债表、利润表季度数据。

## 目录

- [功能特性](#功能特性)
- [安装配置](#安装配置)
- [命令行使用](#命令行使用)
- [Python API](#python-api)
- [数据结构](#数据结构)
- [示例](#示例)

## 功能特性

- 支持下载 2015-2025 年季度财务报表
- 自动处理财报发布时间（年报一般在次年1-4月发布）
- 跳过已存在的文件（增量更新）
- 自动合并为 `_all.parquet` 文件
- 支持指定字段下载

## 安装配置

### 安装依赖

```bash
pip install tushare pandas pyarrow
```

### 配置 Token

```bash
# 方式一：环境变量
export TUSHARE_TOKEN='your_token'

# 方式二：配置文件
echo 'your_token' > ~/.tushare_token
```

## 命令行使用

### 下载全部数据（2015-2025）

```bash
python financial_downloader.py --all
```

### 下载指定时间范围

```bash
# 下载 2020-2025 年数据
python financial_downloader.py --start 20200101 --end 20251231

# 下载 2025 年数据
python financial_downloader.py --start 20250101 --end 20251231
```

### 只下载特定报表

```bash
# 只下载现金流量表
python financial_downloader.py --all --cashflow

# 只下载资产负债表
python financial_downloader.py --all --balance

# 只下载利润表
python financial_downloader.py --all --income

# 下载现金流量表和资产负债表
python financial_downloader.py --all --cashflow --balance
```

### 按季度下载

```bash
# 下载指定季度
python financial_downloader.py --quarters 20250630 20250331 20241231

# 下载中报和年报
python financial_downloader.py --quarters 20250630 20251231
```

### 其他选项

```bash
# 强制重新下载（覆盖已存在的文件）
python financial_downloader.py --start 20250101 --end 20251231 --no-skip

# 合并为一个大文件
python financial_downloader.py --all --merge

# 指定输出目录
python financial_downloader.py --all --output-dir ./my_financial_data
```

### 命令行参数

| 参数 | 短参数 | 说明 |
|------|--------|------|
| `--all` | - | 下载 2015-2025 年全部数据 |
| `--start` | `-s` | 开始日期 YYYYMMDD |
| `--end` | `-e` | 结束日期 YYYYMMDD |
| `--quarters` | `-q` | 指定季度列表 |
| `--cashflow` | - | 只下载现金流量表 |
| `--balance` | - | 只下载资产负债表 |
| `--income` | - | 只下载利润表 |
| `--output-dir` | `-o` | 输出目录 |
| `--merge` | - | 合并为一个大文件 |
| `--no-skip` | - | 不跳过已存在的文件 |

## Python API

### 导入

```python
from financial_downloader import (
    download_cashflow,
    download_balance,
    download_income,
    download_all_financials,
    download_by_quarters,
    merge_to_all,
    set_token
)
```

### 下载全部报表

```python
from financial_downloader import download_all_financials

# 设置 Token
set_token('your_token')

# 下载 2015-2025 年全部数据
download_all_financials(
    start_date='20150101',
    end_date='20251231'
)
```

### 只下载现金流量表

```python
from financial_downloader import download_cashflow

download_cashflow(
    start_date='20150101',
    end_date='20251231',
    output_dir='./daily_data/cashflow'
)
```

### 按季度下载

```python
from financial_downloader import download_by_quarters

# 下载指定季度
download_by_quarters(
    quarters=['20250630', '20250331', '20241231']
)
```

### 合并数据

```python
from financial_downloader import merge_to_all

# 合并所有季度数据为一个大文件
merge_to_all('./daily_data')
```

## 数据结构

### 文件命名

| 报表类型 | 文件格式 | 示例 |
|---------|---------|------|
| 现金流量表 | `cashflow_YYYYMMDD.parquet` | `cashflow_20250630.parquet` |
| 资产负债表 | `balance_YYYYMMDD.parquet` | `balance_20250630.parquet` |
| 利润表 | `income_YYYYMMDD.parquet` | `income_20250630.parquet` |
| 合并文件 | `{table}_all.parquet` | `cashflow_all.parquet` |

### 现金流量表字段 (cashflow)

| 字段 | 说明 |
|------|------|
| ts_code | 股票代码 |
| ann_date | 公告日期 |
| f_ann_date | 最终公告日期 |
| end_date | 报告期结束日期 |
| report_type | 报告类型 |
| comp_type | 公司类型 |
| n_cashflow_act | 经营活动产生的现金流量净额 |
| n_cashflow_inv_act | 投资活动产生的现金流量净额 |
| n_cash_flows_fnc_act | 筹资活动产生的现金流量净额 |
| c_fr_sale_sg | 销售商品、提供劳务收到的现金 |
| c_paid_goods_s | 购买商品、接受劳务支付的现金 |
| c_paid_to_for_empl | 支付给职工以及为职工支付的现金 |

### 资产负债表字段 (balance)

| 字段 | 说明 |
|------|------|
| ts_code | 股票代码 |
| ann_date | 公告日期 |
| f_ann_date | 最终公告日期 |
| end_date | 报告期结束日期 |
| report_type | 报告类型 |
| total_assets | 资产总计 |
| total_liab | 负债合计 |
| total_hldr_eqy_exc_min_int | 归属于母公司所有者权益合计 |
| total_cur_assets | 流动资产合计 |
| total_nca | 非流动资产合计 |
| cash_reser_cb | 货币资金 |
| trad_asset | 交易性金融资产 |
| notes_receiv | 应收票据 |
| accounts_receiv | 应收账款 |
| inventories | 存货 |
| fix_assets | 固定资产 |
| intan_assets | 无形资产 |

### 利润表字段 (income)

| 字段 | 说明 |
|------|------|
| ts_code | 股票代码 |
| ann_date | 公告日期 |
| f_ann_date | 最终公告日期 |
| end_date | 报告期结束日期 |
| report_type | 报告类型 |
| total_revenue | 营业总收入 |
| revenue | 营业收入 |
| int_income | 利息收入 |
| operate_profit | 营业利润 |
| total_profit | 利润总额 |
| income_tax | 所得税费用 |
| n_income | 净利润 |
| n_income_attr_p | 归属于母公司的净利润 |
| basic_eps | 基本每股收益 |
| diluted_eps | 稀释每股收益 |

## 示例

### 完整下载流程

```python
from financial_downloader import (
    download_all_financials,
    merge_to_all,
    set_token
)

# 1. 设置 Token
set_token('your_token')

# 2. 下载全部数据
download_all_financials(
    start_date='20150101',
    end_date='20251231'
)

# 3. 合并数据
merge_to_all('./daily_data')
```

### 获取特定股票数据

```python
import pandas as pd
import pyarrow.parquet as pq

# 读取合并文件
df = pq.read_table('./daily_data/cashflow/cashflow_all.parquet').to_pandas()

# 过滤特定股票
stock_df = df[df['ts_code'].isin(['000001.SZ', '000002.SZ'])]
print(stock_df)
```

### 使用 get_local_data 读取数据

```python
from mylib.get_local_data import get_local_data

# 注意：get_local_data 用于日线数据，季度财务数据直接用 pandas 读取
import pyarrow.parquet as pq

df = pq.read_table('./daily_data/cashflow/cashflow_20250630.parquet').to_pandas()
print(df.head())
```

## 注意事项

1. **Tushare 积分限制**: 财务报表接口需要一定积分才能使用
2. **下载频率**: 脚本内置 0.3 秒延时，避免触发限流
3. **财报发布时间**:
   - Q1 (0331): 4月底前发布
   - Q2 (0630): 8月底前发布
   - Q3 (0930): 10月底前发布
   - Q4 (1231): 次年4月底前发布（年报）
4. **增量更新**: 使用 `--no-skip` 强制重新下载已存在的文件
