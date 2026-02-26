# 本地数据读取指南

> 文档更新时间: 2026-02-13

---

## 目录

1. [数据目录结构](#1-数据目录结构)
2. [数据类型与字段](#2-数据类型与字段)
3. [读取函数](#3-读取函数)
4. [使用示例](#4-使用示例)
5. [性能测试](#5-性能测试)
6. [数据转换](#6-数据转换)

---

## 1. 数据目录结构

```
daily_data/
├── daily/                      # 日线数据（交易数据）
│   ├── 2020_full.parquet
│   ├── 2021_full.parquet
│   ├── ...
│   └── 2026_full.parquet
│
├── daily_basic/                # 每日基本面数据
│   ├── 2020_full.parquet
│   ├── 2021_full.parquet
│   └── ...
│
├── cashflow/                   # 现金流量表（季度原始数据）
│   ├── 2020/03/cashflow_20200331.parquet
│   ├── 2020/06/cashflow_20200630.parquet
│   └── ...
│
├── cashflow_daily/             # 现金流量表（每日填充数据）
│   ├── 2024_full.parquet
│   └── 2025_full.parquet
│
├── income/                     # 利润表（季度原始数据）
│   ├── 2020/03/income_20200331.parquet
│   └── ...
│
├── income_daily/               # 利润表（每日填充数据）
│   ├── 2024_full.parquet
│   └── 2025_full.parquet
│
├── balance/                    # 资产负债表（季度原始数据）
│   ├── 2020/03/balance_20200331.parquet
│   └── ...
│
└── balance_daily/              # 资产负债表（每日填充数据）
    ├── 2024_full.parquet
    └── 2025_full.parquet
```

### 文件命名规则

| 文件类型 | 命名格式 | 说明 |
|----------|----------|------|
| Annual合并文件 | `{year}_full.parquet` | 按年合并的完整数据 |
| 原始日线文件 | `daily_{YYYYMMDD}.parquet` | 按日存储的日线数据 |
| 季度数据文件 | `{table}_{YYYYMMDD}.parquet` | 按季度存储的财务数据 |

---

## 2. 数据类型与字段

### 2.1 daily - 日线数据

| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 股票代码 (如 '000001.SZ') |
| trade_date | int | 交易日期 (YYYYMMDD格式) |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| pre_close | float | 昨收盘价 |
| change | float | 涨跌额 |
| pct_chg | float | 涨跌幅 (%) |
| vol | float | 成交量 (手) |
| amount | float | 成交额 (千元) |

**数据范围:** 2020-01-02 ~ 2026-02-10

### 2.2 daily_basic - 每日基本面数据

| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 股票代码 |
| trade_date | int | 交易日期 |
| close | float | 收盘价 |
| turnover_rate | float | 换手率 (%) |
| turnover_rate_f | float | 换手率 (自由流通股) |
| volume_ratio | float | 量比 |
| pe | float | 市盈率 (PE) |
| pe_ttm | float | 市盈率 (TTM) |
| pb | float | 市净率 (PB) |
| ps | float | 市销率 (PS) |
| ps_ttm | float | 市销率 (TTM) |
| dv_ratio | float | 股息率 (%) |
| dv_ttm | float | 股息率 (TTM) |
| total_share | float | 总股本 (万股) |
| float_share | float | 流通股本 (万股) |
| free_share | float | 限售股本 (万股) |
| total_mv | float | 总市值 (万元) |
| circ_mv | float | 流通市值 (万元) |

**数据范围:** 2020-01-02 ~ 2026-02-10

### 2.3 cashflow_daily - 现金流量表（每日）

| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 股票代码 |
| trade_date | int | 交易日期 |
| end_date | int | 报告期结束日期 |
| report_type | int | 报告类型 |
| n_cashflow_act | float | 经营活动产生的现金流量净额 |
| n_cashflow_inv_act | float | 投资活动产生的现金流量净额 |
| n_cash_flows_fnc_act | float | 筹资活动产生的现金流量净额 |
| c_fr_sale_sg | float | 销售商品、提供劳务收到的现金 |
| c_paid_goods_s | float | 购买商品、接受劳务支付的现金 |
| c_paid_to_for_empl | float | 支付给职工以及为职工支付的现金 |
| c_recp_borrow | float | 取得借款收到的现金 |
| proc_issue_bonds | float | 发行债券收到的现金 |
| net_profit | float | 净利润 |
| depr_fa_coga_dpba | float | 固定资产折旧、油气资产折耗、生产性生物资产折旧 |
| amint_intang_assets | float | 无形资产摊销 |

**数据范围:** 2024-01-02 ~ 2025-12-31

### 2.4 income_daily - 利润表（每日）

| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 股票代码 |
| trade_date | int | 交易日期 |
| end_date | int | 报告期结束日期 |
| report_type | int | 报告类型 |
| total_revenue | float | 营业总收入 |
| revenue | float | 营业收入 |
| int_income | float | 利息收入 |
| comm_income | float | 手续费及佣金收入 |
| oper_cost | float | 营业成本 |
| operate_profit | float | 营业利润 |
| total_profit | float | 利润总额 |
| income_tax | float | 所得税费用 |
| n_income | float | 净利润 |
| n_income_attr_p | float | 归属于母公司的净利润 |
| minority_gain | float | 少数股东损益 |
| basic_eps | float | 基本每股收益 |
| diluted_eps | float | 稀释每股收益 |
| ebit | float | 息税前利润 (EBIT) |
| ebitda | float | 息税折旧摊销前利润 (EBITDA) |

**数据范围:** 2024-01-02 ~ 2025-12-31

### 2.5 balance_daily - 资产负债表（每日）

| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | str | 股票代码 |
| trade_date | int | 交易日期 |
| end_date | int | 报告期结束日期 |
| report_type | int | 报告类型 |
| total_assets | float | 资产总计 |
| total_liab | float | 负债合计 |
| total_hldr_eqy_exc_min_int | float | 归属于母公司所有者权益合计 |
| total_cur_assets | float | 流动资产合计 |
| total_nca | float | 非流动资产合计 |
| total_cur_liab | float | 流动负债合计 |
| total_ncl | float | 非流动负债合计 |
| cash_reser_cb | float | 货币资金 |
| trad_asset | float | 交易性金融资产 |
| notes_receiv | float | 应收票据 |
| accounts_receiv | float | 应收账款 |
| inventories | float | 存货 |
| fix_assets | float | 固定资产 |
| intan_assets | float | 无形资产 |
| st_borr | float | 短期借款 |
| lt_borr | float | 长期借款 |
| notes_payable | float | 应付票据 |
| bond_payable | float | 应付债券 |

**数据范围:** 2024-01-02 ~ 2025-12-31

---

## 3. 读取函数

### 3.1 get_all_data() - 主读取函数

```python
from mylib.get_local_data import get_all_data

data = get_all_data(
    data_type: str,          # 数据类型
    start: str = None,       # 开始日期 (YYYYMMDD)
    end: str = None,         # 结束日期 (YYYYMMDD)
    sec_list: List[str] = None,  # 股票代码列表
    fields: List[str] = None,     # 字段列表
    parallel: bool = True    # 是否并行读取
) -> Dict[str, pd.DataFrame]
```

**参数说明:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| data_type | str | 是 | 数据类型: 'daily', 'daily_basic', 'cashflow_daily', 'income_daily', 'balance_daily' |
| start | str | 是 | 开始日期，格式 'YYYYMMDD' |
| end | str | 是 | 结束日期，格式 'YYYYMMDD' |
| sec_list | List[str] | 否 | 股票代码列表，如 ['000001.SZ', '000002.SZ'] |
| fields | List[str] | 否 | 要读取的字段列表，None表示所有字段 |
| parallel | bool | 否 | 是否并行读取，默认True |

**返回值:**

返回字典，key为字段名，value为DataFrame:
- Index: 交易日 (DatetimeIndex)
- Columns: 股票代码 (str)
- Values: 对应字段的值

### 3.2 便捷函数

```python
# 获取日线数据
data = get_daily_all(start='20200101', end='20201231')

# 获取每日基本面数据
data = get_daily_basic_all(start='20200101', end='20201231')

# 获取单个字段
data = get_local_data('close', 'daily', start='20200101', end='20201231')
```

### 3.3 完整API列表

| 函数 | 说明 |
|------|------|
| `get_all_data(data_type, ...)` | 通用读取函数，返回字典 |
| `get_daily_all(start, end, ...)` | 读取日线所有字段 |
| `get_daily_basic_all(start, end, ...)` | 读取基本面所有字段 |
| `get_local_data(field, data_type, ...)` | 读取单个字段 |

---

## 4. 使用示例

### 4.1 读取所有日线数据

```python
from mylib.get_local_data import get_all_data

# 获取2020年日线数据
data = get_all_data('daily', start='20200101', end='20201231')

# 查看结果
print(f"字段数: {len(data)}")
print(f"close shape: {data['close'].shape}")
print(f"日期范围: {data['close'].index[0]} ~ {data['close'].index[-1]}")

# 访问具体数据
close_df = data['close']      # 收盘价 DataFrame
open_df = data['open']        # 开盘价 DataFrame
```

### 4.2 读取单个字段

```python
from mylib.get_local_data import get_local_data

# 只获取收盘价
close = get_local_data(
    field='close',
    data_type='daily',
    start='20200101',
    end='20201231'
)

print(close.shape)  # (交易日数, 股票数)
```

### 4.3 筛选特定股票

```python
# 只获取指定股票
data = get_all_data(
    'daily',
    start='20200101',
    end='20201231',
    sec_list=['000001.SZ', '600000.SH', '300750.SZ']
)

print(f"股票数: {len(data['close'].columns)}")
```

### 4.4 筛选特定字段

```python
# 只获取PE和PB
data = get_all_data(
    'daily_basic',
    start='20200101',
    end='20201231',
    fields=['pe', 'pb']
)

print(f"字段: {list(data.keys())}")  # ['pe', 'pb']
```

### 4.5 读取财务数据

```python
# 获取现金流量数据
cashflow_data = get_all_data(
    'cashflow_daily',
    start='20240101',
    end='20240630'
)

# 获取利润表数据
income_data = get_all_data(
    'income_daily',
    start='20240101',
    end='20240630'
)

# 获取资产负债表数据
balance_data = get_all_data(
    'balance_daily',
    start='20240101',
    end='20240630'
)
```

### 4.6 访问DataFrame数据

```python
# 获取收盘价DataFrame
close_df = data['close']

# 获取某只股票的数据
stock_close = close_df['000001.SZ']

# 获取某天的数据
day_close = close_df.loc['2020-01-02']

# 计算收益率
returns = close_df.pct_change()
```

---

## 5. 性能测试

### 5.1 读取性能

| 数据类型 | 数据量 | 读取时间 |
|----------|--------|----------|
| daily | 723万条 (1482交易日 × 5693股票 × 9字段) | **3.7秒** |
| daily_basic | 850万条 (1482交易日 × 5693股票 × 16字段) | **4.6秒** |
| cashflow_daily | 131万条 (242交易日 × 5471股票 × 8字段) | **0.5秒** |
| income_daily | 131万条 (242交易日 × 5471股票 × 11字段) | **0.5秒** |
| balance_daily | 131万条 (242交易日 × 5476股票 × 10字段) | **0.5秒** |

### 5.2 性能优化说明

- **Annual文件**: 每个年份的数据合并为单个文件，减少I/O开销
- **并行读取**: 使用ThreadPoolExecutor并行读取多个年份
- **批量Unstack**: 使用一次性展开所有字段，比逐个字段快6倍

### 5.3 加速建议

1. **指定日期范围**: 只读取需要的日期范围
2. **筛选股票**: 使用sec_list限制股票范围
3. **选择字段**: 使用fields参数只读取需要的字段

```python
# 最快方式：只读取需要的
data = get_all_data(
    'daily',
    start='20250101',
    end='20250131',
    fields=['close']
)
```

---

## 6. 数据转换

### 6.1 季度数据转每日数据

将原始季度财务数据转换为每日填充数据：

```python
# 方法1: 使用命令行
python convert_quarterly_to_daily.py --start 20240101 --end 20241231 --all

# 方法2: 使用API
from convert_quarterly_to_daily import convert_all

convert_all(
    start_date='20240101',
    end_date='20241231',
    tables=['cashflow', 'income', 'balance'],
    save_annual=True
)
```

### 6.2 手动创建Annual文件

对于已经分散存储的日数据，可以合并为Annual文件：

```python
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

DATA_DIR = './daily_data/daily/'
year = '2020'

# 读取该年所有文件
files = list(Path(DATA_DIR).glob(f'{year}*/**/*.parquet'))
dfs = [pd.read_parquet(f) for f in files]

# 合并
df = pd.concat(dfs, ignore_index=True)
df = df.drop_duplicates(subset=['trade_date', 'ts_code'], keep='first')
df = df.sort_values('trade_date')

# 保存Annual文件
output = Path(DATA_DIR) / f'{year}_full.parquet'
table = pa.Table.from_pandas(df, preserve_index=False)
pq.write_table(table, output)
```

---

## 常见问题

### Q1: 如何查看有哪些股票？

```python
data = get_all_data('daily', start='20200101', end='20200110')
stocks = data['close'].columns.tolist()
print(f"股票数: {len(stocks)}")
print(stocks[:10])  # 前10只股票
```

### Q2: 数据缺失怎么办？

```python
# 检查缺失值
data = get_all_data('daily', start='20200101', end='20200110')
close = data['close']

# 缺失值统计
missing = close.isna().sum()
print(f"缺失率: {(missing / close.notna().sum() * 100).round(2)}%")
```

### Q3: 如何处理复权？

当前数据为不复权数据。如需复权处理：

```python
# 获取原始收盘价
data = get_all_data('daily', start='20200101', end='20201231')
close = data['close']

# 需要使用Tushare获取复权因子进行计算
# 或使用专业数据库的复权数据
```

### Q4: 报错 "Unknown data type"

确保使用正确的类型名称：

```python
# 正确
get_all_data('daily', ...)
get_all_data('daily_basic', ...)
get_all_data('cashflow_daily', ...)
get_all_data('income_daily', ...)
get_all_data('balance_daily', ...)

# 错误（不要加空格或其他字符）
# get_all_data(' daily', ...)  # 错误
# get_all_data('dailybasic', ...)  # 错误
```

---

## 附录

### A. 数据来源

- **日线数据**: Tushare Pro `daily` 接口
- **基本面数据**: Tushare Pro `daily_basic` 接口
- **财务报表**: Tushare Pro `cashflow`, `income`, `balance` 接口

### B. 相关文件

| 文件 | 说明 |
|------|------|
| `mylib/get_local_data.py` | 数据读取模块 |
| `mylib/constants.py` | 常量定义 |
| `convert_quarterly_to_daily.py` | 季度转每日数据工具 |
| `update_data.py` | 数据更新工具 |

### C. 依赖

```
pandas>=1.0
pyarrow>=5.0
tushare>=1.2
```

---

*文档版本: 1.0*
*最后更新: 2026-02-13*
