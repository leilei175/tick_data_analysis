# 现金流季度数据转每日数据

将 Tushare 的 `cashflow` 季度数据转换为每日数据，保存为 Parquet 格式。

## 目录

- [背景](#背景)
- [配置](#配置)
- [命令行使用](#命令行使用)
- [Python API](#python-api)
- [公告日期配置](#公告日期配置)
- [示例](#示例)

## 背景

Tushare 的 `cashflow` 接口返回的是**季度财务数据**：

| ts_code | ann_date | end_date | n_cashflow_act | ... |
|---------|----------|----------|----------------|-----|
| 000001.SZ | 20250715 | 20250630 | 1234567.89 | ... |

实际使用时，我们希望每天都显示最新的财报数据：

| 日期 | 000001.SZ 经营现金流 |
|------|---------------------|
| 2025-01-01 | 2024Q4 数据 |
| 2025-07-01 | 2025Q1 数据 |
| 2025-10-15 | 2025Q3 数据 |

## 配置

### 修改公告日期

在脚本中配置每个季度财报的公告日期：

```python
# cashflow_daily_converter.py 第 46-52 行

ANNOUNCEMENT_DATES: Dict[str, str] = {
    '2025-02-28': '20241231',  # 2024年年报，2月28日发布
    '2025-04-30': '20250331',  # 2025年一季报，4月30日发布
    '2025-08-30': '20250630',  # 2025年中报，8月30日发布
    '2025-10-10': '20250930',  # 2025年三季报，10月10日发布
}
```

**配置格式说明：**
- 键 (Key): 财报发布日期，格式 `YYYY-MM-DD`
- 值 (Value): 财报对应的季度结束日期，格式 `YYYYMMDD`

## 命令行使用

```bash
# 转换 2025 年全年数据
python cashflow_daily_converter.py --start 20250101 --end 20251231

# 指定自定义公告日期
python cashflow_daily_converter.py \
    --start 20250101 \
    --end 20251231 \
    --ann-date 20250228:20241231 \
    --ann-date 20250430:20250331 \
    --ann-date 20250830:20250630 \
    --ann-date 20251010:20250930

# 仅更新模式（当新财报发布后）
python cashflow_daily_converter.py \
    --start 20251010 \
    --end 20251231 \
    --ann-date 20251010:20250930 \
    --update
```

### 命令行参数

| 参数 | 短参数 | 说明 |
|------|--------|------|
| `--start` | `-s` | 开始日期，必填 |
| `--end` | `-e` | 结束日期，必填 |
| `--cashflow-dir` | `-c` | 季度数据目录，默认 `./daily_data/cashflow/` |
| `--output-dir` | `-o` | 输出目录，默认 `./daily_data/cashflow_daily/` |
| `--ann-date` | - | 自定义公告日期，可多次指定 |
| `--update` | - | 仅更新模式 |
| `--fetch-missing` | - | 自动下载缺失数据 |

## Python API

### 导入

```python
from cashflow_daily_converter import (
    convert_cashflow_to_daily,
    update_cashflow_daily,
    build_announcement_map
)
```

### 完整转换

```python
from cashflow_daily_converter import convert_cashflow_to_daily

# 转换 2025 年全年数据
result = convert_cashflow_to_daily(
    start_date='20250101',
    end_date='20251231',
    cashflow_dir='./daily_data/cashflow/',
    output_dir='./daily_data/cashflow_daily/'
)
print(result.head())
```

### 更新每日数据

当新财报发布时，只更新公告日期之后的数据：

```python
from cashflow_daily_converter import update_cashflow_daily

# 假设 2025年三季报 10月10日发布
update_cashflow_daily(
    start_date='20251010',
    end_date='20251231',
    announcement_date='20251010',
    quarter_end_date='20250930',
    cashflow_dir='./daily_data/cashflow/',
    cashflow_daily_dir='./daily_data/cashflow_daily/'
)
```

### 自定义公告日期

```python
from cashflow_daily_converter import convert_cashflow_to_daily

custom_announcements = {
    '2025-02-28': '20241231',  # 年报
    '2025-04-30': '20250331',  # Q1
    '2025-08-30': '20250630',  # Q2
    '2025-10-10': '20250930',  # Q3
}

convert_cashflow_to_daily(
    start_date='20250101',
    end_date='20251231',
    announcement_dates=custom_announcements
)
```

## 公告日期配置

### 配置规则

1. **公告日期 (ann_date)**: 上市公司发布财报的实际日期
2. **季度日期 (end_date)**: 财报覆盖的季度最后一天

| 财报类型 | end_date | ann_date 示例 | 数据含义 |
|---------|----------|---------------|----------|
| 年报 | 20241231 | 2025-02-28 | 2024年全年数据 |
| 一季报 | 20250331 | 2025-04-30 | 2025年Q1数据 |
| 中报 | 20250630 | 2025-08-30 | 2025年Q1-Q2数据 |
| 三季报 | 20250930 | 2025-10-10 | 2025年Q1-Q3数据 |

### 数据覆盖规则

```
交易日 >= 公告日期  --> 使用对应季度数据
交易日 <  公告日期  --> 使用最近一次已发布的季度数据
```

示例：
- 2025-10-09 及之前：使用 Q2 (20250630) 数据
- 2025-10-10 及之后：使用 Q3 (20250930) 数据

## 数据结构

### 输出格式

每日现金流数据保存在 `cashflow_daily/` 目录：

| 文件名 | 内容 |
|--------|------|
| `cashflow_daily_YYYYMMDD.parquet` | 单日所有股票的现金流数据 |

### 数据列

| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | string | 股票代码 |
| trade_date | string | 交易日期 YYYYMMDD |
| n_cashflow_act | double | 经营活动产生的现金流量净额 |
| n_cashflow_inv_act | double | 投资活动产生的现金流量净额 |
| n_cash_flows_fnc_act | double | 筹资活动产生的现金流量净额 |
| c_fr_sale_sg | double | 销售商品、提供劳务收到的现金 |
| c_paid_goods_s | double | 购买商品、接受劳务支付的现金 |
| c_paid_to_for_empl | double | 支付给职工以及为职工支付的现金 |
| c_recp_borrow | double | 取得借款收到的现金 |
| proc_issue_bonds | double | 发行债券收到的现金 |

## 与 get_local_data 配合使用

```python
from mylib.get_local_data import get_local_data

# 获取经营活动现金流
cf_ops = get_local_data(
    sec_list=['000001.SZ', '000002.SZ'],
    start='20250101',
    end='20251231',
    filed='n_cashflow_act',
    data_dir='./daily_data/cashflow_daily',
    data_type='cashflow_daily'  # 需要在 get_local_data 中添加此类型
)
print(cf_ops)
```

## 注意事项

1. **公告日期需提前配置**: 在财报发布前配置好公告日期
2. **交易日历**: 当前使用自然日，实际使用时应替换为交易日历
3. **数据更新**: 新财报发布后，使用 `--update` 模式更新后续数据
4. **缺失数据**: 如果某季度数据缺失，脚本会从 `cashflow_all.parquet` 尝试提取
