# 财务报表季度数据转每日数据

将季度财务报表转换为每日数据，根据财报公告日期决定每天使用哪个季度的数据。

## 目录

- [背景](#背景)
- [使用方式](#使用方式)
- [命令行使用](#命令行使用)
- [Python API](#python-api)
- [公告日期配置](#公告日期配置)
- [数据结构](#数据结构)

## 背景

财务报表按季度披露，例如：
- 2024年年报：2025年2月28日发布
- 2025年一季报：2025年4月30日发布
- 2025年中报：2025年8月30日发布
- 2025年三季报：2025年10月10日发布

转换后：
| 日期 | 使用数据 |
|------|---------|
| 2025-01-01 ~ 2025-02-27 | 2024年报 (20241231) |
| 2025-02-28 ~ 2025-04-29 | 2025一季报 (20250331) |
| 2025-04-30 ~ 2025-08-29 | 2025中报 (20250630) |
| 2025-08-30 ~ 2025-10-09 | 2025中报 (20250630) |
| 2025-10-10 ~ 2025-12-31 | 2025三季报 (20250930) |

## 使用方式

### 1. 配置公告日期

在 `financial_daily_converter.py` 中修改 `ANNOUNCEMENT_DATES` 配置：

```python
ANNOUNCEMENT_DATES: Dict[str, Dict[str, str]] = {
    'cashflow': {
        '20250228': '20241231',  # 2024年年报
        '20250430': '20250331',  # 2025年一季报
        '20250830': '20250630',  # 2025年中报
        '20251010': '20250930',  # 2025年三季报
    },
    'income': {...},
    'balance': {...},
}
```

### 2. 运行转换

```bash
# 转换全部三张表
python financial_daily_converter.py --start 20250101 --end 20251231 --all

# 只转换现金流量表
python financial_daily_converter.py --start 20250101 --end 20251231 --cashflow

# 跳过已存在的文件（增量更新）
python financial_daily_converter.py --start 20250101 --end 20251231 --all --skip
```

## 命令行使用

```bash
# 全部三张表
python financial_daily_converter.py -s 20250101 -e 20251231 --all

# 指定表
python financial_daily_converter.py -s 20250101 -e 20251231 --cashflow --income

# 使用默认配置（自动生成公告日期）
python financial_daily_converter.py -s 20250101 -e 20251231 --all
```

## Python API

```python
from financial_daily_converter import convert_to_daily, convert_table_to_daily

# 转换全部三张表
convert_to_daily(
    start_date='20250101',
    end_date='20251231',
    tables=['cashflow', 'income', 'balance'],
    skip_existing=True
)

# 单独转换一张表
convert_table_to_daily(
    table_name='cashflow',
    start_date='20250101',
    end_date='20251231',
    data_dir='./daily_data/cashflow/',
    output_dir='./daily_data/cashflow_daily/',
    use_auto_ann=True  # 自动生成公告日期
)
```

### 更新每日数据（新财报发布后）

```python
from financial_daily_converter import update_to_latest_quarter

# 当新财报发布后，更新后续每日数据
update_to_latest_quarter(
    table_name='cashflow',
    announcement_date='20251010',  # 公告日期
    quarter_end_date='20250930',   # 财报结束日期
    start_date='20251010',         # 从哪天开始更新
    end_date='20251231'
)
```

## 公告日期配置

### 配置格式

```python
ANNOUNCEMENT_DATES = {
    'cashflow': {
        '公告日期(YYYYMMDD)': '财报结束日期(YYYYMMDD)',
    },
    'income': {...},
    'balance': {...},
}
```

### 自动生成

使用 `use_auto_ann=True` 参数时，脚本会根据文件名自动推断公告日期：

- Q1 (0331) → 4月30日发布
- Q2 (0630) → 8月31日发布
- Q3 (0930) → 10月31日发布
- Q4 (1231) → 次年4月30日发布

## 数据结构

### 文件命名

| 原始表 | 每日数据文件 |
|--------|------------|
| cashflow | `cashflow_daily_YYYYMMDD.parquet` |
| income | `income_daily_YYYYMMDD.parquet` |
| balance | `balance_daily_YYYYMMDD.parquet` |

### 数据列

转换后的每日数据只保留财务指标列，去除公告相关列：

```python
EXCLUDE_COLUMNS = [
    'ts_code', 'ann_date', 'f_ann_date', 'end_date',
    'report_type', 'comp_type', 'end_type', 'update_flag'
]
```

### 输出目录

```
daily_data/
├── cashflow/           # 季度数据
│   ├── cashflow_20250331.parquet
│   └── ...
├── cashflow_daily/     # 每日数据
│   ├── cashflow_daily_20250101.parquet
│   ├── cashflow_daily_20250102.parquet
│   └── ...
├── income/
├── income_daily/
├── balance/
└── balance_daily/
```

## 与 get_local_data 配合使用

```python
from mylib.get_local_data import get_local_data

# 获取每日现金流
cf = get_local_data(
    sec_list=['000001.SZ', '000002.SZ'],
    start='20250101',
    end='20251231',
    filed='n_cashflow_act',
    data_type='cashflow_daily'
)

# 获取每日利润
income = get_local_data(
    sec_list=['000001.SZ', '000002.SZ'],
    start='20250101',
    end='20251231',
    filed='n_income',
    data_type='income_daily'
)
```

## 注意事项

1. **公告日期需提前配置**：在新财报发布前配置好公告日期
2. **交易日历**：当前使用自然日，可根据需要接入交易日历
3. **增量更新**：使用 `--skip` 参数跳过已存在的文件
4. **数据量**：每年每张表约 3000-4000 条每日记录
