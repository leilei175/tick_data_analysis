# 数据更新指南

## 概述

本文档介绍如何使用 `update_data.py` 脚本自动更新 `daily_data` 目录下的所有数据。

## 数据目录结构

```
daily_data/
├── daily/              # 日线行情数据 (OHLCV)
│   ├── 2025/
│   │   ├── 01/
│   │   │   └── daily_20250102.parquet
│   │   └── ...
│   └── 2026/
│       └── 02/
│           └── daily_20260206.parquet
│
├── daily_basic/        # 每日基本面数据
│   ├── 2025/
│   └── 2026/
│       └── 02/
│           └── daily_basic_20260206.parquet
│
├── cashflow/           # 现金流量表季度数据
│   └── cashflow_20241231.parquet
│   └── cashflow_20250331.parquet
│
├── cashflow_daily/     # 现金流量表每日数据
│   ├── 2025/
│   └── 2025/
│
├── income/             # 利润表季度数据
│   └── income_*.parquet
│
├── income_daily/       # 利润表每日数据
│   └── ...
│
├── balance/           # 资产负债表季度数据
│   └── balance_*.parquet
│
└── balance_daily/      # 资产负债表每日数据
    └── ...
```

## 数据更新逻辑

### 日线数据和基本面数据

1. **自动检测最新日期**：脚本会扫描 `daily/` 和 `daily_basic/` 目录，找到最新数据日期
2. **交易日判断**：使用 Tushare API 获取交易日历，只更新交易日
3. **时间规则**：
   - **18:00 前运行**：更新到前一个交易日
   - **18:00 后运行**：更新到今天（如果今天是交易日）

### 财务报表数据

财务报表数据更新遵循**公告日期机制**：

1. 季度数据下载后，通过 `financial_daily_converter.py` 转换为每日数据
2. 每日使用哪个季度的数据，由公告日期决定
3. 公告日期配置在 `financial_daily_converter.py` 中

## 快速开始

### 1. 更新所有数据

```bash
# 自动检测最新日期并更新
python update_data.py
```

### 2. 只更新日线数据

```bash
python update_data.py --daily
```

### 3. 只更新基本面数据

```bash
python update_data.py --daily-basic
```

### 4. 只更新财务报表数据

```bash
python update_data.py --financial
```

### 5. 指定日期范围

```bash
# 更新 2026年2月10日到2026年2月11日
python update_data.py --start 20260210 --end 20260211
```

### 6. 强制包含今天数据

```bash
# 即使没到18:00也更新今天的数据
python update_data.py --include-today
```

## Python API

### 更新所有数据

```python
from update_data import update_all_data

# 自动检测并更新所有数据
update_all_data()

# 只更新日线和基本面（不含财务报表）
update_all_data(financial=False)

# 指定日期范围更新
update_all_data(start_date='20260210', end_date='20260211')

# 强制包含今天
update_all_data(include_today=True)
```

### 只更新日线数据

```python
from update_data import update_daily_data

# 自动检测并更新
update_daily_data()

# 指定日期范围
update_daily_data(start_date='20260210', end_date='20260211')

# 包含今天
update_daily_data(include_today=True)
```

### 获取当前数据状态

```python
from update_data import get_all_latest_dates

dates = get_all_latest_dates()
print(dates)
# 输出: {'daily': '20260206', 'daily_basic': '20260206', ...}
```

## 配置文件

脚本从 `config.py` 读取 Tushare Token：

```python
# config.py
tushare_tk = 'your_token_here'
```

## 工作流程

### 日线数据更新流程

```
┌─────────────────────────────────────────────────────────────┐
│                    update_data.py                           │
├─────────────────────────────────────────────────────────────┤
│  1. 检测 daily/ 最新日期                                    │
│  2. 检测 daily_basic/ 最新日期                              │
│  3. 获取交易日历 (pro.trade_cal)                           │
│  4. 计算需要更新的日期范围                                    │
│  5. 下载日线数据 (pro.daily)                               │
│  6. 下载基本面数据 (pro.daily_basic)                        │
│  7. 按年/月保存为 Parquet 文件                              │
└─────────────────────────────────────────────────────────────┘
```

### 财务报表数据更新流程

```
┌─────────────────────────────────────────────────────────────┐
│                 financial_vip_downloader.py                 │
├─────────────────────────────────────────────────────────────┤
│  1. 下载季度数据 (balancesheet_vip, income_vip, cashflow)  │
│  2. 保存为: balance/xxx.parquet                             │
│                    ↓                                        │
├─────────────────────────────────────────────────────────────┤
│                 financial_daily_converter.py                │
├─────────────────────────────────────────────────────────────┤
│  1. 读取季度数据                                            │
│  2. 根据公告日期配置决定每天使用哪个季度                     │
│  3. 生成每日数据                                            │
│  4. 保存为: cashflow_daily/xxx.parquet                      │
└─────────────────────────────────────────────────────────────┘
```

## 当前数据状态

| 数据类型 | 最新日期 | 需要更新 |
|---------|---------|---------|
| daily | 20260206 | 是 (需更新到 20260210) |
| daily_basic | 20260206 | 是 |
| cashflow_daily | 20251231 | 是 |
| income_daily | 20251231 | 是 |
| balance_daily | 20251231 | 是 |

## 常见问题

### Q1: 更新时提示 "未找到 Tushare Token"

确保 `config.py` 中配置了有效的 Token：
```python
tushare_tk = 'your_token_here'
```

### Q2: 财务报表数据没有更新

财务报表数据需要两步：
1. 先用 `financial_vip_downloader.py` 下载季度数据
2. 再用 `financial_daily_converter.py` 转换为每日数据

`update_data.py` 会自动调用转换逻辑。

### Q3: 如何更新历史数据

```bash
# 更新整个2026年1月的数据
python update_data.py --start 20260101 --end 20260131
```

### Q4: 跳过已存在的文件

脚本默认会跳过已存在的文件，使用 `--no-skip` 参数可强制重新下载。

## 相关脚本

| 脚本 | 功能 |
|-----|------|
| `update_data.py` | **主更新脚本** - 自动更新所有数据 |
| `tushare_downloader.py` | 下载日线和基本面数据 |
| `financial_vip_downloader.py` | 下载 VIP 财务报表 |
| `financial_daily_converter.py` | 季度数据转每日数据 |

## 命令行参数参考

```
update_data.py 参数:
  --daily          只更新日线数据
  --daily-basic    只更新基本面数据
  --financial      只更新财务报表数据
  --start, -s      开始日期 (YYYYMMDD)
  --end, -e        结束日期 (YYYYMMDD)
  --include-today  包含今天的数据
  --no-financial   不更新财务报表数据
  --help, -h       显示帮助
```

## 自动任务建议

可以设置 cron 任务自动执行更新：

```bash
# 每天 18:30 自动更新数据
30 18 * * 1-5 cd /path/to/project && python update_data.py >> /var/log/data_update.log 2>&1
```

或者使用 systemd timer 实现更复杂的调度。
