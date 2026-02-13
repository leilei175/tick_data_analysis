# Tushare 数据下载工具

从 Tushare 下载日线行情和每日基本面数据，保存为 Parquet 格式。

## 目录

- [安装](#安装)
- [配置](#配置)
- [命令行使用](#命令行使用)
- [Python API](#python-api)
- [数据结构](#数据结构)
- [示例](#示例)

## 安装

```bash
# 安装依赖
pip install tushare pandas pyarrow
```

## 配置

### 获取 Tushare Token

1. 注册 [Tushare 账号](https://tushare.pro/)
2. 在个人中心获取 API Token

### 设置 Token（三选一）

**方式一：环境变量（推荐）**
```bash
export TUSHARE_TOKEN='你的token'
```

**方式二：配置文件**
```bash
echo '你的token' > ~/.tushare_token
```

**方式三：代码中设置**
```python
from tushare_downloader import set_token
set_token('你的token')
```

## 命令行使用

```bash
# 下载日线数据
python tushare_downloader.py daily -s 20260101 -e 20261231

# 下载每日基本面数据
python tushare_downloader.py daily_basic -s 20260101 -e 20261231

# 下载全部数据
python tushare_downloader.py all -s 20260101 -e 20261231

# 指定输出目录
python tushare_downloader.py daily -s 20260101 -e 20260131 -d ./my_data/daily
```

### 命令行参数

| 参数 | 短参数 | 说明 | 默认值 |
|------|--------|------|--------|
| `data_type` | - | 数据类型: daily/daily_basic/all | 必填 |
| `--start` | `-s` | 开始日期 (YYYYMMDD) | 必填 |
| `--end` | `-e` | 结束日期 (YYYYMMDD) | 必填 |
| `--daily-dir` | `-d` | 日线数据输出目录 | `./daily_data/daily/` |
| `--basic-dir` | `-b` | 基本面数据输出目录 | `./daily_data/daily_basic/` |
| `--token` | `-t` | Tushare Token | 优先使用环境变量 |

## Python API

### 导入

```python
from tushare_downloader import (
    download_daily,
    download_daily_basic,
    download_all,
    set_token
)
```

### 下载日线数据

```python
download_daily(
    start_date='20260101',
    end_date='20261231',
    output_dir='./daily_data/daily',
    fields=None  # 使用默认字段
)
```

### 下载每日基本面

```python
download_daily_basic(
    start_date='20260101',
    end_date='20261231',
    output_dir='./daily_data/daily_basic',
    fields=None  # 使用默认字段
)
```

### 批量下载

```python
download_all(
    start_date='20260101',
    end_date='20261231',
    daily_dir='./daily_data/daily',
    daily_basic_dir='./daily_data/daily_basic'
)
```

## 数据结构

### 日线数据 (daily)

| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | string | 股票代码 (如 000001.SZ) |
| trade_date | string | 交易日期 (YYYYMMDD) |
| open | double | 开盘价 |
| high | double | 最高价 |
| low | double | 最低价 |
| close | double | 收盘价 |
| pre_close | double | 昨收盘价 |
| change | double | 涨跌额 |
| pct_chg | double | 涨跌幅 (%) |
| vol | double | 成交量 (手) |
| amount | double | 成交额 (千元) |

### 每日基本面 (daily_basic)

| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | string | 股票代码 |
| trade_date | string | 交易日期 |
| close | double | 收盘价 |
| turnover_rate | double | 换手率 (%) |
| turnover_rate_f | double | 换手率 (自由流通股) |
| volume_ratio | double | 量比 |
| pe | double | 市盈率 |
| pe_ttm | double | 市盈率 TTM |
| pb | double | 市净率 |
| ps | double | 市销率 |
| ps_ttm | double | 市销率 TTM |
| dv_ratio | double | 股息率 (%) |
| dv_ttm | double | 股息率 TTM |
| total_share | double | 总股本 (万股) |
| float_share | double | 流通股本 (万股) |
| free_share | double | 自由流通股本 (万股) |
| total_mv | double | 总市值 (百万元) |
| circ_mv | double | 流通市值 (百万元) |

### 文件命名

- 日线: `daily_YYYYMMDD.parquet`
- 基本面: `daily_basic_YYYYMMDD.parquet`

## 示例

### 完整下载流程

```python
from tushare_downloader import set_token, download_all

# 1. 设置 Token
set_token('your_token_here')

# 2. 下载 2026 年全年数据
download_all(
    start_date='20260101',
    end_date='20261231'
)

# 3. 使用 get_local_data 读取数据
from mylib.get_local_data import get_local_data

# 读取收盘价
close_prices = get_local_data(
    sec_list=['000001.SZ', '000002.SZ'],
    start='20260102',
    end='20260110',
    data_type='daily',
    filed='close'
)
print(close_prices)

# 读取换手率
turnover = get_local_data(
    sec_list=['000001.SZ', '000002.SZ'],
    start='20260102',
    end='20260110',
    data_type='daily_basic',
    filed='turnover_rate'
)
print(turnover)
```

### 输出示例

```
开始下载日线数据: 20260101 ~ 20261231
交易日数量: 244
按 12 个月份分组下载
  202601: 5241 条记录
  202602: 5032 条记录
...
下载完成! 总计 127823 条记录
```

## 注意事项

1. **Tushare 积分限制**: 不同数据接口有不同的积分要求，请确保账户有足够权限
2. **下载频率**: 脚本已内置 0.3 秒延时，避免触发限流
3. **数据范围**: Tushare 只能获取历史数据，无法获取实时数据
4. **网络问题**: 如下载中断，可重新运行，脚本会跳过已存在的文件
