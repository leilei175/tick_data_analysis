# 09:24~09:25 最新价格提取说明

本文档说明如何从本地 `tick_2026` 数据中，提取所有股票在 `09:24:00` 到 `09:25:00` 之间每个 tick 时点的“最新价格”，并保存为单个 parquet 文件。

## 1. 目标

生成一个长表 parquet 文件，每一行代表：

1. 一个交易日
2. 一只股票
3. 在 `09:24:00 <= tick_time <= 09:25:00` 窗口内的一个 tick 快照

输出文件默认路径：

- `factor/high_frequency/auction_latest_price_924_925/auction_latest_price_924_925_all.parquet`

同时输出一个 summary：

- `factor/high_frequency/auction_latest_price_924_925/auction_latest_price_924_925_summary.json`

脚本：

- `build_auction_latest_price_924_925.py`

## 2. “最新价格”定义

集合竞价尾部很多股票的原始 `lastPrice` 仍然为 `0`，因此脚本采用分层回退逻辑：

1. 若 `lastPrice > 0`，使用 `lastPrice`
2. 否则若 `ask1` 和 `bid1` 都可用，使用 `(ask1 + bid1) / 2`
3. 若 mid 不可用，则退化为 `ask1`
4. 若 `ask1` 也不可用，则退化为 `bid1`

对应输出字段 `price_source` 会记录来源：

- `lastPrice`
- `mid`
- `ask1`
- `bid1`

## 3. 输出字段

| 字段名 | 含义 |
| --- | --- |
| `trade_date` | 交易日 |
| `stock_code` | 股票代码，如 `000001.SZ` |
| `tick_seq` | 该股票在窗口内的 tick 序号，从 1 开始 |
| `datetime` | 北京时间的完整时间戳 |
| `tick_time` | 时间字符串，格式 `HH:MM:SS` |
| `time_ms` | 原始毫秒时间戳 |
| `latest_price` | 按回退逻辑得到的最新价格 |
| `price_source` | 最新价格来源 |
| `lastPrice` | 原始逐笔里的 `lastPrice` |
| `lastClose` | 昨收 |
| `ask1` | 卖一价 |
| `bid1` | 买一价 |
| `pct_from_last_close` | `latest_price / lastClose - 1` |

## 4. 使用方式

### 4.1 全量提取

```bash
python build_auction_latest_price_924_925.py
```

默认会扫描 `./tick_2026` 下所有年份目录，并把结果写到单个 parquet 文件。

### 4.2 指定日期范围

```bash
python build_auction_latest_price_924_925.py --start 2025-08-01 --end 2026-03-09
```

### 4.3 指定年份

```bash
python build_auction_latest_price_924_925.py --years 2025,2026
```

### 4.4 指定输出路径

```bash
python build_auction_latest_price_924_925.py \
  --output-file ./tmp/auction_latest_price_924_925.parquet \
  --summary-file ./tmp/auction_latest_price_924_925_summary.json
```

## 5. 实现细节

脚本对每个股票 parquet 文件执行以下步骤：

1. 读取 `time / lastPrice / lastClose / askPrice / bidPrice`
2. 将 `time` 从 epoch 毫秒转换为 `Asia/Shanghai` 本地时间
3. 过滤 `09:24:00 <= t <= 09:25:00`
4. 解析盘口数组，提取 `ask1 / bid1`
5. 生成 `latest_price` 和 `price_source`
6. 把所有交易日、所有股票的结果增量写入一个 parquet 文件

之所以采用“增量写单个 parquet”，是为了避免一次性把全量明细全部放进内存。

## 6. 适用场景

这个文件适合用于：

1. 研究集合竞价尾部价格路径
2. 计算 `09:24:50`、`09:24:57`、`09:25:00` 等尾部时点信号
3. 构造集合竞价末段动量、反转、盘口一致性等高频因子
4. 对比原始 `lastPrice` 与盘口推导价格在尾部阶段的差异

## 7. 注意事项

1. 这里保存的是窗口内“每个实际 tick 快照”的价格，不会人为补齐所有股票在所有全市场时间点的缺失时刻。
2. `09:25:01` 及之后的数据不会被纳入，因为文档目标窗口是 `09:24:00~09:25:00`。
3. 若某只股票在该窗口内没有任何 tick，则该股票当天不会出现在输出文件中。
4. 由于 `lastPrice` 在集合竞价尾部经常为 `0`，请优先使用 `latest_price`，而不是直接使用原始 `lastPrice`。
