# ETF 数据更新指南

本文档对应脚本 [update_etf_data.py](/data1/code_git/tick_data_analysis/update_etf_data.py)。

## 能拿到什么

脚本会在 `daily_data/` 下生成 4 类日文件：

```text
daily_data/
├── etf_daily/     # ETF 日线行情
├── etf_nav/       # ETF 单位净值等净值字段
├── etf_share/     # ETF 份额字段
└── etf_metrics/   # 合并后的日频指标，含收盘折溢价率
```

`etf_metrics` 里重点字段：

- `close`: 收盘价
- `unit_nav`: 单位净值
- `fd_share`: 基金份额
- `premium_rate_close_nav`: 收盘相对净值的折溢价率
- `iopv`: 可选外部接入的 IOPV
- `premium_rate_close_iopv`: 收盘相对 IOPV 的折溢价率

## 数据源设计

优先方案：

- `Tushare etf_share_size`
  直接给出 ETF 收盘价、净值、份额、规模，最适合做日频收盘折溢价。

回退方案：

- `Tushare fund_daily`
- `Tushare fund_nav`
- `Tushare fund_share`

原因：

- `etf_share_size` 的信息最完整，但有积分门槛。
- 如果积分不够，仍可以用 `fund_daily + fund_nav + fund_share` 组合出日频结果。

## 折溢价率口径

收盘折溢价率：

```text
premium_rate_close_nav = close / unit_nav - 1
```

盘中折溢价率：

```text
premium_rate_close_iopv = close / iopv - 1
```

注意：

- `unit_nav` 是基金净值，适合做日频收盘分析。
- `IOPV` 是盘中参考净值，更适合做盘中折溢价监控。
- 交易所通常按约 15 秒频率发布 IOPV，但历史 IOPV 明细一般需要券商/行情商/交易所授权行情。

## 使用方式

初始化或补历史：

```bash
python update_etf_data.py --start 20250101 --end 20250331
```

日常增量更新：

```bash
python update_etf_data.py
```

如果你有外部 IOPV 文件：

```bash
python update_etf_data.py --iopv-dir ./daily_data/iopv
```

如果账号没有 `etf_share_size` 权限：

```bash
python update_etf_data.py --no-share-size
```

## IOPV 外部文件格式

文件名支持：

- `iopv_YYYYMMDD.parquet`
- `YYYYMMDD.parquet`
- `iopv_YYYYMMDD.csv`
- `YYYYMMDD.csv`

至少包含 3 列：

```text
ts_code,trade_date,iopv
```

示例：

```csv
ts_code,trade_date,iopv
510300.SH,20260316,3.9987
159915.SZ,20260316,2.4561
```

## 定时自动更新

建议在 19:10 之后跑一次。

示例 crontab：

```cron
10 19 * * 1-5 cd /data1/code_git/tick_data_analysis && /usr/bin/python3 update_etf_data.py >> log/etf_update.log 2>&1
```

如果还要合并 IOPV：

```cron
15 19 * * 1-5 cd /data1/code_git/tick_data_analysis && /usr/bin/python3 update_etf_data.py --iopv-dir ./daily_data/iopv >> log/etf_update.log 2>&1
```

## 适用边界

这个脚本已经解决：

- ETF 日线收盘价
- ETF 单位净值
- ETF 份额
- ETF 收盘折溢价率
- 每天自动增量更新

这个脚本没有替代：

- 盘中历史 IOPV 采集
- Level-1/Level-2 实时行情订阅
- 券商专有行情接口鉴权

如果你后面要把 IOPV 也做成完整历史库，建议把券商或行情商的 snapshot 数据在盘中按分钟或按 tick 落库，再复用这里的 `--iopv-dir` 合并逻辑。
