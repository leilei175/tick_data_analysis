# 可转债集合竞价成交额因子（日更）

## 1. 因子定义

- 因子名：`kzz_call_auction_amount`
- 含义：可转债在集合竞价阶段的成交金额
- 计算口径：`09:15:00 <= t < 09:26:00` 的成交额增量之和
  - 由于 tick `amount` 是累计成交额，先做 `diff()` 得到增量，再截断负值为 0
- 标的范围：代码前缀 `110/111/113/118/123/127/128`，且后缀为 `.SH/.SZ`
- 数据源：`/data1/quant-data/tick_2026/{year}/{month}/{day}/*.parquet`

## 2. 存储位置

### 2.1 日频明细（每日一文件）

目录：`factor/high_frequency/kzz_call_auction_amount/`

文件名：
- `kzz_call_auction_amount_YYYY_MM_DD.parquet`

字段：
- `date`：交易日（YYYY-MM-DD）
- `stock_code`：转债代码
- `kzz_call_auction_amount`：集合竞价成交金额

### 2.2 年度宽表（按因子）

目录：`factor/by_factor/`

文件名：
- `kzz_call_auction_amount_2025.parquet`
- `kzz_call_auction_amount_2026.parquet`

格式：
- index：`date`
- columns：`stock_code`
- values：`kzz_call_auction_amount`

## 3. 脚本说明

### 3.1 历史回填脚本

文件：`build_kzz_call_auction_factor.py`

常用参数：
- `--mode backfill|update`
- `--years 2025,2026`
- `--start-date YYYY-MM-DD`（backfill）
- `--end-date YYYY-MM-DD`（backfill）
- `--skip-existing`（backfill 时跳过已存在日文件）

### 3.2 每日更新脚本

文件：`update_kzz_call_auction_factor.py`

行为：
- 自动扫描 tick 交易日
- 仅处理“有 tick 数据但因子文件缺失”的日期
- 更新后自动重建受影响年份的年度宽表

## 4. 使用方法

### 4.1 首次全量构建（2025 + 2026）

```bash
python build_kzz_call_auction_factor.py --mode backfill --years 2025,2026 --skip-existing
```

### 4.2 指定区间重算

```bash
python build_kzz_call_auction_factor.py --mode backfill --years 2026 --start-date 2026-02-01 --end-date 2026-02-28
```

### 4.3 每日增量更新

```bash
python update_kzz_call_auction_factor.py
```

或使用统一脚本的 update 模式：

```bash
python build_kzz_call_auction_factor.py --mode update --years 2025,2026
```

## 5. 自动化日更（crontab 示例）

每天收盘后 18:20 执行：

```cron
20 18 * * 1-5 cd /data1/code_git/tick_data_analysis && /home/zxx/miniconda3/envs/quant/bin/python update_kzz_call_auction_factor.py >> log/kzz_call_auction_factor.log 2>&1
```

## 6. 快速校验

### 6.1 查看日频文件数量与区间

```bash
python - <<'PY'
from pathlib import Path
files = sorted(Path('factor/high_frequency/kzz_call_auction_amount').glob('kzz_call_auction_amount_*.parquet'))
print('files=', len(files))
print('first=', files[0].name if files else None)
print('last=', files[-1].name if files else None)
PY
```

### 6.2 查看宽表规模

```bash
python - <<'PY'
import pandas as pd
for y in [2025, 2026]:
    df = pd.read_parquet(f'factor/by_factor/kzz_call_auction_amount_{y}.parquet')
    print(y, df.shape, df.index.min(), df.index.max())
PY
```

## 7. 当前已生成结果

已完成 `2025 + 2026` 全量构建：
- 日频文件：`139` 个交易日（2025 年 101 天，2026 年 38 天）
- 宽表：
  - `factor/by_factor/kzz_call_auction_amount_2025.parquet`（101 x 450）
  - `factor/by_factor/kzz_call_auction_amount_2026.parquet`（37 x 393）

