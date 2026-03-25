# 高频因子定时更新方案（18:30）

## 1. 目标

每天 `18:30` 自动执行高频因子更新，包含：
- 现有高频因子（日文件）：`factor/high_frequency/YYYY_MM_DD.parquet`
- 可转债集合竞价成交额因子（日文件）：`factor/high_frequency/kzz_call_auction_amount/kzz_call_auction_amount_YYYY_MM_DD.parquet`

并支持：
- 若缺失多天数据，自动补齐所有缺失交易日
- 记录结构化日志
- 在 Flask 数据管理页面查看日志并手动触发更新

## 2. 核心脚本

### 2.1 自动更新主脚本

文件：`hf_factor_auto_update.py`

能力：
- 扫描 tick 数据目录中的交易日（`/data1/quant-data/tick_2026/{year}/{month}/{day}`）
- 对比本地因子文件，找出缺失日期
- 逐日补算两类因子
- 自动重建可转债因子的年度宽表
- 写入 JSON/JSONL 日志

常用命令：

```bash
# 默认运行（18:30后自动包含当天）
python hf_factor_auto_update.py

# 指定年份范围
python hf_factor_auto_update.py --years 2025,2026

# 强制不包含当天，仅更新到昨天
python hf_factor_auto_update.py --exclude-today

# 指定截止日期（用于回溯或测试）
python hf_factor_auto_update.py --years 2026 --cutoff-date 2026-01-05
```

### 2.2 安装定时任务脚本

文件：`install_hf_factor_update_cron.sh`

```bash
# 查看当前配置
./install_hf_factor_update_cron.sh --show

# 安装 crontab（交易日 18:30）
./install_hf_factor_update_cron.sh --install

# 删除 crontab
./install_hf_factor_update_cron.sh --remove
```

安装后的 cron：

```cron
30 18 * * 1-5 cd /data1/code_git/tick_data_analysis && /home/zxx/miniconda3/envs/quant/bin/python hf_factor_auto_update.py --include-today >> log/hf_factor_update_cron.log 2>&1 # HF_FACTOR_AUTO_UPDATE
```

## 3. 日志系统

日志目录：`log/hf_factor_updates/`

- 单次执行日志：`hf_update_YYYYMMDD_HHMMSS.json`
- 历史汇总：`hf_update_history.jsonl`
- 最近摘要：`hf_update_latest.log`

JSON 字段包括：
- 执行状态、开始/结束时间、耗时
- 配置参数（年份、截止日、是否含当天）
- 统计信息（缺失天数、更新天数、失败天数）
- 明细信息（成功日期、失败原因、宽表输出路径）

## 4. Flask 集成

已新增 API：

- `GET /api/hf-update/logs?limit=30`
  - 返回自动更新日志列表与最新一次结果
- `POST /api/hf-update/run-sync`
  - 同步执行更新（支持补缺）
  - 请求体示例：

```json
{
  "include_today": true,
  "years": "2025,2026",
  "cutoff_date": "2026-01-05"
}
```

前端页面：
- `数据管理` 页新增“高频因子自动更新（18:30）”模块
- 支持：
  - 查看最近执行摘要
  - 查看历史日志列表
  - 手动触发一次补缺更新

## 5. 运维建议

1. 首次上线先执行一次手动补缺：

```bash
python hf_factor_auto_update.py --years 2025,2026 --include-today
```

2. 再安装 cron：

```bash
./install_hf_factor_update_cron.sh --install
```

3. 每日巡检：
- 查看 `log/hf_factor_update_cron.log`
- 在 Flask 数据管理页面查看“高频因子自动更新（18:30）”模块

