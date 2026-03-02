# 财务全字段中文数据说明

## 1. 目标

将 Tushare 财务三张表的 **2015-2025 全字段数据** 下载到本地，并生成中文列名版本，和原始英文列文件区分保存。

涉及表：

- `cashflow`（现金流量表）
- `income`（利润表）
- `balance`（资产负债表，对应 `balancesheet`）

## 2. 数据范围

- 时间范围：`20150101 ~ 20251231`
- 数据粒度：财报周期（按 `end_date` 季度拆分）
- 股票范围：全部上市股票（接口可返回范围内）

## 3. 输出目录与命名

根目录：`daily_data/`

每张表输出两类中文文件（`_cn` 后缀）：

1. 全量文件（便于一次性读取）

- `daily_data/cashflow/cashflow_all_cn.parquet`
- `daily_data/income/income_all_cn.parquet`
- `daily_data/balance/balance_all_cn.parquet`

2. 季度文件（便于按季度/年份读取）

- `daily_data/cashflow/YYYY/MM/cashflow_YYYYMMDD_cn.parquet`
- `daily_data/income/YYYY/MM/income_YYYYMMDD_cn.parquet`
- `daily_data/balance/YYYY/MM/balance_YYYYMMDD_cn.parquet`

说明：

- 原始英文列文件（无 `_cn`）保留不变。
- `_cn` 文件仅做“列名中文化 + 同期范围过滤”。

## 4. 中文字段映射来源

字段中文名基于 Tushare 文档页的字段定义生成：

- Income: `doc_id=33`
- BalanceSheet: `doc_id=36`
- Cashflow: `doc_id=44`

并对公共字段统一中文名：

- `ts_code` -> `TS代码`
- `ann_date` -> `公告日期`
- `f_ann_date` -> `实际公告日期`
- `end_date` -> `报告期`
- `report_type` -> `报告类型`
- `comp_type` -> `公司类型`
- `end_type` -> `报告期类型`
- `update_flag` -> `更新标识`

## 5. 已完成结果（本次）

- `cashflow_all_cn.parquet`: `265952` 行，`97` 列
- `income_all_cn.parquet`: `258692` 行，`84` 列
- `balance_all_cn.parquet`: `266055` 行，`152` 列

中文列检查结果：三张表 `_all_cn.parquet` 均无英文缩写列残留。

## 6. 读取示例

```python
import pandas as pd

cf = pd.read_parquet("daily_data/cashflow/cashflow_all_cn.parquet")
inc = pd.read_parquet("daily_data/income/income_all_cn.parquet")
bal = pd.read_parquet("daily_data/balance/balance_all_cn.parquet")

print(cf.columns[:10])
print(inc.columns[:10])
print(bal.columns[:10])
```

## 7. 增量更新建议

当需要更新到新季度时，建议按以下顺序执行：

1. 重新下载英文原始全字段（覆盖或更新 `*_all.parquet`）
2. 重新生成 `_all_cn.parquet`
3. 重新按 `end_date` 拆分写入 `*_YYYYMMDD_cn.parquet`

这样可确保中文文件与原始英文文件完全一致且可追溯。

