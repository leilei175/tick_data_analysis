# ROIC 与 企业价值指标说明

本文档说明如何基于仓库现有的 `balance_daily_cn`、`income_daily_cn`、`cashflow_daily_cn` 财务数据，以及 `daily_basic` 市值数据，计算投入资本回报率（ROIC）与企业价值（Enterprise Value, EV）。

## 定义

### ROIC

ROIC 衡量企业使用投入资本创造税后经营利润的效率，常见定义为：

`ROIC = NOPAT / Average Invested Capital`

- `NOPAT`：税后净营业利润，使用 `EBIT * (1 - Tax Rate)` 近似。
- `Average Invested Capital`：投入资本的期初期末平均值。

本项目实现采用的拆解为：

`ROIC = [息税前利润 * (1 - 所得税费用 / 利润总额)] / 平均(股东权益合计(不含少数股东权益) + 短期借款 + 长期借款 + 应付债券 - 货币资金)`

其中税率会被截断在 `[0, 1]`，缺失值使用当日截面中位数回填，若仍缺失则回填为 `25%`。

### 企业价值

企业价值衡量收购一家企业的理论总体成本，常见定义为：

`EV = Market Capitalization + Total Debt - Cash and Cash Equivalents`

本项目实现采用的拆解为：

`企业价值 = total_mv + 短期借款 + 长期借款 + 应付债券 - 货币资金`

说明：

- `total_mv` 来自 `daily_basic`，表示总市值。
- 三张中文财务表里没有股权市值字段，因此 EV 无法仅依赖 `balance_daily_cn`、`income_daily_cn`、`cashflow_daily_cn` 三表完成，必须补充 `daily_basic.total_mv`。
- `cashflow_daily_cn` 在这两个指标的标准定义中没有直接参与计算，因此当前实现未使用其字段。

## 字段映射

### ROIC 使用字段

来自 `income_daily_cn`：

- `息税前利润`
- `所得税费用`
- `利润总额`

来自 `balance_daily_cn`：

- `股东权益合计(不含少数股东权益)`
- `短期借款`
- `长期借款`
- `应付债券`
- `货币资金`

### 企业价值使用字段

来自 `balance_daily_cn`：

- `短期借款`
- `长期借款`
- `应付债券`
- `货币资金`

来自 `daily_basic`：

- `total_mv`

## 代码位置

- 指标生成脚本：[build_derivative_financial_metrics.py](/data1/code_git/tick_data_analysis/build_derivative_financial_metrics.py)
- 本地读取字段注册：[mylib/get_local_data.py](/data1/code_git/tick_data_analysis/mylib/get_local_data.py)

## 输出格式

输出目录：

- `daily_data/derivative/YYYY/MM/derivative_YYYYMMDD.parquet`
- `daily_data/derivative/YYYY_full.parquet`

输出字段：

- `ts_code`
- `trade_date`
- `roe`
- `roa`
- `gross_margin`
- `roic`
- `enterprise_value`

## 实现细节

- 分母接近 0 时结果置为 `NaN`，避免无意义极值。
- ROIC 的投入资本使用相邻两个交易日的平均值近似。
- EV 为时点值指标，不做平均处理。
- 指标构建日期范围取 `income_daily_cn`、`balance_daily_cn`、`daily_basic` 三者的公共区间。

## 参考定义来源

- ROIC 定义参考 CFI 的 ROIC 说明页：https://corporatefinanceinstitute.com/resources/accounting/what-is-roic/
- 企业价值定义参考 Investopedia 的 Enterprise Value 说明页：https://www.investopedia.com/terms/e/enterprisevalue.asp
