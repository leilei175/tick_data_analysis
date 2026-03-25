# 项目数据资产说明

- 生成时间: 2026-03-10 20:39:07
- 项目根目录: `/data1/code_git/tick_data_analysis`

## 1. 总览

| 目录 | 是否存在 | Parquet文件数 | 目录体积 |
|------|----------|--------------|----------|
| `mylib` | 是 | 0 | 264.66 KB |
| `tick_2026` | 是 | 842811 | 154.78 GB |
| `daily_data` | 是 | 28700 | 41.78 GB |
| `factor` | 是 | 431 | 10.78 GB |

## 2. mylib 库文件

| 文件 | 作用 | 最近更新时间 |
|------|------|--------------|
| `mylib/analysis_engine.py` | 封装因子分析主流程，连接读取、预处理、收益率与报告。 | 2026-02-14 16:59:08 |
| `mylib/constants.py` | 集中管理数据目录、文件模式、字段名和项目常量。 | 2026-02-13 10:17:26 |
| `mylib/date_utils.py` | 统一日期解析、格式转换与边界处理。 | 2026-02-13 10:18:23 |
| `mylib/factor_factory.py` | 统一封装多类因子来源的访问入口。 | 2026-02-14 16:27:45 |
| `mylib/factor_preprocessor.py` | 执行 winsorize、标准化等因子预处理。 | 2026-02-14 16:27:53 |
| `mylib/financial_column_mapper.py` | 维护 Tushare 财务字段的中文映射。 | 2026-02-26 22:30:40 |
| `mylib/get_local_data.py` | 读取本地日频、财务、衍生与中文财务 parquet 数据。 | 2026-03-04 16:36:14 |
| `mylib/get_remote_data.py` | 通过 Flask parquet 接口远程读取本地数据。 | 2026-03-03 15:48:40 |
| `mylib/get_tick_data.py` | 读取 tick_2026 目录中的逐股 tick parquet 数据。 | 2026-03-10 19:22:20 |
| `mylib/plotting_utils.py` | 统一图表样式与可视化配置。 | 2026-02-13 10:18:35 |
| `mylib/returns_calculator.py` | 计算远期收益率与相关收益指标。 | 2026-02-14 16:27:27 |
| `mylib/tushare_client.py` | 集中初始化 Tushare 与交易日历访问。 | 2026-02-13 10:17:54 |

## 3. 数据集清单

### tick_raw

- 数据分类: `tick`
- 数据保存地址: `tick_2026`
- 绝对路径: `/data1/quant-data/tick_2026`
- 文件命名模式: `*/*/*/*.parquet`
- 文件名示例: `000001.SZ.parquet `
- 数据频度: tick
- 数据来源: 本地 tick 原始数据目录（当前为软链接到 /data1/quant-data/tick_2026）
- 更新方式: 外部落库；项目内脚本以读取为主，不负责原始 tick 下载。
- 相关脚本: `mylib/get_tick_data.py`, `tick_reader.py`, `high_frequency_factors.py`
- 文件数: 842811
- 日期范围: 2025-07-31 -> 2026-03-10
- 最新文件: `tick_2026/2026/03/10/920992.BJ.parquet `
- 最近更新时间: 2026-03-10 16:07:18
- 目录体积: 154.78 GB
- 说明: 按 年/月/日/股票代码 组织的原始逐笔数据，是所有高频因子的上游数据源。
- 备注: 目录为软链接；本仓库主要消费该数据。

### daily_market

- 数据分类: `daily_data`
- 数据保存地址: `daily_data/daily`
- 绝对路径: `/data1/code_git/tick_data_analysis/daily_data/daily`
- 文件命名模式: `**/daily_*.parquet`
- 文件名示例: `daily_20200102.parquet `
- 数据频度: 日频
- 数据来源: Tushare 日线行情接口
- 更新方式: 通过 update_data.py 或 tushare_downloader.py 增量更新交易日文件。
- 相关脚本: `update_data.py`, `tushare_downloader.py`
- 文件数: 1494
- 日期范围: 20200102 -> 20260309
- 最新文件: `daily_data/daily/2026/03/daily_20260309.parquet `
- 最近更新时间: 2026-03-10 16:38:57
- 目录体积: 897.60 MB
- 说明: A 股日线行情，文件按交易日切分。

### daily_basic

- 数据分类: `daily_data`
- 数据保存地址: `daily_data/daily_basic`
- 绝对路径: `/data1/code_git/tick_data_analysis/daily_data/daily_basic`
- 文件命名模式: `**/daily_basic_*.parquet`
- 文件名示例: `daily_basic_20200102.parquet `
- 数据频度: 日频
- 数据来源: Tushare daily_basic 接口
- 更新方式: 通过 update_data.py 或 tushare_downloader.py 增量更新。
- 相关脚本: `update_data.py`, `tushare_downloader.py`
- 文件数: 1494
- 日期范围: 20200102 -> 20260309
- 最新文件: `daily_data/daily_basic/2026/03/daily_basic_20260309.parquet `
- 最近更新时间: 2026-03-10 16:39:16
- 目录体积: 1.45 GB
- 说明: 每日基本面、市值、换手率等横截面数据。

### cashflow_quarter

- 数据分类: `financial_quarterly`
- 数据保存地址: `daily_data/cashflow`
- 绝对路径: `/data1/code_git/tick_data_analysis/daily_data/cashflow`
- 文件命名模式: `**/cashflow_*.parquet`
- 文件名示例: `cashflow_20091231.parquet `
- 数据频度: 季频/公告期
- 数据来源: Tushare 现金流量表
- 更新方式: 通过 financial_downloader.py 或 update_data.py 下载季度文件，并合并 all 文件。
- 相关脚本: `financial_downloader.py`, `update_data.py`
- 文件数: 98
- 日期范围: 20091231 -> 20251231
- 最新文件: `daily_data/cashflow/2025/09/cashflow_20250930_cn.parquet `
- 最近更新时间: 2026-02-26 22:43:21
- 目录体积: 233.68 MB
- 说明: 原始季度现金流数据，目录按 年/月 组织。

### income_quarter

- 数据分类: `financial_quarterly`
- 数据保存地址: `daily_data/income`
- 绝对路径: `/data1/code_git/tick_data_analysis/daily_data/income`
- 文件命名模式: `**/income_*.parquet`
- 文件名示例: `income_20081231.parquet `
- 数据频度: 季频/公告期
- 数据来源: Tushare 利润表
- 更新方式: 通过 financial_downloader.py 或 update_data.py 下载和增量补齐。
- 相关脚本: `financial_downloader.py`, `update_data.py`
- 文件数: 120
- 日期范围: 20081231 -> 20251231
- 最新文件: `daily_data/income/2025/09/income_20250930_cn.parquet `
- 最近更新时间: 2026-02-26 22:43:23
- 目录体积: 182.39 MB
- 说明: 原始季度利润表数据。

### balance_quarter

- 数据分类: `financial_quarterly`
- 数据保存地址: `daily_data/balance`
- 绝对路径: `/data1/code_git/tick_data_analysis/daily_data/balance`
- 文件命名模式: `**/balance_*.parquet`
- 文件名示例: `balance_20081231.parquet `
- 数据频度: 季频/公告期
- 数据来源: Tushare 资产负债表
- 更新方式: 通过 financial_downloader.py 或 update_data.py 下载和增量补齐。
- 相关脚本: `financial_downloader.py`, `update_data.py`
- 文件数: 102
- 日期范围: 20081231 -> 20251231
- 最新文件: `daily_data/balance/2025/09/balance_20250930_cn.parquet `
- 最近更新时间: 2026-02-26 22:43:25
- 目录体积: 299.96 MB
- 说明: 原始季度资产负债表数据。

### cashflow_daily

- 数据分类: `financial_daily`
- 数据保存地址: `daily_data/cashflow_daily`
- 绝对路径: `/data1/code_git/tick_data_analysis/daily_data/cashflow_daily`
- 文件命名模式: `cashflow_daily_*.parquet`
- 文件名示例: `cashflow_daily_20150105.parquet `
- 数据频度: 日频
- 数据来源: 由 cashflow 季度财务数据按公告日展开得到
- 更新方式: 使用 financial_daily_converter.py 或 cashflow_daily_converter.py 从季度表生成；新财报后可增量更新。
- 相关脚本: `financial_daily_converter.py`, `cashflow_daily_converter.py`, `update_data.py`
- 文件数: 2674
- 日期范围: 20150105 -> 20251231
- 最新文件: `daily_data/cashflow_daily/cashflow_daily_20251231.parquet `
- 最近更新时间: 2026-02-27 10:42:43
- 目录体积: 7.11 GB
- 说明: 英文列名版本的日频现金流数据。
- 备注: 同目录还包含 yearly full 文件与 `_cn` 变体文件。

### income_daily

- 数据分类: `financial_daily`
- 数据保存地址: `daily_data/income_daily`
- 绝对路径: `/data1/code_git/tick_data_analysis/daily_data/income_daily`
- 文件命名模式: `income_daily_*.parquet`
- 文件名示例: `income_daily_20150105.parquet `
- 数据频度: 日频
- 数据来源: 由 income 季度财务数据按公告日展开得到
- 更新方式: 使用 financial_daily_converter.py 从季度表生成；支持公告日后的增量更新。
- 相关脚本: `financial_daily_converter.py`, `update_data.py`
- 文件数: 2674
- 日期范围: 20150105 -> 20251231
- 最新文件: `daily_data/income_daily/income_daily_20251231.parquet `
- 最近更新时间: 2026-02-27 10:43:38
- 目录体积: 5.54 GB
- 说明: 英文列名版本的日频利润表数据。
- 备注: 同目录还包含 yearly full 文件与 `_cn` 变体文件。

### balance_daily

- 数据分类: `financial_daily`
- 数据保存地址: `daily_data/balance_daily`
- 绝对路径: `/data1/code_git/tick_data_analysis/daily_data/balance_daily`
- 文件命名模式: `balance_daily_*.parquet`
- 文件名示例: `balance_daily_20150105.parquet `
- 数据频度: 日频
- 数据来源: 由 balance 季度财务数据按公告日展开得到
- 更新方式: 使用 financial_daily_converter.py 从季度表生成；支持公告日后的增量更新。
- 相关脚本: `financial_daily_converter.py`, `update_data.py`
- 文件数: 2674
- 日期范围: 20150105 -> 20251231
- 最新文件: `daily_data/balance_daily/balance_daily_20251231.parquet `
- 最近更新时间: 2026-02-27 10:45:09
- 目录体积: 10.06 GB
- 说明: 英文列名版本的日频资产负债表数据。
- 备注: 同目录还包含 yearly full 文件与 `_cn` 变体文件。

### cashflow_daily_cn

- 数据分类: `financial_daily_cn`
- 数据保存地址: `daily_data/cashflow_daily_cn`
- 绝对路径: `/data1/code_git/tick_data_analysis/daily_data/cashflow_daily_cn`
- 文件命名模式: `**/cashflow_daily_cn_*.parquet`
- 文件名示例: `cashflow_daily_cn_20150105.parquet `
- 数据频度: 日频
- 数据来源: 由英文版日频财务数据重组/转中文列名得到
- 更新方式: 使用 reorganize_financial_daily_cn.py 重新整理目录与年汇总文件。
- 相关脚本: `reorganize_financial_daily_cn.py`
- 文件数: 2674
- 日期范围: 20150105 -> 20251231
- 最新文件: `daily_data/cashflow_daily_cn/2025/12/cashflow_daily_cn_20251231.parquet `
- 最近更新时间: 2026-02-27 11:08:02
- 目录体积: 4.95 GB
- 说明: 中文字段版本的日频现金流数据，用于基本面因子计算。

### income_daily_cn

- 数据分类: `financial_daily_cn`
- 数据保存地址: `daily_data/income_daily_cn`
- 绝对路径: `/data1/code_git/tick_data_analysis/daily_data/income_daily_cn`
- 文件命名模式: `**/income_daily_cn_*.parquet`
- 文件名示例: `income_daily_cn_20150105.parquet `
- 数据频度: 日频
- 数据来源: 由英文版日频财务数据重组/转中文列名得到
- 更新方式: 使用 reorganize_financial_daily_cn.py 重新整理目录与年汇总文件。
- 相关脚本: `reorganize_financial_daily_cn.py`
- 文件数: 2674
- 日期范围: 20150105 -> 20251231
- 最新文件: `daily_data/income_daily_cn/2025/12/income_daily_cn_20251231.parquet `
- 最近更新时间: 2026-02-27 11:09:38
- 目录体积: 3.88 GB
- 说明: 中文字段版本的日频利润表数据。

### balance_daily_cn

- 数据分类: `financial_daily_cn`
- 数据保存地址: `daily_data/balance_daily_cn`
- 绝对路径: `/data1/code_git/tick_data_analysis/daily_data/balance_daily_cn`
- 文件命名模式: `**/balance_daily_cn_*.parquet`
- 文件名示例: `balance_daily_cn_20150105.parquet `
- 数据频度: 日频
- 数据来源: 由英文版日频财务数据重组/转中文列名得到
- 更新方式: 使用 reorganize_financial_daily_cn.py 重新整理目录与年汇总文件。
- 相关脚本: `reorganize_financial_daily_cn.py`
- 文件数: 2674
- 日期范围: 20150105 -> 20251231
- 最新文件: `daily_data/balance_daily_cn/2025/12/balance_daily_cn_20251231.parquet `
- 最近更新时间: 2026-02-27 11:11:40
- 目录体积: 6.87 GB
- 说明: 中文字段版本的日频资产负债表数据。

### derivative_financial_metrics

- 数据分类: `derived_daily`
- 数据保存地址: `daily_data/derivative`
- 绝对路径: `/data1/code_git/tick_data_analysis/daily_data/derivative`
- 文件命名模式: `**/derivative_*.parquet`
- 文件名示例: `derivative_20200102.parquet `
- 数据频度: 日频
- 数据来源: 由 income_daily_cn、balance_daily_cn 和 daily_basic 派生计算
- 更新方式: 使用 build_derivative_financial_metrics.py 全量重建日文件和年度 full 文件。
- 相关脚本: `build_derivative_financial_metrics.py`
- 文件数: 1212
- 日期范围: 20200102 -> 20250630
- 最新文件: `daily_data/derivative/2025/06/derivative_20250630.parquet `
- 最近更新时间: 2026-03-04 09:03:12
- 目录体积: 320.16 MB
- 说明: 衍生财务指标，当前包含 roe、roa、gross_margin、roic、enterprise_value。

### wind_hub_imports

- 数据分类: `external_import`
- 数据保存地址: `daily_data/wind_hub`
- 绝对路径: `/data1/code_git/tick_data_analysis/daily_data/wind_hub`
- 文件命名模式: `*.parquet`
- 文件名示例: `定期报告实际披露日期_2026_converted.parquet `
- 数据频度: 季度/不规则
- 数据来源: Wind Hub 外部导入数据
- 更新方式: 手工导入后，使用 convert_wind_hub.py 转为时间 x 股票宽表。
- 相关脚本: `convert_wind_hub.py`
- 文件数: 5
- 日期范围: - -> -
- 最新文件: `daily_data/wind_hub/销售毛利率_2026_converted.parquet `
- 最近更新时间: 2026-03-03 17:24:59
- 目录体积: 23.69 MB
- 说明: 当前包含 ROIC、销售毛利率、定期报告实际披露日期等外部数据。

### high_frequency_daily_factors

- 数据分类: `factor_daily`
- 数据保存地址: `factor/high_frequency`
- 绝对路径: `/data1/code_git/tick_data_analysis/factor/high_frequency`
- 文件命名模式: `*.parquet`
- 文件名示例: `2025_07_31.parquet `
- 数据频度: 日频
- 数据来源: 由 tick_raw 计算得到
- 更新方式: 单日用 high_frequency_factors.py 计算，补齐缺口用 hf_factor_auto_update.py 自动更新。
- 相关脚本: `high_frequency_factors.py`, `hf_factor_auto_update.py`
- 文件数: 142
- 日期范围: 2025_07_31 -> 2026_03_06
- 最新文件: `factor/high_frequency/2025_08_01.parquet `
- 最近更新时间: 2026-03-10 18:45:27
- 目录体积: 88.00 MB
- 说明: 逐日保存的高频因子明细文件，每行通常对应单只股票当天的因子结果。

### high_frequency_daily_panels

- 数据分类: `factor_daily`
- 数据保存地址: `factor/daily`
- 绝对路径: `/data1/code_git/tick_data_analysis/factor/daily`
- 文件命名模式: `*.parquet`
- 文件名示例: `factors_20260202.parquet `
- 数据频度: 日频
- 数据来源: 由 tick_raw 计算得到的单日因子面板
- 更新方式: 主要由 compute_zz1000_factors.py 或 high_frequency_factors.py 生成。
- 相关脚本: `compute_zz1000_factors.py`, `high_frequency_factors.py`
- 文件数: 33
- 日期范围: 20251201 -> 20260206
- 最新文件: `factor/daily/zz1000_factors_20260206.parquet `
- 最近更新时间: 2026-02-08 18:04:21
- 目录体积: 10.40 GB
- 说明: 中证1000或全市场单日因子面板，以及汇总文件 zz1000_all_factors.parquet。

### by_factor_wide_tables

- 数据分类: `factor_wide`
- 数据保存地址: `factor/by_factor`
- 绝对路径: `/data1/code_git/tick_data_analysis/factor/by_factor`
- 文件命名模式: `zz1000_*.parquet`
- 文件名示例: `zz1000_bid_ask_spread.parquet `
- 数据频度: 日频宽表
- 数据来源: 由 factor/daily 聚合得到
- 更新方式: 使用 batch_aggregate_factors.py 将日面板转为按因子宽表，含整段和按年文件。
- 相关脚本: `batch_aggregate_factors.py`
- 文件数: 30
- 日期范围: - -> -
- 最新文件: `factor/by_factor/zz1000_trade_flow_intensity_2026.parquet `
- 最近更新时间: 2026-02-09 12:54:32
- 目录体积: 45.96 MB
- 说明: 高频因子宽表，按因子拆文件，行是日期，列是股票代码。
- 备注: 包含 bid_ask_spread、vwap_deviation、trade_imbalance 等 10 个高频因子。

### forward_returns

- 数据分类: `factor_aux`
- 数据保存地址: `factor/by_factor`
- 绝对路径: `/data1/code_git/tick_data_analysis/factor/by_factor`
- 文件命名模式: `return_*d.parquet`
- 文件名示例: `return_10d.parquet `
- 数据频度: 日频标签
- 数据来源: 由 factor/daily 中的 lastPrice 计算得到
- 更新方式: 使用 calculate_returns.py 批量重建 1/5/10 日远期收益率。
- 相关脚本: `calculate_returns.py`
- 文件数: 3
- 日期范围: - -> -
- 最新文件: `factor/by_factor/return_10d.parquet `
- 最近更新时间: 2026-02-09 12:20:41
- 目录体积: 45.96 MB
- 说明: 因子分析标签文件，当前包括 1d、5d、10d 未来收益率。

### kzz_call_auction_factor

- 数据分类: `factor_aux`
- 数据保存地址: `factor/by_factor`
- 绝对路径: `/data1/code_git/tick_data_analysis/factor/by_factor`
- 文件命名模式: `*call_auction*.parquet`
- 文件名示例: `call_auction_amount_all_2026.parquet `
- 数据频度: 日频宽表
- 数据来源: 由 tick_raw 中可转债集合竞价成交额计算得到
- 更新方式: 使用 build_kzz_call_auction_factor.py 回填或 update_kzz_call_auction_factor.py 增量更新。
- 相关脚本: `build_kzz_call_auction_factor.py`, `update_kzz_call_auction_factor.py`, `hf_factor_auto_update.py`
- 文件数: 3
- 日期范围: - -> -
- 最新文件: `factor/by_factor/call_auction_amount_all_2026.parquet `
- 最近更新时间: 2026-03-05 22:51:57
- 目录体积: 45.96 MB
- 说明: 可转债集合竞价成交额因子宽表及年度文件。

### call_auction_snapshot_daily

- 数据分类: `factor_daily`
- 数据保存地址: `factor/high_frequency/call_auction_snapshot`
- 绝对路径: `/data1/code_git/tick_data_analysis/factor/high_frequency/call_auction_snapshot`
- 文件命名模式: `call_auction_snapshot_*.parquet`
- 文件名示例: `call_auction_snapshot_2026_03_02.parquet `
- 数据频度: 日频
- 数据来源: 由 tick_raw 中集合竞价阶段盘口快照计算得到
- 更新方式: 使用 build_call_auction_snapshot_factors.py 按日期范围回填或增量计算。
- 相关脚本: `build_call_auction_snapshot_factors.py`
- 文件数: 8
- 日期范围: 2026_03_02 -> 2026_03_10
- 最新文件: `factor/high_frequency/call_auction_snapshot/call_auction_snapshot_2026_03_10.parquet `
- 最近更新时间: 2026-03-10 20:18:13
- 目录体积: 1.19 MB
- 说明: 集合竞价盘口快照日频明细文件，每行对应单只股票在 09:15:00 <= t < 09:25:00 窗口内的 4 个快照指标。
- 备注: 字段包括 auction_last1_ask1_ret、auction_last2_ask1_ret、auction_last1_askVol1、auction_last2_askVol1。

### call_auction_snapshot_wide

- 数据分类: `factor_wide`
- 数据保存地址: `factor/by_factor`
- 绝对路径: `/data1/code_git/tick_data_analysis/factor/by_factor`
- 文件命名模式: `auction_last*.parquet`
- 文件名示例: `auction_last1_ask1_ret_2026.parquet `
- 数据频度: 日频宽表
- 数据来源: 由 call_auction_snapshot 日频明细透视得到
- 更新方式: 使用 build_call_auction_snapshot_factors.py 在生成日频明细后同步更新年度宽表。
- 相关脚本: `build_call_auction_snapshot_factors.py`
- 文件数: 4
- 日期范围: - -> -
- 最新文件: `factor/by_factor/auction_last2_askVol1_2026.parquet `
- 最近更新时间: 2026-03-10 20:18:14
- 目录体积: 45.96 MB
- 说明: 集合竞价盘口快照因子年度宽表，行是日期，列是股票代码，每个指标单独保存一个 parquet 文件。
- 备注: 当前包含 4 个宽表：auction_last1_ask1_ret、auction_last2_ask1_ret、auction_last1_askVol1、auction_last2_askVol1。

### fundamental_factor_tables

- 数据分类: `factor_fundamental`
- 数据保存地址: `factor/fundamental`
- 绝对路径: `/data1/code_git/tick_data_analysis/factor/fundamental`
- 文件命名模式: `*.parquet`
- 文件名示例: `allmarket_accruals.parquet `
- 数据频度: 日频宽表
- 数据来源: 由中文财务日频数据和 daily_basic 推导
- 更新方式: 使用 build_fundamental_factors.py 重建，支持全市场和 zz1000 两套输出。
- 相关脚本: `build_fundamental_factors.py`
- 文件数: 27
- 日期范围: - -> -
- 最新文件: `factor/fundamental/allmarket_earnings_yield.parquet `
- 最近更新时间: 2026-02-27 16:58:12
- 目录体积: 257.60 MB
- 说明: 基本面因子宽表，当前包含 ROE、ROA、Book-to-Market、FCF Yield 等。

### preprocessed_factors

- 数据分类: `factor_preprocessed`
- 数据保存地址: `factor/preprocessed`
- 绝对路径: `/data1/code_git/tick_data_analysis/factor/preprocessed`
- 文件命名模式: `*.parquet`
- 文件名示例: `pp_zz1000_bid_ask_spread_zscore_std3.parquet `
- 数据频度: 日频宽表
- 数据来源: 由原始因子宽表经过预处理得到
- 更新方式: 由因子预处理流程生成，当前仓库以结果文件为主。
- 相关脚本: `mylib/factor_preprocessor.py`, `convert_factors.py`
- 文件数: 2
- 日期范围: - -> -
- 最新文件: `factor/preprocessed/pp_zz1000_pe_winsorize_low0.025_upp0.975.parquet `
- 最近更新时间: 2026-02-14 16:39:58
- 目录体积: 1.27 MB
- 说明: 标准化或裁剪后的因子文件，例如 zscore、winsorize 结果。

### factor_analysis_outputs

- 数据分类: `analysis_output`
- 数据保存地址: `factor/analysis`
- 绝对路径: `/data1/code_git/tick_data_analysis/factor/analysis`
- 文件命名模式: `**/*`
- 文件名示例: `full_report.md `
- 数据频度: 按分析任务生成
- 数据来源: 因子分析结果输出
- 更新方式: 由 factor_analysis.py、financial_factor_analysis.py 等分析脚本生成。
- 相关脚本: `factor_analysis.py`, `financial_factor_analysis.py`, `zz1000_factor_analysis.py`
- 文件数: 3
- 日期范围: - -> -
- 最新文件: `factor/analysis/20260214_pe_raw/full_report.md `
- 最近更新时间: 2026-02-14 18:43:36
- 目录体积: 1.62 KB
- 说明: 分析报告、IC 统计、分层收益等结果目录。

## 4. 说明

- `tick_2026` 当前是软链接，项目内高频因子与 tick 读取逻辑默认消费该目录。
- `daily_data/*_daily` 目录中既有逐日文件，也混有 `YYYY_full.parquet`、`*_cn.parquet` 等汇总或变体文件；上面的条目已按主要命名模式拆分说明。
- `factor/by_factor` 同时承载高频因子宽表、收益率标签和可转债集合竞价因子宽表。
- 如果后续数据继续增长，可重复运行 `python generate_data_inventory.py` 刷新文档与 JSON 元数据。
