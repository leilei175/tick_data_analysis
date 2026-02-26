# 项目完整使用手册（数据下载、因子分析、数据管理与保存）

> 项目：`tick_data_analysis`
> 适用对象：从 0 到 1 搭建并运行本项目的数据链路、分析链路和看板
> 更新时间：2026-02-14

## 1. 项目目标与总体流程

本项目用于构建并分析股票因子，核心流程如下：

1. 下载基础数据（行情、基本面、财务报表）
2. 将季度财务数据转换为每日可用数据
3. 计算高频因子并按因子宽表保存
4. 计算未来收益率
5. 执行因子分析（IC、分层、多空、报告）
6. 通过网站进行因子看板与文档中心管理

---

## 2. 环境准备

### 2.1 Python 依赖

建议至少安装：

```bash
pip install pandas numpy pyarrow scipy tushare flask plotly markdown
```

### 2.2 Tushare Token 配置

三种方式任选其一：

1. 环境变量：

```bash
export TUSHARE_TOKEN='你的token'
```

2. 用户目录文件：

```bash
echo '你的token' > ~/.tushare_token
```

3. 项目配置文件：`config.py` 中维护 `tushare_tk`

---

## 3. 目录结构与数据落地规范

### 3.1 关键目录

- 原始与转换数据：`daily_data/`
- 高频因子：`factor/daily/`
- 按因子宽表：`factor/by_factor/`
- 分析结果：`factor_analysis_results/`
- 可视化网站：`factor_dashboard/`
- 文档：`doc/`

### 3.2 推荐的数据保存路径

- 日线行情：`daily_data/daily/`
- 每日基本面：`daily_data/daily_basic/`
- 财务季度数据：`daily_data/{cashflow|income|balance}/`
- 财务日度数据：`daily_data/{cashflow_daily|income_daily|balance_daily}/`
- 因子宽表：`factor/by_factor/zz1000_<factor>[_<year>].parquet`
- 收益率：`factor/by_factor/return_{1|5|10}d.parquet`

---

## 4. 数据下载与更新

### 4.1 下载日线与每日基本面（推荐入口）

脚本：`tushare_downloader.py`

```bash
# 日线
python tushare_downloader.py daily -s 20260101 -e 20261231

# 每日基本面
python tushare_downloader.py daily_basic -s 20260101 -e 20261231

# 一次性下载两者
python tushare_downloader.py all -s 20260101 -e 20261231
```

### 4.2 下载财务报表季度数据

脚本：`financial_downloader.py`

```bash
# 全量范围下载（三张表）
python financial_downloader.py --all

# 指定日期范围
python financial_downloader.py -s 20150101 -e 20251231 --all

# 只下载某一类
python financial_downloader.py -s 20150101 -e 20251231 --cashflow
python financial_downloader.py -s 20150101 -e 20251231 --income
python financial_downloader.py -s 20150101 -e 20251231 --balance

# 按季度补数
python financial_downloader.py --quarters 20250630 20250331
```

### 4.3 VIP 财务接口下载（兼容入口）

脚本：`financial_vip_downloader.py`（已并入主逻辑，保留兼容）

```bash
python financial_vip_downloader.py --all
python financial_vip_downloader.py --start 20150101 --end 20251231 --all-fields
python financial_vip_downloader.py --split
```

### 4.4 一体化增量更新（生产推荐）

脚本：`update_data.py`

```bash
# 自动更新（默认：日线 + 基本面 + 财务）
python update_data.py

# 分类型更新
python update_data.py --daily
python update_data.py --daily-basic
python update_data.py --financial

# 指定日期范围
python update_data.py -s 20260210 -e 20260211

# 包含当日（通常收盘后）
python update_data.py --include-today
```

---

## 5. 财务季度数据转每日数据

### 5.1 多表转换（推荐）

脚本：`financial_daily_converter.py`

```bash
# 转换全部三张表
python financial_daily_converter.py -s 20250101 -e 20251231 --all

# 单表转换
python financial_daily_converter.py -s 20250101 -e 20251231 --cashflow
python financial_daily_converter.py -s 20250101 -e 20251231 --income
python financial_daily_converter.py -s 20250101 -e 20251231 --balance
```

### 5.2 现金流单表转换

脚本：`cashflow_daily_converter.py`

```bash
python cashflow_daily_converter.py -s 20250101 -e 20251231

# 自定义公告日期（可多次）
python cashflow_daily_converter.py -s 20250101 -e 20251231 \
  --ann-date 20250228:20241231 \
  --ann-date 20250430:20250331

# 增量更新模式
python cashflow_daily_converter.py -s 20251010 -e 20251231 --update \
  --ann-date 20251010:20250930
```

### 5.3 历史前向填充转换（不依赖公告日）

脚本：`convert_quarterly_to_daily.py`

```bash
python convert_quarterly_to_daily.py -s 20200101 -e 20251231 --all
```

---

## 6. 高频因子生成与保存

### 6.1 按日计算高频因子

脚本：`high_frequency_factors.py`

```bash
# 单日
python high_frequency_factors.py --date 2025-12-01

# 指定股票
python high_frequency_factors.py --date 2025-12-01 --stock 000001.SZ

# 批量区间
python high_frequency_factors.py --start 2025-12-01 --end 2025-12-31
```

输出默认保存到 `factor/daily/`。

### 6.2 按因子宽表聚合

脚本：`batch_aggregate_factors.py`

```bash
python batch_aggregate_factors.py
```

输出到 `factor/by_factor/`，文件格式示例：

- `zz1000_order_imbalance.parquet`
- `zz1000_order_imbalance_2025.parquet`

### 6.3 未来收益率计算

脚本：`calculate_returns.py`

```bash
python calculate_returns.py
```

输出：

- `factor/by_factor/return_1d.parquet`
- `factor/by_factor/return_5d.parquet`
- `factor/by_factor/return_10d.parquet`

---

## 7. 因子分析与报告

### 7.1 中证1000综合分析

脚本：`zz1000_factor_analysis.py`

```bash
python zz1000_factor_analysis.py --start 20240101 --end 20251231

# 指定因子
python zz1000_factor_analysis.py --factors close turnover_rate pe pb

# 仅汇总
python zz1000_factor_analysis.py --summary-only
```

### 7.2 财务因子分析

脚本：`financial_factor_analysis.py`

```bash
python financial_factor_analysis.py --start 20250101 --end 20251231 --stocks 1000
```

### 7.3 结果目录

主要在 `factor_analysis_results/`，包括：

- `summary_report.md`
- `reports/*.md`
- `financial_reports/*.md`
- 统计图和 CSV

---

## 8. 数据管理与网站使用

### 8.1 启动网站

脚本：`factor_dashboard/app.py`

```bash
python factor_dashboard/app.py
```

默认地址：`http://localhost:9999`

### 8.2 登录（如启用）

- 用户：`admin` 密码：`admin`
- 用户：`user` 密码：`user`

### 8.3 网站模块

- 因子看板：仪表盘、相关性、IC、分层、多空
- 数据管理：查看数据状态，触发更新
- 文档中心：统一检索 `doc/` 与分析报告文档

---

## 9. 推荐标准运行顺序（生产模板）

```bash
# 1) 增量更新基础数据
python update_data.py --include-today

# 2) 财务季度转每日（可按需）
python financial_daily_converter.py -s 20250101 -e 20251231 --all

# 3) 计算高频因子
python high_frequency_factors.py --start 2025-12-01 --end 2025-12-31

# 4) 聚合成宽表
python batch_aggregate_factors.py

# 5) 计算收益率
python calculate_returns.py

# 6) 执行分析
python zz1000_factor_analysis.py --start 20240101 --end 20251231

# 7) 启动看板
python factor_dashboard/app.py
```

---

## 10. 常见问题排查

### 10.1 `未找到 Tushare Token`

检查：

- `echo $TUSHARE_TOKEN`
- `cat ~/.tushare_token`
- `config.py` 中是否含 `tushare_tk`

### 10.2 因子分析提示“收益率数据为空”

先执行：

```bash
python batch_aggregate_factors.py
python calculate_returns.py
```

并确认 `factor/by_factor/return_1d.parquet` 存在。

### 10.3 看板能打开但图表无数据

优先检查：

- `factor/by_factor/zz1000_*.parquet` 是否存在
- 是否存在收益率文件 `return_{1,5,10}d.parquet`
- 浏览器请求 `/api/factors/list` 是否返回 `success`

### 10.4 文档中心未显示新文档

- 确认文件位于 `doc/*.md`
- 刷新页面并重新搜索关键词

---

## 11. 代码入口速查

- 基础数据下载：`tushare_downloader.py`
- 财务数据下载：`financial_downloader.py`
- 增量更新：`update_data.py`
- 财务日度转换：`financial_daily_converter.py`
- 高频因子计算：`high_frequency_factors.py`
- 因子宽表聚合：`batch_aggregate_factors.py`
- 收益率计算：`calculate_returns.py`
- 因子分析：`zz1000_factor_analysis.py`、`financial_factor_analysis.py`
- 网站：`factor_dashboard/app.py`

