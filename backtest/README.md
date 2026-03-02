# 回测框架选型与实现说明

## 框架选择

本仓库回测框架选择为 **Backtrader**。

结合你的目标（直接使用本地 `daily_data/` parquet 数据、快速可运行）选择理由如下：

1. `Backtesting.py` 更偏单标的 DataFrame 流程，多标的轮动场景接入成本更高。
2. `Zipline-Reloaded` 功能完整，但自定义本地数据接入前需要较重的 bundle 与交易日历配置。
3. `Backtrader` 可直接接收每个标的的 pandas OHLCV 数据，适合快速搭建多标的调仓策略。

## 选型参考资料

1. Backtesting.py 文档：https://kernc.github.io/backtesting.py/
2. Zipline-Reloaded 文档：https://zipline.ml4trading.io/
3. Backtrader 文档：https://www.backtrader.com/docu/
4. Backtrader GitHub：https://github.com/mementum/backtrader
5. RQAlpha 文档（中文框架参考）：https://www.ricequant.com/doc/rqalpha-plus/api/basics

## 已实现文件

1. `backtest/data_source.py`：本地行情读取与清洗。
2. `backtest/run_backtest.py`：均线策略回测入口（等权持仓）。
3. `backtest/run_factor_topn_demo.py`：因子打分 + TopN 调仓 demo。

## 数据流说明

1. 从 `daily_data/daily/{year}_all.parquet` 读取年级别行情（若不存在则回退 `{year}_full.parquet`）。
2. 按 `start/end` 和可选股票列表过滤。
3. 转换为 Backtrader 所需的单标的 OHLCV DataFrame。
4. 执行策略并导出结果。

## 输出文件

均线策略默认输出目录：`backtest/output`

1. `backtest/output/metrics.json`
2. `backtest/output/equity_curve.csv`

TopN 因子策略默认输出目录：`backtest/output_factor_topn`

1. `backtest/output_factor_topn/metrics.json`
2. `backtest/output_factor_topn/equity_curve.csv`
3. `backtest/output_factor_topn/rebalance_log.csv`

`metrics.json` 现在包含扩展绩效指标：年化波动率、Sortino、Calmar、胜率、盈亏比、VaR/CVaR、Ulcer Index，以及基准对比指标（Alpha/Beta/信息比率/跟踪误差/捕获比）等。

## 快速开始

均线策略：

```bash
python backtest/run_backtest.py \
  --start 20220101 \
  --end 20221231 \
  --symbol-limit 20 \
  --cash 1000000 \
  --commission 0.001
```

指定股票：

```bash
python backtest/run_backtest.py \
  --start 20230101 \
  --end 20231231 \
  --symbols 000001.SZ,600000.SH,000333.SZ
```

因子打分 + TopN 调仓 demo：

```bash
python backtest/run_factor_topn_demo.py \
  --start 20220101 \
  --end 20221231 \
  --symbol-limit 50 \
  --lookback 20 \
  --topn 10 \
  --rebalance-days 5
```

## 策略说明

### 1. 均线策略 `EqualWeightSmaCross`

1. 对每个标的计算短期与长期均线。
2. 在调仓日买入满足 `SMA(short) > SMA(long)` 的标的，并等权分配。
3. 不满足条件的标的清仓。

### 2. 因子策略 `FactorScoreTopN`

1. 使用过去 `lookback` 天收益率作为示例因子分数。
2. 在每个调仓日做截面排序，选取 TopN 标的。
3. 对 TopN 标的等权持有，其余清仓。
