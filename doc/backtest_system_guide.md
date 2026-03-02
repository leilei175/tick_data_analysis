# 回测系统使用手册

## 1. 系统概览

本项目回测系统由三部分组成：

1. 回测执行脚本（`backtest/*.py`）
2. Web 管理与调度（`factor_dashboard/app.py` + `factor_dashboard/templates/backtest.html`）
3. 回测结果持久化目录（`backtest/web_runs/<task_id>/`）

系统支持：

1. 运行内置回测脚本
2. 在网页中手动编写并保存自定义脚本
3. 参数化启动回测任务并实时查看进度
4. 历史回测结果本地保存与点击查看


## 2. 目录结构

```text
backtest/
  data_source.py
  run_backtest.py
  run_factor_topn_demo.py
  custom_scripts/
    demo_custom_backtest.py
    <your_script>.py
  web_runs/
    <task_id>/
      task_meta.json
      run.log
      metrics.json
      equity_curve.csv
      rebalance_log.csv   # 可选
```


## 3. Web 使用流程

入口：`/backtest`

1. 在“脚本选择”中选择内置脚本或 `custom:<脚本名>`
2. 填写参数
3. 点击“启动回测”
4. 页面显示任务进度与实时日志
5. 完成后自动跳转结果页 `/backtest/result/<task_id>`
6. 也可在“历史回测结果”表中点击“查看结果”


## 4. 内置脚本参数说明

### 4.1 `run_backtest.py`（均线等权）

参数：

1. `--start`：开始日期，`YYYYMMDD`
2. `--end`：结束日期，`YYYYMMDD`
3. `--daily-dir`：行情目录，默认 `daily_data/daily`
4. `--symbols`：逗号分隔股票列表（可空）
5. `--symbol-limit`：未指定 `symbols` 时自动选股数量
6. `--cash`：初始资金
7. `--commission`：手续费率
8. `--short-window`：短均线窗口
9. `--long-window`：长均线窗口
10. `--rebalance-days`：每隔 N 个交易日调仓
11. `--output-dir`：输出目录（由 Web 自动注入）


### 4.2 `run_factor_topn_demo.py`（因子打分 TopN）

参数：

1. `--start`：开始日期，`YYYYMMDD`
2. `--end`：结束日期，`YYYYMMDD`
3. `--daily-dir`：行情目录
4. `--symbols`：逗号分隔股票列表（可空）
5. `--symbol-limit`：未指定 `symbols` 时自动选股数量
6. `--cash`：初始资金
7. `--commission`：手续费率
8. `--lookback`：因子回看窗口（天）
9. `--topn`：每次调仓持有 TopN
10. `--rebalance-days`：每隔 N 个交易日调仓
11. `--output-dir`：输出目录（由 Web 自动注入）


### 4.3 自定义脚本通用参数（Web 注入）

对于 `custom:<script>.py`，网页默认传入：

1. `--start`
2. `--end`
3. `--daily-dir`
4. `--symbols`
5. `--symbol-limit`
6. `--cash`
7. `--commission`
8. `--output-dir`

可选：

1. `--extra-args`：会拆分为额外命令行参数附加到脚本后


## 5. 自定义回测脚本编写规范

### 5.1 必须遵守的接口约定

1. 脚本放在 `backtest/custom_scripts/*.py`
2. 文件名必须匹配：`[A-Za-z0-9_.-]+.py`
3. 应支持通用参数（至少 `--output-dir`）
4. 成功时进程退出码必须为 `0`
5. 失败时抛异常或非 0 退出码
6. 关键日志输出到 `stdout`（会被采集到网页）


### 5.2 结果文件约定（强烈建议）

建议脚本产出：

1. `metrics.json`（建议必出）
2. `equity_curve.csv`（建议必出）
3. `rebalance_log.csv`（可选）

其中：

1. `metrics.json` 是结果页指标卡的数据源
2. `equity_curve.csv` 是净值曲线图的数据源
3. `rebalance_log.csv` 是调仓记录表的数据源


### 5.3 推荐 `metrics.json` 字段

```json
{
  "initial_cash": 1000000.0,
  "final_value": 1085000.0,
  "total_return": 0.085,
  "annual_return": 0.121,
  "max_drawdown_pct": 8.7,
  "sharpe": 1.15,
  "symbols_count": 20
}
```


### 5.4 `equity_curve.csv` 格式

至少两列：

1. `date`：日期
2. `nav`：净值

示例：

```csv
date,nav
2022-01-04,1.0000
2022-01-05,1.0032
2022-01-06,0.9987
```


### 5.5 `rebalance_log.csv`（可选）

推荐列：

1. `date`
2. `selected_count`
3. `selected_symbols`


### 5.6 最小可运行模板

```python
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20220101")
    parser.add_argument("--end", default="20221231")
    parser.add_argument("--daily-dir", default="daily_data/daily")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbol-limit", type=int, default=50)
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--output-dir", required=True)
    args, _ = parser.parse_known_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = {
        "initial_cash": args.cash,
        "final_value": args.cash,
        "total_return": 0.0,
        "annual_return": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe": 0.0,
        "symbols_count": 0
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "equity_curve.csv").write_text("date,nav\\n2022-01-01,1.0\\n", encoding="utf-8")
    print("backtest done")

if __name__ == "__main__":
    main()
```


## 6. 回测任务运行规则

1. 每次启动都会生成唯一 `task_id`
2. 对应输出目录：`backtest/web_runs/<task_id>/`
3. 任务状态流转：`queued -> running -> success/error`
4. 进度条为估算进度，成功/失败后置为 `100%`
5. 实时日志来自进程标准输出并写入 `run.log`
6. 元数据持久化到 `task_meta.json`，重启后仍可恢复历史记录


## 7. 结果页面解读

结果页展示三类信息：

1. 指标卡：`metrics.json`
2. 净值曲线：`equity_curve.csv`
3. 调仓记录：`rebalance_log.csv`（若存在）

常见指标解释：

1. `initial_cash`：初始资金
2. `final_value`：结束时资产
3. `total_return`：总收益率（小数制，`0.1`=10%）
4. `annual_return`：年化收益率（小数制）
5. `max_drawdown_pct`：最大回撤（百分数制）
6. `sharpe`：夏普比率
7. `symbols_count`：实际参与回测标的数量

高级指标（新增）：

1. 风险调整收益：`sortino`、`calmar`、`omega_ratio`
2. 风险暴露：`annual_volatility`、`downside_volatility`、`ulcer_index`
3. 回撤结构：`max_drawdown_duration_days`
4. 收益分布：`skewness`、`kurtosis`
5. 尾部风险：`var_95_daily`、`cvar_95_daily`
6. 交易质量：`win_rate`、`payoff_ratio`、`profit_factor`
7. 相对基准：`alpha`、`beta`、`tracking_error`、`information_ratio`、`treynor_ratio`、`up_capture_ratio`、`down_capture_ratio`


## 8. 常见问题与排查

1. 启动失败：检查脚本名是否合法、脚本是否已保存到 `backtest/custom_scripts/`
2. 任务报错：查看结果目录下 `run.log`
3. 无结果图：确认 `equity_curve.csv` 存在且包含 `date/nav`
4. 历史看不到：确认 `backtest/web_runs/<task_id>/task_meta.json` 或至少有 `metrics.json`
5. 参数不生效：检查脚本是否正确解析参数（建议 `parse_known_args()`）


## 9. 推荐实践

1. 固定随机种子，保证结果可复现
2. 在日志中打印关键配置与数据规模
3. 将交易成本、滑点、停牌处理明确写入脚本
4. 每次迭代保留核心指标与回测版本信息（可附加到 `metrics.json`）


## 10. 参考文件

1. `factor_dashboard/app.py`（任务调度与 API）
2. `factor_dashboard/templates/backtest.html`（回测配置页）
3. `factor_dashboard/templates/backtest_result.html`（结果页）
4. `backtest/run_backtest.py`（均线策略）
5. `backtest/run_factor_topn_demo.py`（TopN 因子策略）
6. `backtest/custom_scripts/demo_custom_backtest.py`（自定义脚本示例）
