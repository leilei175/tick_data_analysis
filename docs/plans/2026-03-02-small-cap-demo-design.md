# 小市值Demo策略设计

## 1. 策略概述

**策略名称**：小市值Demo策略
**策略类型**：日频选股策略
**核心思想**：每天从成交活跃的股票中选取市值最小的50只，等权配置

## 2. 策略逻辑

### 2.1 每日筛选条件
- **成交额过滤**：当日成交额 >= 1000万元
- **ST过滤**：排除股票代码以 ST、*ST、PT 开头的股票
- **涨跌停过滤**：排除当日涨停或跌停的股票

### 2.2 选股规则
- 从满足筛选条件的股票中，选取**市值（total_mv）最小**的50只
- 若满足条件的股票不足50只，则全部买入

### 2.3 调仓规则
- **调仓频率**：每日调仓（每日重新计算持仓）
- **权重分配**：等权配置（每只股票权重 = 1/N）

### 2.4 回测参数
- 初始资金：1000万
- 手续费：万分之一（0.0001）
- 滑点：暂不设置

## 3. 数据需求

### 3.1 现有数据
- `daily_data/daily/`：日线行情数据（open, high, low, close, vol）
- `daily_data/daily_basic/`：每日基本面数据（包含 total_mv, amount）

### 3.2 新增数据接口
在 `backtest/data_source.py` 中新增：
- `load_daily_basic_panel()`：加载每日基本面数据
- `load_combined_panel()`：合并行情和基本面数据

## 4. 实现方案

### 4.1 文件结构
```
backtest/
├── data_source.py          # 扩展：新增数据加载函数
├── small_cap_strategy.py  # 新增：策略实现
└── run_small_cap_demo.py  # 新增：运行脚本
```

### 4.2 核心函数

#### data_source.py 新增
```python
def load_daily_basic_panel(daily_basic_dir: str, start: str, end: str) -> pd.DataFrame:
    """加载每日基本面数据"""

def load_combined_panel(
    daily_dir: str,
    daily_basic_dir: str,
    start: str,
    end: str,
    symbols: Optional[List[str]] = None
) -> pd.DataFrame:
    """合并行情和基本面数据"""
```

#### small_cap_strategy.py
```python
class SmallCapStrategy:
    params = (
        ("min_amount", 10_000_000),    # 最小成交额
        ("max_stocks", 50),            # 最大持仓数量
    )

    def compute_daily_signals(self, panel: pd.DataFrame) -> List[str]:
        """计算每日选股信号"""
```

## 5. 风险提示

- 小市值股票流动性风险
- 极端行情下可能无法按照收盘价成交
- 手续费和滑点会对收益产生显著影响

## 6. 待定

- [ ] 是否需要添加行业/概念过滤？
- [ ] 是否需要添加止损/止盈逻辑？
