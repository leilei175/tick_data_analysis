# get_remote_data 函数使用文档

## 1. 功能概述

`get_remote_data` 用于从远端 Flask 服务读取数据，底层调用二进制接口：

- `POST /api/local-data/query/parquet`

你可以在本地脚本中直接像调用本地函数一样请求远端数据，例如：

```python
df = get_remote_data(data_type='daily', start='20250101', end='20250110')
```

函数会把 API 返回值自动转换为 `pandas.DataFrame`（或多字段时的 `Dict[str, DataFrame]`）。

## 2. 函数位置

- 实现文件：`mylib/get_remote_data.py`
- 导入方式：

```python
from mylib.get_remote_data import get_remote_data, RemoteDataError
```

## 3. 函数签名

```python
def get_remote_data(
    data_type: str = "daily",
    start: Optional[str] = None,
    end: Optional[str] = None,
    field: Union[str, List[str]] = "close",
    stocks: Optional[Union[List[str], str]] = None,
    base_url: str = "http://127.0.0.1:9999",
    timeout: int = 60,
    limit: int = 10000000,
    output_format: str = "wide",
    parallel: bool = True,
    max_workers: int = 8,
    warn_on_truncated: bool = True,
    disable_proxy: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]
```

## 4. 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data_type` | `str \| None` | `"daily"` | 数据类型；可留空（`None` 或 `''`）由后端按 `field` 自动推断，例如 `未分配利润 -> balance_daily_cn`。支持季度类型：`balance_q/income_q/cashflow_q`（或别名 `balance/income/cashflow`）及对应 `_cn` 版本 |
| `start` | `str \| None` | `None` | 开始日期，支持 `YYYYMMDD` 或 `YYYY-MM-DD` |
| `end` | `str \| None` | `None` | 结束日期，支持 `YYYYMMDD` 或 `YYYY-MM-DD` |
| `field` | `str \| List[str]` | `"close"` | 单字段或多字段。多字段时返回字典 |
| `stocks` | `List[str] \| str \| None` | `None` | 股票列表，或逗号分隔字符串，如 `"000001.SZ,000002.SZ"` |
| `base_url` | `str` | `"http://127.0.0.1:9999"` | 远端 Flask 服务地址 |
| `timeout` | `int` | `60` | HTTP 请求超时时间（秒） |
| `limit` | `int` | `10000000` | 服务端返回记录上限 |
| `output_format` | `str` | `"wide"` | 远端输出格式：`long` 或 `wide` |
| `parallel` | `bool` | `True` | 是否让远端并行读取数据文件 |
| `max_workers` | `int` | `8` | 远端并行线程数 |
| `warn_on_truncated` | `bool` | `True` | 数据被 `limit` 截断时是否发出 warning |
| `disable_proxy` | `bool` | `True` | 是否禁用系统 HTTP 代理。访问内网 Flask 服务时建议保持开启 |

## 5. 返回值说明

### 5.1 单字段返回

当 `field` 是字符串时，返回 `DataFrame`：

- `index`: `date`（`DatetimeIndex`）
- `columns`: 股票代码，如 `000001.SZ`
- `values`: 对应字段数值

### 5.2 多字段返回

当 `field` 是列表时，返回 `Dict[str, DataFrame]`：

- 键：字段名（如 `close`, `vol`）
- 值：对应字段的 `DataFrame`

## 6. 异常处理

请求失败时会抛出 `RemoteDataError`，常见场景：

- 远端服务不可达（网络/端口问题）
- 接口返回错误（参数非法、数据类型不支持等）
- 远端返回非 JSON

建议按如下方式捕获：

```python
from mylib.get_remote_data import get_remote_data, RemoteDataError

try:
    df = get_remote_data(data_type='daily', start='20250101', end='20250110')
except RemoteDataError as e:
    print("远程取数失败:", e)
```

## 7. 使用示例

### 7.1 最简调用（你的使用方式）

```python
from mylib.get_remote_data import get_remote_data

df = get_remote_data(
    data_type='daily',
    start='20250101',
    end='20250110'
)
print(df.shape)
print(df.head())
```

### 7.2 指定远端服务器地址

```python
df = get_remote_data(
    data_type='daily',
    field='close',
    start='20250101',
    end='20250110',
    base_url='http://10.10.20.8:9999'
)
```

### 7.3 指定股票池

```python
df = get_remote_data(
    data_type='daily',
    field='close',
    start='20250101',
    end='20250110',
    stocks=['000001.SZ', '000002.SZ', '600000.SH']
)
```

### 7.4 多字段批量读取

```python
data = get_remote_data(
    data_type='daily',
    field=['close', 'vol', 'amount'],
    start='20250101',
    end='20250110'
)

close_df = data['close']
vol_df = data['vol']
```

### 7.5 使用宽格式减少透视开销

```python
df = get_remote_data(
    data_type='daily_basic',
    field='turnover_rate',
    start='20250101',
    end='20250110',
    output_format='wide'
)
```

## 8. 建议与注意事项

1. 跨网络使用时优先设置 `base_url` 为固定 IP 或域名。
2. 大区间查询建议提高 `limit`，并关注截断 warning。
3. 如果网络质量一般，建议增大 `timeout`，例如 `timeout=120`。
4. 若字段较多，优先使用多字段一次请求，减少网络往返次数。

## 9. 对应服务端接口

`get_remote_data` 依赖服务端已部署以下接口：

- `POST /api/local-data/query/parquet`

如果服务端未更新到包含该接口的版本，会报 `RemoteDataError`。
