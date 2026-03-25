# get_remote_tick_data 函数使用文档

## 1. 功能概述

`get_remote_tick_data` 用于从远端 Flask 服务读取 tick 数据，底层调用二进制 parquet 接口：

- `POST /api/tick-data/query/parquet`

它会把远端返回的 parquet 自动还原成与本地 `get_tick_data` / `get_tick_data_short` 一致的数据结构。

## 2. 函数位置

- 实现文件：`mylib/get_remote_data.py`
- 导入方式：

```python
from mylib.get_remote_data import get_remote_tick_data, RemoteDataError
```

## 3. 函数签名

```python
def get_remote_tick_data(
    stock_codes: Union[str, List[str]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    tick_dir: Optional[str] = None,
    short: bool = False,
    base_url: str = "http://127.0.0.1:9999",
    timeout: int = 300,
    disable_proxy: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]
```

## 4. 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `stock_codes` | `str \| List[str]` | 必填 | 单个股票代码、股票列表，或逗号分隔字符串 |
| `start_date` | `str \| None` | `None` | 开始日期，支持 `YYYYMMDD` 或 `YYYY-MM-DD` |
| `end_date` | `str \| None` | `None` | 结束日期，支持 `YYYYMMDD` 或 `YYYY-MM-DD` |
| `tick_dir` | `str \| None` | `None` | 服务端 tick 数据目录；不传时默认使用服务端配置的 `tick_2026` |
| `short` | `bool` | `False` | 是否返回精简版 tick 数据；`False` 对应 `get_tick_data()`，`True` 对应 `get_tick_data_short()` |
| `base_url` | `str` | `"http://127.0.0.1:9999"` | 远端 Flask 服务地址 |
| `timeout` | `int` | `300` | HTTP 请求超时时间（秒） |
| `disable_proxy` | `bool` | `True` | 是否禁用系统 HTTP 代理，访问内网服务时建议保持开启 |

## 5. 返回值说明

### 5.1 单股票返回

当 `stock_codes` 只包含 1 只股票时，返回 `DataFrame`：

- `index`: `datetime`，带时区的 `DatetimeIndex`
- 普通版列：与 `get_tick_data()` 一致
- 精简版列：与 `get_tick_data_short()` 一致

### 5.2 多股票返回

当 `stock_codes` 包含多只股票时，返回 `Dict[str, DataFrame]`：

- 键：股票代码，如 `000001.SZ`
- 值：该股票对应的 tick `DataFrame`

## 6. 普通版与精简版区别

### 6.1 `short=False`

对应本地函数 `get_tick_data()`，返回较完整的 tick 数据列。

当前返回结果通常包含：

- `day`
- `time`
- `time_ms`
- 原始行情列
- 原始五档数组列，如 `askPrice`、`bidPrice`、`askVol`、`bidVol`

### 6.2 `short=True`

对应本地函数 `get_tick_data_short()`，返回精简后的高频分析友好格式：

- 基础列：`day`, `time`, `lastPrice`, `open`, `high`, `low`, `lastClose`, `amount`, `volume`
- 五档盘口拆分列：
  - `askPrice1` 到 `askPrice5`
  - `bidPrice1` 到 `bidPrice5`
  - `askVol1` 到 `askVol5`
  - `bidVol1` 到 `bidVol5`

如果是多股票返回，每个子 DataFrame 也会保留 `stock_code`（若服务端结果中存在该列）。

## 7. 异常处理

请求失败时会抛出 `RemoteDataError`，常见场景：

- 远端服务不可达
- 服务端没有部署 `/api/tick-data/query/parquet`
- 参数非法
- 远端返回内容不是合法 parquet

建议这样捕获：

```python
from mylib.get_remote_data import get_remote_tick_data, RemoteDataError

try:
    df = get_remote_tick_data("000001.SZ", start_date="2026-03-02")
except RemoteDataError as e:
    print("远程 tick 取数失败:", e)
```

## 8. 使用示例

### 8.1 读取单只股票单日 tick

```python
from mylib.get_remote_data import get_remote_tick_data

df = get_remote_tick_data(
    stock_codes="000001.SZ",
    start_date="2026-03-02",
    end_date="2026-03-02",
    base_url="http://127.0.0.1:9999",
)

print(df.index.dtype)
print(df.head())
```

### 8.2 读取单只股票一段时间 tick

```python
df = get_remote_tick_data(
    stock_codes="000001.SZ",
    start_date="2026-03-01",
    end_date="2026-03-10",
    base_url="http://127.0.0.1:9999",
)
```

### 8.3 读取多只股票 tick

```python
data = get_remote_tick_data(
    stock_codes=["000001.SZ", "300044.SZ"],
    start_date="2026-03-02",
    end_date="2026-03-02",
    base_url="http://127.0.0.1:9999",
)

df_000001 = data["000001.SZ"]
df_300044 = data["300044.SZ"]
```

### 8.4 读取精简版 tick 数据

```python
short_df = get_remote_tick_data(
    stock_codes="000001.SZ",
    start_date="2026-03-02",
    end_date="2026-03-02",
    short=True,
    base_url="http://127.0.0.1:9999",
)

print(short_df.columns.tolist())
```

### 8.5 指定服务端 tick 目录

```python
df = get_remote_tick_data(
    stock_codes="000001.SZ",
    start_date="2026-03-02",
    tick_dir="/data1/quant-data/tick_2026",
    base_url="http://10.10.20.8:9999",
)
```

## 9. 注意事项

1. 远端服务端必须已经部署 `/api/tick-data/query/parquet`。
2. tick 数据量通常远大于日频数据，跨较长日期范围时建议适当增大 `timeout`。
3. 如果你主要用于高频特征计算，优先考虑 `short=True`，网络传输量会更小。
4. 服务端默认返回带时区的 `datetime` 索引，客户端会自动恢复为 `DatetimeIndex`。

## 10. 对应服务端接口

`get_remote_tick_data` 依赖服务端已部署以下接口：

- `POST /api/tick-data/query/parquet`

如果服务端版本较旧，不包含这个接口，调用时会抛出 `RemoteDataError`。
