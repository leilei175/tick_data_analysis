import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from glob import glob
from typing import Union, List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor

DAILY_DIR = './daily_data/daily/'
KZZ_DAILY_DIR = './daily_data/kzz_daily/'
DAILY_BASIC_DIR = './daily_data/daily_basic/'
ETF_DAILY_DIR = './daily_data/etf_daily/'
ETF_NAV_DIR = './daily_data/etf_nav/'
ETF_SHARE_DIR = './daily_data/etf_share/'
ETF_METRICS_DIR = './daily_data/etf_metrics/'
CASHFLOW_DAILY_DIR = './daily_data/cashflow_daily/'
INCOME_DAILY_DIR = './daily_data/income_daily/'
BALANCE_DAILY_DIR = './daily_data/balance_daily/'
BALANCE_QUARTER_DIR = './daily_data/balance/'
INCOME_QUARTER_DIR = './daily_data/income/'
CASHFLOW_QUARTER_DIR = './daily_data/cashflow/'
CASHFLOW_DAILY_CN_DIR = './daily_data/cashflow_daily_cn/'
INCOME_DAILY_CN_DIR = './daily_data/income_daily_cn/'
BALANCE_DAILY_CN_DIR = './daily_data/balance_daily_cn/'
DERIVATIVE_DIR = './daily_data/derivative/'

# 可用数据类型
DATA_TYPE_META = {
    'daily': {'data_dir': DAILY_DIR, 'prefix': 'daily', 'code_col': 'ts_code', 'date_col': 'trade_date'},
    'kzz_daily': {'data_dir': KZZ_DAILY_DIR, 'prefix': 'kzz_daily', 'code_col': 'ts_code', 'date_col': 'trade_date'},
    'daily_basic': {'data_dir': DAILY_BASIC_DIR, 'prefix': 'daily_basic', 'code_col': 'ts_code', 'date_col': 'trade_date'},
    'etf_daily': {'data_dir': ETF_DAILY_DIR, 'prefix': 'etf_daily', 'code_col': 'ts_code', 'date_col': 'trade_date'},
    'etf_nav': {'data_dir': ETF_NAV_DIR, 'prefix': 'etf_nav', 'code_col': 'ts_code', 'date_col': 'nav_date'},
    'etf_share': {'data_dir': ETF_SHARE_DIR, 'prefix': 'etf_share', 'code_col': 'ts_code', 'date_col': 'trade_date'},
    'etf_metrics': {'data_dir': ETF_METRICS_DIR, 'prefix': 'etf_metrics', 'code_col': 'ts_code', 'date_col': 'trade_date'},
    'cashflow_daily': {'data_dir': CASHFLOW_DAILY_DIR, 'prefix': 'cashflow_daily', 'code_col': 'ts_code', 'date_col': 'trade_date'},
    'income_daily': {'data_dir': INCOME_DAILY_DIR, 'prefix': 'income_daily', 'code_col': 'ts_code', 'date_col': 'trade_date'},
    'balance_daily': {'data_dir': BALANCE_DAILY_DIR, 'prefix': 'balance_daily', 'code_col': 'ts_code', 'date_col': 'trade_date'},
    'cashflow_q': {'data_dir': CASHFLOW_QUARTER_DIR, 'prefix': 'cashflow', 'code_col': 'ts_code', 'date_col': 'end_date', 'all_file': 'cashflow_all.parquet', 'is_quarterly': True},
    'income_q': {'data_dir': INCOME_QUARTER_DIR, 'prefix': 'income', 'code_col': 'ts_code', 'date_col': 'end_date', 'all_file': 'income_all.parquet', 'is_quarterly': True},
    'balance_q': {'data_dir': BALANCE_QUARTER_DIR, 'prefix': 'balance', 'code_col': 'ts_code', 'date_col': 'end_date', 'all_file': 'balance_all.parquet', 'is_quarterly': True},
    'cashflow_daily_cn': {'data_dir': CASHFLOW_DAILY_CN_DIR, 'prefix': 'cashflow_daily_cn', 'code_col': 'TS代码', 'date_col': '交易日期'},
    'income_daily_cn': {'data_dir': INCOME_DAILY_CN_DIR, 'prefix': 'income_daily_cn', 'code_col': 'TS代码', 'date_col': '交易日期'},
    'balance_daily_cn': {'data_dir': BALANCE_DAILY_CN_DIR, 'prefix': 'balance_daily_cn', 'code_col': 'TS代码', 'date_col': '交易日期'},
    'derivative': {'data_dir': DERIVATIVE_DIR, 'prefix': 'derivative', 'code_col': 'ts_code', 'date_col': 'trade_date'},
    'cashflow_q_cn': {'data_dir': CASHFLOW_QUARTER_DIR, 'prefix': 'cashflow', 'code_col': 'TS代码', 'date_col': '报告期', 'all_file': 'cashflow_all_cn.parquet', 'is_quarterly': True},
    'income_q_cn': {'data_dir': INCOME_QUARTER_DIR, 'prefix': 'income', 'code_col': 'TS代码', 'date_col': '报告期', 'all_file': 'income_all_cn.parquet', 'is_quarterly': True},
    'balance_q_cn': {'data_dir': BALANCE_QUARTER_DIR, 'prefix': 'balance', 'code_col': 'TS代码', 'date_col': '报告期', 'all_file': 'balance_all_cn.parquet', 'is_quarterly': True},
}
DATA_TYPES = list(DATA_TYPE_META.keys())
DATA_TYPE_ALIASES = {
    'balance': 'balance_q',
    'income': 'income_q',
    'cashflow': 'cashflow_q',
    'balance_cn': 'balance_q_cn',
    'income_cn': 'income_q_cn',
    'cashflow_cn': 'cashflow_q_cn',
}


def normalize_data_type(data_type: Optional[str]) -> Optional[str]:
    """标准化 data_type（支持别名）"""
    if data_type is None:
        return None
    key = str(data_type).strip()
    if key == '':
        return ''
    return DATA_TYPE_ALIASES.get(key, key)


def _contains_chinese(text: str) -> bool:
    """是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]', str(text or '')))


def infer_data_type_from_field(
    filed: Union[str, List[str]],
    start: Optional[str] = None,
    end: Optional[str] = None
) -> str:
    """
    根据字段名自动推断 data_type。

    规则：
    - filed 为列表时，选择同时包含所有字段的 data_type
    - filed 为单字段时，选择包含该字段的 data_type
    - 若字段含中文，优先 *_cn；否则优先非 *_cn
    """
    if isinstance(filed, str) and filed.lower() == 'all':
        raise ValueError("filed='all' 时必须显式指定 data_type")

    if isinstance(filed, (list, tuple, set)):
        targets = [str(f).strip() for f in filed if str(f).strip()]
        if not targets:
            raise ValueError("字段列表为空，无法推断 data_type")
    else:
        targets = [str(filed).strip()]
        if not targets[0]:
            raise ValueError("字段名为空，无法推断 data_type")

    prefer_cn = any(_contains_chinese(t) for t in targets)
    ordered_types = (
        [t for t in DATA_TYPES if t.endswith('_cn')] + [t for t in DATA_TYPES if not t.endswith('_cn')]
        if prefer_cn else
        [t for t in DATA_TYPES if not t.endswith('_cn')] + [t for t in DATA_TYPES if t.endswith('_cn')]
    )

    candidates = []
    for dt in ordered_types:
        fields = set(list_data_fields(data_type=dt, include_meta=False, start=start, end=end))
        if all(t in fields for t in targets):
            candidates.append(dt)

    if not candidates:
        raise ValueError(f"无法根据字段 {targets} 推断 data_type，请显式指定 data_type")
    return candidates[0]


def _parse_date_from_filename(filename: str, prefix: str = 'daily') -> Optional[int]:
    """从文件名解析日期，返回8位日期整数，解析失败返回None"""
    pattern = rf'{prefix}_(\d{{8}})\.parquet$'
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))
    return None


def _find_data_files(
    data_dir: str,
    prefix: str,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> List[tuple]:
    """
    查找数据文件，支持年/月目录结构

    Args:
        data_dir: 数据根目录
        prefix: 文件前缀
        start: 开始日期
        end: 结束日期

    Returns:
        List of (date, filepath) tuples, sorted by date
    """
    data_path = Path(data_dir)
    files_with_dates = []

    # 支持两种目录结构：
    # 1. flat: daily_data/daily/*.parquet
    # 2. hierarchical: daily_data/daily/2025/01/*.parquet

    # 查找所有匹配的文件
    patterns = [
        str(data_path / f'{prefix}_*.parquet'),  # 扁平结构
        str(data_path / '*' / '*' / f'{prefix}_*.parquet'),  # 年/月结构
    ]

    for pattern in patterns:
        for f in glob(pattern):
            fname = Path(f).name
            date = _parse_date_from_filename(fname, prefix)
            if date is not None:
                files_with_dates.append((date, f))

    # 根据日期范围过滤
    if start:
        start_date = int(start)
        files_with_dates = [(d, f) for d, f in files_with_dates if d >= start_date]
    if end:
        end_date = int(end)
        files_with_dates = [(d, f) for d, f in files_with_dates if d <= end_date]

    # 按日期排序
    files_with_dates.sort(key=lambda x: x[0])

    return files_with_dates


def _read_single_file(args: tuple) -> tuple:
    """读取单个parquet文件"""
    date, filepath, filed, code_col = args
    try:
        table = pq.read_table(filepath, columns=[code_col, filed])
        df = table.to_pandas()
        if code_col != 'ts_code':
            df = df.rename(columns={code_col: 'ts_code'})
        df['trade_date'] = date
        return df
    except Exception:
        return None


def _get_merged_file_path(data_dir: str, year: str) -> Path:
    """获取合并后的年度文件路径"""
    return Path(data_dir) / f'{year}_all.parquet'

def _read_merged_file(
    merged_file: Path,
    columns: List[str],
    code_col: str,
    sec_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    读取年度合并文件（仅读取必要列）。
    优先使用 pyarrow 并下推股票过滤，失败时回退 pandas.read_parquet。
    """
    read_columns = [c for c in dict.fromkeys(columns) if c]
    filters = None
    if sec_list:
        filters = [(code_col, 'in', sec_list)]

    try:
        table = pq.read_table(str(merged_file), columns=read_columns, filters=filters)
        return table.to_pandas()
    except Exception:
        # 回退路径：至少保证列裁剪仍生效
        return pd.read_parquet(merged_file, columns=read_columns)

def _get_merged_max_date(data_dir: str, year: str, date_col: str = 'trade_date') -> Optional[int]:
    """获取年度合并文件中的最大trade_date，失败时返回None"""
    merged_file = _get_merged_file_path(data_dir, year)
    if not merged_file.exists():
        return None
    try:
        table = pq.read_table(merged_file, columns=[date_col])
        if table.num_rows == 0:
            return None
        arr = table.column(date_col).to_pandas()
        if arr.empty:
            return None
        return int(pd.to_numeric(arr, errors='coerce').dropna().max())
    except Exception:
        return None


def _is_merged_file_available(data_dir: str, start: Optional[str], end: Optional[str]) -> bool:
    """检查合并文件是否可用"""
    if not start or not end:
        return False

    # 判断是否跨年
    start_year = start[:4]
    end_year = end[:4]

    if start_year == end_year:
        # 单年，检查合并文件是否存在
        merged_file = _get_merged_file_path(data_dir, start_year)
        return merged_file.exists()

    return False


def _normalize_sec_list(sec_list: Optional[List[str]]) -> Optional[List[str]]:
    """规范化股票代码列表：去空白、转大写、去重。"""
    if sec_list is None:
        return None
    norm = []
    seen = set()
    for s in sec_list:
        if s is None:
            continue
        v = str(s).strip().upper()
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            norm.append(v)
    return norm


def list_data_fields(
    data_type: str = 'daily',
    data_dir: Optional[str] = None,
    include_meta: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> List[str]:
    """
    查看指定 data_type 的可用字段名。

    Args:
        data_type: 数据类型
        data_dir: 自定义目录，None 时使用默认目录
        include_meta: 是否包含代码/日期列
        start: 可选，日期范围起点（用于辅助选择样本文件）
        end: 可选，日期范围终点（用于辅助选择样本文件）

    Returns:
        List[str]: 字段名列表
    """
    data_type = normalize_data_type(data_type)
    if data_type not in DATA_TYPES:
        raise ValueError(f"不支持的数据类型: {data_type}，支持: {DATA_TYPES}")

    meta = DATA_TYPE_META[data_type]
    code_col = meta['code_col']
    date_col = meta['date_col']

    if data_dir is None:
        data_dir = meta['data_dir']
        prefix = meta['prefix']
    else:
        prefix = data_type

    data_path = Path(data_dir)

    # 季度类型优先按固定 all_file 读取 schema
    all_file_name = meta.get('all_file')
    if all_file_name:
        all_file = data_path / all_file_name
        if all_file.exists():
            try:
                schema_names = pq.read_schema(str(all_file)).names
                if include_meta:
                    return schema_names
                return [c for c in schema_names if c not in [code_col, date_col]]
            except Exception:
                pass

    # 优先使用年度合并文件读取 schema（更快）
    merged_files = sorted(data_path.glob('*_all*.parquet'))
    if merged_files:
        try:
            schema_names = pq.read_schema(str(merged_files[-1])).names
            if include_meta:
                return schema_names
            return [c for c in schema_names if c not in [code_col, date_col]]
        except Exception:
            pass

    # 回退到任意一个日文件
    files_with_dates = _find_data_files(data_dir, prefix, start, end)
    if not files_with_dates and (start or end):
        files_with_dates = _find_data_files(data_dir, prefix, None, None)
    if not files_with_dates:
        return []

    sample_file = files_with_dates[-1][1]
    try:
        schema_names = pq.read_schema(sample_file).names
        if include_meta:
            return schema_names
        return [c for c in schema_names if c not in [code_col, date_col]]
    except Exception:
        return []


def _to_date_key_int(series: pd.Series) -> pd.Series:
    """将日期列标准化为 YYYYMMDD int（无法解析则 NaN）"""
    s = series.astype(str).str.replace('-', '', regex=False).str.replace('/', '', regex=False).str.strip()
    return pd.to_numeric(s, errors='coerce')


def _get_quarterly_all_file(data_type: str, data_dir: str) -> Optional[Path]:
    """获取季度数据 all 文件路径"""
    meta = DATA_TYPE_META[data_type]
    all_file_name = meta.get('all_file')
    base = Path(data_dir)
    if all_file_name:
        p = base / all_file_name
        if p.exists():
            return p
    # 回退：按前缀匹配 *_all*.parquet
    cands = sorted(base.glob(f"{meta['prefix']}_all*.parquet"))
    if cands:
        return cands[0]
    return None


def _get_quarterly_local_data(
    sec_list: Optional[List[str]],
    start: Optional[str],
    end: Optional[str],
    filed: str,
    data_type: str,
    data_dir: Optional[str] = None
) -> pd.DataFrame:
    """读取季度财报数据并转宽表（index=end_date, columns=ts_code）"""
    meta = DATA_TYPE_META[data_type]
    code_col = meta['code_col']
    date_col = meta['date_col']
    if data_dir is None:
        data_dir = meta['data_dir']

    all_file = _get_quarterly_all_file(data_type, data_dir)
    if all_file is None:
        return pd.DataFrame()

    read_cols = [code_col, date_col, filed]
    filters = None
    if sec_list:
        filters = [(code_col, 'in', sec_list)]

    try:
        table = pq.read_table(str(all_file), columns=read_cols, filters=filters)
        df_all = table.to_pandas()
    except Exception:
        try:
            df_all = pd.read_parquet(all_file, columns=read_cols)
        except Exception:
            return pd.DataFrame()

    if df_all.empty:
        return pd.DataFrame()
    if code_col != 'ts_code' and code_col in df_all.columns:
        df_all = df_all.rename(columns={code_col: 'ts_code'})
    if date_col != 'end_date' and date_col in df_all.columns:
        df_all = df_all.rename(columns={date_col: 'end_date'})

    if sec_list is not None and len(sec_list) > 0:
        df_all = df_all[df_all['ts_code'].isin(sec_list)]

    date_key = _to_date_key_int(df_all['end_date'])
    df_all = df_all[date_key.notna()].copy()
    if df_all.empty:
        return pd.DataFrame()
    df_all['end_date_key'] = date_key.loc[df_all.index].astype(int)

    if start:
        df_all = df_all[df_all['end_date_key'] >= int(start)]
    if end:
        df_all = df_all[df_all['end_date_key'] <= int(end)]

    if df_all.empty:
        return pd.DataFrame()
    df_all = df_all.dropna(subset=[filed])
    if df_all.empty:
        return pd.DataFrame()

    df_all = df_all.drop_duplicates(subset=['end_date_key', 'ts_code'], keep='last')
    df_pivot = df_all.pivot(index='end_date_key', columns='ts_code', values=filed)
    df_pivot.index = pd.to_datetime(df_pivot.index.astype(str), format='%Y%m%d', errors='coerce')
    df_pivot = df_pivot[~pd.isna(df_pivot.index)].sort_index()
    df_pivot.index.name = 'date'
    return df_pivot


def get_local_data(
    sec_list: Union[List[str], None] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    filed: Union[str, List[str]] = 'close',
    data_type: Optional[str] = 'daily',
    data_dir: Optional[str] = None,
    parallel: bool = True,
    max_workers: int = 8
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    获取日频数据

    Args:
        sec_list: 股票代码列表，如 ['000001.SZ', '000002.SZ']，None表示所有股票
        start: 开始日期，格式 'YYYYMMDD'
        end: 结束日期，格式 'YYYYMMDD'
        filed: 要获取的字段名，默认'close'
        data_type: 数据类型，支持 'daily', 'daily_basic', 'cashflow_daily', 'income_daily', 'balance_daily'
        data_dir: 数据目录路径，默认为None，会根据data_type自动选择
        parallel: 是否并行读取（默认True）
        max_workers: 并行线程数（默认8）

    Returns:
        filed 为单字段时:
            DataFrame: index为日期，columns为股票代码
        filed='all' 或 filed 为字段列表时:
            Dict[str, DataFrame]: {字段名: 宽表DataFrame}

    Example:
        >>> # 日线收盘价
        >>> get_local_data(['000001.SZ'], '20250101', '20250110', 'close', 'daily')
        >>> # 每日基本面换手率
        >>> get_local_data(['000001.SZ'], '20250101', '20250110', 'turnover_rate', 'daily_basic')
        >>> # 每日现金流
        >>> get_local_data(['000001.SZ'], '20250101', '20250110', 'n_cashflow_act', 'cashflow_daily')
    """
    # 允许 data_type 为空：按字段自动推断
    data_type = normalize_data_type(data_type)
    data_type = (str(data_type).strip() if data_type is not None else '')
    if not data_type:
        data_type = infer_data_type_from_field(filed=filed, start=start, end=end)
    if data_type not in DATA_TYPES:
        raise ValueError(f"不支持的数据类型: {data_type}，支持: {DATA_TYPES}")
    sec_list = _normalize_sec_list(sec_list)

    # 批量字段读取
    if isinstance(filed, str) and filed.lower() == 'all':
        fields = list_data_fields(
            data_type=data_type,
            data_dir=data_dir,
            include_meta=False,
            start=start,
            end=end
        )
        if not fields:
            return {}

        # 优先使用批量读取路径
        try:
            return get_all_data(
                data_type=data_type,
                start=start,
                end=end,
                sec_list=sec_list,
                fields=fields,
                parallel=parallel
            )
        except Exception:
            pass

        return {
            f: get_local_data(
                sec_list=sec_list,
                start=start,
                end=end,
                filed=f,
                data_type=data_type,
                data_dir=data_dir,
                parallel=parallel,
                max_workers=max_workers
            )
            for f in fields
        }

    if isinstance(filed, (list, tuple, set)):
        return {
            str(f): get_local_data(
                sec_list=sec_list,
                start=start,
                end=end,
                filed=str(f),
                data_type=data_type,
                data_dir=data_dir,
                parallel=parallel,
                max_workers=max_workers
            )
            for f in filed
        }

    meta = DATA_TYPE_META[data_type]
    if bool(meta.get('is_quarterly')):
        return _get_quarterly_local_data(
            sec_list=sec_list,
            start=start,
            end=end,
            filed=str(filed),
            data_type=data_type,
            data_dir=data_dir
        )

    meta = DATA_TYPE_META[data_type]

    # 根据 data_type 确定默认目录和文件前缀
    if data_dir is None:
        data_dir = meta['data_dir']
        prefix = meta['prefix']
        code_col = meta['code_col']
        date_col = meta['date_col']
    else:
        # 从 data_dir 推断 prefix
        prefix = data_type
        code_col = meta['code_col']
        date_col = meta['date_col']

    # 构建数据
    dfs = []

    # 根据日期范围决定读取策略
    start_year = start[:4] if start else None
    end_year = end[:4] if end else None
    is_single_year = (start_year == end_year and
                      _is_merged_file_available(data_dir, start, end))
    read_columns = [code_col, date_col, filed]

    if is_single_year and parallel:
        # 单年：使用合并文件
        year = start_year
        merged_file = _get_merged_file_path(data_dir, year)
        print(f"[优化] 使用合并文件: {merged_file.name}")

        df_all = _read_merged_file(
            merged_file=merged_file,
            columns=read_columns,
            code_col=code_col,
            sec_list=sec_list
        )

        # 按日期过滤
        start_int = int(start)
        end_int = int(end)
        if date_col != 'trade_date' and date_col in df_all.columns:
            df_all = df_all.rename(columns={date_col: 'trade_date'})
        if code_col != 'ts_code' and code_col in df_all.columns:
            df_all = df_all.rename(columns={code_col: 'ts_code'})
        if df_all['trade_date'].dtype == object:
            df_all['trade_date'] = df_all['trade_date'].astype(int)
        df_all = df_all[(df_all['trade_date'] >= start_int) &
                        (df_all['trade_date'] <= end_int)]

    elif parallel and start_year != end_year and start_year and end_year:
        # 跨年：并行读取多个合并文件
        def read_year_file(year):
            merged_file = _get_merged_file_path(data_dir, year)
            if merged_file.exists():
                return _read_merged_file(
                    merged_file=merged_file,
                    columns=read_columns,
                    code_col=code_col,
                    sec_list=sec_list
                )
            return None

        def read_year_daily(year):
            """读取某年所有每日小文件"""
            year_dir = Path(data_dir) / year
            if not year_dir.exists():
                return None
            files = list(year_dir.glob(f'*/{prefix}_*.parquet'))
            if not files:
                return None
            dfs = []
            for f in sorted(files):
                try:
                    df = pd.read_parquet(f, columns=[code_col, filed])
                    if code_col != 'ts_code':
                        df = df.rename(columns={code_col: 'ts_code'})
                    file_date = _parse_date_from_filename(f.name, prefix)
                    if file_date is None:
                        continue
                    df['trade_date'] = file_date
                    dfs.append(df)
                except:
                    pass
            if dfs:
                return pd.concat(dfs, ignore_index=True)
            return None

        # 收集所有需要读取的年份
        start_yr = int(start_year)
        end_yr = int(end_year)
        all_years = list(range(start_yr, end_yr + 1))

        # 检查哪些年份有合并文件
        years_with_merged = []
        years_without_merged = []
        for yr in all_years:
            merged_file = _get_merged_file_path(data_dir, str(yr))
            if merged_file.exists():
                # 若合并文件落后于查询end日期，则回退到每日文件，避免读到旧数据
                if end and yr == end_yr:
                    merged_max_date = _get_merged_max_date(data_dir, str(yr), date_col=date_col)
                    if merged_max_date is None or merged_max_date < int(end):
                        years_without_merged.append(yr)
                        continue
                years_with_merged.append(yr)
            else:
                years_without_merged.append(yr)

        print(f"[优化] 读取 {len(all_years)} 年数据 ({len(years_with_merged)} 个合并文件, {len(years_without_merged)} 个每日文件)")

        all_dfs = []

        # 并行读取有合并文件的年份
        if years_with_merged:
            with ThreadPoolExecutor(max_workers=min(len(years_with_merged), 6)) as executor:
                dfs_merged = list(executor.map(read_year_file, years_with_merged))
            all_dfs.extend([d for d in dfs_merged if d is not None])

        # 读取没有合并文件的年份
        for yr in years_without_merged:
            df = read_year_daily(str(yr))
            if df is not None:
                all_dfs.append(df)

        df_all = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        # 按日期过滤
        if not df_all.empty:
            if date_col != 'trade_date' and date_col in df_all.columns:
                df_all = df_all.rename(columns={date_col: 'trade_date'})
            if code_col != 'ts_code' and code_col in df_all.columns:
                df_all = df_all.rename(columns={code_col: 'ts_code'})
            if df_all['trade_date'].dtype == object:
                df_all['trade_date'] = df_all['trade_date'].astype(int)
            if start:
                df_all = df_all[df_all['trade_date'] >= int(start)]
            if end:
                df_all = df_all[df_all['trade_date'] <= int(end)]

    else:
        # 仅在未命中年度合并优化路径时，才扫描日文件
        files_with_dates = _find_data_files(data_dir, prefix, start, end)
        if not files_with_dates:
            return pd.DataFrame()

        if parallel and len(files_with_dates) > 10:
            # 并行读取每日小文件
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                args_list = [(date, f, filed, code_col) for date, f in files_with_dates]
                results = list(executor.map(_read_single_file, args_list))
                dfs = [r for r in results if r is not None]
            df_all = pd.concat(dfs, ignore_index=True)
        else:
            # 串行读取
            for date, f in files_with_dates:
                table = pq.read_table(f, columns=[code_col, filed])
                df = table.to_pandas()
                if code_col != 'ts_code':
                    df = df.rename(columns={code_col: 'ts_code'})
                df['trade_date'] = date
                dfs.append(df)
            df_all = pd.concat(dfs, ignore_index=True)

    if df_all.empty:
        return pd.DataFrame()

    # 过滤股票
    if sec_list is not None and len(sec_list) > 0:
        df_all = df_all[df_all['ts_code'].isin(sec_list)]

    # 过滤空值并去重
    df_all = df_all.dropna(subset=[filed])
    if df_all.empty:
        return pd.DataFrame()
    df_all = df_all.drop_duplicates(subset=['trade_date', 'ts_code'], keep='first')

    # 确保trade_date是有效整数
    trade_date_num = pd.to_numeric(df_all['trade_date'], errors='coerce')
    df_all = df_all[trade_date_num.notna()].copy()
    if df_all.empty:
        return pd.DataFrame()
    df_all['trade_date'] = trade_date_num.loc[df_all.index].astype(int)

    # 转为宽表
    df_pivot = df_all.pivot(index='trade_date', columns='ts_code', values=filed)

    # 安全转换日期
    try:
        df_pivot.index = pd.to_datetime(df_pivot.index.astype(str), format='%Y%m%d', errors='raise')
    except ValueError:
        # 如果直接转换失败，使用mixed格式
        df_pivot.index = pd.to_datetime(df_pivot.index.astype(str), format='mixed', errors='coerce')

    df_pivot.index.name = 'date'

    # 按日期排序索引
    df_pivot = df_pivot.sort_index()

    return df_pivot


def list_data_files(
    data_type: str = 'daily',
    data_dir: Optional[str] = None,
    year: Optional[str] = None,
    month: Optional[str] = None
) -> List[tuple]:
    """
    列出数据文件

    Args:
        data_type: 数据类型
        data_dir: 自定义目录
        year: 筛选年份 (YYYY)
        month: 筛选月份 (MM)

    Returns:
        List of (date, filepath) tuples
    """
    data_type = normalize_data_type(data_type)
    if data_type not in DATA_TYPES:
        raise ValueError(f"不支持的数据类型: {data_type}，支持: {DATA_TYPES}")

    if data_dir is None:
        data_dir = DATA_TYPE_META[data_type]['data_dir']
        prefix = DATA_TYPE_META[data_type]['prefix']
    else:
        prefix = data_type

    if DATA_TYPE_META[data_type].get('is_quarterly'):
        # 季度类型无按日文件结构
        return []

    files = _find_data_files(data_dir, prefix)

    if year:
        files = [(d, f) for d, f in files if str(d).startswith(year)]
    if month:
        files = [(d, f) for d, f in files if str(d)[4:6] == month]

    return files


def _get_full_file_path(data_dir: str, year: str) -> Path:
    """获取完整合并文件的路径"""
    return Path(data_dir) / f'{year}_full.parquet'

def _get_full_max_date(data_dir: str, year: str, date_col: str = 'trade_date') -> Optional[int]:
    """获取年度full文件中的最大trade_date，失败时返回None"""
    full_file = _get_full_file_path(data_dir, year)
    if not full_file.exists():
        return None
    try:
        table = pq.read_table(full_file, columns=[date_col])
        if table.num_rows == 0:
            return None
        arr = table.column(date_col).to_pandas()
        if arr.empty:
            return None
        return int(pd.to_numeric(arr, errors='coerce').dropna().max())
    except Exception:
        return None


def get_all_data(
    data_type: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    sec_list: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
    parallel: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    快速获取指定类型的所有数据

    Args:
        data_type: 数据类型 ('daily', 'daily_basic', 'cashflow_daily', 'income_daily', 'balance_daily')
        start: 开始日期，格式 'YYYYMMDD'
        end: 结束日期，格式 'YYYYMMDD'
        sec_list: 股票代码列表，None表示所有股票
        fields: 字段列表，None表示获取所有可用字段
        parallel: 是否并行读取（仍会一次性读取所有字段）

    Returns:
        Dict: {field_name: DataFrame}

    Example:
        >>> data = get_all_data('daily', start='20200101', end='20201231')
        >>> close_df = data['close']
    """
    from concurrent.futures import ThreadPoolExecutor

    data_type = normalize_data_type(data_type)
    if data_type not in DATA_TYPES:
        raise ValueError(f"未知数据类型: {data_type}")
    sec_list = _normalize_sec_list(sec_list)

    meta = DATA_TYPE_META[data_type]
    if bool(meta.get('is_quarterly')):
        # 季度类型回退到逐字段读取
        if fields is None:
            fields = list_data_fields(data_type=data_type, include_meta=False, start=start, end=end)
        return {
            str(f): get_local_data(
                sec_list=sec_list,
                start=start,
                end=end,
                filed=str(f),
                data_type=data_type,
                parallel=parallel
            ) for f in (fields or [])
        }
    data_dir = meta['data_dir']
    code_col = meta['code_col']
    date_col = meta['date_col']
    prefix = meta['prefix']

    # 定义各数据类型的可用字段（英文类型使用固定列表，中文类型自动推断）
    DATA_FIELDS = {
        'daily': ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                  'pre_close', 'change', 'pct_chg', 'vol', 'amount'],
        'kzz_daily': ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                      'pre_close', 'change', 'pct_chg', 'vol', 'amount'],
        'daily_basic': ['ts_code', 'trade_date', 'close', 'turnover_rate', 'turnover_rate_f',
                       'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio',
                       'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv'],
        'cashflow_daily': ['ts_code', 'trade_date', 'n_cashflow_act', 'n_cashflow_inv_act',
                          'n_cash_flows_fnc_act', 'c_fr_sale_sg', 'c_paid_goods_s',
                          'c_paid_to_for_empl', 'c_recp_borrow', 'proc_issue_bonds'],
        'income_daily': ['ts_code', 'trade_date', 'total_revenue', 'revenue', 'int_income',
                         'operate_profit', 'total_profit', 'income_tax', 'n_income',
                         'basic_eps', 'diluted_eps', 'ebit', 'total_cogs', 'oper_cost'],
        'balance_daily': ['ts_code', 'trade_date', 'total_assets', 'total_liab', 'total_cur_assets',
                         'total_cur_liab', 'cash_reser_cb', 'accounts_receiv', 'inventories',
                         'total_hldr_eqy_exc_min_int', 'st_borr', 'lt_borr', 'bond_payable'],
        'derivative': ['ts_code', 'trade_date', 'roe', 'roa', 'gross_margin', 'roic', 'enterprise_value']
    }

    # 确定要读取的字段
    if data_type in DATA_FIELDS:
        all_fields = DATA_FIELDS[data_type]
    else:
        all_fields = list_data_fields(
            data_type=data_type,
            data_dir=data_dir,
            include_meta=True,
            start=start,
            end=end
        )
        if not all_fields:
            return {}

    if fields is None:
        fields = [f for f in all_fields if f not in [code_col, date_col]]
    else:
        # 验证字段
        fields = [f for f in fields if f in all_fields and f not in [code_col, date_col]]

    # 确定年份范围
    start_year = start[:4] if start else None
    end_year = end[:4] if end else None
    if not start_year or not end_year:
        raise ValueError("需要指定start和end日期")

    start_yr = int(start_year)
    end_yr = int(end_year)
    all_years = list(range(start_yr, end_yr + 1))

    # 检查哪些年份有full文件
    years_with_full = []
    years_without_full = []
    for yr in all_years:
        full_file = _get_full_file_path(data_dir, str(yr))
        if full_file.exists():
            # 若结束年份full文件落后于查询end日期，则回退到每日文件，避免读到旧数据
            if yr == end_yr:
                full_max_date = _get_full_max_date(data_dir, str(yr), date_col=date_col)
                if full_max_date is None or full_max_date < int(end):
                    years_without_full.append(yr)
                    continue
            years_with_full.append(yr)
        else:
            years_without_full.append(yr)

    print(f"[get_all_data] {data_type}: {len(all_years)}年数据, {len(years_with_full)}个full文件, {len(years_without_full)}个需合并")

    # 读取函数
    def read_full_year(yr: int) -> pd.DataFrame:
        full_file = _get_full_file_path(data_dir, str(yr))
        if full_file.exists():
            return _read_merged_file(
                merged_file=full_file,
                columns=all_fields,
                code_col=code_col,
                sec_list=sec_list
            )
        return None

    all_dfs = []

    # 并行读取有full文件的年份
    if years_with_full:
        with ThreadPoolExecutor(max_workers=min(len(years_with_full), 6)) as executor:
            for df in executor.map(read_full_year, years_with_full):
                if df is not None:
                    all_dfs.append(df)

    # 处理没有full文件的年份（如果有的话）
    for yr in years_without_full:
        year_files = _find_data_files(data_dir, prefix, f'{yr}0101', f'{yr}1231')
        if year_files:
            dfs = []
            for date, f in year_files:
                df = _read_merged_file(
                    merged_file=Path(f),
                    columns=all_fields,
                    code_col=code_col,
                    sec_list=sec_list
                )
                dfs.append(df)
            if dfs:
                all_dfs.append(pd.concat(dfs, ignore_index=True))

    if not all_dfs:
        if sec_list:
            print("[提示] sec_list 与当前数据代码无匹配，请检查是否为 000001.SZ/600000.SH 格式")
        return {f: pd.DataFrame() for f in fields}

    # 合并所有年份
    df_all = pd.concat(all_dfs, ignore_index=True)
    if date_col != 'trade_date' and date_col in df_all.columns:
        df_all = df_all.rename(columns={date_col: 'trade_date'})
    if code_col != 'ts_code' and code_col in df_all.columns:
        df_all = df_all.rename(columns={code_col: 'ts_code'})

    # 按日期过滤
    start_int = int(start)
    end_int = int(end)
    if df_all['trade_date'].dtype == object:
        df_all['trade_date'] = df_all['trade_date'].astype(int)
    df_all = df_all[(df_all['trade_date'] >= start_int) & (df_all['trade_date'] <= end_int)]

    # 过滤股票
    if sec_list:
        df_all = df_all[df_all['ts_code'].isin(sec_list)]
        if df_all.empty:
            print("[提示] sec_list 过滤后为空，请检查代码格式（大小写/空格/是否含 .SZ/.SH）")

    # 去重
    df_all = df_all.drop_duplicates(subset=['trade_date', 'ts_code'], keep='first')

    # 拆分字段 - 批量unstack优化版本
    result = {}

    # 获取唯一的日期和股票（注意：按整数排序而非字符串排序）
    unique_dates_int = sorted(df_all['trade_date'].unique())
    unique_stocks = df_all['ts_code'].unique()

    # 创建日期索引（用于最终结果）
    date_index = pd.to_datetime([str(d) for d in unique_dates_int], format='%Y%m%d')
    date_index.name = 'date'

    # 设置索引
    df_indexed = df_all.set_index(['trade_date', 'ts_code'])

    # 只选择需要的字段
    available_fields = [f for f in fields if f in df_indexed.columns]

    if available_fields:
        # 批量unstack：一次性展开所有字段，性能提升6倍
        df_multi = df_indexed[available_fields]
        df_unstack = df_multi.unstack(level='ts_code')  # 结果是 MultiIndex 列

        # 转换索引
        df_unstack.index = date_index

        # 拆分每个字段（先提取再转换列类型）
        for field in available_fields:
            # 从MultiIndex列中提取单个字段
            df_field = df_unstack[field]
            df_field.columns = df_field.columns.astype(str)
            result[field] = df_field

    # 对于不存在的字段，创建空DataFrame
    for field in fields:
        if field not in result:
            result[field] = pd.DataFrame(
                np.full((len(unique_dates_int), len(unique_stocks)), np.nan),
                index=date_index,
                columns=[str(s) for s in unique_stocks]
            )

    return result


def get_daily_all(
    start: Optional[str] = None,
    end: Optional[str] = None,
    sec_list: Optional[List[str]] = None,
    parallel: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    快速获取日线所有数据（便捷函数）

    等价于: get_all_data('daily', start=start, end=end, sec_list=sec_list)

    Args:
        start: 开始日期
        end: 结束日期
        sec_list: 股票代码列表
        parallel: 是否并行

    Returns:
        Dict: {'open': df, 'high': df, 'low': df, 'close': df, ...}
    """
    return get_all_data('daily', start=start, end=end, sec_list=sec_list, parallel=parallel)


def get_kzz_daily_all(
    start: Optional[str] = None,
    end: Optional[str] = None,
    sec_list: Optional[List[str]] = None,
    parallel: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    快速获取可转债日线所有数据（便捷函数）

    等价于: get_all_data('kzz_daily', start=start, end=end, sec_list=sec_list)
    """
    return get_all_data('kzz_daily', start=start, end=end, sec_list=sec_list, parallel=parallel)


def get_daily_basic_all(
    start: Optional[str] = None,
    end: Optional[str] = None,
    sec_list: Optional[List[str]] = None,
    parallel: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    快速获取每日基本面所有数据（便捷函数）

    等价于: get_all_data('daily_basic', start=start, end=end, sec_list=sec_list)

    Returns:
        Dict: {'close': df, 'turnover_rate': df, 'pe': df, ...}
    """
    return get_all_data('daily_basic', start=start, end=end, sec_list=sec_list, parallel=parallel)
