"""
Tushare VIP 财务报表数据下载脚本
=============================

功能：
- 使用 Tushare VIP 接口下载现金流量表、资产负债表、利润表
- 直接获取 2015-2025 年历史数据
- 保存为 Parquet 格式

使用说明：
---------

1. 配置 Token（从 config.py 加载）：
   from config import tushare_tk
   from financial_vip_downloader import set_token
   set_token(tushare_tk)

2. 命令行运行：

   # 下载全部三张表 (2015-2025)
   python financial_vip_downloader.py --all

   # 只下载资产负债表
   python financial_vip_downloader.py --all --balance

   # 指定时间范围
   python financial_vip_downloader.py --start 20150101 --end 20251231

   # 合并为一个大文件
   python financial_vip_downloader.py --all --merge

3. Python API：

   from financial_vip_downloader import (
       download_balancesheet_vip,
       download_income_vip,
       download_cashflow_vip,
       download_all_vip
   )
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import tushare as ts
    HAS_TUSHARE = True
except ImportError:
    HAS_TUSHARE = False
    print("警告: 未安装 tushare")

# =============================================================================
# 配置
# =============================================================================

DEFAULT_DATA_DIR = './daily_data/'

# 默认时间范围
DEFAULT_START = '20150101'
DEFAULT_END = '20251231'

# 季度结束日期列表
QUARTER_ENDS = [
    # 2015年
    '20150331', '20150630', '20150930', '20151231',
    # 2016年
    '20160331', '20160630', '20160930', '20161231',
    # 2017年
    '20170331', '20170630', '20170930', '20171231',
    # 2018年
    '20180331', '20180630', '20180930', '20181231',
    # 2019年
    '20190331', '20190630', '20190930', '20191231',
    # 2020年
    '20200331', '20200630', '20200930', '20201231',
    # 2021年
    '20210331', '20210630', '20210930', '20211231',
    # 2022年
    '20220331', '20220630', '20220930', '20221231',
    # 2023年
    '20230331', '20230630', '20230930', '20231231',
    # 2024年
    '20240331', '20240630', '20240930', '20241231',
    # 2025年
    '20250331', '20250630', '20250930', '20251231',
]

# 核心字段列表（常用指标）
BALANCE_CORE_FIELDS = [
    'ts_code', 'end_date', 'report_type',
    'total_assets', 'total_liab', 'total_hldr_eqy_exc_min_int',
    'total_cur_assets', 'total_nca', 'total_cur_liab', 'total_ncl',
    'cash_reser_cb', 'trad_asset', 'notes_receiv', 'accounts_receiv',
    'inventories', 'fix_assets', 'intan_assets',
    'st_borr', 'lt_borr', 'notes_payable', 'bond_payable',
    'accounts_payable', 'total_share', 'cap_rese', 'surplus_rese',
    'undistr_porfit', 'minority_int'
]

INCOME_CORE_FIELDS = [
    'ts_code', 'end_date', 'report_type',
    'total_revenue', 'revenue', 'int_income', 'comm_income',
    'oper_cost', 'operate_profit', 'total_profit', 'income_tax',
    'n_income', 'n_income_attr_p', 'minority_gain',
    'basic_eps', 'diluted_eps', 'ebit', 'ebitda',
    'fin_exp', 'sell_exp', 'admin_exp', 'int_exp',
    'invest_income', 'total_cogs'
]

CASHFLOW_CORE_FIELDS = [
    'ts_code', 'end_date', 'report_type',
    'n_cashflow_act', 'n_cashflow_inv_act', 'n_cash_flows_fnc_act',
    'c_fr_sale_sg', 'c_paid_goods_s', 'c_paid_to_for_empl',
    'finan_exp', 'c_recp_borrow', 'proc_issue_bonds',
    'net_profit', 'depr_fa_coga_dpba', 'amort_intang_assets'
]

# 所有字段
BALANCE_ALL_FIELDS = None  # None 表示获取所有字段
INCOME_ALL_FIELDS = None
CASHFLOW_ALL_FIELDS = None

# =============================================================================
# Token 管理
# =============================================================================

def set_token(token: str):
    """设置 Tushare Token"""
    ts.set_token(token)
    print(f"Token 已设置")


def get_token_from_config(config_path: str = 'config.py') -> str:
    """从配置文件读取 Token"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config.tushare_tk
    except Exception as e:
        raise ValueError(f"无法从 {config_path} 读取 Token: {e}")


def init_tushare(token: str = None):
    """初始化 Tushare"""
    if not HAS_TUSHARE:
        raise ImportError("未安装 tushare")

    if token is None:
        token = get_token_from_config()

    ts.set_token(token)
    pro = ts.pro_api()
    return pro


# =============================================================================
# 辅助函数
# =============================================================================

def get_date_list(start_date: str, end_date: str) -> List[str]:
    """获取日期列表（按年分段，避免数据量过大）"""
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    dates = []
    for year in range(start_year, end_year + 1):
        dates.append(f'{year}0101')
        dates.append(f'{year}1231')

    return sorted(list(set(dates)))


def get_ts_code_list(pro) -> List[str]:
    """获取所有股票代码列表"""
    try:
        df = pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code'
        )
        return df['ts_code'].tolist()
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return []


def batch_download(pro, ts_codes: List[str], start_date: str, end_date: str,
                   interface: str, fields: List[str] = None,
                   batch_size: int = 100) -> pd.DataFrame:
    """
    批量下载数据

    Args:
        pro: Tushare Pro API
        ts_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        interface: 接口名 (balancesheet_vip, income_vip, cashflow_vip)
        fields: 字段列表
        batch_size: 每批股票数量
    """
    all_data = []
    total = len(ts_codes)
    batches = [ts_codes[i:i + batch_size] for i in range(0, total, batch_size)]

    for i, batch in enumerate(batches):
        ts_code_str = ','.join(batch)

        try:
            if interface == 'balancesheet_vip':
                df = pro.balancesheet_vip(
                    ts_code=ts_code_str,
                    start_date=start_date,
                    end_date=end_date,
                    fields=','.join(fields) if fields else None
                )
            elif interface == 'income_vip':
                df = pro.income_vip(
                    ts_code=ts_code_str,
                    start_date=start_date,
                    end_date=end_date,
                    fields=','.join(fields) if fields else None
                )
            elif interface == 'cashflow_vip':
                df = pro.cashflow_vip(
                    ts_code=ts_code_str,
                    start_date=start_date,
                    end_date=end_date,
                    fields=','.join(fields) if fields else None
                )
            else:
                raise ValueError(f"未知接口: {interface}")

            if not df.empty:
                all_data.append(df)

            # 进度
            processed = min((i + 1) * batch_size, total)
            print(f"\r  进度: {processed}/{total} ({processed * 100 // total}%)", end='', flush=True)

            # 避免限流
            time.sleep(0.5)

        except Exception as e:
            print(f"\n  批次 {i+1} 失败: {str(e)[:50]}")

    print()
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# 下载函数
# =============================================================================

def download_balancesheet_vip(
    start_date: str,
    end_date: str,
    output_dir: str = None,
    fields: List[str] = None,
    use_all_fields: bool = False,
    skip_existing: bool = True
) -> pd.DataFrame:
    """
    下载资产负债表数据 (balancesheet_vip)

    Args:
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
        fields: 字段列表，None 表示使用核心字段
        use_all_fields: 是否使用所有字段
        skip_existing: 跳过已存在的合并文件
    """
    output_dir = output_dir or os.path.join(DEFAULT_DATA_DIR, 'balance')
    interface = 'balancesheet_vip'

    print(f"\n{'='*60}")
    print(f"下载资产负债表 (balancesheet_vip)")
    print(f"时间范围: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    # 选择字段
    if use_all_fields:
        selected_fields = None
        fields_desc = "全部字段"
    else:
        selected_fields = fields or BALANCE_CORE_FIELDS
        fields_desc = f"核心字段 ({len(selected_fields)}个)"
    print(f"字段模式: {fields_desc}")

    pro = init_tushare()

    # 获取股票列表
    print("获取股票列表...")
    ts_codes = get_ts_code_list(pro)
    print(f"股票数量: {len(ts_codes)}")

    # 检查现有文件
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged_file = output_path / 'balance_all.parquet'

    if skip_existing and merged_file.exists():
        print(f"文件已存在: {merged_file}")
        return pq.read_table(merged_file).to_pandas()

    # 下载数据
    print(f"\n开始下载...")
    df = batch_download(
        pro, ts_codes, start_date, end_date,
        interface, selected_fields
    )

    if df.empty:
        print("未下载到数据")
        return pd.DataFrame()

    # 保存
    print(f"\n保存数据...")
    filename = f'balance_all.parquet'
    filepath = output_path / filename
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(filepath))

    print(f"✓ 下载完成: {len(df):,} 条记录 -> {filepath}")
    return df


def download_income_vip(
    start_date: str,
    end_date: str,
    output_dir: str = None,
    fields: List[str] = None,
    use_all_fields: bool = False,
    skip_existing: bool = True
) -> pd.DataFrame:
    """
    下载利润表数据 (income_vip)
    """
    output_dir = output_dir or os.path.join(DEFAULT_DATA_DIR, 'income')
    interface = 'income_vip'

    print(f"\n{'='*60}")
    print(f"下载利润表 (income_vip)")
    print(f"时间范围: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    if use_all_fields:
        selected_fields = None
        fields_desc = "全部字段"
    else:
        selected_fields = fields or INCOME_CORE_FIELDS
        fields_desc = f"核心字段 ({len(selected_fields)}个)"
    print(f"字段模式: {fields_desc}")

    pro = init_tushare()

    print("获取股票列表...")
    ts_codes = get_ts_code_list(pro)
    print(f"股票数量: {len(ts_codes)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged_file = output_path / 'income_all.parquet'

    if skip_existing and merged_file.exists():
        print(f"文件已存在: {merged_file}")
        return pq.read_table(merged_file).to_pandas()

    print(f"\n开始下载...")
    df = batch_download(
        pro, ts_codes, start_date, end_date,
        interface, selected_fields
    )

    if df.empty:
        print("未下载到数据")
        return pd.DataFrame()

    print(f"\n保存数据...")
    filename = f'income_all.parquet'
    filepath = output_path / filename
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(filepath))

    print(f"✓ 下载完成: {len(df):,} 条记录 -> {filepath}")
    return df


def download_cashflow_vip(
    start_date: str,
    end_date: str,
    output_dir: str = None,
    fields: List[str] = None,
    use_all_fields: bool = False,
    skip_existing: bool = True
) -> pd.DataFrame:
    """
    下载现金流量表数据 (cashflow_vip)
    """
    output_dir = output_dir or os.path.join(DEFAULT_DATA_DIR, 'cashflow')
    interface = 'cashflow_vip'

    print(f"\n{'='*60}")
    print(f"下载现金流量表 (cashflow_vip)")
    print(f"时间范围: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    if use_all_fields:
        selected_fields = None
        fields_desc = "全部字段"
    else:
        selected_fields = fields or CASHFLOW_CORE_FIELDS
        fields_desc = f"核心字段 ({len(selected_fields)}个)"
    print(f"字段模式: {fields_desc}")

    pro = init_tushare()

    print("获取股票列表...")
    ts_codes = get_ts_code_list(pro)
    print(f"股票数量: {len(ts_codes)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged_file = output_path / 'cashflow_all.parquet'

    if skip_existing and merged_file.exists():
        print(f"文件已存在: {merged_file}")
        return pq.read_table(merged_file).to_pandas()

    print(f"\n开始下载...")
    df = batch_download(
        pro, ts_codes, start_date, end_date,
        interface, selected_fields
    )

    if df.empty:
        print("未下载到数据")
        return pd.DataFrame()

    print(f"\n保存数据...")
    filename = f'cashflow_all.parquet'
    filepath = output_path / filename
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(filepath))

    print(f"✓ 下载完成: {len(df):,} 条记录 -> {filepath}")
    return df


def download_all_vip(
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    output_dir: str = DEFAULT_DATA_DIR,
    balance: bool = True,
    income: bool = True,
    cashflow: bool = True,
    use_all_fields: bool = False
):
    """
    下载全部 VIP 财务报表
    """
    print("="*60)
    print(f"批量下载 Tushare VIP 财务报表")
    print(f"时间范围: {start_date} ~ {end_date}")
    print("="*60)

    if balance:
        download_balancesheet_vip(
            start_date, end_date,
            os.path.join(output_dir, 'balance'),
            use_all_fields=use_all_fields
        )

    if income:
        download_income_vip(
            start_date, end_date,
            os.path.join(output_dir, 'income'),
            use_all_fields=use_all_fields
        )

    if cashflow:
        download_cashflow_vip(
            start_date, end_date,
            os.path.join(output_dir, 'cashflow'),
            use_all_fields=use_all_fields
        )

    print("\n" + "="*60)
    print("全部下载完成!")
    print("="*60)


def split_to_quarters(
    data_dir: str = DEFAULT_DATA_DIR,
    balance: bool = True,
    income: bool = True,
    cashflow: bool = True
):
    """
    将合并数据按季度拆分保存
    """
    print(f"\n拆分季度数据...")

    for table_name in ['balance', 'income', 'cashflow']:
        if table_name == 'balance' and not balance:
            continue
        if table_name == 'income' and not income:
            continue
        if table_name == 'cashflow' and not cashflow:
            continue

        print(f"\n处理 {table_name}...")
        table_dir = Path(data_dir) / table_name
        all_file = table_dir / f'{table_name}_all.parquet'

        if not all_file.exists():
            print(f"  文件不存在: {all_file}")
            continue

        df = pq.read_table(all_file).to_pandas()

        if 'end_date' not in df.columns:
            print(f"  无 end_date 字段")
            continue

        # 按季度分组
        for quarter_end in sorted(df['end_date'].unique()):
            quarter_df = df[df['end_date'] == quarter_end]
            if quarter_df.empty:
                continue

            filename = f'{table_name}_{quarter_end}.parquet'
            filepath = table_dir / filename
            table = pa.Table.from_pandas(quarter_df, preserve_index=False)
            pq.write_table(table, str(filepath))

        print(f"  已拆分 {df['end_date'].nunique()} 个季度")


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='Tushare VIP 财务报表下载',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--start', '-s',
        default=DEFAULT_START,
        help=f'开始日期 (默认: {DEFAULT_START})'
    )

    parser.add_argument(
        '--end', '-e',
        default=DEFAULT_END,
        help=f'结束日期 (默认: {DEFAULT_END})'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default=DEFAULT_DATA_DIR,
        help=f'输出目录 (默认: {DEFAULT_DATA_DIR})'
    )

    parser.add_argument(
        '--balance',
        action='store_true',
        help='只下载资产负债表'
    )

    parser.add_argument(
        '--income',
        action='store_true',
        help='只下载利润表'
    )

    parser.add_argument(
        '--cashflow',
        action='store_true',
        help='只下载现金流量表'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='下载全部三张表'
    )

    parser.add_argument(
        '--all-fields',
        action='store_true',
        help='使用所有字段（默认使用核心字段）'
    )

    parser.add_argument(
        '--split',
        action='store_true',
        help='按季度拆分保存'
    )

    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='不跳过已存在的文件'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 确定下载哪些表
    download_bal = args.balance or args.all
    download_inc = args.income or args.all
    download_cf = args.cashflow or args.all

    if not any([download_bal, download_inc, download_cf]):
        download_bal = download_inc = download_cf = True

    skip_existing = not args.no_skip

    # 下载
    if download_bal:
        download_balancesheet_vip(
            args.start, args.end,
            os.path.join(args.output_dir, 'balance'),
            use_all_fields=args.all_fields,
            skip_existing=skip_existing
        )

    if download_inc:
        download_income_vip(
            args.start, args.end,
            os.path.join(args.output_dir, 'income'),
            use_all_fields=args.all_fields,
            skip_existing=skip_existing
        )

    if download_cf:
        download_cashflow_vip(
            args.start, args.end,
            os.path.join(args.output_dir, 'cashflow'),
            use_all_fields=args.all_fields,
            skip_existing=skip_existing
        )

    # 拆分季度
    if args.split:
        split_to_quarters(
            args.output_dir,
            download_bal, download_inc, download_cf
        )


if __name__ == '__main__':
    main()
