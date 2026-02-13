"""
下载Tushare财务报表数据
- 利润表 (income)
- 资产负债表 (balancesheet)
- 现金流量表 (cashflow)

保存到 daily_data 目录
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

import tushare as ts
import pandas as pd

# 读取配置文件
config_path = Path(__file__).parent / "config.py"
exec(config_path.read_text())

TOKEN = tushare_tk
print(f"Tushare Token: {TOKEN[:10]}...")

# 初始化Tushare
pro = ts.pro_api(TOKEN)

# 数据保存目录
DATA_DIR = Path(__file__).parent / "daily_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INCOME_DIR = DATA_DIR / "income"
BALANCE_DIR = DATA_DIR / "balance"
CASHFLOW_DIR = DATA_DIR / "cashflow"
INCOME_DIR.mkdir(parents=True, exist_ok=True)
BALANCE_DIR.mkdir(parents=True, exist_ok=True)
CASHFLOW_DIR.mkdir(parents=True, exist_ok=True)

# 季度日期列表 (2024-2026)
QUARTERS = [
    # 2024年
    '20240331', '20240630', '20240930', '20241231',
    # 2025年
    '20250331', '20250630', '20250930', '20251231',
    # 2026年
    '20260331',
]

# 利润表关键字段
INCOME_FIELDS = ",".join([
    "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
    "total_revenue", "revenue", "int_income", "prem_earned", "comm_income",
    "n_commis_income", "operate_profit", "total_profit", "income_tax",
    "n_income", "n_income_attr_p", "minority_gain",
    "basic_eps", "diluted_eps",
    "ebit", "ebitda"
])

# 资产负债表关键字段
BALANCE_FIELDS = ",".join([
    "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
    "total_assets", "total_liab", "total_hldr_eqy_exc_min_int",
    "total_cur_assets", "total_nca", "total_cur_liab", "total_ncl",
    "cash_reser_cb", "trad_asset", "notes_receiv", "accounts_receiv",
    "inventories", "fix_assets", "intan_assets",
    "st_borr", "lt_borr", "notes_payable", "accounts_payable"
])

# 现金流量表关键字段
CASHFLOW_FIELDS = ",".join([
    "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
    "n_cashflow_act", "n_cashflow_inv_act", "n_cash_flows_fnc_act",
    "c_fr_sale_sg", "c_paid_goods_s", "c_paid_to_for_empl",
    "c_recp_borrow", "proc_issue_bonds"
])

def download_income_vip(period):
    """下载单季度利润表数据"""
    try:
        df = pro.income_vip(period=period, fields=INCOME_FIELDS)
        if df is not None and len(df) > 0:
            return df
        return None
    except Exception as e:
        print(f"下载 {period} 利润表失败: {e}")
        return None

def download_balancesheet_vip(period):
    """下载单季度资产负债表数据"""
    try:
        df = pro.balancesheet_vip(period=period, fields=BALANCE_FIELDS)
        if df is not None and len(df) > 0:
            return df
        return None
    except Exception as e:
        print(f"下载 {period} 资产负债表失败: {e}")
        return None

def download_cashflow_vip(period):
    """下载单季度现金流量表数据"""
    try:
        df = pro.cashflow_vip(period=period, fields=CASHFLOW_FIELDS)
        if df is not None and len(df) > 0:
            return df
        return None
    except Exception as e:
        print(f"下载 {period} 现金流量表失败: {e}")
        return None

def download_all_income():
    """下载所有季度利润表"""
    print(f"\n{'='*60}")
    print(f"下载利润表数据 (income)")
    print(f"季度范围: {QUARTERS[0]} - {QUARTERS[-1]}")
    print(f"保存目录: {INCOME_DIR}")
    print(f"{'='*60}")

    all_data = []
    for i, period in enumerate(QUARTERS):
        print(f"\r处理中: {i+1}/{len(QUARTERS)} {period}", end="")

        df = download_income_vip(period)
        if df is not None and len(df) > 0:
            all_data.append(df)
            # 保存单季度数据
            file_path = INCOME_DIR / f"income_{period}.parquet"
            df.to_parquet(file_path, index=False)

        time.sleep(0.5)  # API请求间隔

    print(f"\n\n完成! 成功下载 {len(all_data)} 个季度")

    # 保存完整数据
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        all_df = all_df.sort_values(['end_date', 'ts_code'])

        output_file = INCOME_DIR / f"income_all.parquet"
        all_df.to_parquet(output_file, index=False)
        print(f"完整数据已保存: {output_file}")
        print(f"总记录数: {len(all_df):,}")
        print(f"股票数量: {all_df['ts_code'].nunique():,}")

    return all_df if all_data else None

def download_all_balancesheet():
    """下载所有季度资产负债表"""
    print(f"\n{'='*60}")
    print(f"下载资产负债表数据 (balancesheet)")
    print(f"季度范围: {QUARTERS[0]} - {QUARTERS[-1]}")
    print(f"保存目录: {BALANCE_DIR}")
    print(f"{'='*60}")

    all_data = []
    for i, period in enumerate(QUARTERS):
        print(f"\r处理中: {i+1}/{len(QUARTERS)} {period}", end="")

        df = download_balancesheet_vip(period)
        if df is not None and len(df) > 0:
            all_data.append(df)
            # 保存单季度数据
            file_path = BALANCE_DIR / f"balance_{period}.parquet"
            df.to_parquet(file_path, index=False)

        time.sleep(0.5)  # API请求间隔

    print(f"\n\n完成! 成功下载 {len(all_data)} 个季度")

    # 保存完整数据
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        all_df = all_df.sort_values(['end_date', 'ts_code'])

        output_file = BALANCE_DIR / f"balance_all.parquet"
        all_df.to_parquet(output_file, index=False)
        print(f"完整数据已保存: {output_file}")
        print(f"总记录数: {len(all_df):,}")
        print(f"股票数量: {all_df['ts_code'].nunique():,}")

    return all_df if all_data else None

def download_all_cashflow():
    """下载所有季度现金流量表"""
    print(f"\n{'='*60}")
    print(f"下载现金流量表数据 (cashflow)")
    print(f"季度范围: {QUARTERS[0]} - {QUARTERS[-1]}")
    print(f"保存目录: {CASHFLOW_DIR}")
    print(f"{'='*60}")

    all_data = []
    for i, period in enumerate(QUARTERS):
        print(f"\r处理中: {i+1}/{len(QUARTERS)} {period}", end="")

        df = download_cashflow_vip(period)
        if df is not None and len(df) > 0:
            all_data.append(df)
            # 保存单季度数据
            file_path = CASHFLOW_DIR / f"cashflow_{period}.parquet"
            df.to_parquet(file_path, index=False)

        time.sleep(0.5)  # API请求间隔

    print(f"\n\n完成! 成功下载 {len(all_data)} 个季度")

    # 保存完整数据
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        all_df = all_df.sort_values(['end_date', 'ts_code'])

        output_file = CASHFLOW_DIR / f"cashflow_all.parquet"
        all_df.to_parquet(output_file, index=False)
        print(f"完整数据已保存: {output_file}")
        print(f"总记录数: {len(all_df):,}")
        print(f"股票数量: {all_df['ts_code'].nunique():,}")

    return all_df if all_data else None


# ==================== 数据读取函数 ====================

def get_income(sec_list: list = None, start_period: str = None, end_period: str = None,
               data_dir: str = None) -> pd.DataFrame:
    """
    获取利润表数据

    Args:
        sec_list: 股票代码列表，为空则获取全部
        start_period: 开始季度，如 '20240331'
        end_period: 结束季度，如 '20240630'
        data_dir: 数据目录路径

    Returns:
        DataFrame

    Example:
        >>> df = get_income(['600000.SH'], '20240101', '20240630')
    """
    if data_dir is None:
        data_dir = INCOME_DIR
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_files = sorted(data_dir.glob("income_*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到数据文件")

    start_period = int(start_period) if start_period else 0
    end_period = int(end_period) if end_period else 99991231

    filtered_files = []
    for f in all_files:
        fname = f.name.replace("income_", "").replace(".parquet", "")
        if fname == "all":
            filtered_files.append(f)
        else:
            file_period = int(fname)
            if start_period <= file_period <= end_period:
                filtered_files.append(f)

    print(f"加载 {len(filtered_files)} 个数据文件...")

    all_dfs = []
    for f in filtered_files:
        try:
            df = pd.read_parquet(f)
            all_dfs.append(df)
        except Exception as e:
            print(f"读取 {f.name} 失败: {e}")

    if not all_dfs:
        raise ValueError("无法读取任何数据")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    if sec_list:
        combined_df = combined_df[combined_df['ts_code'].isin(sec_list)]

    result_df = combined_df.drop_duplicates(subset=['ts_code', 'end_date'], keep='first')
    result_df = result_df.sort_values(['ts_code', 'end_date'])

    print(f"返回 {len(result_df):,} 条记录，{result_df['ts_code'].nunique()} 只股票")

    return result_df


def get_balance(sec_list: list = None, start_period: str = None, end_period: str = None,
                data_dir: str = None) -> pd.DataFrame:
    """
    获取资产负债表数据
    """
    if data_dir is None:
        data_dir = BALANCE_DIR
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_files = sorted(data_dir.glob("balance_*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到数据文件")

    start_period = int(start_period) if start_period else 0
    end_period = int(end_period) if end_period else 99991231

    filtered_files = []
    for f in all_files:
        fname = f.name.replace("balance_", "").replace(".parquet", "")
        if fname == "all":
            filtered_files.append(f)
        else:
            file_period = int(fname)
            if start_period <= file_period <= end_period:
                filtered_files.append(f)

    print(f"加载 {len(filtered_files)} 个数据文件...")

    all_dfs = []
    for f in filtered_files:
        try:
            df = pd.read_parquet(f)
            all_dfs.append(df)
        except Exception as e:
            print(f"读取 {f.name} 失败: {e}")

    if not all_dfs:
        raise ValueError("无法读取任何数据")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    if sec_list:
        combined_df = combined_df[combined_df['ts_code'].isin(sec_list)]

    result_df = combined_df.drop_duplicates(subset=['ts_code', 'end_date'], keep='first')
    result_df = result_df.sort_values(['ts_code', 'end_date'])

    print(f"返回 {len(result_df):,} 条记录，{result_df['ts_code'].nunique()} 只股票")

    return result_df


def get_cashflow(sec_list: list = None, start_period: str = None, end_period: str = None,
                 data_dir: str = None) -> pd.DataFrame:
    """
    获取现金流量表数据
    """
    if data_dir is None:
        data_dir = CASHFLOW_DIR
    else:
        data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_files = sorted(data_dir.glob("cashflow_*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到数据文件")

    start_period = int(start_period) if start_period else 0
    end_period = int(end_period) if end_period else 99991231

    filtered_files = []
    for f in all_files:
        fname = f.name.replace("cashflow_", "").replace(".parquet", "")
        if fname == "all":
            filtered_files.append(f)
        else:
            file_period = int(fname)
            if start_period <= file_period <= end_period:
                filtered_files.append(f)

    print(f"加载 {len(filtered_files)} 个数据文件...")

    all_dfs = []
    for f in filtered_files:
        try:
            df = pd.read_parquet(f)
            all_dfs.append(df)
        except Exception as e:
            print(f"读取 {f.name} 失败: {e}")

    if not all_dfs:
        raise ValueError("无法读取任何数据")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    if sec_list:
        combined_df = combined_df[combined_df['ts_code'].isin(sec_list)]

    result_df = combined_df.drop_duplicates(subset=['ts_code', 'end_date'], keep='first')
    result_df = result_df.sort_values(['ts_code', 'end_date'])

    print(f"返回 {len(result_df):,} 条记录，{result_df['ts_code'].nunique()} 只股票")

    return result_df


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Tushare 财务报表数据下载")
    print("="*60)
    print(f"季度列表: {QUARTERS}")
    print("注意: VIP接口需要足够积分，下载可能需要较长时间")

    # 下载利润表
    download_all_income()

    # 下载资产负债表
    download_all_balancesheet()

    # 下载现金流量表
    download_all_cashflow()

    print("\n" + "="*60)
    print("全部财务报表下载完成!")
    print(f"数据保存位置: {DATA_DIR}")
    print(f"  - income: 利润表")
    print(f"  - balance: 资产负债表")
    print(f"  - cashflow: 现金流量表")
    print("="*60)
