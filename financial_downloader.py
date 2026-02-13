"""
Tushare 财务报表数据下载脚本
=========================

功能：
- 从 Tushare 下载现金流量表、资产负债表、利润表季度数据
- 保存为 Parquet 格式
- 支持 2015-2025 年数据

使用说明：
---------

1. 配置 Tushare Token：
   - 环境变量: export TUSHARE_TOKEN='your_token'
   - 配置文件: ~/.tushare_token
   - 代码调用: set_token('your_token')

2. 运行下载：

   # 下载全部三张表
   python financial_downloader.py --start 20150101 --end 20251231 --all

   # 只下载现金流量表
   python financial_downloader.py --start 20150101 --end 20251231 --cashflow

   # 只下载资产负债表
   python financial_downloader.py --start 20150101 --end 20251231 --balance

   # 只下载利润表
   python financial_downloader.py --start 20150101 --end 20251231 --income

   # 下载指定季度
   python financial_downloader.py --quarters 20250630 20250331

3. Python API 使用：

   from financial_downloader import (
       download_cashflow,
       download_balance,
       download_income,
       download_all_financials
   )

   # 下载全部数据
   download_all_financials(
       start_date='20150101',
       end_date='20251231',
       output_dir='./daily_data'
   )

   # 只下载现金流量表
   download_cashflow(
       start_date='20150101',
       end_date='20251231',
       output_dir='./daily_data/cashflow'
   )
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Tushare
try:
    import tushare as ts
    HAS_TUSHARE = True
except ImportError:
    HAS_TUSHARE = False
    print("警告: 未安装 tushare，请运行: pip install tushare")

# =============================================================================
# 配置
# =============================================================================

# 默认输出目录
DEFAULT_DATA_DIR = './daily_data/'

# 季度结束日期列表 (用于增量更新)
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

# 现金流量表字段
CASHFLOW_FIELDS = [
    'ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type',
    # 经营活动
    'n_cashflow_act',           # 经营活动产生的现金流量净额
    'n_cashflow_inv_act',       # 投资活动产生的现金流量净额
    'n_cash_flows_fnc_act',     # 筹资活动产生的现金流量净额
    'c_fr_sale_sg',             # 销售商品、提供劳务收到的现金
    'c_paid_goods_s',           # 购买商品、接受劳务支付的现金
    'c_paid_to_for_empl',       # 支付给职工以及为职工支付的现金
    'c_recp_borrow',            # 取得借款收到的现金
    'proc_issue_bonds',         # 发行债券收到的现金
    'dep_draw_deposits',        # 吸收存款收到的现金
    'c_pay_empl_sal',          # 支付给职工以及为职工支付的现金
    'pay_rdexp',                # 研发支出
    'pay_int_exp',             # 利息支出
    'pay_tax',                  # 支付的各项税费
    'rec_hd_handle',           # 处置交易性金融资产净增加额
    'stot_cashflow_act',       # 经营活动现金流量净额
    's_cashflow_inv_act',      # 投资活动现金流量净额
    's_cash_flows_fnc_act',    # 筹资活动现金流量净额
    'c_oth_pay',               # 支付其他与经营活动有关的现金
    'c_recp_tax_rf',           # 收到的税费返还
    'c_pay_tax_rf',            # 支付的税费返还
    'decr_inventories',        # 存货的减少
    'decr_ar',                 # 应收账款及其他应收款减少
    'incr_ar',                 # 应收账款及其他应收款增加
    'decr_pay',                # 应付账款及其他应付款增加
    'decr_adv_pay',            # 预收款项减少
    'incr_adv_pay',            # 预收款项增加
    'c_icb',                   # 收回投资收到的现金
    'c_lend_deposits',         # 存放央行和同业款项净增加额
    'c_dbcass_remt',          # 汇兑收益
    'c_clct_pledge',          # 质押贷款净增加额
    'c_dc_invest',            # 长期股权投资净增加额
    'c_dc_fair_value_gain',   # 以公允价值计量且其变动计入当期损益的金融资产净增加额
    'c_dc_assets_imp',        # 非流动资产净增加额
    'c_dc_right_use_assets', # 使用权资产净增加额
    'c_dc_liab_inc',          # 租赁负债净增加额
]

# 资产负债表字段
BALANCE_FIELDS = [
    'ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type',
    # 资产
    'total_assets',           # 资产总计
    'total_liab',             # 负债合计
    'total_hldr_eqy_exc_min_int',  # 归属于母公司所有者权益合计
    'total_cur_assets',       # 流动资产合计
    'total_nca',              # 非流动资产合计
    'cash_reser_cb',          # 货币资金
    'trad_asset',             # 交易性金融资产
    'notes_receiv',           # 应收票据
    'accounts_receiv',        # 应收账款
    'inventories',            # 存货
    'fix_assets',             # 固定资产
    'intan_assets',           # 无形资产
    'total_cur_liab',         # 流动负债合计
    'total_ncl',              # 非流动负债合计
    'st_borr',                # 短期借款
    'lt_borr',                # 长期借款
    'notes_payable',          # 应付票据
    'bonds_payable',          # 应付债券
    'preferred_stock',        # 优先股
    'perpetual_bond',         # 永续债
    'capital_reser',          # 资本公积
    'surplus_reser',          # 盈余公积
    'special_reser',          # 专项储备
    'retained_earnings',      # 未分配利润
    'oth_rvnu',               # 其他权益工具
    'oth_comprecome',         # 其他综合收益
]

# 利润表字段
INCOME_FIELDS = [
    'ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type', 'comp_type',
    # 收入
    'total_revenue',          # 营业总收入
    'revenue',                # 营业收入
    'int_income',             # 利息收入
    'prem_earned',            # 保费收入
    'comm_income',            # 手续费及佣金收入
    'n_commis_income',        # 手续费及佣金收入
    'prem_cbl',               # 赔付支出净额
    'prem_recd_agd',          # 保费收入
    'reinsur_income',         # 分保费收入
    # 成本与费用
    'operate_profit',         # 营业利润
    'total_profit',           # 利润总额
    'income_tax',             # 所得税费用
    'n_income',               # 净利润
    'n_income_attr_p',       # 归属于母公司的净利润
    'minority_gain',          # 少数股东损益
    'basic_eps',              # 基本每股收益
    'diluted_eps',            # 稀释每股收益
    'ebit',                   # 息税前利润
    'ebitda',                 # 息税折旧摊销前利润
    'cost_expense',           # 营业成本及费用合计
    'bus_tax',                # 营业税金及附加
    'sell_exp',               # 销售费用
    'admin_exp',              # 管理费用
    'fin_exp',                # 财务费用
    'asset_impair_loss',      # 资产减值损失
    'credit_impair_loss',     # 信用减值损失
    'oth_income',             # 其他收益
    'invest_income',          # 投资收益
    'fair_value_gain',       # 公允价值变动收益
    'asset_disp_income',     # 资产处置收益
    'cur_liab',               # 流动负债
    'nca_deduct',             # 非流动资产处置净损益
]

# 季度日期到描述的映射
QUARTER_DESCRIPTION = {
    '0331': 'Q1',
    '0630': 'Q2 (中报)',
    '0930': 'Q3',
    '1231': 'Q4 (年报)',
}

# =============================================================================
# Token 管理
# =============================================================================

def set_token(token: str):
    """设置 Tushare Token"""
    ts.set_token(token)
    print(f"Token 已设置")


def get_token() -> str:
    """获取 Tushare Token"""
    # 1. 环境变量
    token = os.environ.get('TUSHARE_TOKEN')
    if token:
        return token

    # 2. 配置文件
    token_path = Path.home() / '.tushare_token'
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token

    raise ValueError(
        "未找到 Tushare Token，请通过以下方式之一设置：\n"
        "1. 环境变量: export TUSHARE_TOKEN='your_token'\n"
        "2. 配置文件: echo 'your_token' > ~/.tushare_token\n"
        "3. 代码调用: set_token('your_token')"
    )


def init_tushare():
    """初始化 Tushare"""
    if not HAS_TUSHARE:
        raise ImportError("未安装 tushare，请运行: pip install tushare")

    token = get_token()
    ts.set_token(token)
    pro = ts.pro_api()
    return pro


# =============================================================================
# 辅助函数
# =============================================================================

def get_quarter_ends_between(start_date: str, end_date: str) -> List[str]:
    """获取指定日期范围内的所有季度结束日期"""
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])

    quarters = []
    for year in range(start_year, end_year + 1):
        for q in ['0331', '0630', '0930', '1231']:
            quarter_end = f'{year}{q}'
            if start_date <= quarter_end <= end_date:
                quarters.append(quarter_end)

    return sorted(quarters)


def get_quarter_from_date(date: str) -> str:
    """根据日期获取最近的季度结束日期"""
    month = int(date[4:6])
    year = date[:4]

    if month <= 3:
        return f'{year}0331'
    elif month <= 6:
        return f'{year}0630'
    elif month <= 9:
        return f'{year}0930'
    else:
        return f'{year}1231'


def parse_quarter_end(quarter_end: str) -> Tuple[str, str]:
    """解析季度结束日期，返回 (年份, 季度)"""
    year = quarter_end[:4]
    quarter = QUARTER_DESCRIPTION.get(quarter_end[4:], quarter_end[4:])
    return year, quarter


def format_quarter(quarter_end: str) -> str:
    """格式化季度描述"""
    year, quarter = parse_quarter_end(quarter_end)
    return f"{year}{quarter}"


# =============================================================================
# 数据下载函数
# =============================================================================

def download_cashflow(
    start_date: str,
    end_date: str,
    output_dir: str = None,
    fields: List[str] = None,
    skip_existing: bool = True
) -> pd.DataFrame:
    """
    下载现金流量表数据

    Args:
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        output_dir: 输出目录
        fields: 字段列表
        skip_existing: 跳过已存在的文件

    Returns:
        pd.DataFrame: 下载的数据
    """
    if fields is None:
        fields = CASHFLOW_FIELDS

    output_dir = output_dir or os.path.join(DEFAULT_DATA_DIR, 'cashflow')
    print(f"\n{'='*60}")
    print(f"下载现金流量表: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    pro = init_tushare()

    # 获取季度结束日期列表
    quarter_ends = get_quarter_ends_between(start_date, end_date)
    print(f"需要下载 {len(quarter_ends)} 个季度数据")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_data = []
    success_count = 0
    fail_count = 0

    for quarter_end in quarter_ends:
        print(f"\n[{format_quarter(quarter_end)}] ", end='', flush=True)

        # 检查是否已存在
        filename = f'cashflow_{quarter_end}.parquet'
        filepath = Path(output_dir) / filename

        if skip_existing and filepath.exists():
            print("✓ 已存在，跳过")
            # 读取现有数据
            table = pq.read_table(filepath)
            all_data.append(table.to_pandas())
            success_count += 1
            continue

        # 获取该季度的报表日期范围
        # 对于季报，通常是季度结束后的1-2个月内发布
        quarter_start = quarter_end[:4] + '0101' if quarter_end.endswith('1231') else \
                        str(int(quarter_end) - 8999)[:4] + '0101'
        # 年报通常在次年3-4月发布
        if quarter_end.endswith('1231'):
            ann_start = str(int(quarter_end) + 10000)[:4] + '0101'
            ann_end = str(int(quarter_end) + 20000)[:4] + '0430'
        else:
            ann_start = str(int(quarter_end) + 1)[:4] + '0101'
            ann_end = str(int(quarter_end) + 10000)[:4] + '0531'

        try:
            # 下载数据
            df = pro.cashflow(
                start_date=ann_start,
                end_date=ann_end,
                fields=','.join(fields)
            )

            if df.empty:
                print("✗ 无数据")
                fail_count += 1
                continue

            # 过滤只保留指定季度的数据
            if 'end_date' in df.columns:
                df = df[df['end_date'] == quarter_end]

            if df.empty:
                print("✗ 无该季度数据")
                fail_count += 1
                continue

            # 保存为 Parquet
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, str(filepath))

            all_data.append(df)
            print(f"✓ {len(df)} 条记录")
            success_count += 1

            # 避免请求过快
            time.sleep(0.3)

        except Exception as e:
            print(f"✗ 错误: {str(e)[:50]}")
            fail_count += 1

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"\n下载完成! 成功: {success_count}, 失败: {fail_count}, 总计 {len(result)} 条")
        return result
    else:
        print(f"\n未下载到任何数据")
        return pd.DataFrame()


def download_balance(
    start_date: str,
    end_date: str,
    output_dir: str = None,
    fields: List[str] = None,
    skip_existing: bool = True
) -> pd.DataFrame:
    """
    下载资产负债表数据

    Args:
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        output_dir: 输出目录
        fields: 字段列表
        skip_existing: 跳过已存在的文件

    Returns:
        pd.DataFrame: 下载的数据
    """
    if fields is None:
        fields = BALANCE_FIELDS

    output_dir = output_dir or os.path.join(DEFAULT_DATA_DIR, 'balance')
    print(f"\n{'='*60}")
    print(f"下载资产负债表: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    pro = init_tushare()

    quarter_ends = get_quarter_ends_between(start_date, end_date)
    print(f"需要下载 {len(quarter_ends)} 个季度数据")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_data = []
    success_count = 0
    fail_count = 0

    for quarter_end in quarter_ends:
        print(f"\n[{format_quarter(quarter_end)}] ", end='', flush=True)

        filename = f'balance_{quarter_end}.parquet'
        filepath = Path(output_dir) / filename

        if skip_existing and filepath.exists():
            print("✓ 已存在，跳过")
            table = pq.read_table(filepath)
            all_data.append(table.to_pandas())
            success_count += 1
            continue

        # 确定公告日期范围
        if quarter_end.endswith('1231'):
            ann_start = str(int(quarter_end) + 10000)[:4] + '0101'
            ann_end = str(int(quarter_end) + 20000)[:4] + '0430'
        else:
            ann_start = str(int(quarter_end) + 1)[:4] + '0101'
            ann_end = str(int(quarter_end) + 10000)[:4] + '0531'

        try:
            df = pro.balance(
                start_date=ann_start,
                end_date=ann_end,
                fields=','.join(fields)
            )

            if df.empty:
                print("✗ 无数据")
                fail_count += 1
                continue

            if 'end_date' in df.columns:
                df = df[df['end_date'] == quarter_end]

            if df.empty:
                print("✗ 无该季度数据")
                fail_count += 1
                continue

            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, str(filepath))

            all_data.append(df)
            print(f"✓ {len(df)} 条记录")
            success_count += 1

            time.sleep(0.3)

        except Exception as e:
            print(f"✗ 错误: {str(e)[:50]}")
            fail_count += 1

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"\n下载完成! 成功: {success_count}, 失败: {fail_count}, 总计 {len(result)} 条")
        return result
    else:
        print(f"\n未下载到任何数据")
        return pd.DataFrame()


def download_income(
    start_date: str,
    end_date: str,
    output_dir: str = None,
    fields: List[str] = None,
    skip_existing: bool = True
) -> pd.DataFrame:
    """
    下载利润表数据

    Args:
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        output_dir: 输出目录
        fields: 字段列表
        skip_existing: 跳过已存在的文件

    Returns:
        pd.DataFrame: 下载的数据
    """
    if fields is None:
        fields = INCOME_FIELDS

    output_dir = output_dir or os.path.join(DEFAULT_DATA_DIR, 'income')
    print(f"\n{'='*60}")
    print(f"下载利润表: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    pro = init_tushare()

    quarter_ends = get_quarter_ends_between(start_date, end_date)
    print(f"需要下载 {len(quarter_ends)} 个季度数据")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_data = []
    success_count = 0
    fail_count = 0

    for quarter_end in quarter_ends:
        print(f"\n[{format_quarter(quarter_end)}] ", end='', flush=True)

        filename = f'income_{quarter_end}.parquet'
        filepath = Path(output_dir) / filename

        if skip_existing and filepath.exists():
            print("✓ 已存在，跳过")
            table = pq.read_table(filepath)
            all_data.append(table.to_pandas())
            success_count += 1
            continue

        if quarter_end.endswith('1231'):
            ann_start = str(int(quarter_end) + 10000)[:4] + '0101'
            ann_end = str(int(quarter_end) + 20000)[:4] + '0430'
        else:
            ann_start = str(int(quarter_end) + 1)[:4] + '0101'
            ann_end = str(int(quarter_end) + 10000)[:4] + '0531'

        try:
            df = pro.income(
                start_date=ann_start,
                end_date=ann_end,
                fields=','.join(fields)
            )

            if df.empty:
                print("✗ 无数据")
                fail_count += 1
                continue

            if 'end_date' in df.columns:
                df = df[df['end_date'] == quarter_end]

            if df.empty:
                print("✗ 无该季度数据")
                fail_count += 1
                continue

            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, str(filepath))

            all_data.append(df)
            print(f"✓ {len(df)} 条记录")
            success_count += 1

            time.sleep(0.3)

        except Exception as e:
            print(f"✗ 错误: {str(e)[:50]}")
            fail_count += 1

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"\n下载完成! 成功: {success_count}, 失败: {fail_count}, 总计 {len(result)} 条")
        return result
    else:
        print(f"\n未下载到任何数据")
        return pd.DataFrame()


def download_all_financials(
    start_date: str,
    end_date: str,
    output_dir: str = None,
    cashflow: bool = True,
    balance: bool = True,
    income: bool = True
):
    """
    下载全部财务报表数据

    Args:
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出根目录
        cashflow: 是否下载现金流量表
        balance: 是否下载资产负债表
        income: 是否下载利润表
    """
    if output_dir is None:
        output_dir = DEFAULT_DATA_DIR

    print("=" * 60)
    print(f"批量下载财务报表: {start_date} ~ {end_date}")
    print(f"时间范围: {start_date} ~ {end_date}")
    print("=" * 60)

    if cashflow:
        cashflow_dir = os.path.join(output_dir, 'cashflow')
        download_cashflow(start_date, end_date, cashflow_dir)

    if balance:
        balance_dir = os.path.join(output_dir, 'balance')
        download_balance(start_date, end_date, balance_dir)

    if income:
        income_dir = os.path.join(output_dir, 'income')
        download_income(start_date, end_date, income_dir)

    print("\n" + "=" * 60)
    print("全部下载完成!")
    print("=" * 60)


def download_by_quarters(
    quarters: List[str],
    output_dir: str = None,
    cashflow: bool = True,
    balance: bool = True,
    income: bool = True
):
    """
    按指定季度下载数据

    Args:
        quarters: 季度结束日期列表，如 ['20250630', '20250331']
        output_dir: 输出根目录
        cashflow: 是否下载现金流量表
        balance: 是否下载资产负债表
        income: 是否下载利润表
    """
    if output_dir is None:
        output_dir = DEFAULT_DATA_DIR

    print("=" * 60)
    print(f"按季度下载: {', '.join(quarters)}")
    print("=" * 60)

    for quarter_end in quarters:
        print(f"\n{'='*60}")
        print(f"季度: {format_quarter(quarter_end)}")
        print(f"{'='*60}")

        # 确定该季度的公告日期范围
        if quarter_end.endswith('1231'):
            ann_start = str(int(quarter_end) + 10000)[:4] + '0101'
            ann_end = str(int(quarter_end) + 20000)[:4] + '0430'
        else:
            ann_start = str(int(quarter_end) + 1)[:4] + '0101'
            ann_end = str(int(quarter_end) + 10000)[:4] + '0531'

        if cashflow:
            cashflow_dir = os.path.join(output_dir, 'cashflow')
            download_cashflow(ann_start, ann_end, cashflow_dir)

        if balance:
            balance_dir = os.path.join(output_dir, 'balance')
            download_balance(ann_start, ann_end, balance_dir)

        if income:
            income_dir = os.path.join(output_dir, 'income')
            download_income(ann_start, ann_end, income_dir)


def merge_to_all(output_dir: str = None):
    """
    合并所有季度数据为一个大文件

    Args:
        output_dir: 数据目录
    """
    if output_dir is None:
        output_dir = DEFAULT_DATA_DIR

    for table_name in ['cashflow', 'balance', 'income']:
        table_dir = Path(output_dir) / table_name
        if not table_dir.exists():
            print(f"{table_name}: 目录不存在")
            continue

        # 查找所有季度文件
        files = sorted(table_dir.glob(f'{table_name}_*.parquet'))
        if not files:
            print(f"{table_name}: 无数据文件")
            continue

        # 合并
        dfs = []
        for f in files:
            table = pq.read_table(f)
            dfs.append(table.to_pandas())

        all_df = pd.concat(dfs, ignore_index=True)
        all_file = table_dir / f'{table_name}_all.parquet'
        table = pa.Table.from_pandas(all_df, preserve_index=False)
        pq.write_table(table, str(all_file))

        print(f"{table_name}: 合并完成, {len(all_df)} 条记录 -> {all_file}")


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='从 Tushare 下载财务报表数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--start', '-s',
        help='开始日期，格式 YYYYMMDD (与 --all 或 --quarters 互斥)'
    )

    parser.add_argument(
        '--end', '-e',
        help='结束日期，格式 YYYYMMDD (与 --all 或 --quarters 互斥)'
    )

    parser.add_argument(
        '--quarters', '-q',
        nargs='+',
        help='指定季度日期列表，如 20250630 20250331'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='下载 2015-2025 年全部数据'
    )

    parser.add_argument(
        '--cashflow',
        action='store_true',
        help='只下载现金流量表'
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
        '--output-dir', '-o',
        default=DEFAULT_DATA_DIR,
        help=f'输出目录 (默认: {DEFAULT_DATA_DIR})'
    )

    parser.add_argument(
        '--token', '-t',
        help='Tushare Token'
    )

    parser.add_argument(
        '--merge',
        action='store_true',
        help='合并所有季度数据为一个大文件'
    )

    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='不跳过已存在的文件'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置 Token
    if args.token:
        set_token(args.token)

    # 合并模式
    if args.merge:
        merge_to_all(args.output_dir)
        return

    # 确定下载范围
    if args.all:
        start = '20150101'
        end = '20251231'
    elif args.quarters:
        quarters = args.quarters
        download_by_quarters(
            quarters,
            output_dir=args.output_dir,
            cashflow=args.cashflow or not (args.balance or args.income),
            balance=args.balance or not (args.cashflow or args.income),
            income=args.income or not (args.cashflow or args.balance)
        )
        return
    elif args.start and args.end:
        start = args.start
        end = args.end
    else:
        print("错误: 请指定 --start/--end 或 --all 或 --quarters")
        sys.exit(1)

    # 确定下载哪些表
    download_cf = args.cashflow or not (args.balance or args.income)
    download_bal = args.balance or not (args.cashflow or args.income)
    download_inc = args.income or not (args.cashflow or args.balance)

    skip_existing = not args.no_skip

    # 执行下载
    download_all_financials(
        start_date=start,
        end_date=end,
        output_dir=args.output_dir,
        cashflow=download_cf,
        balance=download_bal,
        income=download_inc
    )

    # 合并数据
    merge_to_all(args.output_dir)


if __name__ == '__main__':
    main()
