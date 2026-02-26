"""
Tushare VIP 财务报表下载（兼容封装）

说明：
- 本模块已精简为对 `financial_downloader.py` 的兼容层。
- 所有 VIP 下载实现统一收敛在 `financial_downloader.py`，避免重复代码。
"""

from financial_downloader import (
    set_token,
    download_balancesheet_vip,
    download_income_vip,
    download_cashflow_vip,
    download_all_vip,
    split_to_quarters,
    VIP_DEFAULT_START,
    VIP_DEFAULT_END,
    DEFAULT_DATA_DIR,
)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Tushare VIP 财务报表下载（兼容入口）')
    parser.add_argument('--start', '-s', default=VIP_DEFAULT_START, help=f'开始日期 (默认: {VIP_DEFAULT_START})')
    parser.add_argument('--end', '-e', default=VIP_DEFAULT_END, help=f'结束日期 (默认: {VIP_DEFAULT_END})')
    parser.add_argument('--output-dir', '-o', default=DEFAULT_DATA_DIR, help=f'输出目录 (默认: {DEFAULT_DATA_DIR})')

    parser.add_argument('--balance', action='store_true', help='只下载资产负债表')
    parser.add_argument('--income', action='store_true', help='只下载利润表')
    parser.add_argument('--cashflow', action='store_true', help='只下载现金流量表')
    parser.add_argument('--all', action='store_true', help='下载全部三张表')

    parser.add_argument('--all-fields', action='store_true', help='使用所有字段（默认核心字段）')
    parser.add_argument('--split', action='store_true', help='按季度拆分 *_all.parquet 文件')
    parser.add_argument('--token', '-t', help='Tushare Token')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.token:
        set_token(args.token)

    if args.split:
        split_to_quarters(
            data_dir=args.output_dir,
            balance=args.balance or args.all or not (args.income or args.cashflow),
            income=args.income or args.all or not (args.balance or args.cashflow),
            cashflow=args.cashflow or args.all or not (args.balance or args.income),
        )
        return

    download_balance = args.balance or (not args.income and not args.cashflow)
    download_income_flag = args.income or (not args.balance and not args.cashflow)
    download_cashflow_flag = args.cashflow or (not args.balance and not args.income)

    if args.all:
        download_balance = True
        download_income_flag = True
        download_cashflow_flag = True

    download_all_vip(
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
        balance=download_balance,
        income=download_income_flag,
        cashflow=download_cashflow_flag,
        use_all_fields=args.all_fields,
    )


if __name__ == '__main__':
    main()
