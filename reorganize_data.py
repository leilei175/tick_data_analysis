"""
重新组织数据目录结构
==================

功能：
- 将散落的 parquet 文件按 年/月 目录结构组织
- 支持 daily, daily_basic, cashflow_daily, income_daily, balance_daily

使用方式：
---------

# 重新组织全部数据
python reorganize_data.py --all

# 只重新组织 daily
python reorganize_data.py --daily

# 只重新组织 daily_basic
python reorganize_data.py --daily-basic

# 只重新组织财务每日数据
python reorganize_daily.py --financial-daily

# 同时处理
python reorganize_data.py --all --dry-run  # 先预览
python reorganize_data.py --all  # 实际执行
"""

import os
import shutil
from pathlib import Path
import argparse
from typing import List, Tuple

# =============================================================================
# 配置
# =============================================================================

DATA_DIR = './daily_data/'

# 需要重新组织的目录配置
# 格式: {目录名: (文件名前缀, 是否是每日数据)}
DIRECTORIES = {
    'daily': ('daily_', False),
    'daily_basic': ('daily_basic_', False),
    'cashflow': ('cashflow_', False),
    'cashflow_daily': ('cashflow_daily_', True),
    'income': ('income_', False),
    'income_daily': ('income_daily_', True),
    'balance': ('balance_', False),
    'balance_daily': ('balance_daily_', True),
}

# =============================================================================
# 辅助函数
# =============================================================================

def get_year_month(filename: str) -> Tuple[str, str]:
    """从文件名解析年月"""
    import re

    # 匹配格式: daily_20250101.parquet -> YYYY, MM
    # 也匹配: cashflow_daily_20250101.parquet
    patterns = [
        (r'(\w+)_(\d{4})(\d{2})\d{2}\.parquet$', 2),  # daily_20250101.parquet
    ]

    for pattern, month_group in patterns:
        match = re.match(pattern, filename)
        if match:
            year = match.group(2)
            month = match.group(3)
            return year, month

    return None, None


def reorganize_dir(
    dir_name: str,
    prefix: str,
    is_daily: bool = False,
    dry_run: bool = True
) -> Tuple[int, int]:
    """
    重新组织单个目录

    Args:
        dir_name: 目录名
        prefix: 文件名前缀
        is_daily: 是否是每日数据（从日期解析年月）
        dry_run: 预览模式

    Returns:
        Tuple[移动文件数, 错误数]
    """
    src_dir = Path(DATA_DIR) / dir_name
    if not src_dir.exists():
        print(f"目录不存在: {src_dir}")
        return 0, 0

    moved = 0
    errors = 0

    # 获取所有 parquet 文件
    files = list(src_dir.glob('*.parquet'))

    print(f"\n处理 {dir_name}/ ({len(files)} 个文件)")

    for file in files:
        year, month = get_year_month(file.name)

        if year is None or month is None:
            print(f"  无法解析: {file.name}")
            errors += 1
            continue

        # 创建目标目录
        if is_daily:
            dest_dir = src_dir / year / month
        else:
            # 季度数据也按年/月组织
            # 例如 balance_20250331.parquet -> 2025/03
            dest_dir = src_dir / year / month

        dest_dir.mkdir(parents=True, exist_ok=True)

        # 移动文件
        dest_file = dest_dir / file.name
        if dest_file.exists():
            print(f"  已存在，跳过: {file.name}")
            continue

        if dry_run:
            print(f"  [预览] {file.name} -> {year}/{month}/")
        else:
            file.rename(dest_file)
            print(f"  移动: {file.name} -> {year}/{month}/")

        moved += 1

    return moved, errors


def reorganize_all(dry_run: bool = True):
    """重新组织所有目录"""
    total_moved = 0
    total_errors = 0

    print("=" * 60)
    if dry_run:
        print("预览模式")
    else:
        print("执行模式")
    print("=" * 60)

    for dir_name, (prefix, is_daily) in DIRECTORIES.items():
        moved, errors = reorganize_dir(dir_name, prefix, is_daily, dry_run)
        total_moved += moved
        total_errors += errors

    print("\n" + "=" * 60)
    print(f"总计: 移动 {total_moved} 个文件, {total_errors} 个错误")
    print("=" * 60)

    return total_moved, total_errors


# =============================================================================
# 清理空目录
# =============================================================================

def cleanup_empty_dirs():
    """清理空目录"""
    for dir_name in DIRECTORIES.keys():
        src_dir = Path(DATA_DIR) / dir_name
        if not src_dir.exists():
            continue

        # 查找并删除空目录
        for year_dir in src_dir.iterdir():
            if year_dir.is_dir():
                for month_dir in year_dir.iterdir():
                    if month_dir.is_dir() and not any(month_dir.iterdir()):
                        print(f"删除空目录: {month_dir}")
                        month_dir.rmdir()
                        # 检查年份目录是否为空
                        if not any(year_dir.iterdir()):
                            print(f"删除空目录: {year_dir}")
                            year_dir.rmdir()


# =============================================================================
# 恢复原状（可选）
# =============================================================================

def restore_structure(dry_run: bool = True):
    """将文件从年月目录恢复到根目录"""
    for dir_name, (prefix, _) in DIRECTORIES.items():
        src_dir = Path(DATA_DIR) / dir_name
        if not src_dir.exists():
            continue

        # 查找所有 parquet 文件
        for year_dir in src_dir.iterdir():
            if not year_dir.is_dir():
                continue
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue
                for file in month_dir.iterdir():
                    if file.suffix == '.parquet':
                        dest_file = src_dir / file.name
                        if dry_run:
                            print(f"[预览] {file.name} <- {year_dir.name}/{month_dir.name}/")
                        else:
                            file.rename(dest_file)
                            print(f"恢复: {file.name}")

        # 删除空目录
        if not dry_run:
            for month_dir in year_dir.iterdir():
                if month_dir.is_dir():
                    month_dir.rmdir()
            year_dir.rmdir()


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='重新组织数据目录结构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='预览模式，不实际执行'
    )

    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='重新组织全部数据'
    )

    parser.add_argument(
        '--daily',
        action='store_true',
        help='只处理 daily'
    )

    parser.add_argument(
        '--daily-basic',
        action='store_true',
        help='只处理 daily_basic'
    )

    parser.add_argument(
        '--financial-daily',
        action='store_true',
        help='只处理财务每日数据 (cashflow_daily, income_daily, balance_daily)'
    )

    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='清理空目录'
    )

    parser.add_argument(
        '--restore',
        action='store_true',
        help='恢复原状（从年月目录恢复到根目录）'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.restore:
        print("恢复原状...")
        restore_structure(args.dry_run)
        return

    if args.cleanup:
        print("清理空目录...")
        cleanup_empty_dirs()
        return

    # 确定要处理哪些目录
    dirs_to_process = []

    if args.all:
        dirs_to_process = list(DIRECTORIES.keys())
    elif args.daily:
        dirs_to_process = ['daily']
    elif args.daily_basic:
        dirs_to_process = ['daily_basic']
    elif args.financial_daily:
        dirs_to_process = ['cashflow_daily', 'income_daily', 'balance_daily']
    else:
        # 默认处理所有
        dirs_to_process = list(DIRECTORIES.keys())

    # 执行
    total_moved = 0
    total_errors = 0

    for dir_name in dirs_to_process:
        prefix, is_daily = DIRECTORIES[dir_name]
        moved, errors = reorganize_dir(dir_name, prefix, is_daily, args.dry_run)
        total_moved += moved
        total_errors += errors

    if not args.dry_run:
        print("\n清理空目录...")
        cleanup_empty_dirs()

    print("\n" + "=" * 60)
    print(f"完成: 移动 {total_moved} 个文件, {total_errors} 个错误")
    print("=" * 60)


if __name__ == '__main__':
    main()
