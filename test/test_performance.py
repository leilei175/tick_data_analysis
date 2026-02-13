import time
import tracemalloc
import sys
import os
from glob import glob
from pathlib import Path

# 脚本所在目录
SCRIPT_DIR = Path(__file__).resolve().parent
# 数据目录 (基于脚本位置)
DATA_DIR = SCRIPT_DIR.parent / 'daily_data' / 'daily'

# 添加父目录到路径
sys.path.insert(0, str(SCRIPT_DIR.parent))

import pandas as pd
import pyarrow.parquet as pq
from mylib.get_local_data import get_local_data


def get_all_stocks():
    """从数据中获取全量股票列表"""
    files = sorted(DATA_DIR.glob('daily_*.parquet'))
    # 过滤汇总文件 (daily_YYYYMMDD_YYYYMMDD.parquet)
    files = [f for f in files if f.stem.split('_')[1].isdigit() and len(f.stem.split('_')[1]) == 8]
    if not files:
        return []

    t = pq.read_table(files[0], columns=['ts_code'])
    df = t.to_pandas()
    return df['ts_code'].unique().tolist()


def test_performance():
    """性能测试"""

    # 获取全量股票列表
    all_stocks = get_all_stocks()
    print(f"全量股票数量: {len(all_stocks)}")

    # 测试场景配置
    test_cases = {
        '1只股票_1天': {'sec_list': all_stocks[:1], 'start': '20250102', 'end': '20250102'},
        '1只股票_1个月': {'sec_list': all_stocks[:1], 'start': '20250101', 'end': '20250131'},
        '1只股票_1年': {'sec_list': all_stocks[:1], 'start': '20250101', 'end': '20251231'},
        '10只股票_1天': {'sec_list': all_stocks[:10], 'start': '20250102', 'end': '20250102'},
        '10只股票_1个月': {'sec_list': all_stocks[:10], 'start': '20250101', 'end': '20250131'},
        '10只股票_1年': {'sec_list': all_stocks[:10], 'start': '20250101', 'end': '20251231'},
        '100只股票_1天': {'sec_list': all_stocks[:100], 'start': '20250102', 'end': '20250102'},
        '100只股票_1个月': {'sec_list': all_stocks[:100], 'start': '20250101', 'end': '20250131'},
        '100只股票_1年': {'sec_list': all_stocks[:100], 'start': '20250101', 'end': '20251231'},
        '1000只股票_1天': {'sec_list': all_stocks[:1000], 'start': '20250102', 'end': '20250102'},
        '1000只股票_1个月': {'sec_list': all_stocks[:1000], 'start': '20250101', 'end': '20250131'},
        '1000只股票_1年': {'sec_list': all_stocks[:1000], 'start': '20250101', 'end': '20251231'},
        '5000只股票_1天': {'sec_list': all_stocks[:5000], 'start': '20250102', 'end': '20250102'},
        '5000只股票_1个月': {'sec_list': all_stocks[:5000], 'start': '20250101', 'end': '20250131'},
        '5000只股票_1年': {'sec_list': all_stocks[:5000], 'start': '20250101', 'end': '20251231'},
    }

    results = []

    for name, params in test_cases.items():
        print(f"测试中: {name}...")

        # 开始内存追踪
        tracemalloc.start()

        # 记录开始时间
        start_time = time.perf_counter()

        # 执行查询
        df = get_local_data(
            data_dir=str(DATA_DIR),
            **params
        )

        # 记录结束时间
        end_time = time.perf_counter()

        # 获取内存使用
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = end_time - start_time
        memory_mb = peak / 1024 / 1024

        result = {
            '测试场景': name,
            '耗时(秒)': round(elapsed, 3),
            '内存峰值(MB)': round(memory_mb, 2),
            '结果行数': df.shape[0],
            '结果列数': df.shape[1]
        }
        results.append(result)

        print(f"  耗时: {elapsed:.3f}s, 内存峰值: {memory_mb:.2f}MB, 数据形状: {df.shape}")

    # 输出汇总表格
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("性能测试汇总表")
    print("="*80)
    print(df_results.to_string(index=False))

    # 保存结果
    df_results.to_csv('performance_results.csv', index=False)
    print(f"\n结果已保存到 performance_results.csv")


if __name__ == '__main__':
    test_performance()
