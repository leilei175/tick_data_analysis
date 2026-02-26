"""
演示脚本 - 展示如何使用数据适配器获取股票数据
"""

from data_adapters import get_data, get_turnover_rate, get_close, get_pe

# 获取换手率数据
df = get_turnover_rate(['600000.SH', '000001.SZ'], '20260101', '20260310')
data_df = df.pivot_table('turnover_rate', index='trade_date', columns='ts_code')
print("换手率数据:")
print(df)
print("\n透视表:")
print(data_df)

# 获取收盘价
df_close = get_close(['600000.SH', '000001.SZ'], '20260101', '20260310')
print("\n收盘价数据:")
print(df_close)
