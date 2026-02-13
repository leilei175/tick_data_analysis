from download_daily_basic import get_data, get_turnover_rate, get_close, get_pe

  # 通用函数
df = get_data('turnover_rate', ['600000.SH', '000001.SZ'], '20260101', '20260310')
data_df = df.pivot_table('turnover_rate',index='trade_date',columns='ts_code')
print(df)
print(data_df)
