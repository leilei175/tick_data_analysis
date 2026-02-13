"""
高频因子计算模块
按日期批量计算所有股票的10个高频因子

因子列表:
1. order_imbalance - 订单不平衡因子
2. effective_spread - 有效价差因子
3. realized_volatility - 已实现波动率因子
4. bid_ask_spread - 买卖价差因子
5. vwap_deviation - VWAP偏离因子
6. price_momentum - 价格动量因子
7. trade_flow_intensity - 订单流强度因子
8. micro_price - 微价格因子
9. trade_imbalance - 交易失衡度因子
10. depth_imbalance - 深度失衡度因子

使用示例:
    from high_frequency_factors import calc_high_frequency

    # 计算单日所有股票
    df = calc_high_frequency('2025-12-01', 'all')

    # 计算单只股票
    df = calc_high_frequency('2025-12-01', '000001.SZ')

    # 计算多只股票
    df = calc_high_frequency('2025-12-01', ['000001.SZ', '000002.SZ'])

    # 批量计算日期范围
    results = calc_date_range('2025-12-01', '2025-12-31', 'all')
"""

import os
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import pandas as pd
import numpy as np
from datetime import date, datetime
import warnings
warnings.filterwarnings('ignore')

# 路径配置
TICK_DATA_DIR = "/data1/quant-data/tick_2026"
OUTPUT_DIR = "/data1/code_git/tick_data_analysis/factor/high_frequency"

# 因子列表
FACTORS = [
    'bid_ask_spread',       # 买卖价差
    'vwap_deviation',       # VWAP偏离度
    'trade_imbalance',      # 交易失衡度
    'order_imbalance',      # 订单失衡度
    'depth_imbalance',      # 深度失衡度
    'realized_volatility',  # 已实现波动率
    'effective_spread',     # 有效价差
    'micro_price',          # 微价格
    'price_momentum',       # 价格动量
    'trade_flow_intensity'  # 交易流强度
]


class HighFrequencyFactor:
    """高频因子计算器"""

    def __init__(self, tick_path: str = "./tick_2026"):
        """
        初始化

        Args:
            tick_path: tick数据根目录
        """
        self.tick_path = Path(tick_path)
        self.factors_cache = {}

    def _parse_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析时间列"""
        if 'time' in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['time'], unit='ns')
            df['time_str'] = df['datetime'].dt.strftime('%H:%M:%S.%f')[:-3]
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
        return df

    def _parse_price_level(self, level_str) -> np.ndarray:
        """解析5档价格数组"""
        if isinstance(level_str, str):
            return np.array(eval(level_str))
        return np.array(level_str)

    def _parse_vol_level(self, level_str) -> np.ndarray:
        """解析5档量数组"""
        if isinstance(level_str, str):
            return np.array(eval(level_str))
        return np.array(level_str)

    def _safe_divide(self, a: np.ndarray, b: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        """安全除法"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = a / b
            result = np.where(np.isfinite(result), result, fill_value)
        return result

    # ==================== 因子计算函数 ====================

    def factor_order_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """
        订单不平衡因子 (Order Imbalance)

        公式: OI = (BidVol - AskVol) / (BidVol + AskVol)

        解读:
        - OI > 0: 买方压力较大
        - OI < 0: 卖方压力较大
        """
        df = df.copy()

        bid_vol = np.vstack(df['bidVol'].apply(self._parse_vol_level))
        ask_vol = np.vstack(df['askVol'].apply(self._parse_vol_level))

        total_bid = bid_vol[:, 0]
        total_ask = ask_vol[:, 0]
        total = total_bid + total_ask

        oi = self._safe_divide(total_bid - total_ask, total, 0.0)

        return pd.Series(oi, index=df.index, name='order_imbalance')

    def factor_effective_spread(self, df: pd.DataFrame) -> pd.Series:
        """
        有效价差因子 (Effective Spread)

        公式: ES = 2 * |MidPrice - TradePrice| / MidPrice
        """
        df = df.copy()

        bid_price = np.vstack(df['bidPrice'].apply(self._parse_price_level))
        ask_price = np.vstack(df['askPrice'].apply(self._parse_price_level))

        bid1 = bid_price[:, 0]
        ask1 = ask_price[:, 0]
        mid_price = (bid1 + ask1) / 2

        trade_price = df['lastPrice'].values
        es = 2 * np.abs(mid_price - trade_price) / np.where(mid_price > 0, mid_price, 1)

        return pd.Series(es, index=df.index, name='effective_spread')

    def factor_realized_volatility(self, df: pd.DataFrame, window: int = 100) -> pd.Series:
        """
        已实现波动率因子 (Realized Volatility)

        公式: RV = sqrt(sum(returns^2))
        """
        df = df.copy()

        prices = df['lastPrice'].values
        returns = np.zeros(len(prices))
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns[i] = np.log(prices[i] / prices[i-1])

        rv = np.zeros(len(returns))
        actual_window = min(window, len(returns))
        for i in range(actual_window, len(returns)):
            rv[i] = np.sqrt(np.sum(returns[i-actual_window:i]**2))

        return pd.Series(rv, index=df.index, name='realized_volatility')

    def factor_bid_ask_spread(self, df: pd.DataFrame) -> pd.Series:
        """
        买卖价差因子 (Bid-Ask Spread)

        公式: Spread = Ask1 - Bid1
        """
        df = df.copy()

        bid_price = np.vstack(df['bidPrice'].apply(self._parse_price_level))
        ask_price = np.vstack(df['askPrice'].apply(self._parse_price_level))

        bid1 = bid_price[:, 0]
        ask1 = ask_price[:, 0]

        spread = ask1 - bid1

        return pd.Series(spread, index=df.index, name='bid_ask_spread')

    def factor_vwap_deviation(self, df: pd.DataFrame, window: int = 50) -> pd.Series:
        """
        VWAP偏离因子

        公式: Dev = (Price - VWAP) / VWAP
        """
        df = df.copy()

        prices = df['lastPrice'].values
        amounts = df['amount'].values
        volumes = df['volume'].values

        vwap = np.zeros(len(prices))
        for i in range(len(prices)):
            if i == 0:
                vwap[i] = prices[i]
            else:
                tick_amount = amounts[i] - amounts[i-1]
                tick_vol = volumes[i] - volumes[i-1]
                if i < window:
                    cum_amount = amounts[:i+1]
                    cum_vol = volumes[:i+1]
                else:
                    cum_amount = amounts[i-window+1:i+1]
                    cum_vol = volumes[i-window+1:i+1]

                if cum_vol.sum() > 0:
                    vwap[i] = cum_amount.sum() / cum_vol.sum()
                else:
                    vwap[i] = prices[i]

        deviation = (prices - vwap) / np.where(vwap > 0, vwap, 1)

        return pd.Series(deviation, index=df.index, name='vwap_deviation')

    def factor_price_momentum(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        价格动量因子

        公式: Mom = Price_t / Price_{t-n} - 1
        """
        df = df.copy()

        prices = df['lastPrice'].values
        momentum = np.zeros(len(prices))

        for i in range(window, len(prices)):
            if prices[i - window] > 0:
                momentum[i] = prices[i] / prices[i - window] - 1
            else:
                momentum[i] = 0

        return pd.Series(momentum, index=df.index, name='price_momentum')

    def factor_trade_flow_intensity(self, df: pd.DataFrame, window: int = 30) -> pd.Series:
        """
        订单流强度因子

        公式: TFI = tick_vol的滚动均值
        """
        df = df.copy()

        volumes = df['volume'].values
        intensity = np.zeros(len(volumes))

        for i in range(1, len(volumes)):
            tick_vol = volumes[i] - volumes[i-1]
            intensity[i] = tick_vol

        return pd.Series(intensity, index=df.index, name='trade_flow_intensity')

    def factor_micro_price(self, df: pd.DataFrame) -> pd.Series:
        """
        微价格因子 (Micro Price)

        公式: MicroPrice = (Bid1 * AskVol1 + Ask1 * BidVol1) / (BidVol1 + AskVol1)
        """
        df = df.copy()

        bid_price = np.vstack(df['bidPrice'].apply(self._parse_price_level))
        ask_price = np.vstack(df['askPrice'].apply(self._parse_price_level))
        bid_vol = np.vstack(df['bidVol'].apply(self._parse_vol_level))
        ask_vol = np.vstack(df['askVol'].apply(self._parse_vol_level))

        bid1 = bid_price[:, 0]
        ask1 = ask_price[:, 0]
        bv1 = bid_vol[:, 0]
        av1 = ask_vol[:, 0]

        total_vol = bv1 + av1
        micro_price = (bid1 * av1 + ask1 * bv1) / np.where(total_vol > 0, total_vol, 1)

        mid_price = (bid1 + ask1) / 2
        deviation = (micro_price - mid_price) / np.where(mid_price > 0, mid_price, 1)

        return pd.Series(deviation, index=df.index, name='micro_price')

    def factor_trade_imbalance(self, df: pd.DataFrame, window: int = 50) -> pd.Series:
        """
        交易订单不平衡因子

        基于成交价与买卖盘的关系判断主动买入力度
        - Price > MicroPrice: 主动买入
        - Price < MicroPrice: 主动卖出
        """
        df = df.copy()

        bid_price = np.vstack(df['bidPrice'].apply(self._parse_price_level))
        ask_price = np.vstack(df['askPrice'].apply(self._parse_price_level))
        prices = df['lastPrice'].values

        bid1 = bid_price[:, 0]
        ask1 = ask_price[:, 0]
        mid_price = (bid1 + ask1) / 2

        trade_imbalance = (prices - mid_price) / np.where(ask1 - bid1 > 0, ask1 - bid1, 1)

        return pd.Series(trade_imbalance, index=df.index, name='trade_imbalance')

    def factor_depth_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """
        深度不平衡因子

        公式: Depth_Imb = (BidVol1 - AskVol1) / (BidVol1 + AskVol1)
        """
        df = df.copy()

        bid_vol = np.vstack(df['bidVol'].apply(self._parse_vol_level))
        ask_vol = np.vstack(df['askVol'].apply(self._parse_vol_level))

        bv1 = bid_vol[:, 0].astype(float)
        av1 = ask_vol[:, 0].astype(float)

        total = bv1 + av1
        depth_imb = (bv1 - av1) / np.where(total > 0, total, 1) * 100

        return pd.Series(depth_imb, index=df.index, name='depth_imbalance')

    # ==================== 批量计算 ====================

    def compute_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有因子

        Args:
            df: tick数据

        Returns:
            包含所有因子的DataFrame
        """
        df = self._parse_time(df)
        df = df[df['lastPrice'] > 0].copy()

        if len(df) < 10:
            return df

        factors = pd.DataFrame(index=df.index)

        try:
            factors['order_imbalance'] = self.factor_order_imbalance(df)
            factors['effective_spread'] = self.factor_effective_spread(df)
            factors['realized_volatility'] = self.factor_realized_volatility(df, window=min(100, len(df)//2))
            factors['bid_ask_spread'] = self.factor_bid_ask_spread(df)
            factors['vwap_deviation'] = self.factor_vwap_deviation(df)
            factors['price_momentum'] = self.factor_price_momentum(df)
            factors['trade_flow_intensity'] = self.factor_trade_flow_intensity(df)
            factors['micro_price'] = self.factor_micro_price(df)
            factors['trade_imbalance'] = self.factor_trade_imbalance(df)
            factors['depth_imbalance'] = self.factor_depth_imbalance(df)
        except Exception as e:
            print(f"Error computing factors: {e}")
            import traceback
            traceback.print_exc()
            return df

        result = pd.concat([df[['time', 'datetime', 'lastPrice', 'volume', 'amount', 'stock_code']], factors], axis=1)
        return result

    def compute_daily_factors(
        self,
        target_date: date,
        stock_codes: Optional[List[str]] = None,
        max_stocks: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        计算指定日期的因子

        Args:
            target_date: 目标日期
            stock_codes: 股票列表，为None则计算所有
            max_stocks: 最大股票数量

        Returns:
            日频因子数据
        """
        date_str = f"{target_date.year:04d}/{target_date.month:02d}/{target_date.day:02d}"
        all_factors = []

        if stock_codes is None:
            pattern = f"{self.tick_path}/{date_str}/*.parquet"
            files = list(Path(pattern).glob("*.parquet"))
        else:
            files = [Path(f"{self.tick_path}/{date_str}/{code}.parquet") for code in stock_codes]

        if max_stocks:
            files = files[:max_stocks]

        print(f"Processing {len(files)} files for {date_str}...")

        for idx, f in enumerate(files):
            if not f.exists():
                continue

            try:
                df = pd.read_parquet(f)
                df['stock_code'] = f.stem

                factors_df = self.compute_all_factors(df)
                if not factors_df.empty:
                    factors_df['date'] = target_date
                    all_factors.append(factors_df)

            except Exception as e:
                print(f"Error processing {f.stem}: {e}")
                continue

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(files)} files...")

        if all_factors:
            result = pd.concat(all_factors, ignore_index=True)
            return result.sort_values(['stock_code', 'datetime']).reset_index(drop=True)
        return pd.DataFrame()

    def save_daily_factors(
        self,
        factors_df: pd.DataFrame,
        target_date: date,
        output_dir: str = "./factor/daily",
    ) -> str:
        """
        保存日频因子

        Args:
            factors_df: 因子数据
            target_date: 日期
            output_dir: 输出目录

        Returns:
            保存的文件路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        date_str = target_date.strftime("%Y%m%d")
        file_path = output_path / f"factors_{date_str}.parquet"

        factors_df = factors_df.reset_index(drop=True)
        factors_df.to_parquet(file_path, index=False)

        print(f"Saved factors to {file_path}")
        return str(file_path)

    def get_factor_stats(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        获取因子统计信息
        """
        factor_cols = [c for c in factors_df.columns if c not in
                      ['time', 'datetime', 'lastPrice', 'volume', 'amount', 'stock_code', 'date']]

        stats = []
        for col in factor_cols:
            data = factors_df[col].dropna()
            stats.append({
                'factor': col,
                'count': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                '25%': data.quantile(0.25),
                '50%': data.median(),
                '75%': data.quantile(0.75),
                'max': data.max()
            })

        return pd.DataFrame(stats)


# ==================== 便捷函数 ====================

def calc_single_stock_factors(tick_df: pd.DataFrame) -> Dict[str, float]:
    """
    计算单只股票的10个高频因子（日均值）

    Args:
        tick_df: 单只股票的tick数据DataFrame

    Returns:
        因子名称->因子日均值的字典
    """
    if tick_df.empty or len(tick_df) < 2:
        return {f: np.nan for f in FACTORS}

    engine = HighFrequencyFactor()
    tick_df = tick_df[tick_df['lastPrice'] > 0].copy()

    if len(tick_df) < 10:
        return {f: np.nan for f in FACTORS}

    try:
        result = engine.compute_all_factors(tick_df)

        factors = {}
        for factor in FACTORS:
            if factor in result.columns:
                factors[factor] = result[factor].mean()
            else:
                factors[factor] = np.nan

        return factors
    except Exception as e:
        print(f"Error: {e}")
        return {f: np.nan for f in FACTORS}


def calc_high_frequency(date: str, stock_code: Union[str, List[str]] = 'all',
                        base_dir: str = TICK_DATA_DIR,
                        output_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    """
    计算指定日期的高频因子数据

    Args:
        date: 日期，格式 'YYYY-MM-DD' 或 'YYYY/MM/DD' 或 'YYYYMMDD'
        stock_code: 股票代码
            - 'all': 计算所有股票
            - 具体代码如 '000001.SZ': 只计算该股票
            - 列表如 ['000001.SZ', '000002.SZ']: 计算指定股票
        base_dir: tick数据根目录
        output_dir: 输出目录

    Returns:
        DataFrame: 包含股票代码和10个因子的DataFrame

    Examples:
        >>> # 计算2025年12月1日所有股票
        >>> df = calc_high_frequency('2025-12-01', 'all')

        >>> # 计算单只股票
        >>> df = calc_high_frequency('2025-12-01', '000001.SZ')

        >>> # 计算多只股票
        >>> df = calc_high_frequency('2025-12-01', ['000001.SZ', '000002.SZ'])

        >>> # 使用不同路径
        >>> df = calc_high_frequency('2025-12-01', 'all', '/path/to/tick', '/path/to/output')
    """
    # 解析日期
    date_str = date.replace('-', '').replace('/', '')
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]

    # 构建tick数据路径
    tick_path = Path(base_dir) / year / month / day

    if not tick_path.exists():
        raise FileNotFoundError(f"Tick数据目录不存在: {tick_path}")

    # 获取要处理的股票列表
    if stock_code == 'all':
        stock_files = list(tick_path.glob("*.parquet"))
    elif isinstance(stock_code, list):
        stock_files = [tick_path / f"{s}.parquet" for s in stock_code if (tick_path / f"{s}.parquet").exists()]
    else:
        stock_files = [tick_path / f"{stock_code}.parquet"]

    if not stock_files:
        raise FileNotFoundError(f"没有找到股票tick数据: {date}")

    print(f"处理日期: {date}, 股票数: {len(stock_files)}")

    # 计算每个股票的因子
    engine = HighFrequencyFactor(base_dir)
    results = []
    processed = 0
    errors = 0

    for idx, tick_file in enumerate(stock_files, 1):
        stock_code_result = tick_file.stem

        try:
            # 读取tick数据
            tick_df = pd.read_parquet(tick_file)
            tick_df['stock_code'] = stock_code_result

            if tick_df.empty or len(tick_df) < 10:
                continue

            # 计算因子
            factors = engine.compute_all_factors(tick_df)

            if factors.empty or len(factors) < 10:
                continue

            # 计算日均值
            row = {
                'date': f"{year}-{month}-{day}",
                'stock_code': stock_code_result
            }
            for factor in FACTORS:
                if factor in factors.columns:
                    row[factor] = factors[factor].mean()
                else:
                    row[factor] = np.nan

            results.append(row)
            processed += 1

            if idx % 100 == 0 or idx == len(stock_files):
                print(f"  [{idx}/{len(stock_files)}] 完成: {stock_code_result} (已处理: {processed})")

        except Exception as e:
            errors += 1
            if errors <= 5:  # 只打印前5个错误
                print(f"  [{idx}/{len(stock_files)}] 错误: {stock_code_result} - {e}")
            continue

    if not results:
        raise ValueError(f"没有成功计算任何股票的因子: {date}")

    # 合并结果
    result_df = pd.DataFrame(results)

    # 调整列顺序
    cols = ['date', 'stock_code'] + [f for f in FACTORS if f in result_df.columns]
    result_df = result_df[cols]

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 保存到文件
    output_file = Path(output_dir) / f"{year}_{month}_{day}.parquet"
    result_df.to_parquet(output_file, engine='pyarrow')

    print(f"\n完成! 保存到: {output_file}")
    print(f"数据规模: {len(result_df)} 股票 x {len(FACTORS)} 因子")
    if errors > 0:
        print(f"错误数: {errors}")

    return result_df


def calc_date_range(start_date: str, end_date: str, stock_code: Union[str, List[str]] = 'all',
                    base_dir: str = TICK_DATA_DIR,
                    output_dir: str = OUTPUT_DIR,
                    verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    计算日期范围内所有交易日的高频因子

    Args:
        start_date: 起始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        stock_code: 股票代码 ('all' 或具体代码列表)
        base_dir: tick数据根目录
        output_dir: 输出目录
        verbose: 是否打印进度

    Returns:
        dict: 日期->DataFrame的字典

    Examples:
        >>> # 批量计算整个月
        >>> results = calc_date_range('2025-12-01', '2025-12-31', 'all')

        >>> # 只计算指定股票
        >>> results = calc_date_range('2025-12-01', '2025-12-31', '000001.SZ')
    """
    from datetime import timedelta

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    current = start
    results = {}
    success_count = 0
    error_count = 0

    while current <= end:
        date_str = current.strftime('%Y-%m-%d')

        if verbose:
            print(f"\n{'='*50}")
            print(f"处理日期: {date_str}")

        try:
            df = calc_high_frequency(date_str, stock_code, base_dir, output_dir)
            results[date_str] = df
            success_count += 1
        except FileNotFoundError:
            if verbose:
                print(f"  跳过: {date_str} (无数据)")
        except Exception as e:
            error_count += 1
            if verbose:
                print(f"  错误: {date_str} - {e}")

        current += timedelta(days=1)

    if verbose:
        print(f"\n{'='*50}")
        print(f"处理完成!")
        print(f"  成功: {success_count} 天")
        print(f"  跳过/错误: {error_count} 天")

    return results


# ==================== 主程序 ====================

def compute_all_dates():
    """计算2月份所有日期的因子"""
    from tick_reader import TickDataReader

    factor_engine = HighFrequencyFactor("./tick_2026")
    reader = TickDataReader("./tick_2026")

    dates = [
        date(2026, 2, 2),
        date(2026, 2, 3),
        date(2026, 2, 4),
        date(2026, 2, 5),
        date(2026, 2, 6),
    ]

    for target_date in dates:
        print(f"\n{'='*60}")
        print(f"Processing {target_date}")
        print('='*60)

        stocks = reader.get_available_stocks(target_date)
        print(f"Available stocks: {len(stocks)}")

        if len(stocks) == 0:
            print(f"No data for {target_date}")
            continue

        factors_df = factor_engine.compute_daily_factors(target_date, stocks, max_stocks=100)

        if not factors_df.empty:
            factor_engine.save_daily_factors(factors_df, target_date)
            print("\nFactor Statistics:")
            print(factor_engine.get_factor_stats(factors_df).to_string(index=False))
        else:
            print(f"No factors computed for {target_date}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='计算高频因子')
    parser.add_argument('--date', '-d', help='日期 (YYYY-MM-DD)')
    parser.add_argument('--stock', '-s', default='all', help='股票代码 (all 或具体代码)')
    parser.add_argument('--output', '-o', default=OUTPUT_DIR, help='输出目录')
    parser.add_argument('--start', help='起始日期 (批量模式)')
    parser.add_argument('--end', help='结束日期 (批量模式)')

    args = parser.parse_args()

    if args.start and args.end:
        # 批量模式
        calc_date_range(args.start, args.end, args.stock, TICK_DATA_DIR, args.output)
    elif args.date:
        # 单日模式
        calc_high_frequency(args.date, args.stock, TICK_DATA_DIR, args.output)
    else:
        parser.error("请提供 --date 或 --start/--end 参数")
