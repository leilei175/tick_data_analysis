"""
Tushare客户端统一管理模块
避免多处重复初始化Tushare
"""

import os
from pathlib import Path

# Tushare pro API对象，延迟初始化
_pro_api = None

def get_tushare_token(config_path: str = None) -> str:
    """
    获取Tushare Token
    
    优先级：
    1. 环境变量 TUSHARE_TOKEN
    2. 配置文件 ~/.tushare_token
    3. 项目 config.py
    4. 抛出异常
    
    Args:
        config_path: 配置文件路径，默认为项目根目录的config.py
        
    Returns:
        str: Token字符串
    """
    # 1. 环境变量
    token = os.environ.get('TUSHARE_TOKEN')
    if token:
        return token
    
    # 2. 配置文件 ~/.tushare_token
    token_path = Path.home() / '.tushare_token'
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token
    
    # 3. 项目 config.py
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.py"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        try:
            config_content = config_path.read_text()
            # 安全地执行配置文件
            config_namespace = {}
            exec(config_content, config_namespace)
            if 'tushare_tk' in config_namespace:
                return config_namespace['tushare_tk']
        except Exception:
            pass
    
    raise ValueError(
        "未找到 Tushare Token，请通过以下方式之一设置：\n"
        "1. 环境变量: export TUSHARE_TOKEN='your_token'\n"
        "2. 配置文件: echo 'your_token' > ~/.tushare_token\n"
        "3. 项目配置: 在 config.py 中定义 tushare_tk 变量"
    )


def init_tushare(token: str = None, config_path: str = None):
    """
    初始化Tushare
    
    Args:
        token: Tushare API Token，为None则自动获取
        config_path: 配置文件路径
        
    Returns:
        pro: Tushare pro API对象
    """
    global _pro_api
    
    if _pro_api is not None:
        return _pro_api
    
    try:
        import tushare as ts
    except ImportError:
        raise ImportError("未安装 tushare，请运行: pip install tushare")
    
    if token is None:
        token = get_tushare_token(config_path)
    
    ts.set_token(token)
    _pro_api = ts.pro_api()
    
    return _pro_api


def get_pro_api():
    """
    获取已初始化的Tushare pro API对象
    如果未初始化，会自动初始化
    
    Returns:
        pro: Tushare pro API对象
    """
    global _pro_api
    
    if _pro_api is None:
        return init_tushare()
    
    return _pro_api


def reset_pro_api():
    """
    重置Tushare pro API对象
    用于重新初始化或清理
    """
    global _pro_api
    _pro_api = None


def get_trading_days(start_date: str, end_date: str, exchange: str = 'SSE') -> list:
    """
    获取交易日列表
    
    Args:
        start_date: 开始日期，格式 YYYYMMDD
        end_date: 结束日期，格式 YYYYMMDD
        exchange: 交易所代码，默认'SSE'
        
    Returns:
        list: 交易日列表
    """
    pro = get_pro_api()
    
    try:
        df = pro.trade_cal(exchange=exchange, start_date=start_date, end_date=end_date)
        trading_days = df[df['is_open'] == 1]['cal_date'].tolist()
        return trading_days
    except Exception as e:
        print(f"获取交易日失败: {e}")
        return []


# 向后兼容的别名
set_token = init_tushare
