"""
可视化配置工具模块
统一配置Matplotlib和其他可视化库
"""

import warnings


def setup_matplotlib(
    backend: str = 'Agg',
    font_family: str = 'DejaVu Sans',
    figure_size: tuple = (12, 6),
    dpi: int = 100,
    unicode_minus: bool = False
):
    """
    配置Matplotlib
    
    Args:
        backend: 后端，默认'Agg'（非交互式）
        font_family: 字体，默认'DejaVu Sans'
        figure_size: 图大小，默认(12, 6)
        dpi: DPI，默认100
        unicode_minus: 是否使用Unicode减号，默认False
    """
    import matplotlib
    import matplotlib.pyplot as plt
    
    # 设置后端
    if backend:
        matplotlib.use(backend)
    
    # 配置字体
    plt.rcParams['font.sans-serif'] = [font_family]
    plt.rcParams['axes.unicode_minus'] = unicode_minus
    
    # 配置图大小
    plt.rcParams['figure.figsize'] = figure_size
    plt.rcParams['figure.dpi'] = dpi
    
    return plt


def setup_warnings(filter_warnings: bool = True):
    """
    配置警告过滤
    
    Args:
        filter_warnings: 是否过滤警告，默认True
    """
    if filter_warnings:
        warnings.filterwarnings('ignore')
    else:
        warnings.filterwarnings('default')


def setup_all(
    backend: str = 'Agg',
    font_family: str = 'DejaVu Sans',
    filter_warnings: bool = True
):
    """
    统一设置所有可视化配置
    
    Args:
        backend: Matplotlib后端
        font_family: 字体
        filter_warnings: 是否过滤警告
        
    Returns:
        plt: matplotlib.pyplot对象
    """
    # 配置警告
    setup_warnings(filter_warnings)
    
    # 配置Matplotlib
    plt = setup_matplotlib(backend=backend, font_family=font_family)
    
    return plt


# 默认配置（向后兼容）
DEFAULT_FIGURE_SIZE = (12, 6)
DEFAULT_DPI = 100
DEFAULT_FONT_FAMILY = 'DejaVu Sans'
