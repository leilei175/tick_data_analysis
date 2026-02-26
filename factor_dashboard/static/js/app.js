/**
 * 高频因子分析平台 - 通用JavaScript
 */

// 更新时钟
function updateTime() {
    const timeEl = document.getElementById('currentTime');
    if (timeEl) {
        const now = new Date();
        timeEl.textContent = now.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }
}

// 格式化数字
function formatNumber(num, decimals = 2) {
    if (Math.abs(num) >= 1000000) {
        return (num / 1000000).toFixed(decimals) + 'M';
    } else if (Math.abs(num) >= 1000) {
        return (num / 1000).toFixed(decimals) + 'K';
    }
    return num.toFixed(decimals);
}

// 格式化百分比
function formatPercent(num, decimals = 2) {
    return (num * 100).toFixed(decimals) + '%';
}

// 格式化基点
function formatBPS(num, decimals = 2) {
    return (num * 10000).toFixed(decimals) + ' bps';
}

// API请求封装
async function apiRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        return await response.json();
    } catch (error) {
        console.error('API Request Error:', error);
        return { status: 'error', message: error.message };
    }
}

// 加载状态
function showLoading(container) {
    container.innerHTML = '<div class="loading"><div class="loading-spinner"></div></div>';
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    // 更新时钟
    updateTime();
    setInterval(updateTime, 1000);

    // 添加动画
    const fadeElements = document.querySelectorAll('.fade-in');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    });

    fadeElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(10px)';
        el.style.transition = 'all 0.3s ease';
        observer.observe(el);
    });

    // Plotly全局配置
    Plotly.setPlotConfig({
        displayModeBar: false,
        responsive: true
    });
});

// 深色主题Plotly配置
function getPlotlyTheme() {
    return {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: {
            color: '#94a3b8',
            family: 'IBM Plex Sans, Noto Sans SC, sans-serif'
        }
    };
}

// 渲染统一因子按钮组，减少页面重复代码
function renderFactorButtons(containerId, factors, currentFactor, onSelectFnName) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = factors.map(factor => `
        <button class="factor-btn ${factor === currentFactor ? 'active' : ''}"
                onclick="window['${onSelectFnName}']('${factor}')">${factor}</button>
    `).join('');
}

// 创建热力图
function createHeatmap(data, xLabels, yLabels, options = {}) {
    return {
        type: 'heatmap',
        z: data,
        x: xLabels,
        y: yLabels,
        colorscale: options.colorscale || [
            [0, '#ef4444'],
            [0.5, '#1f2937'],
            [1, '#10b981']
        ],
        zmin: options.zmin || -1,
        zmax: options.zmax || 1,
        hoverongaps: false,
        ...options
    };
}

// 创建折线图
function createLineChart(data, options = {}) {
    return {
        type: 'scatter',
        mode: options.mode || 'lines',
        line: {
            color: options.color || '#3b82f6',
            width: options.width || 2
        },
        fill: options.fill || 'none',
        fillcolor: options.fillcolor || 'transparent',
        ...options
    };
}

// 创建柱状图
function createBarChart(data, options = {}) {
    return {
        type: 'bar',
        marker: {
            color: options.color || '#3b82f6',
            opacity: options.opacity || 0.9
        },
        ...options
    };
}

// 窗口调整时重绘图表
function redrawOnResize(chartId, redrawFn) {
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(redrawFn, 250);
    });
}

// 工具函数：防抖
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// 工具函数：节流
function throttle(func, limit) {
    let inThrottle;
    return function executedFunction(...args) {
        if (!inThrottle) {
            func(...args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}
